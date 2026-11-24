"""
대리모델: 설계 → 시뮬레이션 결과 예측 (PyTorch 신경망)
학습 (optimizer 폴더에서)
    python -m rl.surrogate
결과: rl/surrogate.pt (앙상블), rl/surrogate_meta.json (검증 지표)
"""

import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import GA
import constrain as cs
from rl import features as ft

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "surrogate.pt")
META_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "surrogate_meta.json")   # 검증 지표 (웹 표시용)
PIPE_FEATURES = ft.PIPE_FEATURES
FEATURES = ft.FEATURE_VARS + ["pipe_" + f for f in PIPE_FEATURES]
TARGETS = ["isp", "impulse", "thrust_avg", "max_tank_pressure", "max_chamber_pressure", "dp_ratio_min"]
# 예전 결과에는 없는 출력 → 값이 있는 행에서만 배운다 (rl/backfill.py 로 보충)
OPTIONAL_TARGETS = {"dp_ratio_min"}
# 차압 비율은 꼬리가 길다 (드물게 4~10). 제약 판정(하한 0.2) 근처만 중요하므로 학습 시 이 값에서 자른다
DP_CLIP = 1.0

EPOCHS = 300
BATCH_SIZE = 256
LR = 1e-3
N_MODELS = 5          # 앙상블 크기 (예측 불확실성 추정용)


features = ft.design_features


def _results_rows(pipes):
    """MDO core 결과 (results/*/candidates.csv). 예전 형식 (슈퍼차징 압력 없음, 헬륨 0.025 kg) 은 rl.features 로 변환"""
    fixed = GA.load_fixed_values()
    frames = []
    for f in glob.glob(os.path.join(GA.RESULTS_DIR, "*", "candidates.csv")):
        d = pd.read_csv(f)
        d = d[(d["termination_code"].notna() | d["error"].notna()) & d["pipe_id"].isin(pipes)]
        if d.empty:
            continue
        if "supercharge_pressure" not in d:
            rows = [ft.legacy_design({v: r[v] for v in GA.DESIGN_VARS if v in r} |
                                     {"combution_chamber_length": r["combution_chamber_length"]},
                                     fixed, pipes[r["pipe_id"]]["inner_diameter"]) for r in d.to_dict("records")]
            d = d.assign(supercharge_pressure=[r["supercharge_pressure"] for r in rows],
                         pressurant_M=[r["pressurant_M"] for r in rows])
        if "temperature" not in d:                       # 온도 기록 전 결과는 모두 258 K
            d = d.assign(temperature=ft.LEGACY_TEMPERATURE)
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _ts_rows(pipes, simt=None):
    """
    시계열 데이터 (rl/ts_data, simt 40 s) → MDO 시뮬레이션 시간 (fixedvalue simt, 곡선 길이 40 s 까지) 으로 자른 요약값
    simt 안에 끝나지 않으면 core 처럼 TIME_LIMIT_REACHED (completed = False)
    """
    from rl import ts_surrogate as ts
    simt = ts.surrogate_simt(simt)
    X, S, ok, term, Y, kinds = ts.load_data()
    t_end = Y[:, ts.SCALARS.index("t_end")]
    recs = []
    for i in range(len(X)):
        r = {v: float(X[i, k]) for k, v in enumerate(ft.FEATURE_VARS)}
        r.update({"pipe_" + f: float(X[i, len(ft.FEATURE_VARS) + k]) for k, f in enumerate(PIPE_FEATURES)})
        if not ok[i]:
            r.update(error="core error", termination_code=None)
        elif t_end[i] > simt - ts.DT / 2:
            r.update(error=None, termination_code="TIME_LIMIT_REACHED")
        else:
            der = ts.derive(S[i], simt)
            r.update(error=None, termination_code="OXIDIZER_FULLY_DEPLETED",
                     **{t: der[t] for t in TARGETS})
        recs.append(r)
    df = pd.DataFrame(recs)
    # 파이프 치수는 데이터에 있는 그대로 (파이프 DB 가 바뀌어도 예전 데이터를 버리지 않게) — id 는 치수로 만든 이름
    df["pipe_id"] = ["ts:" + "/".join(f"{v:.5g}" for v in row) for row in df[["pipe_" + f for f in PIPE_FEATURES]].to_numpy()]
    return df


def input_bounds():
    """Surrogate 페이지 입력 범위: 설계 변수 + 초기 온도 (범용 온도 모델), 슈퍼차징은 모든 온도를 덮는 범위"""
    from rl import ts_data
    lo = ts_data.p_sat(ft.TEMP_RANGE[0])
    return {**GA.BOUNDS, "supercharge_pressure": (lo, ts_data.P_MAX), "temperature": ft.TEMP_RANGE}


def load_dataset(use_ts=True):
    pipes = {p["pipe_id"]: p for p in GA.load_candidate_pipes()}
    df = _results_rows(pipes)
    for f in PIPE_FEATURES:
        df["pipe_" + f] = df["pipe_id"].map(lambda i: pipes[i][f])
    for t in OPTIONAL_TARGETS:
        if t not in df:
            df[t] = np.nan
    if use_ts:
        df = pd.concat([df, _ts_rows(pipes)], ignore_index=True)
    # 같은 설계가 여러 번 있으면 선택 출력까지 있는 행을 남긴다
    df = df.assign(_n=df[list(OPTIONAL_TARGETS)].notna().sum(axis=1)).sort_values("_n", ascending=False)
    df = df.drop_duplicates(subset=["pipe_id"] + ft.FEATURE_VARS).drop(columns="_n").copy()
    df["completed"] = df["error"].isna() & (df["termination_code"] != "TIME_LIMIT_REACHED")
    required = [t for t in TARGETS if t not in OPTIONAL_TARGETS]
    df = df[~(df["completed"] & df[required].isna().any(axis=1))]      # 필수 출력이 빈 행은 버림
    df["dp_ratio_min"] = df["dp_ratio_min"].clip(upper=DP_CLIP)
    return df


class SurrogateNet(nn.Module):
    """입력 (rl.features) → completed 로짓 1개 + 수치 6개. 몸통은 함께 쓰고 머리만 둘."""

    def __init__(self, n_in=len(FEATURES), n_out=len(TARGETS), hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.cls_head = nn.Linear(hidden, 1)       # 정상 종료 여부 (분류)
        self.reg_head = nn.Linear(hidden, n_out)   # 5개 수치 (회귀)
        # 정규화 통계. buffer 로 등록하면 학습은 안 되지만 state_dict 에 같이 저장된다
        self.register_buffer("x_mean", torch.zeros(n_in))
        self.register_buffer("x_std", torch.ones(n_in))
        self.register_buffer("y_mean", torch.zeros(n_out))
        self.register_buffer("y_std", torch.ones(n_out))

    def forward(self, x):
        h = self.body((x - self.x_mean) / self.x_std)
        return self.cls_head(h).squeeze(-1), self.reg_head(h)      # 회귀 출력은 정규화된 단위

    @torch.no_grad()
    def predict(self, x):
        logit, y = self(x)
        return logit > 0, y * self.y_std + self.y_mean               # 원래 단위로 되돌림


def _fit(X, c, Y, M, tr, device, seed):
    """신경망 1개 학습. tr 에서 복원 추출(bootstrap)한 행으로 배워 앙상블 구성원끼리 달라지게 한다."""
    torch.manual_seed(seed)
    net = SurrogateNet().to(device)
    ok = tr[c[tr] == 1]
    net.x_mean.copy_(X[tr].mean(0)); net.x_std.copy_(X[tr].std(0) + 1e-8)
    m = M[ok]                                                          # 출력별로 값이 있는 행만으로 통계
    mean = (Y[ok] * m).sum(0) / m.sum(0).clamp(min=1)
    std = (((Y[ok] - mean) ** 2 * m).sum(0) / m.sum(0).clamp(min=1)).sqrt()
    net.y_mean.copy_(mean); net.y_std.copy_(std + 1e-8)
    boot = tr[torch.randint(len(tr), (len(tr),), device=device)]

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()
    for epoch in range(1, EPOCHS + 1):
        net.train()
        for idx in boot[torch.randperm(len(boot), device=device)].split(BATCH_SIZE):
            logit, y = net(X[idx])
            # 수치는 정상 종료 행 + 값이 있는 출력만 배운다. 마스크 곱으로 처리해 GPU→CPU 동기화를 없앤다
            w = c[idx].unsqueeze(-1) * M[idx]
            sq = (y - (Y[idx] - net.y_mean) / net.y_std) ** 2 * w
            loss = bce(logit, c[idx]) + sq.sum() / w.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 100 == 0:
            print(f"    epoch {epoch:>3}  loss {loss.item():.4f}")
    return net.eval()


def train(device=None, n_models=N_MODELS):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)
    df = load_dataset()
    print(f"데이터 {len(df)}건 (정상 종료 {int(df['completed'].sum())}건), 파이프 {df['pipe_id'].nunique()}종, 장치 {device}")

    X = torch.tensor(df[FEATURES].to_numpy(np.float32))
    c = torch.tensor(df["completed"].to_numpy(np.float32))
    M = torch.tensor(df[TARGETS].notna().to_numpy(np.float32))     # 출력별 값 존재 여부
    Y = torch.tensor(df[TARGETS].fillna(0).to_numpy(np.float32))   # 빈 값(0)은 M 으로 손실에서 빠진다
    print("  출력별 학습 행: " + ", ".join(f"{t} {int(n)}" for t, n in zip(TARGETS, (M * c[:, None]).sum(0).tolist())))

    X, c, Y, M = X.to(device), c.to(device), Y.to(device), M.to(device)
    perm = torch.randperm(len(X), device=device)                      # 80% 학습 / 20% 시험
    n_test = len(X) // 5
    test, tr = perm[:n_test], perm[n_test:]

    nets = []
    for k in range(n_models):
        print(f"  모델 {k + 1}/{n_models}")
        nets.append(_fit(X, c, Y, M, tr, device, seed=k))

    p_ok, pred, _ = SurrogateEnsemble(nets).predict_stats(X[test])
    acc = ((p_ok > 0.5) == (c[test] == 1)).float().mean().item()
    print(f"  completed  정확도 {acc:.3f}")
    ok_test = c[test] == 1
    actual, pred = Y[test][ok_test].cpu().numpy(), pred[ok_test].cpu().numpy()
    avail = M[test][ok_test].cpu().numpy() > 0
    metrics = {"acc_completed": acc}
    parity = {}
    for k, target in enumerate(TARGETS):
        a, p = actual[avail[:, k], k], pred[avail[:, k], k]
        parity[target] = [[float(x), float(y)] for x, y in zip(a[:400], p[:400])]
        if len(a) < 2:
            continue
        metrics[target] = {
            "mae": float(np.abs(p - a).mean()),
            # 0 에 가까운 실제값(연소 실패 직전 설계 등)이 MAPE 를 폭주시키지 않도록 분모 하한 = 평균 크기의 5%
            "mape": float(np.mean(np.abs(p - a) / np.maximum(np.abs(a), 0.05 * np.abs(a).mean())) * 100),
            "r2": float(1 - ((p - a) ** 2).sum() / ((a - a.mean()) ** 2).sum()),
        }
        print(f"  {target:<22} 평균 절대 오차 {metrics[target]['mae']:.4g}  R² {metrics[target]['r2']:.4f}")

    torch.save([net.cpu().state_dict() for net in nets], MODEL_PATH)
    meta = {
        "features": FEATURES, "design_vars": GA.DESIGN_VARS, "derived_vars": GA.DERIVED_VARS, "pipe_features": PIPE_FEATURES,
        "targets": TARGETS, "bounds": input_bounds(), "int_vars": sorted(GA.INT_VARS),
        "n_samples": len(df), "n_train": len(tr), "n_test": n_test, "n_models": n_models,
        "hidden": nets[0].cls_head.in_features, "epochs": EPOCHS,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "parity": parity,
        "n_rows": {t: int(n) for t, n in zip(TARGETS, (M[test][ok_test].sum(0)).tolist())},
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"저장: {MODEL_PATH}, {META_PATH}")


class SurrogateEnsemble:
    """신경망 여러 개의 평균 = 예측, 표준편차 = 불확실성"""

    def __init__(self, nets):
        self.nets = nets

    @classmethod
    def load(cls, path=MODEL_PATH):
        meta_path = path.replace(".pt", "_meta.json")
        try:
            with open(meta_path, encoding="utf-8") as f:
                feats = json.load(f).get("features")
        except (OSError, ValueError):
            feats = None
        if feats != FEATURES:
            raise ValueError(f"{os.path.basename(path)} 는 예전 입력 구성으로 학습된 대리모델입니다. "
                             "python -m rl.surrogate 로 다시 학습하세요.")
        states = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(states, list):                   # 예전 형식 (신경망 1개)
            states = [states]
        nets = []
        for state in states:
            net = SurrogateNet(n_out=state["reg_head.weight"].shape[0])   # 예전 모델은 출력이 적다
            net.load_state_dict(state)
            nets.append(net.eval())
        return cls(nets)

    @torch.no_grad()
    def predict_stats(self, x):
        """x: (N, N_FEATURES) → (정상 종료 확률, 평균, 표준편차) — 모두 원래 단위"""
        probs, ys = [], []
        for net in self.nets:
            logit, y = net(x)
            probs.append(torch.sigmoid(logit))
            ys.append(y * net.y_std + net.y_mean)
        ys = torch.stack(ys)
        std = ys.std(0, unbiased=False) if len(self.nets) > 1 else torch.zeros_like(ys[0])
        return torch.stack(probs).mean(0), ys.mean(0), std


class SurrogateSimulator:
    """GA.simulate() 대신 쓰는 대리모델. GA.run_ga(surrogate=SurrogateSimulator()) 로 사용"""

    def __init__(self, model_path=MODEL_PATH):
        # 한 번에 수십 개만 예측 → 스레드 1개면 충분. 기본값(코어 수)이면 스레드들이 CPU 를 붙잡아
        # 탐색 중 백그라운드로 도는 core 검증 워커가 느려진다
        torch.set_num_threads(1)
        self.model = SurrogateEnsemble.load(model_path)
        self.bias = {}       # 출력별 보정값 (core - 예측). GA 검증 실패 시 update_bias 로 갱신

    def update_bias(self, candidates, checked):
        """
        같은 설계의 예측(candidates[i]["sim"]) 과 core 결과(checked[i]["sim"]) 차이의 평균만큼 보정값을 더한다.
        특정 영역에서 대리모델이 한쪽으로 치우친 경우 (예: 평균 추력을 계속 2% 높게) 를 바로잡는다.
        """
        residuals = {}
        for ev_pred, ev_core in zip(candidates, checked):
            core = ev_core["sim"] or {}
            if core.get("error") or core.get("termination_code") == "TIME_LIMIT_REACHED":
                continue
            for t in TARGETS:
                actual = cs.injector_dp_ratio(core) if t == "dp_ratio_min" else core.get(t)
                if actual is None or t not in ev_pred["sim"]:
                    continue
                if t == "dp_ratio_min":
                    actual = min(actual, DP_CLIP)
                residuals.setdefault(t, []).append(actual - ev_pred["sim"][t])
        for t, r in residuals.items():
            self.bias[t] = self.bias.get(t, 0.0) + float(np.mean(r))
        return dict(self.bias)

    def simulate_batch(self, designs, pipes):
        """designs/pipes: GA.decode() 결과 리스트 → GA.simulate() 와 같은 키의 결과 dict 리스트"""
        t0 = time.time()
        X = torch.tensor([features(d, p) for d, p in zip(designs, pipes)], dtype=torch.float32)
        p_ok, mean, std = self.model.predict_stats(X)
        wall = (time.time() - t0) / max(len(designs), 1)
        sims = []
        for p, row, sd in zip(p_ok.tolist(), mean.tolist(), std.tolist()):
            sim = dict(zip(TARGETS, row))
            for t, b in self.bias.items():
                sim[t] += b
            sim.update({
                "error": None,
                # 정상 종료 확률이 낮으면 core 의 "연소 미완료" 와 같은 코드로 → SimCompleted 제약에서 걸러짐
                "termination_code": "SURROGATE" if p > 0.5 else "TIME_LIMIT_REACHED",
                "burn_time": sim["impulse"] / sim["thrust_avg"] if sim["thrust_avg"] > 0 else 0.0,
                "p_completed": p,
                "isp_std": sd[0],
                "wall_time": wall,
            })
            sims.append(sim)
        return sims


class SurrogateEvaluator:
    """RealEvaluator 와 같은 모양. 시뮬레이션 대신 대리모델 예측을 쓴다."""

    def __init__(self, model_path=MODEL_PATH):
        # 한 번에 32개 정도만 예측하므로 CPU 가 GPU 보다 빠르다 (병렬 환경마다 CUDA 를 잡지도 않음)
        self.model = SurrogateEnsemble.load(model_path)
        self.pipes = GA.load_candidate_pipes()
        self.fixed = GA.load_fixed_values()
        self.caches = {}
        self.simulated = 0

    def _predict(self, X):
        p_ok, preds, _ = self.model.predict_stats(torch.tensor(X, dtype=torch.float32))
        sims = []
        for p, row in zip(p_ok.tolist(), preds.tolist()):
            sim = dict(zip(TARGETS, row))
            sim["error"] = None
            sim["termination_code"] = "SURROGATE" if p > 0.5 else "TIME_LIMIT_REACHED"
            sims.append(sim)
        return sims

    def __call__(self, population, targets):
        cache = self.caches.setdefault(tuple(targets), {})
        keys, todo = [], []
        for g in population:
            design, pipe = GA.decode(g, self.pipes, self.fixed)
            key = GA.design_key(design, pipe)
            keys.append(key)
            if key in cache:
                continue
            pre, tank = cs.pre_sim_constraints(design, pipe)
            ok, violation, failed = cs.check(pre)
            cache[key] = {"design": design, "pipe": pipe, "tank": tank, "sim": None,
                          "feasible": False, "violation": violation, "failed": failed}
            if ok:
                todo.append(key)
        if todo:
            sims = self._predict([features(cache[k]["design"], cache[k]["pipe"]) for k in todo])
            for key, sim in zip(todo, sims):
                ev = cache[key]
                # GA 의 대리모델 평가와 같은 안전 여유 (학습 환경 = 실제 사용 환경)
                ok, violation, failed = cs.check(cs.post_sim_constraints(ev["design"], ev["pipe"], sim, *targets,
                                                                         margin=cs.SURROGATE_MARGIN))
                ev.update(sim=sim, feasible=ok, violation=violation, failed=failed)
            self.simulated += len(todo)
        return [dict(cache[k], genome=g) for k, g in zip(keys, population)]


if __name__ == "__main__":
    train()
