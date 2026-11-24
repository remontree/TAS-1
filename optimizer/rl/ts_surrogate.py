"""
시계열 대리모델: 설계 → 연소 전 과정 (추력·챔버압·탱크압·산화제/연료 질량유량, 0~40 s, 0.1 s 간격)
Transformer: 설계 토큰 1개 + 시간 패치 토큰 50개(0.8 s 씩)를 함께 self-attention → 패치마다 8스텝 × 5채널 출력.
설계 토큰에서 core 실패 확률, 종료 유형, 요약값(Isp·임펄스·평균 추력 ...)도 함께 예측한다.

학습 (optimizer 폴더에서, rl/ts_data/* 가 있어야 함 — python -m rl.ts_data)
    python -m rl.ts_surrogate [--epochs 200 --models 3]
결과: rl/ts_surrogate.pt (앙상블), rl/ts_surrogate_meta.json (검증 지표)

사용
    sim = TSSurrogateSimulator()          # GA.run_ga(surrogate=...) 에 그대로 쓸 수 있음 (simulate_batch)
    out = TSSurrogate.load().predict(X)   # X: (N, rl.features.N_FEATURES) → series (N, 6, 400), scalars, p_ok ...

입력 = rl.features (설계 변수 14 + 챔버 길이·헬륨 질량 + 파이프 3). 예전 데이터(헬륨 0.025 kg 고정)도 변환해 함께 쓴다.
채널 6번째 dp_ratio = 인젝터 차압 비율 (Pt − Pc)/Pc (연소 중, 연소 밖은 1). 차압 제약의 최솟값은 액체 소진 직전에
나오므로 곡선으로 직접 배운다 (탱크압·챔버압을 따로 예측해 빼면 오차가 커진다).
예측 차압은 보수적으로: min(요약 머리, 곡선 최솟값) − DP_STD_K × 앙상블 표준편차 (sims_from_output).
"""

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn

import GA
from rl import features as ft

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "ts_surrogate.pt")
META_PATH = os.path.join(HERE, "ts_surrogate_meta.json")
DATA_DIR = os.path.join(HERE, "ts_data")

CHANNELS = GA.SERIES_CHANNELS + ["dp_ratio"]   # thrust, chamber_pressure, tank_pressure, mox_dot, mf_dot, (Pt−Pc)/Pc
DP_FILL = 1.0                            # 연소 밖 dp_ratio 채널 값 (곡선 최솟값이 연소 경계의 0 으로 끌려가지 않게)
DP_CH_W = 3.0                            # 시계열 손실에서 dp_ratio 채널 가중치
DP_STD_K = 1.0                           # 보수적 차압 = min(머리, 곡선) − k × 앙상블 표준편차
DT = GA.SERIES_DT
N_T = 400                                # 40 s
HORIZON = N_T * DT                       # 대리모델이 보는 시간 (곡선 길이)


def surrogate_simt(simt=None):
    """MDO 시뮬레이션 시간 → 대리모델 판정 시간. core simt (600 s) 가 곡선보다 길면 곡선 끝 (40 s) 까지만 본다
    — 그보다 오래 타는 설계는 시간 초과로 본다 (보수적, 목표 추력 범위에서는 나오지 않는 설계)"""
    simt = simt or GA.load_fixed_values().get("simt", 20.0)
    return min(float(simt), HORIZON)
PATCH = 8
N_TOK = N_T // PATCH
SCALARS = ["isp", "impulse", "thrust_avg", "burn_time", "max_tank_pressure", "max_chamber_pressure", "dp_ratio_min",
           "t_end"]
SCALARS_V2 = SCALARS[:7]                 # t_end 이 없던 예전 모델 (blob 에 scalars 가 없으면)
ISP_CLIP = (0.0, 260.0)                  # Isp 목표값 범위 (음수 Isp = 역추력 설계, 손실을 키우지 않게)
HIGH_ISP = 170.0                         # 이보다 높은 설계 (최적화가 가는 곳) 의 손실 가중치 ×HIGH_W
HIGH_W, LOW_W = 3.0, 0.3                 # Isp < 50 인 설계는 ×LOW_W
TERMS = ["OXIDIZER_FULLY_DEPLETED", "FUEL_DEPLETED", "TIME_LIMIT_REACHED"]
DP_CLIP = (-1.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────
# 데이터
# ──────────────────────────────────────────────────────────────────────────

def dp_channel(S):
    """원 시계열 (N, 5, T) → 차압 비율 채널 (N, T): 연소 중 (Pt−Pc)/Pc (DP_CLIP), 연소 밖 DP_FILL"""
    th, pc, pt = S[:, 0], S[:, 1], S[:, 2]
    fmax = th.max(1, keepdims=True)
    on = (th >= GA.BURN_THRESHOLD * fmax) & (fmax > 0) & (pc > 1e4)
    dp = np.clip((pt - pc) / np.maximum(pc, 1e4), *DP_CLIP)
    return np.where(on, dp, DP_FILL).astype(np.float32)


def load_data(data_dir=DATA_DIR):
    Xs, Ss, metas = [], [], []
    fixed = GA.load_fixed_values()
    for run in sorted(glob.glob(os.path.join(data_dir, "*"))):
        paths = sorted(glob.glob(os.path.join(run, "shard_*.npz")))
        if not paths:
            continue
        with open(os.path.join(run, "info.json"), encoding="utf-8") as f:
            info = json.load(f)
        names = info.get("feature_vars") or info["design_vars"]       # 예전 데이터는 설계 변수만 (헬륨 고정)
        for path in paths:
            d = np.load(path)
            Xs.append(ft.convert_X(d["X"], names, fixed, info.get("pipe_features"),
                                   info.get("temperature", ft.LEGACY_TEMPERATURE)))
            Ss.append(d["S"][:, :, :N_T])
            metas += json.loads(str(d["meta"]))
    if not Xs:
        raise FileNotFoundError("rl/ts_data 에 데이터가 없습니다. python -m rl.ts_data 로 만드세요.")
    X = np.concatenate(Xs)
    S = np.concatenate(Ss)
    # 시뮬레이션 종료 시각 (이후는 0 으로 채워져 있음) — core 채널로만 판단 (차압 채널은 연소 밖을 DP_FILL 로 채운다)
    nz = np.abs(S).sum(1) > 0
    S = np.concatenate([S, dp_channel(S)[:, None]], 1)
    ok = np.array([not m["error"] for m in metas])
    term = np.array([TERMS.index(m["termination_code"]) if m["termination_code"] in TERMS else -1 for m in metas])
    Y = np.full((len(metas), len(SCALARS)), np.nan, dtype=np.float32)
    for i, m in enumerate(metas):
        for j, k in enumerate(SCALARS):
            v = m["scalars"].get(k)
            if v is not None and np.isfinite(v):
                Y[i, j] = v
    j = SCALARS.index("dp_ratio_min")
    Y[:, j] = np.clip(Y[:, j], *DP_CLIP)
    j = SCALARS.index("isp")
    Y[:, j] = np.clip(Y[:, j], *ISP_CLIP)
    # simt 보다 길면 core 는 TIME_LIMIT_REACHED
    Y[:, SCALARS.index("t_end")] = np.where(nz.any(1), N_T - np.argmax(nz[:, ::-1], 1), 0) * DT
    # 실패(오류)한 설계는 시계열·요약값이 없다
    Y[~ok] = np.nan
    kinds = np.array([m["kind"] for m in metas])
    return X, S, ok, term, Y, kinds


# ──────────────────────────────────────────────────────────────────────────
# 모델
# ──────────────────────────────────────────────────────────────────────────

class TSNet(nn.Module):
    def __init__(self, n_in=ft.N_FEATURES, d=128, layers=3, heads=4, ff=256, n_sc=len(SCALARS), n_ch=len(CHANNELS)):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.randn(N_TOK, d) * 0.02)
        self.cond = nn.Linear(d, d)                       # 설계 → 시간 토큰에 더하는 조건 (FiLM 의 덧셈 항)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.n_ch = n_ch
        self.series_head = nn.Linear(d, PATCH * n_ch)
        self.ok_head = nn.Linear(d, 1)
        self.term_head = nn.Linear(d, len(TERMS))
        self.scalar_head = nn.Linear(d, n_sc)
        for name, n in (("x_mean", n_in), ("x_std", n_in), ("y_mean", n_sc), ("y_std", n_sc),
                        ("s_scale", n_ch)):
            self.register_buffer(name, torch.zeros(n) if "mean" in name else torch.ones(n))

    def forward(self, x):
        """x: 정규화 전 (B, N_FEATURES) → (series 정규화 (B, C, N_T), ok 로짓, term 로짓, scalars 정규화)"""
        h = self.inp((x - self.x_mean) / self.x_std)
        tok = self.pos.unsqueeze(0) + self.cond(h).unsqueeze(1)
        z = self.enc(torch.cat([h.unsqueeze(1), tok], 1))
        z = self.norm(z)
        g, t = z[:, 0], z[:, 1:]
        s = self.series_head(t).view(x.shape[0], N_TOK, PATCH, self.n_ch)
        s = s.reshape(x.shape[0], N_T, self.n_ch).transpose(1, 2)
        return s, self.ok_head(g).squeeze(-1), self.term_head(g), self.scalar_head(g)


def series_integrals(s_phys):
    """물리 단위 시계열 (B, C, N_T) → (임펄스, 추진제 질량) — derive 와 같은 정의 (음수는 0)"""
    s_phys = s_phys.clamp(min=0)
    return s_phys[:, 0].sum(-1) * DT, (s_phys[:, 3] + s_phys[:, 4]).sum(-1) * DT


def _fit(net, X, S, ok, term, Y, tr, va, device, epochs, seed, log=print, lr=1e-3, w_cons=1.0, on_eval=None):
    torch.manual_seed(seed)
    net.to(device)
    Xt = torch.tensor(X, device=device)
    St = torch.tensor(S, device=device) / net.s_scale.view(1, -1, 1)
    okt = torch.tensor(ok, dtype=torch.float32, device=device)
    tt = torch.tensor(term, device=device)
    Yt = (torch.tensor(Y, device=device) - net.y_mean) / net.y_std
    Ym = ~torch.isnan(Yt)
    Yt = torch.nan_to_num(Yt)
    # 정답 시계열의 적분 (곡선 ↔ 요약값 일관성 손실의 기준)
    with torch.no_grad():
        imp_true, mp_true = series_integrals(torch.tensor(S, device=device))
        isp_true = torch.where(mp_true > 1e-6, imp_true / (mp_true * GA.G0), torch.zeros_like(mp_true)).clamp(*ISP_CLIP)
    imp_scale = float(imp_true[torch.tensor(tr, device=device)].std()) + 1e-6
    mp_scale = float(mp_true[torch.tensor(tr, device=device)].std()) + 1e-6
    # 표본 가중치: 최적화가 가는 고 Isp 영역을 더 정확하게
    isp_raw = torch.nan_to_num(torch.tensor(Y[:, SCALARS.index("isp")], device=device), nan=0.0)
    wt = torch.where(isp_raw > HIGH_ISP, HIGH_W, torch.where(isp_raw < 50, LOW_W, 1.0)) * okt + (1 - okt)
    s_scale = net.s_scale.view(1, -1, 1)
    ch_w = torch.ones(1, len(CHANNELS), 1, device=device)
    ch_w[0, CHANNELS.index("dp_ratio")] = DP_CH_W
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    bs = 256
    steps = epochs * math.ceil(len(tr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tr_t = torch.tensor(tr)

    def losses(idx):
        s, lo, lt, ys = net(Xt[idx])
        m = okt[idx] > 0
        w = wt[idx]
        l_ok = nn.functional.binary_cross_entropy_with_logits(lo, okt[idx])
        tm = tt[idx] >= 0
        l_term = nn.functional.cross_entropy(lt[tm], tt[idx][tm]) if tm.any() else lo.sum() * 0
        zero = lo.sum() * 0
        if m.any():
            wm = w[m] / w[m].mean()
            diff = (s[m] - St[idx][m]) * ch_w
            l_series = (((diff ** 2).mean((1, 2)) + 0.3 * diff.abs().mean((1, 2))) * wm).mean() / ch_w.mean()
            imp, mp = series_integrals(s[m] * s_scale)
            isp = torch.where(mp > 1e-6, imp / (mp.clamp(min=1e-6) * GA.G0), torch.zeros_like(mp)).clamp(*ISP_CLIP)
            l_cons = ((nn.functional.smooth_l1_loss(imp / imp_scale, imp_true[idx][m] / imp_scale, reduction="none")
                       + nn.functional.smooth_l1_loss(mp / mp_scale, mp_true[idx][m] / mp_scale, reduction="none")
                       + nn.functional.smooth_l1_loss(isp / 20.0, isp_true[idx][m] / 20.0, reduction="none")) * wm).mean()
        else:
            l_series, l_cons = zero, zero
        ym = Ym[idx]
        sc = nn.functional.smooth_l1_loss(ys, Yt[idx], reduction="none", beta=0.5) * ym * w.unsqueeze(1)
        l_sc = sc.sum() / (ym * w.unsqueeze(1)).sum().clamp(min=1)
        return l_ok, l_term, l_series, l_sc, l_cons

    def total(parts):
        return 0.3 * parts[0] + 0.2 * parts[1] + 3.0 * parts[2] + parts[3] + w_cons * parts[4]

    best, best_state = float("inf"), None
    for ep in range(epochs):
        net.train()
        perm = tr_t[torch.randperm(len(tr), generator=gen)]
        for k in range(0, len(perm), bs):
            idx = perm[k:k + bs].to(device)
            loss = total(losses(idx))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        if ep % 10 == 9 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                parts = [l.item() for l in losses(torch.tensor(va, device=device))]
            tot = total(parts)
            if on_eval:
                on_eval(ep + 1, parts, tot)
            log(f"    epoch {ep + 1:4d}  val ok {parts[0]:.3f} term {parts[1]:.3f} series {parts[2]:.4f} "
                f"scalar {parts[3]:.4f} cons {parts[4]:.4f}", flush=True)
            if tot < best:
                best = tot
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net.cpu().eval()


# ──────────────────────────────────────────────────────────────────────────
# 시계열 → 요약값 (GA.simulate 와 같은 정의, simt 로 잘라서)
# ──────────────────────────────────────────────────────────────────────────

def derive(series, simt=20.0, burn_threshold=GA.BURN_THRESHOLD):
    """series: (C, N_T) 원래 단위 → GA.simulate 와 같은 정의의 요약값 dict"""
    n = int(round(simt / DT))
    th, pc, pt, mo, mf = (np.maximum(series[i, :n], 0.0) for i in range(5))
    impulse = float(th.sum() * DT)
    has_dp = series.shape[0] > 5
    m_prop = float((mo + mf).sum() * DT)
    f_max = float(th.max()) if n else 0.0
    burn = th >= burn_threshold * f_max if f_max > 0 else np.zeros(n, dtype=bool)
    bt = float(burn.sum() * DT)
    on = burn & (pc > 1e4)
    if has_dp:                                   # 차압 채널을 직접 예측한 모델
        dp = series[5, :n][on]
    else:
        dp = (pt[on] - pc[on]) / pc[on] if on.any() else np.array([])
    return {
        "impulse": impulse,
        "isp": impulse / (m_prop * GA.G0) if m_prop > 0 else 0.0,
        "thrust_avg": float(th[burn].sum() * DT / bt) if bt > 0 else 0.0,
        "burn_time": bt,
        "max_tank_pressure": float(pt.max()),
        "max_chamber_pressure": float(pc.max()),
        "dp_ratio_min": float(dp.min()) if dp.size else -1.0,
        "still_burning": bool(f_max > 0 and th[-1] >= burn_threshold * f_max),
    }


# ──────────────────────────────────────────────────────────────────────────
# 추론
# ──────────────────────────────────────────────────────────────────────────

class TSSurrogate:
    def __init__(self, nets, device="cpu", scalars=SCALARS):
        self.nets = [n.to(device).eval() for n in nets]
        self.device = device
        self.scalars = list(scalars)
        self.has_t_end = "t_end" in self.scalars

    @classmethod
    def load(cls, path=MODEL_PATH, device="cpu"):
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob.get("feature_vars") != ft.FEATURE_VARS:
            raise ValueError(f"{os.path.basename(path)} 는 예전 설계 변수 구성 (챔버 길이 변수 · 헬륨 고정) 으로 학습된 "
                             "시계열 대리모델입니다. python -m rl.ts_surrogate 로 다시 학습하세요.")
        nets = []
        scalars = blob.get("scalars", SCALARS_V2)
        for state in blob["states"]:
            # 예전 모델 (차압 채널 없음, 입력 17) 은 config 에 n_ch 가 없다
            cfg = dict(blob["config"])
            cfg.setdefault("n_ch", len(blob.get("channels", GA.SERIES_CHANNELS)))
            net = TSNet(**cfg, n_sc=len(scalars))
            net.load_state_dict(state)
            nets.append(net)
        return cls(nets, device, scalars)

    @torch.no_grad()
    def predict(self, X):
        """X: (N, N_FEATURES) → dict(series (N,C,N_T) 원래 단위, p_ok, p_term (N,3), scalars (N,K), scalars_std)"""
        x = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
        ser, ok, term, sc = [], [], [], []
        for net in self.nets:
            s, lo, lt, ys = net(x)
            ser.append(s * net.s_scale.view(1, -1, 1))
            ok.append(torch.sigmoid(lo))
            term.append(torch.softmax(lt, -1))
            sc.append(ys * net.y_std + net.y_mean)
        sc = torch.stack(sc)
        if not self.has_t_end:                  # 예전 모델: 현재 SCALARS 순서로 맞추고 없는 값은 nan
            pad = torch.full((*sc.shape[:2], len(SCALARS) - sc.shape[2]), float("nan"), device=sc.device)
            sc = torch.cat([sc, pad], -1)
        return {
            "series": _clamp_series(torch.stack(ser).mean(0)).cpu().numpy(),
            "p_ok": torch.stack(ok).mean(0).cpu().numpy(),
            "p_term": torch.stack(term).mean(0).cpu().numpy(),
            "scalars": sc.mean(0).cpu().numpy(),
            "scalars_std": (sc.std(0, unbiased=False) if len(self.nets) > 1 else torch.zeros_like(sc[0])).cpu().numpy(),
        }


def _clamp_series(s):
    """물리 채널은 0 이상, 차압 채널은 DP_CLIP 범위"""
    if s.shape[1] > 5:
        return torch.cat([s[:, :5].clamp(min=0), s[:, 5:].clamp(*DP_CLIP)], 1)
    return s.clamp(min=0)


design_features = ft.design_features


class TSSurrogateSimulator:
    """
    GA.simulate() 대신 쓰는 시계열 대리모델 (rl.surrogate.SurrogateSimulator 와 같은 인터페이스).
    keep_series=True 면 결과 dict 에 "series" (C, N_T) numpy 배열을 넣는다 (기본은 안 넣음 — GA 캐시가 커지므로).
    요약값은 요약 머리의 예측을 쓰고, 연소가 simt 를 넘으면 core 와 같이 TIME_LIMIT_REACHED.
    """

    def __init__(self, model_path=MODEL_PATH, device=None, simt=None, keep_series=False, t_margin=0.0,
                 ok_threshold=0.5):
        self.keep_series = keep_series
        self.t_margin = t_margin
        self.ok_threshold = ok_threshold      # core 정상 종료 확률이 이보다 낮으면 실패로 본다
        if device is None:
            device = "cpu"
        if device == "cpu":
            torch.set_num_threads(1)
        self.model = TSSurrogate.load(model_path, device)
        self.simt = surrogate_simt(simt)
        self.bias = {}

    def update_bias(self, candidates, checked):
        import constrain as cs
        residuals = {}
        for ev_pred, ev_core in zip(candidates, checked):
            core = ev_core["sim"] or {}
            if core.get("error") or core.get("termination_code") == "TIME_LIMIT_REACHED":
                continue
            for t in SCALARS:
                actual = cs.injector_dp_ratio(core) if t == "dp_ratio_min" else core.get(t)
                if actual is None or t not in ev_pred["sim"]:
                    continue
                residuals.setdefault(t, []).append(actual - ev_pred["sim"][t])
        for t, r in residuals.items():
            self.bias[t] = self.bias.get(t, 0.0) + float(np.mean(r))
        return dict(self.bias)

    def predict_designs(self, designs, pipes):
        X = np.array([design_features(d, p) for d, p in zip(designs, pipes)], dtype=np.float32)
        return self.model.predict(X)

    def simulate_batch(self, designs, pipes):
        t0 = time.time()
        out = self.predict_designs(designs, pipes)
        wall = (time.time() - t0) / max(len(designs), 1)
        sims = sims_from_output(out, self.simt, self.model.has_t_end, self.bias, self.keep_series, self.t_margin,
                                self.ok_threshold)
        for sim in sims:
            sim["wall_time"] = wall
        self.last_output = out          # 같은 배치의 시계열 (에이전트 관측용)
        return sims


def sims_from_output(out, simt, has_t_end, bias=None, keep_series=False, t_margin=0.0, ok_threshold=0.5,
                     dp_std_k=DP_STD_K):
    """TSSurrogate.predict 결과 → GA.simulate 와 같은 형식의 결과 dict 리스트 (MDO 시뮬레이션 시간 simt 기준)"""
    sims = []
    for i in range(len(out["p_ok"])):
        sim = {k: float(v) for k, v in zip(SCALARS, out["scalars"][i])}
        for t, b in (bias or {}).items():
            sim[t] += b
        der = derive(out["series"][i], simt)
        # 보수적 차압: 요약 머리와 곡선 최솟값 중 작은 값 − k × 앙상블 표준편차 (최적화가 예측 오차를 악용하지 않게)
        j = SCALARS.index("dp_ratio_min")
        dp_head = sim["dp_ratio_min"]
        sim["dp_ratio_min"] = min(dp_head, der["dp_ratio_min"] + (bias or {}).get("dp_ratio_min", 0.0)) \
            - dp_std_k * float(out["scalars_std"][i][j])
        sim["dp_ratio_head"] = dp_head
        if has_t_end:                                # 종료 시각 머리: simt 안에 끝나는지 직접 예측
            time_limit = sim["t_end"] > simt - DT / 2 - t_margin
        else:
            time_limit = sim["burn_time"] >= simt - DT or der["still_burning"]
            sim.pop("t_end", None)
        if time_limit:                               # simt 안에 끝나지 않음 → core 도 시간 제한으로 끝난다
            sim.update({k: der[k] for k in ("impulse", "isp", "thrust_avg")})
        ok = out["p_ok"][i] > ok_threshold
        sim.update({
            "error": None,
            "termination_code": ("TIME_LIMIT_REACHED" if time_limit or not ok
                                 else TERMS[int(out["p_term"][i][:2].argmax())]),
            "p_completed": float(out["p_ok"][i] * (1 - out["p_term"][i][2])),
            "isp_std": float(out["scalars_std"][i][0]),
        })
        if keep_series:
            sim["series"] = out["series"][i]
        sims.append(sim)
    return sims


def mdo_metrics(model, X, S, ok, Y, simt=None):
    """
    MDO 에서 중요한 정확도 (시뮬레이션 시간 simt 기준) — 정상 종료 설계만:
    시간 제한 판정 정확도 (특히 '끝난다고 예측했는데 실제로는 시간 초과' 비율), 둘 다 끝난 설계의 Isp 오차 (구간별),
    곡선에서 계산한 Isp 의 오차, 고 Isp 순위 상관
    """
    from scipy.stats import spearmanr
    simt = surrogate_simt(simt)
    import constrain as cs
    out = model.predict(X)
    sims = sims_from_output(out, simt, model.has_t_end)
    t_end = Y[:, SCALARS.index("t_end")]
    tl_true = t_end > simt - DT / 2
    tl_pred = np.array([s["termination_code"] == "TIME_LIMIT_REACHED" for s in sims])
    isp_p = np.array([s["isp"] for s in sims])
    isp_t = Y[:, 0]
    der_isp = np.array([derive(out["series"][i], simt)["isp"] for i in range(len(X))])
    both = ok & ~tl_true & ~tl_pred
    r = {"n_ok": int(ok.sum()), "time_limit_acc": float((tl_true == tl_pred)[ok].mean()),
         "missed_time_limit": float((tl_true & ~tl_pred)[ok].sum() / max((~tl_pred)[ok].sum(), 1)),
         "isp_mae": float(np.abs(isp_p - isp_t)[both].mean()),
         "isp_from_series_mae": float(np.abs(der_isp - isp_t)[both].mean())}
    for lo, hi in ((150, 180), (180, 190), (190, 200), (200, 260)):
        m = both & (isp_t >= lo) & (isp_t < hi)
        if m.sum():
            r[f"isp_{lo}_{hi}"] = {"n": int(m.sum()), "mae": float(np.abs(isp_p - isp_t)[m].mean()),
                                  "bias": float((isp_p - isp_t)[m].mean()),
                                  "series_mae": float(np.abs(der_isp - isp_t)[m].mean())}
    top = both & (isp_t > 185)
    r["spearman_isp_185"] = float(spearmanr(isp_p[top], isp_t[top]).correlation)
    for k in ("impulse", "thrust_avg", "dp_ratio_min"):
        j = SCALARS.index(k)
        p = np.array([s[k] for s in sims])
        m = both & ~np.isnan(Y[:, j])
        r[f"{k}_mae"] = float(np.abs(p - Y[:, j])[m].mean())
    # 차압 제약 판정: 요약 머리 / 곡선 최솟값 / 보수적 (실제로 쓰는 값) — 제약 경계 근처 (실제 0.1~0.4)
    j = SCALARS.index("dp_ratio_min")
    dp_t = Y[:, j]
    est = {"head": np.array([s["dp_ratio_head"] for s in sims]),
           "series": np.array([derive(out["series"][i], simt)["dp_ratio_min"] for i in range(len(X))]),
           "conservative": np.array([s["dp_ratio_min"] for s in sims])}
    m = both & ~np.isnan(dp_t) & (dp_t > 0.1) & (dp_t < 0.4)
    lim = cs.MIN_INJECTOR_DP_RATIO
    for name, p in est.items():
        r[f"dp_{name}"] = {"n": int(m.sum()), "mae": float(np.abs(p - dp_t)[m].mean()),
                           "bias": float((p - dp_t)[m].mean()),
                           # 예측은 합격 (≥ 하한 + 대리모델 여유) 인데 실제는 불합격인 비율
                           "false_pass": float(((p >= lim + cs.SURROGATE_MARGIN["dp_ratio"]) & (dp_t < lim))[m].sum()
                                               / max(((p >= lim + cs.SURROGATE_MARGIN["dp_ratio"]))[m].sum(), 1)),
                           "true_pass_kept": float(((p >= lim + cs.SURROGATE_MARGIN["dp_ratio"]) & (dp_t >= lim))[m].sum()
                                                   / max((dp_t >= lim)[m].sum(), 1))}
    return r


# ──────────────────────────────────────────────────────────────────────────
# 학습
# ──────────────────────────────────────────────────────────────────────────

class _Monitor:
    """대리모델 학습을 웹 학습 모니터(/training)에 — rl/runs/<id>/status.json, live.jsonl (step = 누적 에폭)"""

    def __init__(self, run_dir, total, config):
        os.makedirs(run_dir, exist_ok=True)
        self.dir = run_dir
        self.status = {"id": os.path.basename(run_dir), "kind": "surrogate", "state": "running", "pid": os.getpid(),
                       "started": time.time(), "step": 0, "total_steps": total, "config": config}
        self.f = open(os.path.join(run_dir, "live.jsonl"), "a", encoding="utf-8")
        self._save()

    def _save(self):
        tmp = os.path.join(self.dir, "status.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.status, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(self.dir, "status.json"))

    def epoch(self, k, ep, epochs, parts, tot):
        step = k * epochs + ep
        names = ("ok", "term", "series", "scalar", "cons")
        row = {"step": step, "t": round(time.time(), 1), "sur/model": k + 1, "sur/total": float(tot),
               **{f"sur/{n}": float(v) for n, v in zip(names, parts)}}
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()
        self.status.update(step=step, model=k + 1, fps=None)
        self._save()

    def done(self, meta):
        self.status.update(state="done", finished=time.time(), step=self.status["total_steps"],
                           result={k: meta[k] for k in ("mdo", "classifier", "series_r2") if k in meta})
        self._save()
        self.f.close()


def train(epochs=200, n_models=3, device=None, seed=0, out_path=MODEL_PATH, meta_path=META_PATH,
          d=128, layers=3, heads=4, ff=256, lr=1e-3, w_cons=1.0, run_dir=None):
    """run_dir: 웹 학습 모니터용 기록 (status.json · live.jsonl, kind=surrogate)"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mon = _Monitor(run_dir, epochs * n_models, dict(epochs=epochs, models=n_models, d=d, layers=layers, heads=heads,
                                                    ff=ff, lr=lr, w_cons=w_cons, out=os.path.relpath(out_path, HERE),
                                                    device=device)) if run_dir else None
    X, S, ok, term, Y, kinds = load_data()
    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_te, n_va = n // 10, n // 10
    te, va, tr = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]
    print(f"데이터 {n}개 (정상 {ok.sum()}, 실패 {(~ok).sum()}) · 학습 {len(tr)} / 검증 {len(va)} / 테스트 {len(te)} · {device}")

    x_mean, x_std = X[tr].mean(0), X[tr].std(0) + 1e-9
    y_mean, y_std = np.nanmean(Y[tr], 0), np.nanstd(Y[tr], 0) + 1e-9
    s_scale = np.array([np.percentile(S[tr][ok[tr]][:, c].max(1), 95) for c in range(len(CHANNELS))]) + 1e-9
    config = {"n_in": X.shape[1], "d": d, "layers": layers, "heads": heads, "ff": ff, "n_ch": len(CHANNELS)}
    states = []
    t0 = time.time()
    for k in range(n_models):
        print(f"  모델 {k + 1}/{n_models}")
        net = TSNet(**config)
        for name, v in (("x_mean", x_mean), ("x_std", x_std), ("y_mean", y_mean), ("y_std", y_std), ("s_scale", s_scale)):
            getattr(net, name).copy_(torch.tensor(v, dtype=torch.float32))
        # 앙상블 다양성: 모델마다 다른 부트스트랩 표본
        boot = np.random.default_rng(seed + k).choice(tr, len(tr), replace=True) if n_models > 1 else tr
        on_eval = (lambda ep, parts, tot, k=k: mon.epoch(k, ep, epochs, parts, tot)) if mon else None
        net = _fit(net, X, S, ok, term, Y, boot, va, device, epochs, seed + k, lr=lr, w_cons=w_cons, on_eval=on_eval)
        states.append(net.state_dict())
    print(f"  학습 {time.time() - t0:.0f}s")
    torch.save({"config": config, "states": states, "scalars": SCALARS, "channels": CHANNELS,
                "feature_vars": ft.FEATURE_VARS}, out_path)

    # 테스트 지표
    model = TSSurrogate.load(out_path, device)
    out = model.predict(X[te])
    meta = {"n": int(n), "n_ok": int(ok.sum()), "n_test": int(len(te)), "trained": time.time(),
            "channels": CHANNELS, "dt": DT, "n_t": N_T, "scalars": SCALARS, "metrics": {}, "series_r2": {},
            "classifier": {}}
    okt = ok[te]
    pred_ok = out["p_ok"] > 0.5
    meta["classifier"]["ok_accuracy"] = float((pred_ok == okt).mean())
    tm = term[te] >= 0
    meta["classifier"]["term_accuracy"] = float((out["p_term"][tm].argmax(1) == term[te][tm]).mean())
    for j, k in enumerate(SCALARS):
        m = ~np.isnan(Y[te][:, j])
        err = out["scalars"][m, j] - Y[te][m, j]
        var = np.var(Y[te][m, j])
        meta["metrics"][k] = {"mae": float(np.abs(err).mean()), "r2": float(1 - np.mean(err ** 2) / var) if var > 0 else None}
    for c, name in enumerate(CHANNELS):
        p, t = out["series"][okt, c], S[te][okt, c]
        meta["series_r2"][name] = float(1 - np.mean((p - t) ** 2) / np.var(t))
    # 시계열에서 계산한 요약값의 정확도 (경향성이 맞는지)
    der = [derive(s, 40.0) for s in out["series"][okt]]
    for k in ("impulse", "isp", "burn_time"):
        j = SCALARS.index(k)
        tv = Y[te][okt, j]
        pv = np.array([d[k] for d in der])
        m = ~np.isnan(tv)
        meta["metrics"][k]["from_series_mae"] = float(np.abs(pv[m] - tv[m]).mean())
    # 산화제 조기 소진 등으로 연소 시간이 다양한 경우의 오차 (짧은/중간/긴 연소)
    bt = Y[te][okt, SCALARS.index("burn_time")]
    ip = out["scalars"][okt, SCALARS.index("impulse")] - Y[te][okt, SCALARS.index("impulse")]
    meta["by_burn_time"] = {}
    for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 40)):
        m = (bt >= lo) & (bt < hi)
        if m.sum():
            meta["by_burn_time"][f"{lo}-{hi}s"] = {"n": int(m.sum()), "impulse_mae": float(np.abs(ip[m]).mean())}
    meta["mdo"] = mdo_metrics(model, X[te], S[te], ok[te], Y[te])
    meta["config"] = {**config, "epochs": epochs, "models": n_models, "lr": lr, "w_cons": w_cons}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    if mon:
        mon.done(meta)
    print(json.dumps({k: meta[k] for k in ("classifier", "metrics", "series_r2", "by_burn_time", "mdo")}, indent=1, ensure_ascii=False))
    return meta


def compare(path):
    """저장된 모델을 같은 테스트 분할(seed 0)로 MDO 지표 평가"""
    X, S, ok, term, Y, kinds = load_data()
    n = len(X)
    te = np.random.default_rng(0).permutation(n)[:n // 10]
    model = TSSurrogate.load(path, "cuda" if torch.cuda.is_available() else "cpu")
    return mdo_metrics(model, X[te], S[te], ok[te], Y[te])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--models", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=MODEL_PATH)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w-cons", type=float, default=1.0)
    parser.add_argument("--run", default=None, help="웹 모니터용 실행 이름 (rl/runs/<이름>)")
    parser.add_argument("--compare", default=None, help="학습하지 않고 이 모델의 MDO 지표만")
    args = parser.parse_args()
    if args.compare:
        print(json.dumps(compare(args.compare), indent=1, ensure_ascii=False))
    else:
        meta = args.meta or args.out.replace(".pt", "_meta.json")
        train(args.epochs, args.models, args.device, out_path=args.out, meta_path=meta, d=args.d, layers=args.layers,
              heads=args.heads, ff=args.ff, lr=args.lr, w_cons=args.w_cons,
              run_dir=os.path.join(HERE, "runs", args.run) if args.run else None)
