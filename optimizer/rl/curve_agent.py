"""
곡선을 보고 판단하는 설계 에이전트 (rl/direct.py 의 후속)
    - 관측: 시계열 대리모델이 예측한 0~simt 연소 곡선 8채널 (추력, 챔버압, 탱크압, 산화제·연료 질량유량, Isp(t), O/F(t),
            인젝터 차압 (Pt−Pc)/Pc(t))
            을 0.4 s 패치 토큰 50개로 + 전역 토큰 (설계·파이프·요약 성능·제약 위반·목표·경향 요약)
    - 인코더: Transformer (CurveEncoder) — [CLS, 전역, 시간 패치 50개] self-attention → CLS 특징 벡터 → 정책·가치
              CLS 가 어느 시간 패치를 봤는지 (last_attn) 를 웹에서 판단 근거로 보여 준다
    - 행동: 한 걸음에 '어떤 변수를, 어느 방향으로, 얼마나' 를 최대 2개 (MultiDiscrete [N_MOVES, N_MOVES])
            이동 = 변수 14개 (헬륨 슈퍼차징 압력 포함, 챔버 길이는 연료 길이로 정해져 없음) × 크기 8단계 (×e^±0.02, ±0.05, ±0.15, ±0.40 — 현재 값 대비 −2/−5/−14/−33 %, +2/+5/+16/+49 %)
                   + 파이프 다음/이전 + 아무것도 안 함
    - 사전학습: 대리모델로 계산한 '이동 하나마다의 점수 변화' (한 걸음 앞 보기) 를 정책 로짓으로 증류 (rl/train_curve.py)
보상: rl/direct.py 와 같은 계층 점수의 증가량. 점수의 Isp 는 예측 − isp_penalty × 앙상블 표준편차 (보수적)
"""

import math

import numpy as np

import GA
from rl import features
import constrain as cs
from rl import direct

MAGS = [-0.40, -0.15, -0.05, -0.02, 0.02, 0.05, 0.15, 0.40]     # 로그 배율 (현재 값 × e^m)
N_MAGS = len(MAGS)
N_VARS = len(GA.DESIGN_VARS)
PIPE_UP = N_VARS * N_MAGS
PIPE_DOWN = PIPE_UP + 1
NONE = PIPE_UP + 2
N_MOVES = NONE + 1
MOVES_PER_STEP = 2

VAR_KO = {
    "oxidizer_M": "산화제 질량", "port_num": "포트 수", "outter_radius": "연료 외경", "port_center_radius": "포트 배치 반지름",
    "inner_radius": "포트 반지름", "fuel_length": "연료 길이", "oxidizer_tank_length": "탱크 길이",
    "oxidizer_tank_tickness": "가공 두께", "supercharge_pressure": "헬륨 슈퍼차징", "combution_chamber_length": "챔버 길이",
    "injector_oriffice_n": "오리피스 수",
    "injector_oriffice_radius": "오리피스 반지름", "injector_oriffice_length": "오리피스 길이",
    "nozzle_throat_radius": "노즐 목", "nozzle_exit_radius": "노즐 출구", "pipe+": "파이프 ▲", "pipe-": "파이프 ▼",
}


def move_text(moves):
    """이동들 → '노즐 목 +16%, 오리피스 수 +5%' (같은 이동 두 번은 합쳐서)"""
    acc = {}
    for a in moves:
        var, m = move_label(a)
        if var is None:
            continue
        acc[var] = acc.get(var, 0.0) + m
    out = []
    for var, m in acc.items():
        if var.startswith("pipe"):
            out.append(VAR_KO[var])
        else:
            p = (math.exp(m) - 1) * 100
            out.append(f"{VAR_KO.get(var, var)} {'+' if p >= 0 else '−'}{abs(p):.0f}%")
    return ", ".join(out) or "유지"


SERIES_CH = ["thrust", "chamber_pressure", "tank_pressure", "mox_dot", "mf_dot", "isp", "of", "dp"]
N_TOK = 50
PATCH = 4
N_PTS = N_TOK * PATCH
TOK_DIM = len(SERIES_CH) * PATCH
GLOBAL_DIM = direct.BASE_DIM + 9
OBS_DIM = N_TOK * TOK_DIM + GLOBAL_DIM


def move_label(a):
    """행동 번호 → (변수 이름 / 'pipe+' / 'pipe-' / None, 로그 배율)"""
    a = int(a)
    if a == NONE:
        return None, 0.0
    if a == PIPE_UP:
        return "pipe+", 0.0
    if a == PIPE_DOWN:
        return "pipe-", 0.0
    return GA.DESIGN_VARS[a // N_MAGS], MAGS[a % N_MAGS]


def apply_moves(rng, genome, moves, n_pipes):
    """genome 에 이동들을 적용 (양자화·범위 자르기 포함). 정수 변수는 최소 ±1 (가장 작은 크기만 확률적으로)"""
    g = dict(genome)
    for a in moves:
        var, m = move_label(a)
        if var is None:
            continue
        if var in ("pipe+", "pipe-"):
            if n_pipes > 1:
                g["pipe"] = (g["pipe"] + (1 if var == "pipe+" else -1)) % n_pipes
            continue
        x = g[var]
        if var in GA.INT_VARS:
            delta = x * (math.exp(m) - 1)
            if abs(delta) < 1:
                small = abs(m) == min(abs(v) for v in MAGS)
                delta = math.copysign(1, m) if not small or rng.random() < abs(delta) else 0
            g[var] = x + int(round(delta))
        else:
            g[var] = x * math.exp(m)
    return GA.quantize(g)


def series_tokens(series, targets, simt, dt=GA.SERIES_DT):
    """예측 곡선 (5, N_T) → (N_TOK, TOK_DIM) — 0~simt 를 N_PTS 점으로 다시 뽑아 채널별 정규화"""
    thrust_t = targets[1]
    n = int(round(simt / dt))
    th, pc, pt, mo, mf = (np.maximum(series[i, :n], 0.0) for i in range(5))
    f_max = th.max()
    burn = th >= GA.BURN_THRESHOLD * f_max if f_max > 0 else np.zeros(n, dtype=bool)
    mdot = mo + mf
    ref_mdot = thrust_t / (200.0 * GA.G0)               # 목표 추력을 Isp 200 s 로 낼 때의 질량유량
    isp = np.where(burn & (mdot > 1e-4), th / (np.maximum(mdot, 1e-4) * GA.G0), 0.0)
    of = np.where(burn & (mf > 1e-4), mo / np.maximum(mf, 1e-4), 0.0)
    # 인젝터 차압 비율: 차압 채널을 예측하는 대리모델이면 그 값, 아니면 탱크압·챔버압으로 계산 (연소 밖 0)
    dp_raw = series[5, :n] if series.shape[0] > 5 else (pt - pc) / np.maximum(pc, 1e4)
    dp = np.where(burn & (pc > 1e4), np.clip(dp_raw, -0.5, 1.5), 0.0)
    ch = np.stack([np.clip(th / thrust_t, 0, 3), np.clip(pc / 5e6, 0, 2), np.clip(pt / 6e6, 0, 2),
                   np.clip(mo / ref_mdot, 0, 3), np.clip(mf / ref_mdot, 0, 3), np.clip(isp / 250.0, 0, 1.5),
                   np.clip(of / 10.0, 0, 2), dp])
    if n != N_PTS:
        src = (np.arange(n) + 0.5) / n
        dst = (np.arange(N_PTS) + 0.5) / N_PTS
        ch = np.stack([np.interp(dst, src, c) for c in ch])
    return ch.reshape(len(SERIES_CH), N_TOK, PATCH).transpose(1, 0, 2).reshape(N_TOK, TOK_DIM).astype(np.float32)


def build_obs(genome, sim, cvec, feasible, targets, progress, pipe_norm, series, simt, p_ok, isp_std):
    base = direct.base_features(genome, sim, cvec, feasible, targets, progress, pipe_norm)
    trend = direct.trend_features(series, targets, simt, p_ok, isp_std)[-9:]        # 경향 요약 스칼라만
    return np.concatenate([series_tokens(series, targets, simt).ravel(), base, trend])


# ──────────────────────────────────────────────────────────────────────────
# 인코더 (SB3 features extractor)
# ──────────────────────────────────────────────────────────────────────────

try:
    import torch
    from torch import nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except ImportError:           # 웹 서버 쪽에서 상수만 쓸 때
    BaseFeaturesExtractor = object


class _Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, keep=False):
        h = self.n1(x)
        a, w = self.attn(h, h, h, need_weights=keep, average_attn_weights=True)
        x = x + a
        return x + self.ff(self.n2(x)), w


class CurveEncoder(BaseFeaturesExtractor):
    """obs (B, OBS_DIM) → 특징 (B, 2d) = [CLS 출력, 전역 임베딩]. last_attn: 마지막 층 CLS → 시간 패치 주의 (B, N_TOK)"""

    def __init__(self, observation_space, d=128, layers=3, heads=4, ff=256):
        super().__init__(observation_space, features_dim=2 * d)
        self.tok = nn.Linear(TOK_DIM, d)
        self.pos = nn.Parameter(torch.randn(N_TOK, d) * 0.02)
        self.glob = nn.Sequential(nn.Linear(GLOBAL_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.type_emb = nn.Parameter(torch.randn(2, d) * 0.02)          # CLS / 전역 토큰 구분
        self.blocks = nn.ModuleList([_Block(d, heads, ff) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.keep_attn = False
        self.last_attn = None

    def forward(self, obs):
        b = obs.shape[0]
        ser = obs[:, :N_TOK * TOK_DIM].reshape(b, N_TOK, TOK_DIM)
        hg = self.glob(obs[:, N_TOK * TOK_DIM:])
        x = torch.cat([self.cls.expand(b, 1, -1) + self.type_emb[0], (hg + self.type_emb[1]).unsqueeze(1),
                       self.tok(ser) + self.pos], 1)
        for k, blk in enumerate(self.blocks):
            last = self.keep_attn and k == len(self.blocks) - 1
            x, w = blk(x, keep=last)
            if last:
                self.last_attn = w[:, 0, 2:].detach()
        return torch.cat([self.norm(x[:, 0]), hg], -1)


def policy_kwargs(d=128, layers=3, heads=4):
    return {"features_extractor_class": CurveEncoder,
            "features_extractor_kwargs": {"d": d, "layers": layers, "heads": heads, "ff": 2 * d},
            "net_arch": {"pi": [256], "vf": [256]}}


# ──────────────────────────────────────────────────────────────────────────
# 학습 환경
# ──────────────────────────────────────────────────────────────────────────

class CurveVecEnv(direct.DesignVecEnv):
    obs_dim = OBS_DIM

    def action_space_(self):
        from gymnasium import spaces
        return spaces.MultiDiscrete([N_MOVES] * MOVES_PER_STEP)

    def make_obs(self, i, sim, cvec, ok, tg, out, k):
        return build_obs(self.genomes[i], sim, cvec, ok, tg, self.t[i] / max(1, self.T[i]), self.pipe_norm,
                         out["series"][k], self.simt, out["p_ok"][k], out["scalars_std"][k][0])

    def apply(self, genome, action):
        return apply_moves(self.rng, genome, action, len(self.pipes))

    # ── 한 걸음 앞 보기 (사전학습 정답 · 비교 기준) ──
    def score_designs(self, genomes, targets):
        """genome 리스트 (목표는 각각) → 점수 배열 (evaluate 와 같은 규칙, 관측은 만들지 않음)"""
        decoded = [GA.decode(g, self.pipes, self.fixed) for g in genomes]
        sims = self.simulator.simulate_batch([d for d, _ in decoded], [p for _, p in decoded])
        scores = np.zeros(len(genomes))
        for k, ((design, pipe), tg) in enumerate(zip(decoded, targets)):
            _, v_pre = direct.constraint_vector(design, pipe, None, tg)
            if v_pre > 0:
                scores[k] = direct.score_of(False, 0.0, v_pre, self.isp_scale, False)
                continue
            sim = sims[k]
            _, v = direct.constraint_vector(design, pipe, sim, tg, cs.SURROGATE_MARGIN)
            isp_s = sim["isp"] - self.isp_penalty * sim["isp_std"]
            scores[k] = direct.score_of(v <= 0, isp_s, v, self.isp_scale, True)
        return scores

    def lookahead(self, idx, rng=None):
        """에이전트 idx 들의 현재 설계에서 이동 하나씩 (N_MOVES 개) 해 본 점수 변화 (len(idx), N_MOVES)"""
        rng = rng or self.rng
        genomes, targets = [], []
        for i in idx:
            for a in range(N_MOVES):
                genomes.append(apply_moves(rng, self.genomes[i], [a], len(self.pipes)))
                targets.append(self.targets[i])
        sc = self.score_designs(genomes, targets).reshape(len(idx), N_MOVES)
        return sc - self.score[np.asarray(idx)][:, None]


class CurveStrategy:
    """실제 MDO: GA.run_ga(strategy=CurveStrategy(...)) — 세대마다 개체 하나하나를 에이전트가 '판단' 해서 옮긴다"""

    kind = "curve"

    def __init__(self, model_path, device="cpu", deterministic=True, seed=None, ts_model=None):
        from stable_baselines3 import PPO
        from rl.ts_surrogate import MODEL_PATH, TSSurrogateSimulator
        if seed is not None:
            torch.manual_seed(seed)
        self.model = PPO.load(model_path, device=device)
        if self.model.observation_space.shape[0] != OBS_DIM:
            raise ValueError(f"모델 관측 {self.model.observation_space.shape[0]}차원 ≠ 현재 {OBS_DIM}차원. 다시 학습하세요.")
        self.ts = TSSurrogateSimulator(ts_model or MODEL_PATH, device="cpu")
        self.deterministic = deterministic
        self.pipe_norm = None

    def __call__(self, rng, evaluations, pipes, history, ctx, parents=None):
        if self.pipe_norm is None or self.pipe_norm.n != len(pipes):
            self.pipe_norm = direct.PipeNorm(pipes)
        targets = ctx.get("targets_full") or (*ctx["targets"], cs.TARGET_TOLERANCE)
        margin = cs.SURROGATE_MARGIN if ctx.get("surrogate") else None
        progress = (ctx["gen"] + 1) / max(1, ctx["n_generations"])
        out = self.ts.predict_designs([ev["design"] for ev in evaluations], [ev["pipe"] for ev in evaluations])
        obs = np.zeros((len(evaluations), OBS_DIM), dtype=np.float32)
        for k, ev in enumerate(evaluations):
            cvec, _ = direct.constraint_vector(ev["design"], ev["pipe"], ev["sim"], targets, margin)
            obs[k] = build_obs(ev["genome"], ev["sim"], cvec, ev["feasible"], targets, progress, self.pipe_norm,
                               out["series"][k], self.ts.simt, out["p_ok"][k], out["scalars_std"][k][0])
        enc = self.model.policy.features_extractor
        enc.keep_attn = True
        actions = self.model.predict(obs, deterministic=self.deterministic)[0]
        enc.keep_attn = False
        attn = enc.last_attn.cpu().numpy() if enc.last_attn is not None else None
        population = [apply_moves(rng, ev["genome"], a, len(pipes)) for ev, a in zip(evaluations, actions)]
        # 판단 기록: 변수별 선택 횟수·평균 크기, 최고 개체의 이동과 주의 시간대
        counts = {}
        for a in actions.ravel():
            var, m = move_label(a)
            if var is not None:
                c = counts.setdefault(var, [0, 0.0])
                c[0] += 1
                c[1] += abs(m)
        n = int(round(self.ts.simt / GA.SERIES_DT))
        curve = {c: [round(float(v), 4) for v in direct._pool(out["series"][0][i, :n], n // 10)]
                 for i, c in enumerate(GA.SERIES_CHANNELS)}
        best_moves = [{"var": move_label(a)[0], "mag": move_label(a)[1]} for a in actions[0]]
        info = {"action": -1, "name": "곡선 판단 · 최고 개체: " + move_text(actions[0]),
                "curve_agent": {"counts": {k: v[0] for k, v in counts.items()},
                                "mean_mag": {k: v[1] / v[0] for k, v in counts.items()},
                                "best_moves": best_moves,
                                "attention": [round(float(x), 4) for x in attn[0]] if attn is not None else None,
                                "attn_dt": self.ts.simt / N_TOK,
                                "curve": curve, "curve_dt": GA.SERIES_DT * n / (n // 10),
                                "best_id": GA.design_id(evaluations[0]["design"], evaluations[0]["pipe"])}}
        return population, info


def list_models():
    """웹에서 고를 수 있는 곡선 판단 모델 (rl/runs/curve_*/best_model.zip, final_model.zip)"""
    import csv
    import glob
    import json
    import os
    import zipfile
    here = os.path.dirname(os.path.abspath(__file__))
    opt_dir = os.path.dirname(here)
    models = []
    for run_dir in sorted(glob.glob(os.path.join(here, "runs", "curve_*")), reverse=True):
        name = os.path.basename(run_dir)
        if not features.run_compatible(run_dir):        # 예전 설계 변수 구성 (챔버 길이 변수, 헬륨 고정)
            continue
        rows = []
        prog = os.path.join(run_dir, "progress.csv")
        if os.path.exists(prog):
            with open(prog, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        for fname, label in (("best_model.zip", "best"), ("final_model.zip", "final")):
            path = os.path.join(run_dir, fname)
            if not os.path.exists(path):
                continue
            try:
                with zipfile.ZipFile(path) as z:
                    dim = int(json.loads(z.read("data"))["observation_space"]["_shape"][0])
            except Exception:
                dim = None
            if dim != OBS_DIM:
                continue
            models.append({"id": os.path.relpath(path, opt_dir), "label": f"{name} · {label}",
                           "mtime": os.path.getmtime(path), "steps": int(rows[-1]["step"]) if rows else None})
    return models
