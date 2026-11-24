"""
에이전트가 설계 변수를 직접 조정하는 방식 (GA 전략 선택 대신)
    - 에이전트 1명 = 설계안 1개. 매 스텝 자기 설계의 '연소 전 과정' (시계열 대리모델 rl/ts_surrogate.py 가 예측한
      추력·챔버압·탱크압·질량유량 곡선) 의 경향성과 제약 위반을 보고, 14개 설계 변수를 얼마나 움직일지(Δ),
      한 번에 움직일 크기(스텝), 파이프 변경 여부를 정한다.
    - 학습: DesignVecEnv — 에이전트 수백 명을 한 프로세스에서 시계열 대리모델(GPU)로 한꺼번에 평가
    - 실제 MDO: DirectStrategy — GA.run_ga(strategy=...) 에 넘기면 개체군의 각 개체를 에이전트가 한 걸음씩 옮긴다
      (교배·선택 없음. 평가는 core 시뮬레이션 또는 시계열 대리모델, 관측용 곡선은 항상 시계열 대리모델)

행동 (연속, [-1, 1]^16)
    [0:14]  설계 변수별 이동 방향·비율 (탐색 범위 × 스텝 × a)
    [14]    스텝 크기 (로그 스케일, -1 → STEP_MIN, +1 → STEP_MAX)
    [15]    파이프 (> +PIPE_THRESH: 다음 파이프, < -PIPE_THRESH: 이전 파이프)
관측 (OBS_DIM)
    설계 14 · 파이프 4 · 평가 요약 5 · 제약별 위반 22 · 합격/진행도/목표 4
    + 곡선 경향성 TREND_DIM: 추력 곡선 20점(목표 추력 대비), 챔버압·탱크압·차압비·O/F 각 10점,
      연소 시간/목표 연소 시간, 최대/평균 추력, 후반/초반 추력비(감쇠), 액체 소진(급락) 시점, 급락 후 꼬리 임펄스 비율,
      시간 제한 초과 여부, 정상 종료 확률, 예측 불확실성
보상: 점수(score) 증가량. GA 의 rank_key 와 같은 계층:
       합격 1 + (Isp − 150)/ISP_SCALE  >  시뮬레이션 후 불합격 −min(위반, 5)/5  >  형상 불합격 −1 − min(위반, 5)/5
       (계층이 없으면 형상 불합격 쪽 위반이 더 작아 보여 에이전트가 시뮬레이션 전 탈락 영역으로 도망간다)
       → 에피소드 보상 합 = 마지막 점수 − 처음 점수
"""

import math
import random

import numpy as np

import GA
from rl import features
import constrain as cs
from rl.state import CONSTRAINT_NAMES

ISP_SCALE = 20.0
STEP_MIN, STEP_MAX = 0.002, 0.10        # 한 스텝 이동 크기 (탐색 범위 대비 비율)
PIPE_THRESH = 0.7
N_VARS = len(GA.DESIGN_VARS)
ACTION_DIM = N_VARS + 2
_CIDX = {n: i for i, n in enumerate(CONSTRAINT_NAMES)}
_ALIAS = {"OxtPressure": "OxtFill"}     # 관측 차원을 바꾸지 않도록 탱크 압력 위반은 탱크 충진 칸에 (학습된 모델 호환)

TREND_DIM = 20 + 10 * 4 + 9
BASE_DIM = N_VARS + 4 + 5 + len(CONSTRAINT_NAMES) + 4
OBS_DIM = BASE_DIM + TREND_DIM


def score_of(feasible, isp, violation, isp_scale=ISP_SCALE, simulated=True):
    if feasible:
        return 1.0 + max(isp - 150.0, 0.0) / isp_scale
    return -min(violation, 5.0) / 5.0 - (0.0 if simulated else 1.0)


def step_size(a_step):
    t = (float(np.clip(a_step, -1, 1)) + 1) / 2
    return STEP_MIN * (STEP_MAX / STEP_MIN) ** t


def apply_action(rng, genome, action, n_pipes):
    """genome 에 행동을 적용한 새 genome (양자화 포함). 정수 변수는 확률적 반올림으로 작은 이동도 기대값이 맞게"""
    a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    step = step_size(a[N_VARS])
    g = dict(genome)
    for k, var in enumerate(GA.DESIGN_VARS):
        lo, hi = GA.BOUNDS[var]
        delta = a[k] * step * (hi - lo)
        if var in GA.INT_VARS:
            whole = math.floor(delta)
            g[var] = g[var] + whole + (1 if rng.random() < delta - whole else 0)
        else:
            g[var] = g[var] + delta
    if n_pipes > 1:
        if a[N_VARS + 1] > PIPE_THRESH:
            g["pipe"] = (g["pipe"] + 1) % n_pipes
        elif a[N_VARS + 1] < -PIPE_THRESH:
            g["pipe"] = (g["pipe"] - 1) % n_pipes
    return GA.quantize(g)


# ──────────────────────────────────────────────────────────────────────────
# 관측
# ──────────────────────────────────────────────────────────────────────────

class PipeNorm:
    def __init__(self, pipes):
        self.n = len(pipes)
        arr = np.array([[p["inner_diameter"], p["wall_thickness"], p["length"]] for p in pipes], dtype=float)
        lo, hi = arr.min(0), arr.max(0)
        self.feat = (arr - lo) / np.where(hi > lo, hi - lo, 1.0)

    def __call__(self, idx):
        return list(self.feat[idx]) + [idx / max(1, self.n - 1)]


def _pool(x, n):
    return x[:len(x) // n * n].reshape(n, -1).mean(1)


def trend_features(series, targets, simt, p_ok=1.0, isp_std=0.0, dt=GA.SERIES_DT):
    """
    예측한 연소 곡선 (C, N_T) → 경향성 특징 TREND_DIM 개 (대략 0~1, 일부 음수)
    MDO 와 같은 시뮬레이션 시간(simt) 까지만 본다.
    """
    impulse_t, thrust_t = targets[0], targets[1]
    n = int(round(simt / dt))
    th, pc, pt, mo, mf = (np.maximum(series[i, :n], 0.0) for i in range(5))
    f_max = th.max()
    burn = th >= GA.BURN_THRESHOLD * f_max if f_max > 0 else np.zeros(n, dtype=bool)
    f_curve = np.clip(_pool(th, 20) / thrust_t, 0, 3) / 3
    pc_curve = np.clip(_pool(pc, 10) / 5e6, 0, 2) / 2
    pt_curve = np.clip(_pool(pt, 10) / 6e6, 0, 2) / 2
    on = burn & (pc > 1e4)
    dp = np.where(on, (pt - pc) / np.maximum(pc, 1e4), 0.0)
    dp_curve = np.clip(_pool(dp, 10), -1, 1)
    of = np.where(mf > 1e-4, mo / np.maximum(mf, 1e-4), 0.0)
    of_curve = np.clip(_pool(of, 10) / 10, 0, 2) / 2

    idx = np.flatnonzero(burn)
    if idx.size:
        b0, b1 = idx[0], idx[-1] + 1
        bt = (b1 - b0) * dt
        seg = th[b0:b1]
        avg = seg.mean()
        q = max(1, len(seg) // 4)
        decay = seg[-2 * q:-q].mean() / max(seg[q:2 * q].mean(), 1e-6) if len(seg) >= 4 else 1.0
        # 액체 산화제 소진: 0.5 s 사이 최대 추력의 20% 이상 급락하는 첫 시점
        k = max(1, int(round(0.5 / dt)))
        drop = th[b0 + k:b1] - th[b0:b1 - k] if b1 - b0 > k else np.array([0.0])
        j = int(drop.argmin())
        dropped = drop[j] <= -0.2 * f_max
        t_drop = (j + k) * dt / bt if dropped else 1.0
        tail = th[b0 + j + k:b1].sum() / max(seg.sum(), 1e-6) if dropped else 0.0
        still = float(th[-1] >= GA.BURN_THRESHOLD * f_max)
    else:
        bt, avg, decay, t_drop, tail, still = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    burn_target = impulse_t / thrust_t
    scal = [min(bt / burn_target, 3) / 3, min(f_max / max(avg, 1e-6), 5) / 5 if avg > 0 else 0.0,
            float(np.clip(decay, 0, 2)) / 2, t_drop, float(np.clip(tail, 0, 1)), still,
            float(p_ok), min(isp_std / 5.0, 1.0), min(th.sum() * dt / impulse_t, 3) / 3]
    return np.concatenate([f_curve, pc_curve, pt_curve, dp_curve, of_curve, scal]).astype(np.float32)


def base_features(genome, sim, cvec, feasible, targets, progress, pipe_norm):
    x = [(genome[v] - GA.BOUNDS[v][0]) / (GA.BOUNDS[v][1] - GA.BOUNDS[v][0]) for v in GA.DESIGN_VARS]
    impulse_t, thrust_t = targets[0], targets[1]
    if sim is not None and not sim.get("error"):
        dp = sim.get("dp_ratio_min")
        perf = [1.0,
                float(np.clip((sim["isp"] - 150.0) / 100.0, -1, 1.5)),
                float(np.clip(sim["impulse"] / impulse_t - 1.0, -1, 1)),
                float(np.clip(sim["thrust_avg"] / thrust_t - 1.0, -1, 1)),
                float(np.clip(dp if dp is not None else cs.injector_dp_ratio(sim), -1, 1))]
    else:
        perf = [0.0] * 5
    tail = [float(feasible), float(progress), min(impulse_t / 20000.0, 1.0), min(thrust_t / 2000.0, 1.0)]
    return np.concatenate([np.array(x + pipe_norm(genome["pipe"]) + perf, dtype=np.float32),
                           cvec, np.array(tail, dtype=np.float32)])


def constraint_vector(design, pipe, sim, targets, margin=None):
    """제약별 위반량 (log1p, 0~1) 벡터와 총 위반량. sim 이 None 이면 형상 제약만"""
    vec = np.zeros(len(CONSTRAINT_NAMES), dtype=np.float32)
    cons = cs.pre_sim_constraints(design, pipe)[0]
    if sim is not None:
        cons = cons + cs.post_sim_constraints(design, pipe, sim, *targets, margin=margin)
    total = 0.0
    for c in cons:
        v = c.violation()
        if v > 0:
            total += v
            i = _CIDX.get(_ALIAS.get(c.name(), c.name()))
            if i is not None:
                vec[i] = min(math.log1p(v) / 3.0, 1.0)
    return vec, total


BURN_RANGE = (3.0, 18.0)     # 목표 연소 시간 I/F [s] — MDO 시뮬레이션 시간(20 s) 안에 끝나야 합격 가능


def sample_targets(np_rng, impulse_range, thrust_range):
    while True:
        imp = round(float(np_rng.uniform(*impulse_range)), -2)
        thr = round(float(np_rng.uniform(*thrust_range)), -1)
        if BURN_RANGE[0] <= imp / thr <= BURN_RANGE[1]:
            return imp, thr


# ──────────────────────────────────────────────────────────────────────────
# 학습 환경 — 에이전트 N 명을 한 번에 (SB3 VecEnv)
# ──────────────────────────────────────────────────────────────────────────

try:
    from gymnasium import spaces
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv
except ImportError:          # 웹 서버에서 DirectStrategy 만 쓸 때
    VecEnv = object


class DesignVecEnv(VecEnv):
    """
    에이전트 n_agents 명 = SB3 의 병렬 환경 n_agents 개. 에피소드마다 목표·시작 설계·길이를 새로 뽑는다.
    평가는 GA 의 대리모델 평가와 같다: 형상 제약 불합격이면 시뮬레이션 없음, 아니면 예측값으로 결과 제약(SURROGATE_MARGIN).
    관측용 곡선은 모든 설계에 대해 예측한다 (형상 불합격 설계도 곡선은 있음).
    info (에피소드 끝): best_isp(에피소드 중 최고 합격 Isp), final_isp, feasible, score
    """

    obs_dim = OBS_DIM

    def action_space_(self):
        return spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)

    def __init__(self, n_agents=256, seed=0, impulse_range=(6000, 16000), thrust_range=(400, 1500),
                 len_range=(30, 150), isp_scale=ISP_SCALE, device="cuda", isp_penalty=0.0, model_path=None,
                 t_margin=0.0, ok_threshold=0.5):
        # isp_penalty: 점수에 쓰는 Isp = 예측 − isp_penalty × 앙상블 표준편차 (불확실한 곳을 악용하지 않게)
        # t_margin: 예측 종료 시각이 simt − t_margin 보다 늦으면 시간 초과로 본다
        from rl.ts_surrogate import MODEL_PATH, TSSurrogateSimulator
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.simulator = TSSurrogateSimulator(model_path or MODEL_PATH, device=device, t_margin=t_margin,
                                              ok_threshold=ok_threshold)
        self.isp_penalty = isp_penalty
        self.simt = self.simulator.simt
        self.pipes = GA.load_candidate_pipes()
        self.fixed = GA.load_fixed_values()
        self.pipe_norm = PipeNorm(self.pipes)
        self.impulse_range, self.thrust_range = impulse_range, thrust_range
        self.len_range = len_range
        self.isp_scale = isp_scale
        self.fixed_options = None          # 평가용: 에이전트별 {"targets", "seed", "n_steps"}
        super().__init__(n_agents, spaces.Box(-2.0, 2.0, shape=(self.obs_dim,), dtype=np.float32),
                         self.action_space_())
        self.n = n_agents
        self.genomes = [None] * n_agents
        self.targets = [None] * n_agents
        self.T = np.zeros(n_agents, dtype=int)
        self.t = np.zeros(n_agents, dtype=int)
        self.score = np.zeros(n_agents)
        self.best_isp = np.zeros(n_agents)
        self.best_genome = [None] * n_agents     # 에피소드 중 최고 합격 설계 (core 검증용)
        self.ep_ret = np.zeros(n_agents)
        self.n_sims = 0
        self._actions = None

    def _new_episode(self, i):
        opt = self.fixed_options[i] if self.fixed_options else None
        if opt:
            self.targets[i] = (*opt["targets"], cs.TARGET_TOLERANCE)
            self.T[i] = opt["n_steps"]
            rng = random.Random(opt["seed"])
        else:
            self.targets[i] = (*sample_targets(self.np_rng, self.impulse_range, self.thrust_range), cs.TARGET_TOLERANCE)
            self.T[i] = int(self.np_rng.integers(self.len_range[0], self.len_range[1] + 1))
            rng = self.rng
        # 시작 설계 = MDO 의 초기 개체와 같은 방식 (형상 제약을 만족하는 무작위 설계)
        self.genomes[i] = GA.initial_population(rng, self.pipes, self.fixed, 1)[0]
        self.t[i] = 0
        self.best_isp[i] = 0.0
        self.best_genome[i] = None
        self.ep_ret[i] = 0.0

    def evaluate(self, idx):
        """에이전트 idx 들의 현재 설계 평가 → (obs, score, feasible, isp)"""
        decoded = [GA.decode(self.genomes[i], self.pipes, self.fixed) for i in idx]
        sims = self.simulator.simulate_batch([d for d, _ in decoded], [p for _, p in decoded])
        out = self.simulator.last_output
        self.n_sims += len(idx)
        obs = np.zeros((len(idx), self.obs_dim), dtype=np.float32)
        scores, feas, isps = np.zeros(len(idx)), np.zeros(len(idx), dtype=bool), np.zeros(len(idx))
        for k, i in enumerate(idx):
            design, pipe = decoded[k]
            tg = self.targets[i]
            cvec_pre, v_pre = constraint_vector(design, pipe, None, tg)
            sim = sims[k] if v_pre <= 0 else None            # 형상 불합격 → GA 처럼 시뮬레이션 안 함
            if sim is not None:
                cvec, v = constraint_vector(design, pipe, sim, tg, cs.SURROGATE_MARGIN)
            else:
                cvec, v = cvec_pre, v_pre
            ok = sim is not None and v <= 0
            isp = sim["isp"] if sim is not None else 0.0
            isp_s = isp - self.isp_penalty * sim["isp_std"] if sim is not None else 0.0
            scores[k], feas[k], isps[k] = score_of(ok, isp_s, v, self.isp_scale, sim is not None), ok, isp
            if ok and isp > self.best_isp[i]:
                self.best_isp[i] = isp
                self.best_genome[i] = dict(self.genomes[i])
            obs[k] = self.make_obs(i, sim, cvec, ok, tg, out, k)
        return obs, scores, feas, isps

    def make_obs(self, i, sim, cvec, ok, tg, out, k):
        base = base_features(self.genomes[i], sim, cvec, ok, tg, self.t[i] / max(1, self.T[i]), self.pipe_norm)
        trend = trend_features(out["series"][k], tg, self.simt, out["p_ok"][k], out["scalars_std"][k][0])
        return np.concatenate([base, trend])

    def apply(self, genome, action):
        return apply_action(self.rng, genome, action, len(self.pipes))

    def reset(self):
        for i in range(self.n):
            self._new_episode(i)
        obs, self.score[:], _, _ = self.evaluate(list(range(self.n)))
        return obs

    def step_async(self, actions):
        self._actions = np.asarray(actions)

    def step_wait(self):
        for i in range(self.n):
            self.genomes[i] = self.apply(self.genomes[i], self._actions[i])
        self.t += 1
        obs, new_score, feas, isps = self.evaluate(list(range(self.n)))
        rewards = (new_score - self.score).astype(np.float32)
        self.score = new_score
        self.ep_ret += rewards
        dones = self.t >= self.T
        infos = [{} for _ in range(self.n)]
        if dones.any():
            restart = [int(i) for i in np.flatnonzero(dones)]
            for i in restart:
                infos[i] = {"terminal_observation": obs[i].copy(),
                            "best_isp": float(self.best_isp[i]), "best_genome": self.best_genome[i],
                            "feasible": float(self.best_isp[i] > 0),
                            "final_feasible": float(feas[i]), "final_isp": float(isps[i]) if feas[i] else 0.0,
                            "score": float(self.score[i]),
                            "episode": {"r": float(self.ep_ret[i]), "l": int(self.t[i])}}
                self._new_episode(i)
            obs2, sc2, _, _ = self.evaluate(restart)
            obs[restart] = obs2
            self.score[restart] = sc2
        return obs, rewards, dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self, method_name)(*args, **kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))

    def seed(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
            self.np_rng = np.random.default_rng(seed)
        return [seed] * self.n


# ──────────────────────────────────────────────────────────────────────────
# 실제 MDO 에서 사용 — GA.run_ga(strategy=DirectStrategy(...))
# ──────────────────────────────────────────────────────────────────────────

class DirectStrategy:
    """세대마다 개체 하나하나를 에이전트가 직접 한 걸음 옮긴다 (교배·선택 없음)"""

    kind = "direct"

    def __init__(self, model_path, device="cpu", deterministic=True, seed=None):
        import torch
        from stable_baselines3 import PPO
        from rl.ts_surrogate import TSSurrogateSimulator
        if seed is not None:
            torch.manual_seed(seed)
        self.model = PPO.load(model_path, device=device)
        if self.model.observation_space.shape[0] != OBS_DIM:
            raise ValueError(f"모델 관측 {self.model.observation_space.shape[0]}차원 ≠ 현재 {OBS_DIM}차원. 다시 학습하세요.")
        self.ts = TSSurrogateSimulator(device="cpu")          # 관측용 곡선 예측
        self.model_path = model_path
        self.deterministic = deterministic
        self.pipe_norm = None
        self.last_curves = None

    def __call__(self, rng, evaluations, pipes, history, ctx, parents=None):
        if self.pipe_norm is None or self.pipe_norm.n != len(pipes):
            self.pipe_norm = PipeNorm(pipes)
        targets = ctx.get("targets_full") or (*ctx["targets"], cs.TARGET_TOLERANCE)
        margin = cs.SURROGATE_MARGIN if ctx.get("surrogate") else None
        progress = (ctx["gen"] + 1) / max(1, ctx["n_generations"])
        out = self.ts.predict_designs([ev["design"] for ev in evaluations], [ev["pipe"] for ev in evaluations])
        obs = np.zeros((len(evaluations), OBS_DIM), dtype=np.float32)
        for k, ev in enumerate(evaluations):
            cvec, _ = constraint_vector(ev["design"], ev["pipe"], ev["sim"], targets, margin)
            obs[k, :BASE_DIM] = base_features(ev["genome"], ev["sim"], cvec, ev["feasible"], targets, progress,
                                              self.pipe_norm)
            obs[k, BASE_DIM:] = trend_features(out["series"][k], targets, self.ts.simt, out["p_ok"][k],
                                               out["scalars_std"][k][0])
        actions = self.model.predict(obs, deterministic=self.deterministic)[0]
        population = [apply_action(rng, ev["genome"], a, len(pipes)) for ev, a in zip(evaluations, actions)]
        a = np.clip(actions, -1, 1)
        steps = np.array([step_size(x) for x in a[:, N_VARS]])
        move = np.abs(a[:, :N_VARS]).mean(0) * steps.mean()          # 변수별 평균 이동 (탐색 범위 대비)
        # 이번 세대 최고 개체(evaluations[0])의 예측 곡선 — 웹에서 경향성 표시 (1 s 간격으로 줄여 보냄)
        n = int(round(self.ts.simt / GA.SERIES_DT))
        curve = {c: [round(float(v), 4) for v in _pool(out["series"][0][i, :n], n // 10)]
                 for i, c in enumerate(GA.SERIES_CHANNELS)}
        info = {"action": -1, "name": f"직접 조정 · 평균 스텝 {steps.mean() * 100:.2f}%",
                "direct": {"step_mean": float(steps.mean()), "step_min": float(steps.min()),
                           "step_max": float(steps.max()),
                           "pipe_changes": int(np.sum(np.abs(a[:, N_VARS + 1]) > PIPE_THRESH)),
                           "move": [float(m) for m in move], "curve": curve, "curve_dt": GA.SERIES_DT * n / (n // 10),
                           "best_id": GA.design_id(evaluations[0]["design"], evaluations[0]["pipe"])}}
        return population, info


def list_models():
    """웹에서 고를 수 있는 직접 설계 모델 (rl/runs/direct_*/best_model.zip, final_model.zip) — 현재 관측 규격과 맞는 것만"""
    import csv
    import glob
    import os
    import zipfile
    import json
    here = os.path.dirname(os.path.abspath(__file__))
    opt_dir = os.path.dirname(here)
    models = []
    for run_dir in sorted(glob.glob(os.path.join(here, "runs", "direct_*")), reverse=True):
        name = os.path.basename(run_dir)
        if not features.run_compatible(run_dir):        # 예전 설계 변수 구성 (챔버 길이 변수, 헬륨 고정)
            continue
        rows = []
        prog = os.path.join(run_dir, "progress.csv")
        if os.path.exists(prog):
            with open(prog, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        best = max(rows, key=lambda r: float(r["score"])) if rows else None
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
            r = best if label == "best" else (rows[-1] if rows else None)
            models.append({"id": os.path.relpath(path, opt_dir), "label": f"{name} · {label}",
                           "mtime": os.path.getmtime(path), "steps": int(rows[-1]["step"]) if rows else None,
                           "eval": {k: float(r[k]) for k in ("success", "best_isp", "isp_gain", "win_rate") if k in r} if r else None})
    return models
