"""
에이전트 직접 설계(rl/direct.py) PPO 학습 — 시계열 대리모델(rl/ts_surrogate.pt) 환경
실행 (optimizer 폴더에서)
    python -m rl.train_direct --steps 3000000
모니터링: http://127.0.0.1:10000/training  (status.json / live.jsonl / progress.csv — train_ppo 와 같은 형식, kind=direct)
결과: rl/runs/direct_<시각>/  best_model.zip, final_model.zip

평가: 같은 (목표, 시드) 에피소드마다 에이전트 32명 × 100스텝 (MDO 개체 32 × 100세대와 같은 평가 수) 의
      최고 합격 Isp 를, 같은 대리모델로 평가한 기본 GA (개체 32 × 100세대) 와 짝지어 비교한다.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
import warnings
from collections import deque

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, TensorBoardOutputFormat
from stable_baselines3.common.vec_env import VecNormalize

import GA
import constrain as cs
from rl import direct
from rl.benchmark import TARGETS as EVAL_TARGETS
from rl.controlled_ga import ControlledGA
from rl.train_ppo import LiveWriter, Status

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")
EVAL_SEEDS = [1000, 1001, 1002]
N_EVAL_RANDOM = 40
EVAL_POP = 32


def eval_episodes(impulse_range, thrust_range):
    rng = np.random.default_rng(12345)
    episodes = [(tuple(t), s) for t in EVAL_TARGETS for s in EVAL_SEEDS]
    for k in range(N_EVAL_RANDOM):
        episodes.append((direct.sample_targets(rng, impulse_range, thrust_range), 2000 + k))
    return episodes


class TSEvaluator:
    """ControlledGA 용 평가기 — GA.evaluate 를 시계열 대리모델로 (MDO 의 대리모델 평가와 같은 경로)"""

    def __init__(self, simulator):
        self.sim = simulator
        self.pipes = GA.load_candidate_pipes()
        self.fixed = GA.load_fixed_values()
        self.caches = {}
        self.counters = {"pre_rejected": 0, "simulated": 0}

    @property
    def simulated(self):
        return self.counters["simulated"]

    def __call__(self, population, targets):
        cache = self.caches.setdefault(tuple(targets), {})
        return GA.evaluate(population, self.pipes, self.fixed, tuple(targets), cache, None, self.counters,
                           surrogate=self.sim)


def baseline_eval(simulator, episodes, n_gens):
    """기본 GA (전략 0 고정) 를 같은 에피소드로"""
    ev = TSEvaluator(simulator)
    rows = []
    for targets, seed in episodes:
        ev.caches.clear()
        ga = ControlledGA(ev, targets, EVAL_POP, n_gens, seed=seed)
        ga.reset()
        while not ga.done:
            ga.step(0)
        isp = ga.history[-1]["best_isp"]
        rows.append(isp)
    return rows


def agent_eval(eval_env, predict, episodes, n_steps):
    opts = [{"targets": t, "seed": sd * 1000 + j, "n_steps": n_steps} for t, sd in episodes for j in range(EVAL_POP)]
    eval_env.fixed_options = opts
    obs = eval_env.reset()
    best = np.zeros(len(opts))
    for _ in range(n_steps):
        obs, _, dones, infos = eval_env.step(predict(obs))
        for i, info in enumerate(infos):
            if "best_isp" in info:
                best[i] = max(best[i], info["best_isp"])
        best = np.maximum(best, eval_env.best_isp)      # 끝나면 초기화되므로 직전 값도 같이
    per_ep = best.reshape(len(episodes), EVAL_POP).max(1)
    return [float(x) if x > 0 else None for x in per_ep]


def summarize(isps, base):
    ok = [x for x in isps if x is not None]
    pairs = [(a, b) for a, b in zip(isps, base) if a is not None and b is not None]
    return {"success": float(np.mean([x is not None for x in isps])),
            "best_isp": float(np.mean(ok)) if ok else 0.0,
            "isp_gain": float(np.mean([a - b for a, b in pairs])) if pairs else 0.0,
            "win_rate": float(np.mean([a > b for a, b in pairs])) if pairs else 0.0,
            "score": float(np.mean([direct.score_of(True, x, 0) if x is not None else -1.0 for x in isps]))}


class DirectCallback(BaseCallback):
    def __init__(self, eval_env, episodes, baseline_isps, log_dir, eval_freq, total_steps, eval_steps, status):
        super().__init__()
        self.eval_env, self.episodes, self.base = eval_env, episodes, baseline_isps
        self.log_dir, self.eval_freq, self.total_steps, self.eval_steps = log_dir, eval_freq, total_steps, eval_steps
        self.status = status
        self.episodes_done = deque(maxlen=2000)
        self.abs_act = []
        self.next_eval = eval_freq
        self.best_score = -np.inf
        self.csv_path = os.path.join(log_dir, "progress.csv")
        self.t0 = time.time()

    def _on_step(self):
        a = np.clip(np.asarray(self.locals["actions"]), -1, 1)
        self.abs_act.append(np.concatenate([np.abs(a[:, :direct.N_VARS]).mean(0),
                                            [np.mean([direct.step_size(x) for x in a[:, direct.N_VARS]]),
                                             float(np.mean(np.abs(a[:, direct.N_VARS + 1]) > direct.PIPE_THRESH))]]))
        for info in self.locals["infos"]:
            if "best_isp" in info:
                self.episodes_done.append(info)
        return True

    def _on_rollout_end(self):
        if self.episodes_done:
            eps = list(self.episodes_done)
            isps = [e["best_isp"] for e in eps if e["feasible"]]
            self.logger.record("ga/success_rate", np.mean([e["feasible"] for e in eps]))
            self.logger.record("ga/best_isp", np.mean(isps) if isps else 0.0)
            self.logger.record("ga/final_feasible", np.mean([e["final_feasible"] for e in eps]))
            self.logger.record("ga/ep_score", np.mean([e["score"] for e in eps]))
        if self.abs_act:
            m = np.mean(self.abs_act, 0)
            for var, v in zip(GA.DESIGN_VARS, m[:direct.N_VARS]):
                self.logger.record(f"move/{var}", float(v))
            self.logger.record("direct/step_mean", float(m[direct.N_VARS]))
            self.logger.record("direct/pipe_change", float(m[direct.N_VARS + 1]))
            self.abs_act = []
        std = torch.exp(self.model.policy.log_std).mean().item()
        self.logger.record("direct/action_std", std)
        if self.num_timesteps >= self.next_eval:
            self.next_eval = self.num_timesteps + self.eval_freq
            self._evaluate()

    def _evaluate(self):
        isps = agent_eval(self.eval_env, lambda o: self.model.predict(o, deterministic=True)[0],
                          self.episodes, self.eval_steps)
        res = summarize(isps, self.base)
        res["sims"] = float(EVAL_POP * self.eval_steps)
        for k, v in res.items():
            self.logger.record(f"eval/{k}", v)
        is_best = res["score"] > self.best_score
        if is_best:
            self.best_score = res["score"]
            self.model.save(os.path.join(self.log_dir, "best_model"))
            self.status["best"] = {"step": self.num_timesteps, **{k: res[k] for k in ("isp_gain", "win_rate", "success", "best_isp")}}
        elapsed = time.time() - self.t0
        fps = self.num_timesteps / max(elapsed, 1e-9)
        row = {"step": self.num_timesteps, "elapsed_s": round(elapsed), **{k: round(v, 4) for k, v in res.items()}}
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        b = self.status["baseline"]
        print(f"[{self.num_timesteps:>9,}/{self.total_steps:,}] {fps:5.0f} fps, 남은 {(self.total_steps - self.num_timesteps) / max(fps, 1e-9) / 60:5.1f}분 | "
              f"평가: 성공 {res['success']:.0%} (GA {b['success']:.0%})  Isp {res['best_isp']:.2f} (GA {b['best_isp']:.2f})  "
              f"GA 대비 {res['isp_gain']:+.2f} s, 승률 {res['win_rate']:.0%}{'  ★ best' if is_best else ''}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3_000_000, help="총 스텝 (에이전트 1명의 1걸음 = 1 스텝)")
    p.add_argument("--n-agents", type=int, default=256, help="동시에 움직이는 에이전트 수 (병렬 환경 수)")
    p.add_argument("--n-steps", type=int, default=128, help="에이전트당 롤아웃 길이")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-end-frac", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--target-kl", type=float, default=0.03)
    p.add_argument("--len-min", type=int, default=30, help="에피소드 길이(스텝) 최소")
    p.add_argument("--len-max", type=int, default=150)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--eval-freq", type=int, default=250_000)
    p.add_argument("--isp-scale", type=float, default=direct.ISP_SCALE)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    p.add_argument("--init", default=None, help="이 모델(zip)의 정책 가중치에서 이어서 학습 (예: 대리모델을 다시 학습한 뒤)")
    args = p.parse_args(argv)

    name = args.name or "direct_" + time.strftime("%Y%m%d_%H%M")
    log_dir = os.path.join(RUNS_DIR, name)
    os.makedirs(log_dir, exist_ok=True)
    cfg = dict(vars(args), kind="direct", n_envs=args.n_agents, eval_gens=args.eval_steps,
               gen_min=args.len_min, gen_max=args.len_max, obs_dim=direct.OBS_DIM, action_dim=direct.ACTION_DIM,
               design_vars=GA.DESIGN_VARS, temperature=GA.load_fixed_values()["temperature"])
    with open(os.path.join(log_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    status = Status(os.path.join(log_dir, "status.json"), name=name, kind="direct", state="starting", pid=os.getpid(),
                    started=time.time(), total_steps=args.steps, step=0, config=cfg,
                    vars=GA.DESIGN_VARS)
    status.save()

    env_kw = {"len_range": (args.len_min, args.len_max), "isp_scale": args.isp_scale, "device": args.device}
    env = direct.DesignVecEnv(args.n_agents, seed=args.seed, **env_kw)
    episodes = eval_episodes(env.impulse_range, env.thrust_range)
    eval_env = direct.DesignVecEnv(len(episodes) * EVAL_POP, seed=args.seed + 1, **env_kw)

    print(f"실행 {name} | 장치 {args.device} | 에이전트 {args.n_agents} | 총 {args.steps:,} 스텝 | 관측 {direct.OBS_DIM} · 행동 {direct.ACTION_DIM}")
    print(f"  로그: {log_dir}\n  모니터링: http://127.0.0.1:10000/training", flush=True)

    status.update(state="baseline")
    status.save()
    t = time.time()
    base = baseline_eval(env.simulator, episodes, args.eval_steps)
    b = summarize(base, base)
    status.update(state="running", baseline={k: b[k] for k in ("success", "best_isp", "score")}, eval_episodes=len(episodes))
    status.save()
    print(f"  기준(기본 GA, 같은 대리모델, 개체 {EVAL_POP} × {args.eval_steps}세대): 성공 {b['success']:.0%}  Isp {b['best_isp']:.2f}  ({time.time() - t:.0f}s)", flush=True)

    venv = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=args.gamma)
    lr0, lr1 = args.lr, args.lr * args.lr_end_frac
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*run PPO on the GPU.*")
        model = PPO("MlpPolicy", venv, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=10,
                    gamma=args.gamma, gae_lambda=0.95, learning_rate=lambda pr: lr1 + (lr0 - lr1) * pr,
                    ent_coef=args.ent_coef, target_kl=args.target_kl, seed=args.seed, device=args.device,
                    policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}, "log_std_init": -0.5},
                    verbose=0)
    if args.init:
        model.policy.load_state_dict(PPO.load(args.init, device=args.device).policy.state_dict())
        print(f"  초기 가중치: {args.init}", flush=True)
    live = LiveWriter(log_dir, status)
    model.set_logger(Logger(log_dir, [TensorBoardOutputFormat(os.path.join(log_dir, "tb")), live]))
    cb = DirectCallback(eval_env, episodes, base, log_dir, args.eval_freq, args.steps, args.eval_steps, status)
    try:
        model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False)
        cb._evaluate()
        status["state"] = "done"
    except KeyboardInterrupt:
        print("\n중단됨 — 지금까지의 정책을 저장합니다")
        status["state"] = "stopped"
    except Exception as exc:
        status.update(state="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        model.save(os.path.join(log_dir, "final_model"))
        status.update(finished=time.time(), step=int(model.num_timesteps))
        status.save()
        live.close()
    print(f"저장: {log_dir}/final_model.zip, best_model.zip")


if __name__ == "__main__":
    main()
