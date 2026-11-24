"""
대리모델 환경에서 PPO 학습
실행 (optimizer 폴더에서)
    python -m rl.train_ppo --steps 2000000
모니터링
    웹: http://127.0.0.1:10000/training   (MDO 서버의 학습 모니터, 3초마다 갱신)
    또는 tensorboard --logdir rl/runs
결과: rl/runs/<실행이름>/
    status.json         진행 상태 (웹 모니터가 읽음)
    live.jsonl          업데이트마다 학습 지표 한 줄 (웹 모니터가 읽음)
    progress.csv        평가 기록 (스텝, 성공률, 평균 best Isp, 기본 GA 대비 ...)
    monitor/*.csv       환경별 에피소드 기록
    checkpoints/        주기 저장
    best_model.zip      평가 점수가 가장 좋았던 정책
    final_model.zip     마지막 정책 (rl/ppo_ga.zip 에도 복사 → benchmark 가 읽음)
"""

import argparse
import csv
import functools
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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import KVWriter, Logger, TensorBoardOutputFormat
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from rl.actions import N_ACTIONS
from rl.benchmark import TARGETS as EVAL_TARGETS
from rl.controlled_ga import COST_PER_SIM, ISP_SCALE
import GA
from rl.env import GAEnv
from rl.surrogate import SurrogateEvaluator

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")
# TensorBoard 태그용 영어 이름 (actions.ACTIONS 순서와 같게)
ACTION_TAGS = ["basic", "explore", "exploit", "local", "inject", "finetune", "crossover", "repair"]
EVAL_SEEDS = [1000, 1001, 1002]
N_EVAL_RANDOM = 40    # 평가에 더하는 무작위 목표 수. 고정 목표 몇 개만으로는 표본이 작고 치우쳐 순위가 뒤집힌다


def eval_episodes(impulse_range, thrust_range):
    """평가 에피소드 (목표, 시드) 목록. 기본 GA 와 정책이 같은 목록을 써서 짝지어 비교한다"""
    rng = np.random.default_rng(12345)
    episodes = [(t, s) for t in EVAL_TARGETS for s in EVAL_SEEDS]
    for k in range(N_EVAL_RANDOM):
        target = (round(float(rng.uniform(*impulse_range)), -2), round(float(rng.uniform(*thrust_range)), -1))
        episodes.append((target, 2000 + k))
    return episodes


def make_env(rank, log_dir, seed, env_kwargs, monitor=True):
    """서브프로세스 안에서 환경 생성. 환경 하나가 torch 스레드를 여러 개 잡으면 서로 경쟁해 느려진다."""
    torch.set_num_threads(1)
    env = GAEnv(SurrogateEvaluator(), **env_kwargs)
    if monitor:
        env = Monitor(env, os.path.join(log_dir, "monitor", str(rank)),
                      info_keywords=("best_isp", "feasible", "sims"))
    env.reset(seed=seed + rank)
    return env


def evaluate(vec_env, predict, n_generations):
    """
    평가 에피소드 전체를 병렬 환경으로 돌린다 (환경 수만큼 한 번에).
    predict(obs 배치) → 행동 배치. 에피소드 길이가 모두 같아 한 묶음이 동시에 끝난다.
    """
    episodes = eval_episodes(vec_env.get_attr("impulse_range")[0], vec_env.get_attr("thrust_range")[0])
    n = vec_env.num_envs
    rows = []
    for k in range(0, len(episodes), n):
        wave = episodes[k:k + n]
        wave_full = wave + [wave[-1]] * (n - len(wave))          # 남는 환경은 같은 에피소드를 한 번 더 (결과는 버림)
        vec_env.set_options([{"targets": t, "seed": sd, "n_generations": n_generations} for t, sd in wave_full])
        obs = vec_env.reset()
        done_info = [None] * n
        while any(d is None for d in done_info):
            obs, _, dones, infos = vec_env.step(predict(obs))
            for i, (d, info) in enumerate(zip(dones, infos)):
                if d and done_info[i] is None:
                    done_info[i] = info
        rows += done_info[:len(wave)]
    isps = [r["best_isp"] for r in rows if r["feasible"]]
    return {
        "isps": [r["best_isp"] if r["feasible"] else None for r in rows],
        "success": float(np.mean([r["feasible"] for r in rows])),
        "best_isp": float(np.mean(isps)) if isps else 0.0,
        "score": float(np.mean([r["score"] for r in rows])),
        "sims": float(np.mean([r["sims"] for r in rows])),
    }


class LiveWriter(KVWriter):
    """
    SB3 로거가 업데이트마다 내보내는 지표를 live.jsonl 에 한 줄씩 쓰고 status.json 을 갱신한다.
    웹 학습 모니터(/training)가 이 두 파일을 읽는다.
    """

    def __init__(self, log_dir, status):
        self.path = os.path.join(log_dir, "live.jsonl")
        self.status = status
        self.f = open(self.path, "a", encoding="utf-8")

    def write(self, key_values, key_excluded, step=0):
        row = {"step": int(step), "t": round(time.time(), 1)}
        for k, v in key_values.items():
            if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v):
                row[k] = float(v)
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()
        self.status.update(step=int(step), fps=row.get("time/fps"))
        self.status.save()

    def close(self):
        self.f.close()


class Status(dict):
    """status.json — 실행 상태 (running / done / stopped / error), 진행 스텝, 기준선 등"""

    def __init__(self, path, **kw):
        super().__init__(**kw)
        self.path = path

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self, f, ensure_ascii=False)
        os.replace(tmp, self.path)


class MonitorCallback(BaseCallback):
    """
    - 학습 에피소드 결과 (best Isp, 성공률, 시뮬 수) 와 전략 사용 비율을 TensorBoard 에 기록
    - eval_freq 스텝마다 고정 목표로 평가 → TensorBoard + progress.csv + 콘솔, 최고 정책 저장
    """

    def __init__(self, eval_env, log_dir, eval_freq, baseline, total_steps, eval_gens, status):
        super().__init__()
        self.eval_gens = eval_gens
        self.status = status
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.baseline = baseline
        self.total_steps = total_steps
        self.episodes = deque(maxlen=200)
        self.action_counts = np.zeros(N_ACTIONS)
        self.next_eval = 0
        self.best_score = -np.inf
        self.eval_env = eval_env
        self.csv_path = os.path.join(log_dir, "progress.csv")
        self.t0 = time.time()

    def _on_step(self):
        for a in np.asarray(self.locals["actions"]).ravel():
            self.action_counts[int(a)] += 1
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "best_isp" in info:
                self.episodes.append(info)
        return True

    def _on_rollout_end(self):
        if self.episodes:
            eps = list(self.episodes)
            isps = [e["best_isp"] for e in eps if e["feasible"]]
            self.logger.record("ga/success_rate", np.mean([e["feasible"] for e in eps]))
            self.logger.record("ga/best_isp", np.mean(isps) if isps else 0.0)
            self.logger.record("ga/sims_per_episode", np.mean([e["sims"] for e in eps]))
        total = self.action_counts.sum()
        if total:
            for tag, cnt in zip(ACTION_TAGS, self.action_counts):
                self.logger.record(f"actions/{tag}", cnt / total)
        self.action_counts[:] = 0

        if self.num_timesteps >= self.next_eval:
            self.next_eval = self.num_timesteps + self.eval_freq
            self._evaluate()

    def _evaluate(self):
        res = evaluate(self.eval_env, lambda obs: self.model.predict(obs, deterministic=True)[0], self.eval_gens)
        # 같은 (목표, 시드) 끼리 Isp 차이 → 짝지은 비교 (둘 다 합격한 에피소드만)
        pairs = [(a, b) for a, b in zip(res.pop("isps"), self.baseline["isps"]) if a is not None and b is not None]
        res["isp_gain"] = float(np.mean([a - b for a, b in pairs])) if pairs else 0.0
        res["win_rate"] = float(np.mean([a > b for a, b in pairs])) if pairs else 0.0
        for k, v in res.items():
            self.logger.record(f"eval/{k}", v)
            if k in self.baseline:
                self.logger.record(f"eval/{k}_vs_baseline", v - self.baseline[k])
        is_best = res["score"] > self.best_score
        if is_best:
            self.best_score = res["score"]
            self.model.save(os.path.join(self.log_dir, "best_model"))
            self.status["best"] = {"step": self.num_timesteps, **{k: res[k] for k in ("isp_gain", "win_rate", "success", "best_isp")}}
        self.logger.record("eval/is_best", float(is_best))

        elapsed = time.time() - self.t0
        fps = self.num_timesteps / max(elapsed, 1e-9)
        eta = (self.total_steps - self.num_timesteps) / max(fps, 1e-9)
        row = {"step": self.num_timesteps, "elapsed_s": round(elapsed), **{k: round(v, 4) for k, v in res.items()}}
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)
        b = self.baseline
        print(f"[{self.num_timesteps:>9,}/{self.total_steps:,}] {fps:5.0f} fps, 남은 {eta/60:5.1f}분 | "
              f"평가: 성공 {res['success']:.0%} (기본 {b['success']:.0%})  "
              f"Isp {res['best_isp']:.2f} (기본 {b['best_isp']:.2f})  기본 대비 {res['isp_gain']:+.2f} s, 승률 {res['win_rate']:.0%}  "
              f"점수 {res['score']:+.3f} (기본 {b['score']:+.3f})  시뮬 {res['sims']:.0f}"
              f"{'  ★ best' if is_best else ''}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_000_000, help="총 학습 스텝 (1 스텝 = GA 1세대)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 4)),
                        help="병렬 환경 수 (CPU 프로세스)")
    parser.add_argument("--n-steps", type=int, default=512, help="환경당 롤아웃 길이 (업데이트당 n_steps × n_envs 스텝)")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4, help="시작 학습률 (끝에서 lr × lr-end-frac 까지 선형 감소)")
    parser.add_argument("--lr-end-frac", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.995, help="할인율 (에피소드가 길어 0.99 보다 길게 본다)")
    parser.add_argument("--target-kl", type=float, default=0.02, help="업데이트 중 KL 이 이 값을 넘으면 그 업데이트 중단 (안정화)")
    parser.add_argument("--no-norm-reward", action="store_true", help="보상 정규화(VecNormalize) 끄기")
    parser.add_argument("--gen-min", type=int, default=30, help="학습 에피소드 세대 수 최소")
    parser.add_argument("--gen-max", type=int, default=200, help="학습 에피소드 세대 수 최대 (실제 MDO 의 긴 실행까지 학습)")
    parser.add_argument("--eval-gens", type=int, default=100, help="평가 에피소드 세대 수")
    parser.add_argument("--isp-scale", type=float, default=ISP_SCALE,
                        help="보상: Isp 가 이 값[s] 오를 때 +1 (작을수록 Isp 를 강하게 추구)")
    parser.add_argument("--sim-cost", type=float, default=COST_PER_SIM,
                        help="보상: 시뮬레이션 개체군 1회분(pop 회)당 비용 (0 = Isp 만 최대화)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-freq", type=int, default=100_000, help="평가 주기 (스텝)")
    parser.add_argument("--save-freq", type=int, default=100_000, help="체크포인트 주기 (스텝)")
    parser.add_argument("--name", default=None, help="실행 이름 (기본: 날짜_시간)")
    args = parser.parse_args(argv)

    name = args.name or time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(RUNS_DIR, name)
    os.makedirs(os.path.join(log_dir, "monitor"), exist_ok=True)

    env_kwargs = {"isp_scale": args.isp_scale, "sim_cost": args.sim_cost}
    train_kwargs = {**env_kwargs, "gen_range": (args.gen_min, args.gen_max)}
    with open(os.path.join(log_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(vars(args), design_vars=GA.DESIGN_VARS, temperature=GA.load_fixed_values()["temperature"]), f, indent=2, ensure_ascii=False)
    status = Status(os.path.join(log_dir, "status.json"), name=name, state="starting", pid=os.getpid(),
                    started=time.time(), total_steps=args.steps, step=0, config=vars(args))
    status.save()

    check_env(GAEnv(SurrogateEvaluator(), **env_kwargs))   # 규격 검사: 문제 있으면 여기서 에러
    env_fns = [functools.partial(make_env, i, log_dir, args.seed, train_kwargs) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)
    if not args.no_norm_reward:
        # 보상 크기를 일정하게 → 가치함수 학습이 안정된다 (관측은 이미 0~1 이라 정규화 안 함)
        env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=args.gamma)
    eval_fns = [functools.partial(make_env, 1000 + i, log_dir, args.seed, env_kwargs, False) for i in range(args.n_envs)]
    eval_env = SubprocVecEnv(eval_fns) if args.n_envs > 1 else DummyVecEnv(eval_fns)

    print(f"실행 {name} | 장치 {args.device} | 병렬 환경 {args.n_envs} | 총 {args.steps:,} 스텝")
    print(f"  보상: Isp {args.isp_scale:g} s 당 +1, 시뮬레이션 비용 {args.sim_cost:g}/개체군")
    print(f"  로그: {log_dir}")
    print(f"  모니터링: http://127.0.0.1:10000/training  (또는 tensorboard --logdir {RUNS_DIR})")
    print(f"  에피소드 {args.gen_min}~{args.gen_max}세대 · γ {args.gamma} · lr {args.lr:g}→{args.lr * args.lr_end_frac:g} · "
          f"target_kl {args.target_kl} · 보상 정규화 {'끔' if args.no_norm_reward else '켬'}")

    status.update(state="baseline")
    status.save()
    n_eval = len(EVAL_TARGETS) * len(EVAL_SEEDS) + N_EVAL_RANDOM
    baseline = evaluate(eval_env, lambda obs: np.zeros(len(obs), dtype=int), args.eval_gens)   # 기준: 전략 0(기존 GA) 고정
    print(f"  평가: 고정 목표 {len(EVAL_TARGETS)}×{len(EVAL_SEEDS)} + 무작위 목표 {N_EVAL_RANDOM} = "
          f"{n_eval} 에피소드 × {args.eval_gens}세대, {args.eval_freq:,} 스텝마다")
    status.update(state="running", baseline={k: v for k, v in baseline.items() if k != "isps"},
                  eval_episodes=n_eval)
    status.save()
    print(f"  기준(전략 0 고정): 성공 {baseline['success']:.0%}  Isp {baseline['best_isp']:.2f}  "
          f"점수 {baseline['score']:+.3f}  시뮬 {baseline['sims']:.0f}")

    with warnings.catch_warnings():                    # MlpPolicy 를 GPU 로 돌린다는 경고 (요청한 것이므로 숨김)
        warnings.filterwarnings("ignore", message=".*run PPO on the GPU.*")
        lr0, lr1 = args.lr, args.lr * args.lr_end_frac
        model = PPO("MlpPolicy", env, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=10,
                    gamma=args.gamma, gae_lambda=0.95, learning_rate=lambda p: lr1 + (lr0 - lr1) * p,
                    ent_coef=args.ent_coef, target_kl=args.target_kl, seed=args.seed,
                    device=args.device, verbose=0)
    live = LiveWriter(log_dir, status)
    model.set_logger(Logger(log_dir, [TensorBoardOutputFormat(os.path.join(log_dir, "tb_1")), live]))

    callbacks = [
        MonitorCallback(eval_env, log_dir, args.eval_freq, baseline, args.steps, args.eval_gens, status),
        CheckpointCallback(max(1, args.save_freq // args.n_envs), os.path.join(log_dir, "checkpoints"), "ppo"),
    ]
    status["state"] = "running"
    try:
        model.learn(total_timesteps=args.steps, callback=callbacks,
                    progress_bar=sys.stdout.isatty())     # 로그 파일로 보낼 때는 진행바 끔 (평가 줄에 ETA 있음)
        status["state"] = "done"
    except KeyboardInterrupt:
        print("\n중단됨 — 지금까지의 정책을 저장합니다")
        status["state"] = "stopped"
    except Exception as exc:
        status.update(state="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        model.save(os.path.join(log_dir, "final_model"))
        shutil.copy(os.path.join(log_dir, "final_model.zip"), os.path.join(HERE, "ppo_ga.zip"))
        status.update(finished=time.time(), step=int(model.num_timesteps))
        status.save()
        live.close()
        env.close()
        eval_env.close()
    print(f"저장: {log_dir}/final_model.zip, best_model.zip (→ {os.path.join(HERE, 'ppo_ga.zip')})")


if __name__ == "__main__":
    main()
