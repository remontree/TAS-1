"""
학습된 PPO 정책으로 실제 GA(진짜 core 시뮬레이션)의 세대별 전략 고르기
    from rl.policy import PPOStrategy
    GA.run_ga(12000, 750, strategy=PPOStrategy())
실행 (optimizer 폴더에서)
    python -m rl.policy --impulse 12000 --thrust 750 [--model rl/runs/<실행>/best_model.zip]
"""

import argparse
import csv
import glob
import json
import os
import time
import zipfile

from rl import actions, operators, state

HERE = os.path.dirname(os.path.abspath(__file__))
OPTIMIZER_DIR = os.path.dirname(HERE)
DEFAULT_MODEL = os.path.join(HERE, "ppo_ga.zip")


class PPOStrategy:
    """GA.run_ga(strategy=...) 에 넘기는 콜백. 세대마다 상태 → PPO → 전략 → 다음 개체군"""

    def __init__(self, model_path=DEFAULT_MODEL, device="cpu", deterministic=True, seed=None):
        """
        deterministic=True : 가장 확률 높은 전략 (argmax)
        deterministic=False: 정책 분포에서 전략을 뽑는다 (확률적). seed 를 주면 같은 실행을 재현할 수 있다.
        """
        # 세대마다 상태 1개만 추론하므로 CPU 로 충분 (GPU 를 잡으면 GA 워커 fork 와도 얽힌다)
        import torch
        from stable_baselines3 import PPO
        if seed is not None:
            torch.manual_seed(seed)
        if model_obs_dim(model_path) != state.STATE_DIM:
            raise ValueError(f"{os.path.basename(model_path)} 는 상태 {model_obs_dim(model_path)}차원으로 학습된 모델입니다 "
                             f"(현재 {state.STATE_DIM}차원, 제약 조건이 바뀜). python -m rl.train_ppo 로 다시 학습하세요.")
        self.model = PPO.load(model_path, device=device)
        self.model_path = model_path
        self.deterministic = deterministic

    def __call__(self, rng, evaluations, pipes, history, ctx, parents=None):
        """parents: 부모로 쓸 개체 (기본 = evaluations). 상태는 항상 전체 개체군(evaluations) 기준"""
        obs = state.extract_state(evaluations, history, ctx)
        action = int(self.model.predict(obs, deterministic=self.deterministic)[0])
        params = actions.resolve(action, evaluations)
        population = operators.make_next_population(rng, parents or evaluations, pipes, params, ctx["pop_size"])
        return population, {"action": action, "name": actions.ACTIONS[action]["name"]}


def model_obs_dim(path):
    """SB3 zip 의 data 에서 관측 차원만 읽는다 (모델 전체를 불러오지 않음)"""
    try:
        with zipfile.ZipFile(path) as z:
            data = json.loads(z.read("data"))
        return int(data["observation_space"]["_shape"][0])
    except Exception:
        return None


def _eval_summary(run_dir):
    """train_ppo 의 progress.csv → 마지막 평가 결과"""
    path = os.path.join(run_dir, "progress.csv")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r["score"]))
    last = rows[-1]
    keys = [k for k in ("success", "best_isp", "sims", "score", "isp_gain", "win_rate") if k in last]
    return {"steps": int(last["step"]),
            "best": {k: float(best[k]) for k in keys},
            "last": {k: float(last[k]) for k in keys}}


def list_models():
    """웹에서 고를 수 있는 PPO 모델 목록 (경로는 optimizer 폴더 기준 상대 경로)"""
    models = []
    from rl import features
    # ppo_ga.zip 은 예전 설계 변수 구성 (챔버 길이 변수, 헬륨 고정) 으로 학습됨 → 새로 학습한 실행만
    if False and os.path.exists(DEFAULT_MODEL) and model_obs_dim(DEFAULT_MODEL) == state.STATE_DIM:
        models.append({"id": "rl/ppo_ga.zip", "label": "ppo_ga.zip (최근 학습 최종)",
                       "mtime": os.path.getmtime(DEFAULT_MODEL), "eval": None})
    for run_dir in sorted(glob.glob(os.path.join(HERE, "runs", "*")), reverse=True):
        if not os.path.isdir(run_dir) or not features.run_compatible(run_dir):
            continue
        name = os.path.basename(run_dir)
        ev = _eval_summary(run_dir)
        for fname, label in (("best_model.zip", "best"), ("final_model.zip", "final")):
            path = os.path.join(run_dir, fname)
            if os.path.exists(path) and model_obs_dim(path) == state.STATE_DIM:   # 현재 상태 규격과 맞는 것만
                models.append({"id": os.path.relpath(path, OPTIMIZER_DIR), "label": f"{name} · {label}",
                               "mtime": os.path.getmtime(path),
                               "eval": (ev or {}).get("best" if label == "best" else "last"),
                               "steps": (ev or {}).get("steps")})
    return models


def resolve_model(model_id):
    """웹에서 받은 모델 id → 절대 경로. rl/ 아래 .zip 만 허용"""
    path = os.path.realpath(os.path.join(OPTIMIZER_DIR, model_id or "rl/ppo_ga.zip"))
    if not path.startswith(os.path.realpath(HERE) + os.sep) or not path.endswith(".zip") or not os.path.exists(path):
        raise ValueError(f"PPO 모델을 찾을 수 없습니다: {model_id}")
    return path


def main(argv=None):
    import GA
    parser = argparse.ArgumentParser(description="PPO 가 전략을 고르는 GA (진짜 시뮬레이션)")
    parser.add_argument("--impulse", type=float, default=None)
    parser.add_argument("--thrust", type=float, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pop", type=int, default=None)
    parser.add_argument("--gens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    t0 = time.time()
    result = GA.run_ga(args.impulse, args.thrust, pop_size=args.pop, n_generations=args.gens,
                       seed=args.seed, strategy=PPOStrategy(args.model))
    GA.print_summary(result)
    print(f"총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
