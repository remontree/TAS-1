"""
기존 GA(전략 0 고정) vs 밴딧 vs PPO 비교
실행 (optimizer 폴더에서)
    python -m rl.benchmark --controllers baseline bandit --seeds 0 1 2 --pop 32 --gens 30
결과: rl/bench_results.csv
"""

import argparse
import csv
import os
import time

from rl.bandit import UCB
from rl.actions import N_ACTIONS
from rl.controlled_ga import ControlledGA, RealEvaluator

TARGETS = [(12000, 1000), (10000, 1000), (12000, 750)]
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results.csv")


def make_controller(name, seed):
    """controller(obs, last_action, last_reward) → 행동 번호"""
    if name == "baseline":
        return lambda obs, a, r: 0
    if name == "bandit":
        bandit = UCB(N_ACTIONS, c=1.0, seed=seed)

        def pick(obs, a, r):
            if a is not None:
                bandit.update(a, r)
            return bandit.select()
        return pick
    if name == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_ga"))
        return lambda obs, a, r: int(model.predict(obs, deterministic=True)[0])
    raise ValueError(name)


def run_one(evaluator, controller, targets, pop, gens, seed):
    evaluator.caches.clear()          # 앞 실행의 캐시를 지워야 시뮬레이션 수를 공정하게 비교할 수 있다
    ga = ControlledGA(evaluator, targets, pop, gens, seed=seed)
    obs, action, reward = ga.reset(), None, 0.0
    first_feasible_sims = None
    while True:
        if first_feasible_sims is None and ga.history[-1]["best_isp"] is not None:
            first_feasible_sims = ga.history[-1]["simulated_total"]
        if ga.done:
            break
        action = controller(obs, action, reward)
        obs, reward, _, _ = ga.step(action)
    last = ga.history[-1]
    return {
        "best_isp": last["best_isp"],
        "feasible": last["best_isp"] is not None,
        "sims_total": last["simulated_total"],
        "sims_to_first_feasible": first_feasible_sims,
        "generations": ga.gen + 1,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--controllers", nargs="+", default=["baseline", "bandit"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--pop", type=int, default=32)
    parser.add_argument("--gens", type=int, default=30)
    args = parser.parse_args(argv)

    evaluator = RealEvaluator()
    rows = []
    try:
        for targets in TARGETS:
            for seed in args.seeds:
                for name in args.controllers:
                    t0 = time.time()
                    res = run_one(evaluator, make_controller(name, seed), targets, args.pop, args.gens, seed)
                    row = {"controller": name, "impulse": targets[0], "thrust": targets[1], "seed": seed,
                           **res, "wall_time": round(time.time() - t0, 1)}
                    rows.append(row)
                    print(row)
    finally:
        evaluator.close()

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n저장: {OUT_PATH}")
    for name in args.controllers:
        mine = [r for r in rows if r["controller"] == name]
        isps = [r["best_isp"] for r in mine if r["feasible"]]
        avg = sum(isps) / len(isps) if isps else float("nan")
        print(f"  {name:<9} 성공 {len(isps)}/{len(mine)}  평균 best Isp {avg:.2f} s")


if __name__ == "__main__":
    main()
