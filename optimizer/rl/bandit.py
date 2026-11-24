"""
1단계 AI: UCB 밴딧으로 세대마다 전략 고르기
실행 (optimizer 폴더에서)
    python -m rl.bandit --impulse 12000 --thrust 1000 --pop 16 --gens 10
"""

import argparse
import math
import random

from rl.actions import ACTIONS, N_ACTIONS
from rl.controlled_ga import ControlledGA, RealEvaluator


class UCB:
    def __init__(self, n_actions, c=1.0, seed=None):
        self.c = c
        self.n = [0] * n_actions       # 전략별 사용 횟수 N(a)
        self.q = [0.0] * n_actions     # 전략별 평균 보상 Q(a)
        self.t = 0                     # 전체 사용 횟수
        self.rng = random.Random(seed)

    def select(self):
        untried = [a for a, cnt in enumerate(self.n) if cnt == 0]
        if untried:
            return self.rng.choice(untried)
        ucb = [self.q[a] + self.c * math.sqrt(math.log(self.t) / self.n[a]) for a in range(len(self.n))]
        return max(range(len(ucb)), key=lambda a: ucb[a])

    def update(self, action, reward):
        self.t += 1
        self.n[action] += 1
        self.q[action] += (reward - self.q[action]) / self.n[action]   # 이동 평균


def run_bandit(evaluator, targets, pop_size, n_generations, seed=None, c=1.0, verbose=True):
    ga = ControlledGA(evaluator, targets, pop_size, n_generations, seed=seed)
    bandit = UCB(N_ACTIONS, c=c, seed=seed)
    ga.reset()
    while not ga.done:
        action = bandit.select()
        _, reward, _, info = ga.step(action)
        bandit.update(action, reward)
        if verbose:
            best = ga.history[-1]
            isp = f"{best['best_isp']:.2f} s" if best["best_isp"] else f"위반 {best['best_violation']:.3f}"
            print(f"[GEN {ga.gen:3d}] 전략 {action} {info['action']:<6} | 보상 {reward:+.3f} | "
                  f"best {isp} | 새 시뮬 {info['new_sims']}")
    return ga, bandit


def main(argv=None):
    parser = argparse.ArgumentParser(description="UCB 밴딧으로 GA 전략 고르기")
    parser.add_argument("--impulse", type=float, default=12000.0)
    parser.add_argument("--thrust", type=float, default=1000.0)
    parser.add_argument("--pop", type=int, default=32)
    parser.add_argument("--gens", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--c", type=float, default=1.0)
    args = parser.parse_args(argv)

    evaluator = RealEvaluator()
    try:
        ga, bandit = run_bandit(evaluator, (args.impulse, args.thrust), args.pop, args.gens, args.seed, args.c)
    finally:
        evaluator.close()
    print("\n전략별 사용 횟수 / 평균 보상")
    for a, spec in enumerate(ACTIONS):
        print(f"  {a} {spec['name']:<6} N={bandit.n[a]:3d}  Q={bandit.q[a]:+.3f}")


if __name__ == "__main__":
    main()
