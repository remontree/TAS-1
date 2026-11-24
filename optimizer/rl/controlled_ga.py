"""
한 세대씩 멈추고, 바깥에서 전략을 받아 진행하는 GA
    ga = ControlledGA(evaluator, targets=(12000, 1000))
    obs = ga.reset()
    while not ga.done:
        obs, reward, done, info = ga.step(action_id)
"""

import random
from concurrent.futures import ProcessPoolExecutor

import GA
from rl import actions, operators, state

# 보상 = 세대마다 best 점수의 증가량 - 시뮬레이션 비용
# 예전 기본값(ISP_SCALE=100, COST_PER_SIM=0.01)은 시뮬레이션 32회 비용 = Isp 1 s 라서
# 정책이 Isp 를 포기하고 시뮬레이션을 아끼는 쪽으로 학습됐다 → Isp 를 키우고 비용은 기본 0.
ISP_SCALE = 20.0      # Isp 이 값[s] 만큼 오르면 점수 +1  (1 s = 0.05)
COST_PER_SIM = 0.0    # 시뮬레이션 POP_SIZE 회당 비용 (0 = Isp 만 최대화)


def score(ev, isp_scale=ISP_SCALE):
    """best ev → 점수. 합격이면 1 이상 (Isp 클수록 큼), 불합격이면 0 이하 (위반 작을수록 큼)."""
    if ev["feasible"]:
        return 1.0 + max(ev["sim"]["isp"] - 150.0, 0.0) / isp_scale
    return -min(ev["violation"], 5.0) / 5.0


class RealEvaluator:
    """진짜 core 시뮬레이터로 평가 (GA.evaluate 재사용)"""

    def __init__(self, n_workers=GA.N_WORKERS):
        self.pipes = GA.load_candidate_pipes()
        self.fixed = GA.load_fixed_values()
        self.caches = {}     # 목표값마다 따로 (합격 판정이 목표에 따라 다르므로)
        self.counters = {"pre_rejected": 0, "simulated": 0}
        self.pool = ProcessPoolExecutor(max_workers=n_workers)

    @property
    def simulated(self):
        return self.counters["simulated"]

    def __call__(self, population, targets):
        cache = self.caches.setdefault(tuple(targets), {})
        return GA.evaluate(population, self.pipes, self.fixed, tuple(targets), cache, self.pool, self.counters)

    def close(self):
        self.pool.shutdown()


class ControlledGA:
    def __init__(self, evaluator, targets, pop_size=GA.POP_SIZE, n_generations=GA.N_GENERATIONS,
                 early_stop=0, seed=None, isp_scale=ISP_SCALE, sim_cost=COST_PER_SIM):
        # early_stop: 이 세대 수 동안 개선이 없으면 에피소드 종료 (0 = 안 함 → 실제 MDO 와 같음).
        # 상태의 '정체 세대 수' 정규화는 GA.STALL_NORM 을 써서 MDO 와 항상 같게 한다.
        self.evaluator = evaluator
        self.targets = tuple(targets)
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.early_stop = early_stop
        self.rng = random.Random(seed)
        self.pipes = evaluator.pipes
        self.fixed = evaluator.fixed
        self.budget = pop_size * n_generations
        self.isp_scale = isp_scale
        self.sim_cost = sim_cost

    def score(self, ev=None):
        return score(ev or self.best, self.isp_scale)

    def reset(self):
        self.gen = 0
        self.best, self.stall, self.history = None, 0, []
        self.sim_start = self.evaluator.simulated
        self.population = GA.initial_population(self.rng, self.pipes, self.fixed, self.pop_size)
        self._evaluate_and_record()
        return self.observe()

    def _evaluate_and_record(self):
        before = self.evaluator.simulated
        evaluations = self.evaluator(self.population, self.targets)
        self.evaluations = sorted(evaluations, key=GA.rank_key)
        top = self.evaluations[0]
        if self.best is None or GA.rank_key(top) < GA.rank_key(self.best):
            self.best, self.stall = top, 0
        else:
            self.stall += 1
        new_sims = self.evaluator.simulated - before
        self.history.append({
            "generation": self.gen,
            "n_feasible": sum(ev["feasible"] for ev in evaluations),
            "best_isp": self.best["sim"]["isp"] if self.best["feasible"] else None,
            "best_violation": self.best["violation"],
            "simulated_total": self.evaluator.simulated - self.sim_start,
            "new_sims": new_sims,
        })
        return new_sims

    def observe(self):
        ctx = {
            "gen": self.gen, "n_generations": self.n_generations,
            "simulated": self.evaluator.simulated - self.sim_start, "budget": self.budget,
            "stall": self.stall, "patience": GA.STALL_NORM,
            "n_pipes": len(self.pipes), "targets": self.targets,
        }
        return state.extract_state(self.evaluations, self.history, ctx)

    @property
    def done(self):
        if self.gen >= self.n_generations - 1:
            return True
        return bool(self.early_stop and self.best["feasible"] and self.stall >= self.early_stop)

    def step(self, action_id):
        params = actions.resolve(action_id, self.evaluations)
        old_score = self.score()
        self.population = operators.make_next_population(
            self.rng, self.evaluations, self.pipes, params, self.pop_size)
        self.gen += 1
        new_sims = self._evaluate_and_record()
        reward = self.score() - old_score - self.sim_cost * new_sims / self.pop_size
        info = {"action": actions.ACTIONS[action_id]["name"], "new_sims": new_sims}
        return self.observe(), reward, self.done, info
