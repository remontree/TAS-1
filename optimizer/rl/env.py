"""
ControlledGA 를 Gymnasium 환경 규격으로 포장
    env = GAEnv(SurrogateEvaluator())
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl import actions, state
from rl.controlled_ga import COST_PER_SIM, ISP_SCALE, ControlledGA


class GAEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, evaluator, pop_size=32, n_generations=30,
                 impulse_range=(8000, 14000), thrust_range=(600, 1200),
                 isp_scale=ISP_SCALE, sim_cost=COST_PER_SIM, gen_range=None):
        """gen_range=(최소, 최대): 에피소드마다 세대 수를 이 범위에서 무작위로 (짧은 실행·긴 실행 모두 학습)"""
        super().__init__()
        self.evaluator = evaluator
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.impulse_range = impulse_range
        self.thrust_range = thrust_range
        self.isp_scale = isp_scale
        self.sim_cost = sim_cost
        self.gen_range = gen_range
        self.observation_space = spaces.Box(0.0, 1.0, shape=(state.STATE_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(actions.N_ACTIONS)

    def reset(self, seed=None, options=None):
        if options and "seed" in options:          # 병렬 평가: 환경마다 다른 시드를 options 로 전달
            seed = options["seed"]
        super().reset(seed=seed)
        # 캐시를 에피소드마다 비워야 시뮬레이션 비용(보상)이 앞 에피소드에 따라 달라지지 않고, 메모리도 안 샌다
        self.evaluator.caches.clear()
        if options and "targets" in options:
            targets = tuple(options["targets"])
        else:
            targets = (round(float(self.np_random.uniform(*self.impulse_range)), -2),
                       round(float(self.np_random.uniform(*self.thrust_range)), -1))
        if options and "n_generations" in options:
            n_gen = int(options["n_generations"])
        elif self.gen_range:
            n_gen = int(self.np_random.integers(self.gen_range[0], self.gen_range[1] + 1))
        else:
            n_gen = self.n_generations
        self.ga = ControlledGA(self.evaluator, targets, self.pop_size, n_gen,
                               seed=int(self.np_random.integers(2**31)),
                               isp_scale=self.isp_scale, sim_cost=self.sim_cost)
        obs = self.ga.reset()
        return obs, {"targets": targets, "n_generations": n_gen}

    def step(self, action):
        obs, reward, done, info = self.ga.step(int(action))
        if done:                                   # 에피소드 결과 (Monitor / 콜백에서 기록)
            last = self.ga.history[-1]
            info.update(best_isp=last["best_isp"] or 0.0, feasible=float(last["best_isp"] is not None),
                        sims=last["simulated_total"], score=self.ga.score())
        return obs, float(reward), done, False, info
