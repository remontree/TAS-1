"""
직접 설계 정책 · 기본 GA 의 최종 설계를 진짜 core 시뮬레이션으로 검증 (대리모델 악용 여부 확인)
실행 (optimizer 폴더에서)
    python -m rl.verify_core --model rl/runs/direct_v3_20261019_2042/best_model.zip [--random 20 --steps 100]
에피소드(목표, 시드)마다 에이전트 32명 × steps 걸음의 최고 합격 설계, 같은 대리모델 위 기본 GA (32 × steps 세대) 의
최고 설계를 core 로 돌려 예측 Isp ↔ 실제 Isp, 실제 합격 여부를 비교한다.
"""

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import GA
import constrain as cs
from rl import direct

EVAL_POP = 32


def agent_best(env, predict, episodes, n_steps, pop=EVAL_POP):
    """에피소드마다 (예측 Isp, genome) — 합격 설계가 없으면 (None, None)"""
    opts = [{"targets": t, "seed": sd * 1000 + j, "n_steps": n_steps} for t, sd in episodes for j in range(pop)]
    env.fixed_options = opts
    obs = env.reset()
    best = [(0.0, None)] * len(opts)
    for _ in range(n_steps):
        obs, _, _, infos = env.step(predict(obs))
        for i, info in enumerate(infos):
            if info.get("best_genome") is not None and info["best_isp"] > best[i][0]:
                best[i] = (info["best_isp"], info["best_genome"])
    out = []
    for e in range(len(episodes)):
        isp, g = max(best[e * pop:(e + 1) * pop], key=lambda x: x[0])
        out.append((isp, g) if g is not None else (None, None))
    return out


def ga_best(simulator, episodes, n_gens, pop=EVAL_POP):
    from rl.controlled_ga import ControlledGA
    from rl.train_direct import TSEvaluator
    ev = TSEvaluator(simulator)
    out = []
    for targets, seed in episodes:
        ev.caches.clear()
        ga = ControlledGA(ev, targets, pop, n_gens, seed=seed)
        ga.reset()
        while not ga.done:
            ga.step(0)
        b = ga.best
        out.append((b["sim"]["isp"], b) if b["feasible"] else (None, None))
    return out


def core_check(items, pool):
    """items: [(design, pipe, targets)] → [(core sim, 합격, 실패 제약)]"""
    futs = [pool.submit(GA.simulate, d) for d, _, _ in items]
    res = []
    for (d, p, tg), f in zip(items, futs):
        sim = f.result()
        ok, _, failed = cs.check(cs.post_sim_constraints(d, p, sim, *tg))
        res.append((sim, ok, failed))
    return res


def verify(model_path, episodes, n_steps, device="cuda", pool=None, with_ga=True, env=None, predict=None,
           ga_rows=None):
    """
    에피소드마다 에이전트·GA 최고 설계를 core 로 검증. predict/env 를 주면 model_path 대신 그 정책 (학습 중 평가용).
    ga_rows: 이전에 계산한 GA 쪽 결과 (rows[e]["ga"]) 를 재사용
    """
    if predict is None:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device=device)
        predict = lambda o: model.predict(o, deterministic=True)[0]
        if env is None and model.observation_space.shape[0] != direct.OBS_DIM:
            from rl import curve_agent
            env = curve_agent.CurveVecEnv(len(episodes) * EVAL_POP, seed=1, device=device)
    pipes, fixed = GA.load_candidate_pipes(), GA.load_fixed_values()
    env = env or direct.DesignVecEnv(len(episodes) * EVAL_POP, seed=1, device=device)
    agent = agent_best(env, predict, episodes, n_steps)
    with_ga = with_ga and ga_rows is None
    ga = ga_best(env.simulator, episodes, 100) if with_ga else [(None, None)] * len(episodes)
    items, where = [], []
    for e, (targets, _) in enumerate(episodes):
        tg = (*targets, cs.TARGET_TOLERANCE)
        if agent[e][1] is not None:
            d, p = GA.decode(agent[e][1], pipes, fixed)
            items.append((d, p, tg)); where.append(("agent", e))
        if ga[e][1] is not None:
            items.append((ga[e][1]["design"], ga[e][1]["pipe"], tg)); where.append(("ga", e))
    own = pool is None
    pool = pool or ProcessPoolExecutor(GA.N_WORKERS, mp_context=multiprocessing.get_context("spawn"))
    try:
        checked = core_check(items, pool)
    finally:
        if own:
            pool.shutdown()
    rows = [{"episode": e, "targets": list(episodes[e][0]), "agent": {},
             "ga": ga_rows[e]["ga"] if ga_rows else {}} for e in range(len(episodes))]
    for (who, e), (sim, ok, failed) in zip(where, checked):
        pred = (agent if who == "agent" else ga)[e][0]
        rows[e][who] = {"pred_isp": pred, "core_isp": None if sim.get("error") else sim["isp"],
                        "core_ok": bool(ok), "failed": failed,
                        **({"genome": agent[e][1]} if who == "agent" else {})}
    return rows


def ga_core_rows(simulator, episodes, pool, n_gens=100):
    """기본 GA (대리모델 위, 32 × n_gens) 최고 설계를 core 로 검증한 rows (verify(..., ga_rows=) 에 재사용)"""
    ga = ga_best(simulator, episodes, n_gens)
    items, where = [], []
    for e, (targets, _) in enumerate(episodes):
        if ga[e][1] is not None:
            items.append((ga[e][1]["design"], ga[e][1]["pipe"], (*targets, cs.TARGET_TOLERANCE)))
            where.append(e)
    rows = [{"episode": e, "targets": list(episodes[e][0]), "agent": {}, "ga": {}} for e in range(len(episodes))]
    for e, (sim, ok, failed) in zip(where, core_check(items, pool)):
        rows[e]["ga"] = {"pred_isp": ga[e][0], "core_isp": None if sim.get("error") else sim["isp"],
                         "core_ok": bool(ok), "failed": failed}
    return rows


def core_episodes(n_random=20):
    from rl.benchmark import TARGETS
    rng = np.random.default_rng(777)
    episodes = [(tuple(t), s) for t in TARGETS for s in (1000, 1001, 1002)]
    return episodes + [(direct.sample_targets(rng, (6000, 16000), (400, 1500)), 3000 + k) for k in range(n_random)]


def core_score(rows, who="agent"):
    """core 기준 점수 (합격 1 + (Isp−150)/20, 불합격 0) 의 평균 — 학습 중 최고 모델 선택 기준"""
    return float(np.mean([direct.score_of(True, x[who]["core_isp"], 0) if x[who].get("core_ok") else 0.0
                          for x in rows]))


def summarize(rows, who):
    r = [x[who] for x in rows]
    found = [x for x in r if x]
    ok = [x for x in found if x["core_ok"]]
    err = [x["pred_isp"] - x["core_isp"] for x in found if x["core_isp"] is not None]
    fails = {}
    for x in found:
        for f in x["failed"]:
            fails[f] = fails.get(f, 0) + 1
    return {"found": len(found) / len(r), "core_ok": len(ok) / len(r),
            "core_isp_ok": float(np.mean([x["core_isp"] for x in ok])) if ok else None,
            "pred_minus_core": float(np.mean(err)) if err else None,
            "pred_core_mae": float(np.mean(np.abs(err))) if err else None, "fails": fails}


def paired(rows):
    """둘 다 core 합격인 에피소드의 Isp 차이 (agent − GA), core 기준 점수 (불합격 = 0)"""
    d = [x["agent"]["core_isp"] - x["ga"]["core_isp"] for x in rows
         if x["agent"].get("core_ok") and x["ga"].get("core_ok")]
    sc = lambda x: x["core_isp"] if x and x["core_ok"] else 0.0
    return {"n_both": len(d), "isp_gain": float(np.mean(d)) if d else None,
            "win_rate": float(np.mean([sc(x["agent"]) > sc(x["ga"]) for x in rows]))}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--random", type=int, default=20)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    episodes = core_episodes(a.random)
    t0 = time.time()
    rows = verify(a.model, episodes, a.steps)
    res = {"model": a.model, "episodes": len(episodes), "agent": summarize(rows, "agent"), "ga": summarize(rows, "ga"),
           "paired": paired(rows), "core_score": {"agent": core_score(rows), "ga": core_score(rows, "ga")},
           "elapsed_s": round(time.time() - t0)}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    out = a.out or os.path.join(os.path.dirname(a.model), "verify_core.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({**res, "rows": rows}, f, indent=1, ensure_ascii=False)
    print("저장:", out)


if __name__ == "__main__":
    main()
