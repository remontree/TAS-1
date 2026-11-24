"""
능동 학습 데이터 — 곡선 판단 에이전트가 실제로 가는 곳 (대리모델이 '합격·고 Isp' 라고 믿는 설계) 을 core 로 다시 계산
대리모델 위에서 최적화하면 예측 오차가 유리한 쪽 (차압·평균 추력 과대 예측 등) 으로 몰린다.
그 설계들을 core 정답과 함께 학습 데이터에 넣어 대리모델이 자기 실수를 배우게 한다.

    1. 수집: CurveVecEnv 에서 '한 걸음 앞 보기' 탐욕 이동 (+ 확률 eps 무작위) 으로 에이전트처럼 설계를 옮긴다.
             에피소드마다 예측 합격 최고 설계 + 중간에 예측 합격한 설계 일부 + 작은 섭동
    2. core: rl.ts_data 와 같은 형식 (simt 40 s, 시계열) 으로 시뮬레이션 → rl/ts_data/<시각>_active/shard_*.npz (kind=active)

실행 (optimizer 폴더에서, 새 rl/ts_surrogate.pt 가 있어야 함)
    python -m rl.ts_active --n 2500
모니터링: /training (kind=data)
"""

import argparse
import json
import multiprocessing
import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

import GA
from rl import curve_agent as ca
from rl import features as ft
from rl import ts_data
from rl.data_monitor import DataMonitor

HERE = os.path.dirname(os.path.abspath(__file__))


def collect(n_designs, n_agents=128, eps=0.15, seed=0, model_path=None, device="cuda", log=print, report=None):
    """에이전트식 탐욕 이동으로 대리모델이 좋다고 믿는 설계 genome 목록. report(모은 수, 걸음): 진행 기록"""
    kw = {"len_range": (20, 60), "device": device, "isp_penalty": 0.5, "t_margin": 1.0, "ok_threshold": 0.9}
    if model_path:
        kw["model_path"] = model_path
    env = ca.CurveVecEnv(n_agents, seed=seed, **kw)
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    env.reset()
    out, seen = [], set()

    def add(g):
        key = tuple([g["pipe"]] + [g[v] for v in GA.DESIGN_VARS])
        if key not in seen:
            seen.add(key)
            out.append(dict(g))

    t0, it = time.time(), 0
    while len(out) < n_designs:
        gain = env.lookahead(list(range(env.n)))
        order = np.argsort(-gain, 1)
        a1, a2 = order[:, 0], order[:, 1]
        a2 = np.where(gain[np.arange(env.n), a2] > 0, a2, ca.NONE)
        act = np.stack([a1, a2], 1)
        rand = rng.random(act.shape) < eps
        act[rand] = rng.integers(0, ca.N_MOVES, rand.sum())
        _, _, _, infos = env.step(act)
        for info in infos:
            g = info.get("best_genome")
            if g is not None:
                add(g)
                for _ in range(2):                                  # 경계 근처의 작은 섭동 (제작 공차 수준)
                    add(ts_data.perturb(prng, g, len(env.pipes), prng.choice((0.002, 0.005, 0.01))))
        # 진행 중 예측 합격한 설계 일부 (최고에 이르기 전 경로)
        if it % 5 == 4:
            for i in np.flatnonzero(env.score >= 1.0)[:8]:
                add(env.genomes[int(i)])
        it += 1
        if report:
            report(len(out), it)
        if it % 20 == 0:
            log(f"  수집 {len(out)}/{n_designs} 설계 · {it} 걸음 · {time.time() - t0:.0f}s", flush=True)
    return out[:n_designs]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500, help="core 로 계산할 설계 수")
    ap.add_argument("--agents", type=int, default=128)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=GA.N_WORKERS)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--ts-model", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    out_dir = os.path.join(ts_data.DATA_DIR, time.strftime("%Y%m%d_%H%M%S") + "_active")
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump({"simt": ts_data.SIMT, "series_dt": GA.SERIES_DT, "channels": GA.SERIES_CHANNELS,
                   "design_vars": GA.DESIGN_VARS, "feature_vars": ft.FEATURE_VARS, "pipe_features": ft.DATA_PIPE_FEATURES,
                   "scalars": ts_data.SCALARS, "mix": {"active": 1.0}, "n_target": args.n, "workers": args.workers,
                   "ts_model": args.ts_model or "rl/ts_surrogate.pt"}, f, indent=2)
    mon = DataMonitor("tsdata_" + os.path.basename(out_dir), out_dir, args.n, os.getpid(),
                      {"simt": ts_data.SIMT, "mix": {"active": 1.0}, "n_target": args.n, "workers": args.workers,
                       "pressure_bounds": GA.BOUNDS["supercharge_pressure"], "data_dir": os.path.relpath(out_dir, HERE)})

    def log(msg, **_):
        print(msg, flush=True)
        mon.write_log(msg)

    mon.status["phase"] = "에이전트식 이동으로 설계 수집 (대리모델)"
    mon.save()
    log(f"능동 학습 데이터 {args.n}개 → {out_dir}")

    def report(n, it):
        mon.status.update(phase_step=min(n, args.n), phase_total=args.n, phase_iter=it)
        mon.save()

    genomes = collect(args.n, args.agents, args.eps, args.seed, args.ts_model, args.device, log, report)

    mon.status["phase"] = "core 시뮬레이션"
    pipes = GA.load_candidate_pipes()
    fixed = dict(GA.load_fixed_values(), simt=ts_data.SIMT)
    n_t = int(round(ts_data.SIMT / GA.SERIES_DT))
    os.makedirs(os.path.join(os.path.expanduser("~"), "RocketCEA"), exist_ok=True)
    t0, done, n_ok, shard_k, rows = time.time(), 0, 0, 0, []
    state = "done"
    pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"))
    try:
        futures = {}
        queue = list(genomes)
        while queue or futures:
            while queue and len(futures) < args.workers * 2:
                design, pipe = GA.decode(queue.pop(), pipes, fixed)
                futures[pool.submit(ts_data.run_sim, design)] = (design, pipe)
            fin, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for fut in fin:
                design, pipe = futures.pop(fut)
                sim = fut.result()
                done += 1
                n_ok += not sim.get("error")
                series = sim.get("series") or np.zeros((len(GA.SERIES_CHANNELS), n_t)).tolist()
                rows.append({"x": ts_data.data_row(design, pipe), "series": series, "kind": "active",
                             "pipe_id": pipe["pipe_id"], "termination_code": sim.get("termination_code"),
                             "error": sim.get("error"), "scalars": {s: sim.get(s) for s in ts_data.SCALARS}})
                mon.stats.add(rows[-1]["x"], "active", dict(rows[-1]["scalars"], error=sim.get("error"),
                                                             termination_code=sim.get("termination_code")))
                mon.status["total_steps"] = len(genomes)
                mon.progress(done, n_ok, n_ok / (time.time() - t0), {"active": done}, step=done)
                if len(rows) >= ts_data.SHARD:
                    ts_data.save_shard(out_dir, shard_k, rows)
                    shard_k += 1
                    rows = []
                if done % 200 == 0:
                    log(f"  core {done}/{len(genomes)} (정상 {n_ok})  {n_ok / (time.time() - t0):.1f}개/s")
    except KeyboardInterrupt:
        log("중단됨 — 지금까지의 결과를 저장합니다")
        state = "stopped"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        if rows:
            ts_data.save_shard(out_dir, shard_k, rows)
        mon.progress(done, n_ok, n_ok / max(time.time() - t0, 1e-9), {"active": done}, force=True, step=done)
        log(f"{'완료' if state == 'done' else '중단'}: {out_dir}")
        mon.finish(state)


if __name__ == "__main__":
    main()
