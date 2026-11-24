"""
예전 MDO 결과(results/*/candidates.csv)의 설계를 core 로 다시 시뮬레이션해서
새로 추가된 출력(dp_ratio_min 등)을 채운다 → 대리모델 학습 데이터 보충
실행 (optimizer 폴더에서)
    python -m rl.backfill --n 3000
결과: results/backfill_<시각>/candidates.csv  (진행 중에도 한 줄씩 기록 → 중간에 끊어도 그때까지 사용 가능)
"""

import argparse
import csv
import multiprocessing
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

import GA
from rl.surrogate import load_dataset

SIM_COLS = ["isp", "impulse", "thrust_avg", "burn_time", "max_tank_pressure", "max_chamber_pressure",
            "dp_ratio_min", "dp_ratio_avg", "termination_code", "error"]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000, help="다시 시뮬레이션할 설계 수")
    parser.add_argument("--workers", type=int, default=GA.N_WORKERS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    df = load_dataset()
    if "dp_ratio_min" in df:                          # 이미 값이 있는 설계는 건너뜀
        df = df[df["dp_ratio_min"].isna()]
    df = df[df["completed"]]                          # 차압은 정상 연소한 설계에서만 의미가 있다
    rows = df.sample(n=min(args.n, len(df)), random_state=args.seed).to_dict("records")

    pipes = GA.load_candidate_pipes()
    by_id = {p["pipe_id"]: i for i, p in enumerate(pipes)}
    fixed = GA.load_fixed_values()
    jobs = []
    for r in rows:
        genome = {"pipe": by_id[r["pipe_id"]], **{v: r[v] for v in GA.DESIGN_VARS}}
        genome["port_num"], genome["injector_oriffice_n"] = int(genome["port_num"]), int(genome["injector_oriffice_n"])
        jobs.append(GA.decode(genome, pipes, fixed))

    out_dir = os.path.join(GA.RESULTS_DIR, "backfill_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir)
    path = os.path.join(out_dir, "candidates.csv")
    print(f"설계 {len(jobs)}개 core 재시뮬레이션 (워커 {args.workers}) → {path}", flush=True)

    os.makedirs(os.path.join(os.path.expanduser("~"), "RocketCEA"), exist_ok=True)
    t0, done = time.time(), 0
    with open(path, "w", newline="", encoding="utf-8") as f, \
            ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        writer = csv.writer(f)
        writer.writerow(["pipe_id"] + GA.DESIGN_VARS + GA.PIPE_VARS + ["feasible", "violation", "failed"] + SIM_COLS)
        futures = {pool.submit(GA.simulate, design): (design, pipe) for design, pipe in jobs}
        for fut in as_completed(futures):
            design, pipe = futures[fut]
            sim = fut.result()
            writer.writerow([pipe["pipe_id"]] + [design[v] for v in GA.DESIGN_VARS + GA.PIPE_VARS]
                            + ["", "", ""] + [sim.get(c) for c in SIM_COLS])
            f.flush()
            done += 1
            if done % 100 == 0 or done == len(jobs):
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(jobs)}  {rate:.1f}개/s  남은 {(len(jobs) - done) / rate / 60:.1f}분", flush=True)
    print(f"완료: {path}")


if __name__ == "__main__":
    main()
