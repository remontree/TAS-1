"""
시계열 대리모델 학습 데이터 생성 — core 시뮬레이션의 연소 전 과정(추력·챔버압·탱크압·질량유량)을 저장
실행 (optimizer 폴더에서)
    python -m rl.ts_data --n 16000
결과: rl/ts_data/<실행>/shard_*.npz  (SHARD 개마다 저장 → 중간에 끊어도 그때까지 사용 가능)
모니터링: http://127.0.0.1:10000/training  (kind=data, rl/runs/tsdata_<실행>/ — rl/data_monitor.py)

설계 분포 (다양한 형상·연소 시간) — 제약 조건은 보지 않는다 (설계 범위 BOUNDS 안이면 모두)
    uniform : 설계 범위에서 변수마다 독립 균등 (core 가 실패하는 형상도 '실패' 로 기록 → 분류기가 배움)
    random  : 변수 의존 순서로 뽑은 무작위 설계 (GA.sample_genome, 형상이 그럴듯한 쪽)
    burn    : 산화제 질량·인젝터 면적·노즐 목을 로그 균등으로 뽑아 짧은~긴 연소를 고르게
    archive : 예전 MDO 결과(results/*)의 설계 + 대리모델 MDO 에서 core 로 검증한 설계 + 작은 섭동 (합격 영역 근처)
    archive_he : archive 설계의 헬륨 슈퍼차징 압력만 범위 안에서 새로 뽑음 (합격 영역 근처에서 슈퍼차징의 효과)
    walk    : 무작위/기존 설계에서 출발해 에이전트처럼 작은~큰 걸음으로 이동한 궤적 (RL 이 실제로 가는 곳)
시뮬레이션 시간은 SIMT(40 s) — MDO 기본(20 s)보다 길게 해 장시간 연소의 경향도 배운다.
입력 X 의 열 = rl.features.FEATURE_VARS (설계 변수 + 챔버 길이·헬륨 질량) + 파이프 3개 (info.json 의 feature_vars)
"""

import argparse
import json
import math
import multiprocessing
import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

import GA
from rl import features as ft
from rl.data_monitor import DataMonitor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "ts_data")
SIMT = 40.0
SHARD = 500
PIPE_FEATURES = ft.DATA_PIPE_FEATURES
SCALARS = ["isp", "impulse", "thrust_avg", "burn_time", "max_tank_pressure", "max_chamber_pressure",
           "dp_ratio_min", "m_oxidizer_used", "m_fuel_used"]
# 범용 온도 대리모델: 설계마다 초기 온도를 TEMP_RANGE 에서 고르고, 슈퍼차징 압력은 그 온도의 포화압 ~ 상한 안으로
P_MAX = 7.0e6
MIX = {"uniform": 0.10, "random": 0.15, "burn": 0.10, "archive": 0.20, "archive_he": 0.15, "walk": 0.30}


def _loguniform(rng, var):
    lo, hi = GA.BOUNDS[var]
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def sample_uniform(rng, n_pipes):
    g = {"pipe": rng.randrange(n_pipes)}
    for var in GA.DESIGN_VARS:
        lo, hi = GA.BOUNDS[var]
        g[var] = rng.randint(lo, hi) if var in GA.INT_VARS else rng.uniform(lo, hi)
    return GA.quantize(g)


def sample_any(rng, pipes):
    """GA.sample_genome (실패하면 독립 균등)"""
    try:
        return GA.sample_genome(rng, pipes)
    except ValueError:
        return sample_uniform(rng, len(pipes))


def sample_burn(rng, pipes):
    """연소 시간을 좌우하는 변수만 로그 균등으로 바꿔 짧은/긴 연소를 고르게"""
    g = sample_any(rng, pipes)
    for var in ("oxidizer_M", "injector_oriffice_radius", "nozzle_throat_radius"):
        g[var] = _loguniform(rng, var)
    g["injector_oriffice_n"] = int(round(_loguniform(rng, "injector_oriffice_n")))
    return GA.quantize(g)


def perturb(rng, g, n_pipes, step):
    g = dict(g)
    for var in GA.DESIGN_VARS:
        lo, hi = GA.BOUNDS[var]
        d = rng.gauss(0, step) * (hi - lo)
        if var in GA.INT_VARS:
            d = round(d) if abs(d) >= 0.5 else (rng.choice((-1, 1)) if rng.random() < abs(d) * 2 else 0)
        g[var] = g[var] + d
    if n_pipes > 1 and rng.random() < 0.05:
        g["pipe"] = rng.randrange(n_pipes)
    return GA.quantize(g)


def load_archive(pipes, fixed):
    """예전 MDO 결과(core) 의 설계 + 대리모델 MDO 의 core 검증 설계 → genome (예전 형식은 rl.features 로 변환)"""
    import glob
    import pandas as pd
    by_id = {p["pipe_id"]: i for i, p in enumerate(pipes)}
    files = glob.glob(os.path.join(GA.RESULTS_DIR, "*", "candidates.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["termination_code"].notna() & df["pipe_id"].isin(by_id)]
    rows = df.to_dict("records")
    for f in glob.glob(os.path.join(GA.SURROGATE_RESULTS_DIR, "*", "summary.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                ver = json.load(fh).get("verified") or []
        except (OSError, ValueError):
            continue
        rows += [dict(v["design"], pipe_id=v["pipe_id"]) for v in ver if v.get("pipe_id") in by_id]
    out, seen = [], set()
    for r in rows:
        i = by_id[r["pipe_id"]]
        g = ft.legacy_genome(r, i, fixed, pipes[i])
        key = tuple(g[v] for v in ["pipe"] + GA.DESIGN_VARS)
        if key not in seen:
            seen.add(key)
            out.append(g)
    return out


def design_stream(rng, pipes, fixed, archive):
    """설계를 끝없이 만들어 낸다 (kind, genome)"""
    kinds, weights = list(MIX), list(MIX.values())
    walk = []
    while True:
        kind = rng.choices(kinds, weights)[0]
        if kind == "uniform":
            g = sample_uniform(rng, len(pipes))
        elif kind == "random":
            g = sample_any(rng, pipes)
        elif kind == "burn":
            g = sample_burn(rng, pipes)
        elif kind == "archive":
            g = perturb(rng, rng.choice(archive), len(pipes), rng.choice((0.0, 0.003, 0.01, 0.03)))
        elif kind == "archive_he":
            g = perturb(rng, rng.choice(archive), len(pipes), rng.choice((0.0, 0.003, 0.01)))
            g["supercharge_pressure"] = rng.uniform(*GA.BOUNDS["supercharge_pressure"])
            g = GA.quantize(g)
        else:
            if not walk:
                start = rng.choice(archive) if rng.random() < 0.5 else sample_any(rng, pipes)
                walk = [start, math.exp(rng.uniform(math.log(0.002), math.log(0.08))), rng.randint(5, 25)]
            cur, step, left = walk
            g = perturb(rng, cur, len(pipes), step)
            walk = [g, step, left - 1] if left > 1 else []
        yield kind, g


def p_sat(T):
    import constrain as cs
    return cs._saturated("P", "T", T, "Q", 0, "NitrousOxide") * 1.001


def wide_pressure_bounds():
    """데이터 생성 중 GA.BOUNDS 의 슈퍼차징 범위 = 가장 낮은 온도의 포화압 ~ 상한 (sample/perturb/quantize 가 전 범위를 쓰도록)"""
    lo = math.ceil(p_sat(ft.TEMP_RANGE[0]) / GA.PRESSURE_RESOLUTION) * GA.PRESSURE_RESOLUTION
    GA.BOUNDS["supercharge_pressure"] = (lo, P_MAX)


def with_temperature(rng, g, kind):
    """설계 genome 에 초기 온도를 붙인다. 슈퍼차징 압력이 그 온도의 포화압보다 낮으면 (헬륨 0 g 인 셈) 범위 안에서 다시 고른다"""
    T = round(rng.uniform(*ft.TEMP_RANGE), 1)
    lo = p_sat(T)
    g = dict(g)
    if lo >= P_MAX:
        g["supercharge_pressure"] = math.ceil(lo / GA.PRESSURE_RESOLUTION) * GA.PRESSURE_RESOLUTION
    elif g["supercharge_pressure"] < lo or kind in ("uniform", "archive_he"):
        g["supercharge_pressure"] = math.floor(rng.uniform(lo, P_MAX) / GA.PRESSURE_RESOLUTION) * GA.PRESSURE_RESOLUTION
        g["supercharge_pressure"] = max(g["supercharge_pressure"], math.ceil(lo / GA.PRESSURE_RESOLUTION) * GA.PRESSURE_RESOLUTION)
    return g, T


def data_row(design, pipe):
    """shard 에 저장하는 입력 행: rl.features.FEATURE_VARS + 파이프 3개 (학습 때 convert_X 로 현재 특징만 고른다)"""
    return [design[v] for v in ft.FEATURE_VARS] + [pipe[f] for f in PIPE_FEATURES]


def run_sim(design):
    return GA.simulate(design, series=True)


def save_shard(out_dir, k, rows):
    X = np.array([r["x"] for r in rows], dtype=np.float32)
    S = np.array([r["series"] for r in rows], dtype=np.float32)
    meta = [{k2: r[k2] for k2 in ("kind", "pipe_id", "termination_code", "error", "scalars")} for r in rows]
    np.savez_compressed(os.path.join(out_dir, f"shard_{k:04d}.npz"), X=X, S=S,
                        meta=np.array(json.dumps(meta)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15000, help="정상 종료(오류 아님)한 시뮬레이션 개수 목표 (실패 설계도 함께 저장)")
    parser.add_argument("--workers", type=int, default=GA.N_WORKERS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    wide_pressure_bounds()
    pipes = GA.load_candidate_pipes()
    fixed = dict(GA.load_fixed_values(), simt=SIMT)
    archive = load_archive(pipes, dict(fixed, temperature=ft.LEGACY_TEMPERATURE))
    stream = design_stream(rng, pipes, fixed, archive)
    n_t = int(round(SIMT / GA.SERIES_DT))

    out_dir = os.path.join(DATA_DIR, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump({"simt": SIMT, "series_dt": GA.SERIES_DT, "channels": GA.SERIES_CHANNELS,
                   "design_vars": GA.DESIGN_VARS, "feature_vars": ft.FEATURE_VARS, "pipe_features": PIPE_FEATURES,
                   "scalars": SCALARS, "temperature_range": ft.TEMP_RANGE,
                   "mix": MIX, "n_target": args.n, "workers": args.workers}, f, indent=2)
    msg = f"시계열 데이터 {args.n}개 생성 (워커 {args.workers}, simt {SIMT}s, 기존 설계 {len(archive)}개) → {out_dir}"
    print(msg, flush=True)
    mon = DataMonitor("tsdata_" + os.path.basename(out_dir), out_dir, args.n, os.getpid(),
                      {"simt": SIMT, "mix": MIX, "n_target": args.n, "workers": args.workers,
                       "temperature_range": ft.TEMP_RANGE,
                       "pressure_bounds": GA.BOUNDS["supercharge_pressure"], "data_dir": os.path.relpath(out_dir, HERE)})
    mon.write_log(msg)
    mon.save()

    os.makedirs(os.path.join(os.path.expanduser("~"), "RocketCEA"), exist_ok=True)
    t0, done, n_ok, shard_k, rows = time.time(), 0, 0, 0, []
    kinds_done = {k: 0 for k in MIX}
    state = "done"
    pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"))
    try:
        futures = {}
        while n_ok < args.n:
            while len(futures) < args.workers * 2:
                kind, g = next(stream)
                g, T = with_temperature(rng, g, kind)
                design, pipe = GA.decode(g, pipes, dict(fixed, temperature=T))
                futures[pool.submit(run_sim, design)] = (kind, design, pipe)
            fin, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for fut in fin:
                kind, design, pipe = futures.pop(fut)
                sim = fut.result()
                done += 1
                n_ok += not sim.get("error")
                kinds_done[kind] += 1
                series = sim.get("series") or np.zeros((len(GA.SERIES_CHANNELS), n_t)).tolist()
                rows.append({"x": data_row(design, pipe),
                             "series": series, "kind": kind, "pipe_id": pipe["pipe_id"],
                             "termination_code": sim.get("termination_code"), "error": sim.get("error"),
                             "scalars": {s: sim.get(s) for s in SCALARS}})
                mon.stats.add(rows[-1]["x"], kind, dict(rows[-1]["scalars"], error=sim.get("error"),
                                                         termination_code=sim.get("termination_code")))
                mon.progress(done, n_ok, n_ok / (time.time() - t0), kinds_done)
                if len(rows) >= SHARD:
                    save_shard(out_dir, shard_k, rows)
                    shard_k += 1
                    rows = []
                if done % 200 == 0:
                    rate = n_ok / (time.time() - t0)
                    line = f"  정상 {n_ok}/{args.n} (전체 {done})  {rate:.1f}개/s  남은 {(args.n - n_ok) / max(rate, 1e-9) / 60:.1f}분  {kinds_done}"
                    print(line, flush=True)
                    mon.write_log(line)
    except KeyboardInterrupt:
        print("\n중단됨 — 지금까지의 결과를 저장합니다", flush=True)
        state = "stopped"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        if rows:
            save_shard(out_dir, shard_k, rows)
        mon.progress(done, n_ok, n_ok / max(time.time() - t0, 1e-9), kinds_done, force=True)
        mon.write_log(f"{'완료' if state == 'done' else '중단'}: {out_dir}")
        mon.finish(state)
    print(f"완료: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
