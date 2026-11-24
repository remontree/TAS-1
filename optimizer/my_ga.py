"""
하이브리드 엔진 MDO - 유전 알고리즘(GA) 최적화기
────────────────────────────────────────────────────────────────────────────
사용자 입력 : 목표 총 임펄스 [N·s], 목표 평균 추력 [N]
반환       : 모든 제약 조건을 만족하면서 비추력(Isp)이 최대인 엔진 형상
흐름
    최적화기 → 후보 선정(0.01 mm 단위로 절사) → 형상 제약 검사(constrain.pre_sim_constraints)
    → core 시뮬레이션 → 결과 기반 제약 검사(constrain.post_sim_constraints) → GA 선택/교배/변이
설계 가정
    - 산화제 탱크와 연소 챔버는 PIPEDB.csv 의 같은 파이프 두 개로 만든다.
    - 파이프 내경은 그대로 사용 → oxidizer_tank_radius = combution_chamber_radius = 내경/2
      (그래서 두 반지름은 독립 변수가 아니라 "어떤 파이프를 쓸지" 유전자로 결정된다)
    - 외경을 CNC 로 깍아 두께(oxidizer_tank_tickness)를 정한다. 챔버 두께도 같은 값.
    - 연료 포트는 원형 단일(port_num=1, 중앙) 또는 원형 다공(port_num>=2)만 사용.
    - 설계 변수 외 나머지 core 입력값은 fixedvalue.json 에서 읽는다 (실행할 때마다 새로 읽음).
실행
    python GA.py                          # 아래 TARGET_* 값 사용
    python GA.py --impulse 12000 --thrust 750
모든 값은 SI 단위.
"""

import os

# ══════════════════════════════════════════════════════════════════════════
# 핵심 하이퍼파라미터 (여기만 수정하면 됨)
# ══════════════════════════════════════════════════════════════════════════


# 목표 성능 (CLI 인자로 덮어쓸 수 있음)
TARGET_IMPULSE = 12000.0 # 목표 총 임펄스 [N·s]
TARGET_THRUST_AVG = 750.0 # 목표 평균 추력 [N]


# GA
POP_SIZE = 32 # 개체 수
N_GENERATIONS = 30 # 최대 세대 수
ELITE_COUNT = 2 # 다음 세대로 그대로 넘기는 상위 개체 수
TOURNAMENT_SIZE = 3 # 토너먼트 선택 크기
CROSSOVER_RATE = 0.9 # 교배 확률
BLX_ALPHA = 0.3 # BLX-alpha 교배의 탐색 확장 비율
MUTATION_RATE = 0.15 # 유전자별 변이 확률
MUTATION_SIGMA = 0.10 # 연속 변수 변이 표준 편차 (탐색 범위 대비 비율)
EARLY_STOP_PATIENCE = 10 # 최고 ISP가 이 세대 수 동안 개선 없으면 종료
INIT_MAX_TRIES = 2000 # 초기 개체 1개 당 형상 제약을 만족하는 후보를 찾는 최대 시도 수
RANDOM_SEED = None # 재현이 필요하면 정수


# 실행 환경
N_WORKERS = max(1, (os.cpu_count() or 2) - 2) # 병렬 시뮬레이션 프로세스 수


# 사용할 파이프 (None = PIPEDB.csv 전체, 아니면 pipe_id 리스트)

PIPE_IDS = None


# 가공 해상도
LENGTH_RESOLUTION = 1e-5 # 길이 변수 해상도 [m] = 0.01mm (그 아래 자릿수는 버림)
MASS_RESOLUTION = 0.01 # 산화제 질량 해상도 [kg]

#성능 산출
GO = 9.80665
BURN_THRESHOLD = 0.05 # 최대 추력의 이 비율 이상인 구간을 연소 시간으로 보고 평균 추력 계산


# 설계 변수 탐색 범위 (min, max) [SI]
# oxidizer_tank_radius, combution_chamber_radius 는 파이프 내경으로 결정되므로 여기 없음
BOUNDS = {
    "oxidizer_M":               (1.0,    10.0),    # 산화제 질량 [kg]
    "port_num":                 (1,      8),       # 연료 포트 개수 (정수)
    "outter_radius":            (0.02,   0.065),   # 연료 외부 반지름 [m]
    "port_center_radius":       (0.005,  0.05),    # 다공 포트 중심 배치 반지름 [m] (단일 포트면 0)
    "inner_radius":             (0.003,  0.03),    # 포트 반지름 [m]
    "fuel_length":              (0.1,    1.0),     # 연료 길이 [m]
    "oxidizer_tank_length":     (0.3,    2.0),     # 산화제 탱크 길이 [m]
    "oxidizer_tank_tickness":   (0.001,  0.01),    # 탱크/챔버 가공 두께 [m]
    "combution_chamber_length": (0.2,    1.5),     # 연소 챔버 길이 [m]
    "injector_oriffice_n":      (4,      64),      # 인젝터 오리피스 개수 (정수)
    "injector_oriffice_radius": (0.0003, 0.0015),  # 인젝터 오리피스 반지름 [m]
    "injector_oriffice_length": (0.003,  0.03),    # 인젝터 오리피스 길이 [m]
    "nozzle_throat_radius":     (0.004,  0.02),    # 노즐 목 반지름 [m]
    "nozzle_exit_radius":       (0.008,  0.05),    # 노즐 출구 반지름 [m]
}


import argparse
import contextlib
import csv
import json
import math
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np


OPTIMIZER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIRT = os.path.dirname(OPTIMIZER_DIR)

for _p in (OPTIMIZER_DIR, PROJECT_DIRT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import constrain as cs

from managepipedb import load_pipes

FIXED_VALUE_PATH = os.path.join(OPTIMIZER_DIR, "fixedvalue.json")
RESULT_TS_DIR = os.path.join(OPTIMIZER_DIR, "results")


INT_VARS = {"port_num", "injector_oriffice_n"}
MASS_VARS = {"oxidizer_M"}
DESIGN_VARS = list(BOUNDS.keys())
PIPE_VARS = ["oxidizer_tank_radius", "combution_chamber_radius"]


# ──────────────────────────────────────────────────────────────────────────
# 설정 로드
# ──────────────────────────────────────────────────────────────────────────


def load_fixed_values(path=FIXED_VALUE_PATH):
    with open(path, "r", encoding="utf-8") as f:
        fixed = json.load(f)

    overlap = set(fixed) & (set(DESIGN_VARS) | set(PIPE_VARS))

    if overlap:
        print(f"[경고] fixedvalue.json 의 {sorted(overlap)} 는 설계 변수라 무시됩니다.")
        for k in overlap:
            fixed.pop(k)
    if str(fixed.get("grain_type", "PORT  ")).upper() != "PORT":
        raise ValueError("포트는 원형 단일/다공만 지원합니다. fixedvalue.json 의 grain_type 을 PORT 로 두세요.")
    fixed["grain_type"] = "PORT"
    return fixed


def load_candidate_pipes():
    pipes = load_pipes()
    if PIPE_IDS is not None:
        pipes = [p for p in pipes if p["pipe_id"] in PIPE_IDS]
    if not pipes:
        raise ValueError("사용 가능한 파이프가 없습니다. managepipedb.py 로 PIPEDB.csv 에 등록하세요.")
    return pipes

# ──────────────────────────────────────────────────────────────────────────
# 유전자 표현 / 양자화
#   genome = {"pipe": 파이프 인덱스, 설계변수: 값, ...}
# ──────────────────────────────────────────────────────────────────────────

def _truncate(x, res):
    decimals = max(0, round(-math.log10(res))) # 소숫점 아래 몇 칸 까지 살릴지
    return round(math.floor(x/res + 1e-6)*res, decimals)


def quantize(genome):
    g = dict(genome)

    for var in DESIGN_VARS:
        lo, hi = BOUNDS[var]
        x = min(max(g[var], lo), hi)
        if var in INT_VARS:
            g[var] = int(round(x))
        elif var in MASS_VARS:
            g[var] = _truncate(x, MASS_RESOLUTION)
        else:
            g[var] = _truncate(x, LENGTH_RESOLUTION)
    return g


def decode(genome, pipes, fixed):
    pipe = pipes[genome["pipe"]]
    design = dict(fixed)
    for var in DESIGN_VARS:
        design[var] = genome[var]
    design["oxidizer_tank_radius"] = pipe["inner_diameter"] / 2
    design["combution_chamber_radius"] = pipe["inner_diameter"] / 2

    if design["port_num"] == 1:
        design["port_center_radius"] = 0.0
    return design, pipe


def design_key(design, pipe):
    return (pipe["pipe_id"], ) + tuple(design[v] for v in DESIGN_VARS)



# ──────────────────────────────────────────────────────────────────────────
# 초기 개체 생성 - 변수 간 의존 관계 순서로 뽑아서 형상 제약을 만족할 확률을 높인다
# ──────────────────────────────────────────────────────────────────────────


def _uniform(rng, var, hi_limit=None, lo_limit=None):
    lo, hi = BOUNDS[var]
    if lo_limit is not None:
        lo = max(lo, lo_limit)
    if hi_limit is not None:
        hi = min(hi, hi_limit)
    if lo > hi:
        raise ValueError(var)
    return rng.uniform(lo, hi)


def sample_genome(rng, pipes):
    g = {"pipe": rng.randrange(len(pipes))}
    pipe = pipes[g["pipe"]]
    rcc = pipe["inner_diameter"] / 2

    g["oxidizer_tank_tickness"]   = _uniform(rng, "oxidizer_tank_tickness", pipe["wall_thickness"])
    g["oxidizer_tank_length"]     = _uniform(rng, "oxidizer_tank_length", pipe["length"])
    g["combution_chamber_length"] = _uniform(rng, "combution_chamber_length", pipe["length"])
    g["oxidizer_M"]               = _uniform(rng, "oxidizer_M")

    g["outter_radius"] = _uniform(rng, "outter_radius", rcc)
    g["port_num"]      = rng.randint(*BOUNDS["port_num"])
    g["inner_radius"]  = _uniform(rng, "inner_radius",
                                  g["outter_radius"] * math.sqrt(cs.FUEL_MARGIN / g["port_num"]))
    g["port_center_radius"] = _uniform(rng, "port_center_radius",
                                       g["outter_radius"] - g["inner_radius"])
    g["fuel_length"] = _uniform(rng, "fuel_length",
                                g["combution_chamber_length"] * cs.BULKHEAD_NOZZLE_MARGIN
                                - 2 * g["outter_radius"])

    g["injector_oriffice_n"]      = rng.randint(*BOUNDS["injector_oriffice_n"])
    g["injector_oriffice_radius"] = _uniform(rng, "injector_oriffice_radius")
    g["injector_oriffice_length"] = _uniform(rng, "injector_oriffice_length", cs.MAX_ORI)

    g["nozzle_throat_radius"] = _uniform(rng, "nozzle_throat_radius", rcc)
    g["nozzle_exit_radius"]   = _uniform(rng, "nozzle_exit_radius",
                                         rcc + g["oxidizer_tank_tickness"],
                                         g["nozzle_throat_radius"])
    return quantize(g)



def initial_population(rng, pipes, fixed, size):
    population, tries = [], 0

    while len(population) < size:
        tries += 1
        try:
            g = sample_genome(rng, pipes)
        except ValueError:
            continue
        design, pipe = decode(g, pipes, fixed)
        ok, _ , _ = cs.check(cs.pre_sim_constraints(design, pipe)[0])

        if ok or tries > INIT_MAX_TRIES * size:
            population.append(g)

    if tries > INIT_MAX_TRIES * size:
        print("[경고] 형상 제약을 만족하는 초기 개체를 충분히 찾지 못했습니다. BOUNDS 를 확인하세요.")
    return population


# ──────────────────────────────────────────────────────────────────────────
# 시뮬레이션 (워커 프로세스에서 실행)
# ──────────────────────────────────────────────────────────────────────────


def simulate(config):
    """
    core 시뮬레이션 1회. Rocket.ignite() 는 ../result.csv 를 덮어쓰므로 쓰지 않고
    engine.simulation() 을 직접 호출한다. core 의 스텝별 print 는 버린다.
    """

    started = time.time()

    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            from core.rocket import Rocket
            rocket = Rocket(config)
            rocket.engine.simulation(config["dt"], config["simt"]. config["thrust_coeff"])
        e = rocket.engine
        dt = config["dt"]

        thrust = np.asarray(e.thrust, dtype=float)
        mox    = np.asarray(e.mox_dot, dtype=float)
        mf     = np.asarray(e.mf_dot, dtype=float)
        isp_t  = np.asarray(e.Isp, dtype=float)

        impulse   = float(e.impulse)
        m_ox      = float(np.sum(mox) * dt)
        m_fuel    = float(np.sum(mf) * dt)
        m_prop    = m_ox + m_fuel
        isp       = impulse / (m_prop * GO) if m_prop > 0 else 0.0

        f_max = float(thrust.max()) if thrust.size else 0.0
        burn = thrust >= BURN_THRESHOLD * f_max if f_max > 0 else np.zeros_like(thrust, dtype=bool)
        burn_time = float(np.count_nonzero(burn)*dt)
        thrust_avg = float(np.sum(thrust[burn]) * dt / burn_time) if burn_time > 0 else 0.0


        return {
            "error": None,
            "termination_code": e.termination_code,
            "impulse": impulse,
            "isp": isp,
            "isp_mean": float(isp_t[burn].mean()) if burn.any() else 0.0,
            "thrust_avg": thrust_avg,
            "thrust_max": f_max,
            "burn_time": burn_time,
            "sim_time": float(e.time_step[-1]),
            "m_oxidizer_used": m_ox,
            "m_fuel_used": m_fuel,
            "of_avg": m_ox / m_fuel if m_fuel > 0 else 0.0,
            "max_tank_pressure": float(max(e.tank_pressure)),
            "max_chamber_pressure": float(max(e.chamber_pressure)),
            "wall_time": time.time() - started,
        }
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}"[:300],
            "termination_code": None,
            "wall_time": time.time() - started,
        }


# ──────────────────────────────────────────────────────────────────────────
# 평가
# ──────────────────────────────────────────────────────────────────────────



def rank_key(ev):
    if ev["feasible"]:
        return (0, -ev["sim"]["isp"])
    if ev["sim"] is not None:
        return (1, ev["violation"])
    return (2, ev["violation"])


class StopRequested(Exception):
    """외부(GUI)에서 최적화 중단을 요청했을 때"""



def candidate_summary(ev):
    """GUI/로그용 후보 요약 (JSON 직렬화 가능)"""
    sim = ev.get("sim") or {}
    if ev["feasible"]:
        state = "feasible"
    elif ev.get("sim") is not None:
        state = "infeasible"
    else:
        state = "rejected"
    return {
        "state": state,
        "pipe_id": ev["pipe"]["pipe_id"],
        "violation": ev["violation"],
        "failed": ev["failed"],
        "isp": sim.get("isp"),
        "impulse": sim.get("impulse"),
        "thrust_avg": sim.get("thrust_avg"),
        "burn_time": sim.get("burn_time"),
        "error": sim.get("error"),
        "design": {v: ev["design"][v] for v in DESIGN_VARS + PIPE_VARS},
    }


def evaluate(population, pipes, fixed, targets, cache, pool, counters, notify=None, should_stop=None):
    notify = notify or (lambda i ,state, ev : None)
    evaluations = [None] * len(population)

    keys = []

    waiting = {}
    futures = {}

    for i, g in enumerate(population):
        design, pipe = decode(g, pipes, fixed)
        key = design_key(design, pipe)
        keys.append(key)

        if key in cache:
            notify(i, None, cache[key])
            continue

        if key in waiting:
            waiting[key].append(i)
            notify(i,"simulating", None)
            continue

        pre, tank = cs.pre_sim_constraints(design, pipe)
        ok, violation, failed = cs.check(pre)
        if not ok:
            cache[key] = {"design": design, "pipe": pipe, "tank": tank, "sim": None,
                          "feasible": False, "violation": violation, "failed": failed}
            counters["pre_rejected"] += 1
            notify(i, None, cache[key])
            continue
        waiting[key] = [i]
        futures[pool.submit(simulate, design)] = (key, design, pipe, tank)
        notify(i, "simulating", None)

    remaining = set(futures)
    while remaining:
        if should_stop and should_stop():
            for fut in remaining:
                fut.cancel()
            raise StopRequested()
        done, remaining = wait(remaining, timeout=0.5, return_when=FIRST_COMPLETED)
        for fut in done:
            key, design, pipe, tank = futures[fut]
            sim = fut.result()
            counters["simulated"] += 1
            post = cs.post_sim_constraints(design, pipe, sim, *targets)
            ok, violation, failed = cs.check(post)
            cache[key] = {"design": design, "pipe": pipe, "tank": tank, "sim": sim,
                          "feasible": ok, "violation": violation, "failed": failed}
            for i in waiting[key]:
                notify(i, None, cache[key])

    for i, g in enumerate(population):
        evaluations[i] = dict(cache[keys[i]], genome=g)
    return evaluations



# ──────────────────────────────────────────────────────────────────────────
# GA 연산자
# ──────────────────────────────────────────────────────────────────────────

def tournament(rng, evaluations):
    contenders = rng.sample(evaluations, min(TOURNAMENT_SIZE, len(evaluations)))
    return min(contenders, key=rank_key)["genome"]


def crossover(rng, a, b):
    if rng.random() > CROSSOVER_RATE:
        return dict(a), dict(b)
    c1, c2 = {}, {}
    c1["pipe"], c2["pipe"] = (a["pipe"], b["pipe"]) if rng.random() < 0.5 else (b["pipe"], a["pipe"])
    for var in DESIGN_VARS:
        lo, hi = min(a[var], b[var]), max(a[var], b[var])
        span = hi - lo
        c1[var] = rng.uniform(lo - BLX_ALPHA * span, hi + BLX_ALPHA * span)
        c2[var] = rng.uniform(lo - BLX_ALPHA * span, hi + BLX_ALPHA * span)
    return c1, c2


def mutate(rng, g, n_pipes):
    g = dict(g)
    if n_pipes > 1 and rng.random() < MUTATION_RATE:
        g["pipe"] = rng.randrange(n_pipes)
    for var in DESIGN_VARS:
        if rng.random() >= MUTATION_RATE:
            continue
        lo, hi = BOUNDS[var]
        if var in INT_VARS:
            step = max(1, round(MUTATION_SIGMA * (hi - lo)))
            g[var] = g[var] + rng.randint(-step, step)
        else:
            g[var] = g[var] + rng.gauss(0.0, MUTATION_SIGMA * (hi - lo))
    return quantize(g)




# ──────────────────────────────────────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────────────────────────────────────

def run_ga(target_impulse=None, target_thrust_avg=None, verbose=True,
           pop_size=None, n_generations=None, seed=None, n_workers=None,
           progress=None, should_stop=None):
    """
    progress(event, data): 진행 상황 콜백 (GUI 용). event 종류
        start            {pipes, pop_size, n_generations, n_workers, targets}
        generation_start {generation}
        candidate        {generation, index, state, summary}
        generation_end   {generation, history, ranking, best}
        done             {stopped, best, output_dir, elapsed, counters}
    should_stop(): True 를 반환하면 현재까지의 최선 결과로 종료
    """
    target_impulse    = TARGET_IMPULSE if target_impulse is None else target_impulse
    target_thrust_avg = TARGET_THRUST_AVG if target_thrust_avg is None else target_thrust_avg
    pop_size          = pop_size or POP_SIZE
    n_generations     = n_generations or N_GENERATIONS
    n_workers         = n_workers or N_WORKERS
    seed              = RANDOM_SEED if seed is None else seed
    targets = (target_impulse, target_thrust_avg)
    emit = progress or (lambda event, data: None)

    rng    = random.Random(seed)
    fixed  = load_fixed_values()
    pipes  = load_candidate_pipes()
    cache  = {}
    counters = {"pre_rejected": 0, "simulated": 0}
    history = []
    best, stall = None, 0
    stopped = False
    started = time.time()

    # RocketCEA 가 첫 실행 시 ~/RocketCEA 를 만드는데, 여러 프로세스가 동시에 만들면 충돌
    os.makedirs(os.path.join(os.path.expanduser("~"), "RocketCEA"), exist_ok=True)

    if verbose:
        print(f"[GA] 목표 임펄스 {target_impulse:.1f} N·s, 목표 평균 추력 {target_thrust_avg:.1f} N")
        print(f"[GA] 파이프 {len(pipes)}종, 개체 {pop_size}, 최대 세대 {n_generations}, 워커 {n_workers}")
    emit("start", {"pipes": [p["pipe_id"] for p in pipes], "pop_size": pop_size,
                   "n_generations": n_generations, "n_workers": n_workers,
                   "targets": {"impulse": target_impulse, "thrust_avg": target_thrust_avg}})

    population = initial_population(rng, pipes, fixed, pop_size)

    pool = ProcessPoolExecutor(max_workers=n_workers)
    try:
        for gen in range(n_generations):
            t_gen = time.time()
            emit("generation_start", {"generation": gen})

            def notify(i, state, ev, gen=gen):
                if state == "simulating":
                    emit("candidate", {"generation": gen, "index": i, "state": "simulating", "summary": None})
                else:
                    summary = candidate_summary(ev)
                    emit("candidate", {"generation": gen, "index": i, "state": summary["state"], "summary": summary})

            try:
                evaluations = evaluate(population, pipes, fixed, targets, cache, pool, counters,
                                       notify, should_stop)
            except StopRequested:
                stopped = True
                if verbose:
                    print("[GA] 사용자 요청으로 중단")
                break

            order = sorted(range(len(evaluations)), key=lambda i: rank_key(evaluations[i]))
            evaluations = [evaluations[i] for i in order]

            gen_best = evaluations[0]
            n_feasible = sum(ev["feasible"] for ev in evaluations)
            if best is None or rank_key(gen_best) < rank_key(best):
                best, stall = gen_best, 0
            else:
                stall += 1

            feasible_isp = [ev["sim"]["isp"] for ev in evaluations if ev["feasible"]]
            history.append({
                "generation": gen,
                "n_feasible": n_feasible,
                "n_rejected": sum(ev["sim"] is None for ev in evaluations),
                "best_feasible": best["feasible"],
                "best_isp": best["sim"]["isp"] if best["feasible"] else None,
                "gen_best_isp": max(feasible_isp) if feasible_isp else None,
                "mean_feasible_isp": sum(feasible_isp) / len(feasible_isp) if feasible_isp else None,
                "best_violation": best["violation"],
                "simulated_total": counters["simulated"],
                "rejected_total": counters["pre_rejected"],
                "duration": time.time() - t_gen,
                "elapsed": time.time() - started,
            })
            if verbose:
                print_generation(gen, evaluations, best, n_feasible, time.time() - t_gen)
            emit("generation_end", {"generation": gen, "history": history[-1],
                                    "ranking": order, "elite": order[:ELITE_COUNT],
                                    "best": candidate_summary(best)})

            if EARLY_STOP_PATIENCE and best["feasible"] and stall >= EARLY_STOP_PATIENCE:
                if verbose:
                    print(f"[GA] {EARLY_STOP_PATIENCE} 세대 동안 개선이 없어 조기 종료")
                break
            if gen == n_generations - 1:
                break

            next_pop = [ev["genome"] for ev in evaluations[:ELITE_COUNT]]
            while len(next_pop) < pop_size:
                c1, c2 = crossover(rng, tournament(rng, evaluations), tournament(rng, evaluations))
                next_pop.append(mutate(rng, c1, len(pipes)))
                if len(next_pop) < pop_size:
                    next_pop.append(mutate(rng, c2, len(pipes)))
            population = next_pop
    finally:
        # 중단 시에는 실행 중인 시뮬레이션을 기다리지 않는다 (남은 작업은 취소)
        pool.shutdown(wait=not stopped, cancel_futures=True)

    result = {
        "targets": {"impulse": target_impulse, "thrust_avg": target_thrust_avg},
        "best": best,
        "stopped": stopped,
        "history": history,
        "counters": counters,
        "elapsed": time.time() - started,
        "all_evaluations": list(cache.values()),
    }
    result["output_dir"] = save_result(result) if best is not None else None
    emit("done", {"stopped": stopped, "best": candidate_summary(best) if best else None,
                  "output_dir": result["output_dir"], "elapsed": result["elapsed"],
                  "counters": counters})
    return result
