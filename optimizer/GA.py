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
TARGET_IMPULSE      = 12000.0   # 목표 총 임펄스 [N·s]
TARGET_THRUST_AVG   = 750.0     # 목표 평균 추력 [N]

# GA
POP_SIZE            = 32        # 개체 수
N_GENERATIONS       = 30        # 최대 세대 수
ELITE_COUNT         = 2         # 다음 세대로 그대로 넘기는 상위 개체 수
TOURNAMENT_SIZE     = 3         # 토너먼트 선택 크기
CROSSOVER_RATE      = 0.9       # 교배 확률
BLX_ALPHA           = 0.3       # BLX-alpha 교배의 탐색 확장 비율
MUTATION_RATE       = 0.15      # 유전자별 변이 확률
MUTATION_SIGMA      = 0.10      # 연속 변수 변이 표준편차 (탐색 범위 대비 비율)
EARLY_STOP_PATIENCE = 0         # 최고 Isp 가 이 세대 수 동안 개선 없으면 종료 (0 = 사용 안 함 → 항상 지정 세대까지)
STALL_NORM          = 10        # PPO 상태의 '정체 세대 수' 정규화 기준 (학습 환경과 같아야 함, 조기 종료와 무관)
INIT_MAX_TRIES      = 2000      # 초기 개체 1개당 형상 제약을 만족하는 후보를 찾는 최대 시도 수
RANDOM_SEED         = None      # 재현이 필요하면 정수
VERIFY_TOP          = 5         # 대리모델 평가 모드: 끝난 뒤 core 시뮬레이션으로 다시 확인할 상위 후보 수
# 대리모델 모드 실시간 검증: 검증 슬롯이 비면 그 시점의 '예측 최고 합격 형상' 을 백그라운드로 core 검증한다.
# 탐색(초당 수백 세대)이 core 시뮬레이션(10초 안팎)보다 훨씬 빨라, 초반 후보로 슬롯이 차지 않게 시작 조건을 둔다.
LIVE_VERIFY_MAX     = 2         # 동시에 검증하는 후보 수
LIVE_VERIFY_WARMUP  = 20        # 이 세대 이후부터
LIVE_VERIFY_STABLE  = 5         # 예측 최고가 이 세대 수 동안 그대로일 때 (막 바뀐 후보는 곧 더 좋은 것으로 바뀐다)
FEASIBLE_PARENTS    = False     # True: 합격 형상이 FEASIBLE_PARENT_MIN 개 이상인 세대부터 부모를 합격 형상 안에서만 고른다
FEASIBLE_PARENT_MIN = 3         #       (그 전에는 위반이 작은 쪽을 부모로 — 합격 영역을 찾아가는 단계)
ROBUST_PICK         = False     # True: Isp 가 최고와 ROBUST_ISP_BAND 이내인 합격 형상 중 제약 여유가 가장 큰 것을 최종 형상으로
ROBUST_ISP_BAND     = 1.5       # [s] 이 차이 이하는 대리모델 오차·제작 공차 안이라 사실상 같은 성능으로 본다
CORRECTION_ROUNDS   = 2         # 검증에서 합격이 없으면: core 결과로 대리모델 편향 보정 → 재탐색 → 재검증 (최대 횟수)
REFINE_GENERATIONS  = 30        # 편향 보정 후 추가로 진행할 세대 수
VERIFY_MIN_DIST     = 0.01      # 검증 후보끼리 설계 변수가 탐색 범위의 1% 이상 달라야 함

# 실행 환경
N_WORKERS           = max(1, (os.cpu_count() or 2) - 2)   # 병렬 시뮬레이션 프로세스 수

# 사용할 파이프 (None = PIPEDB.csv 전체, 아니면 pipe_id 리스트)
PIPE_IDS            = None

# 가공 해상도
LENGTH_RESOLUTION   = 1e-5      # 길이 변수 해상도 [m] = 0.01 mm (그 아래 자릿수는 버림)
MASS_RESOLUTION     = 0.01      # 산화제 질량 해상도 [kg]
PRESSURE_RESOLUTION = 1e4       # 슈퍼차징 압력 해상도 [Pa] = 0.01 MPa
HELIUM_RESOLUTION   = 1e-6      # 헬륨 질량 해상도 [kg] = 1 mg

# 예전 결과(설계 변수에 슈퍼차징 압력이 없던 때)의 고정 헬륨 질량 [kg] — 예전 데이터를 새 변수로 바꿀 때만 사용
LEGACY_PRESSURANT_M = 0.025

# 성능 산출
G0                  = 9.80665   # 표준 중력 가속도 [m/s^2]
BURN_THRESHOLD      = 0.05      # 최대 추력의 이 비율 이상인 구간을 연소 시간으로 보고 평균 추력 계산

# 설계 변수 탐색 범위 (min, max) [SI]
# oxidizer_tank_radius, combution_chamber_radius 는 파이프 내경으로 결정되므로 여기 없음
# 연소 챔버 길이와 헬륨 질량은 설계 변수가 아니라 다른 변수로 정해진다 (DERIVED_VARS, derive()):
#   combution_chamber_length = (연료 길이 + 2 × 연료 외경 반지름) × cs.CHAMBER_LENGTH_FACTOR (가공 해상도로 올림)
#   pressurant_M             = 탱크를 supercharge_pressure 까지 올리는 헬륨 질량 (N2O 포화압 위의 분압 × 기체 공간)
BOUNDS = {
    "oxidizer_M":               (1.0,    10.0),    # 산화제 질량 [kg]
    "port_num":                 (2,      8),       # 연료 포트 개수 (정수, 단일 포트 제외)
    "outter_radius":            (0.02,   0.065),   # 연료 외부 반지름 [m]
    "port_center_radius":       (0.005,  0.05),    # 다공 포트 중심 배치 반지름 [m] (단일 포트면 0)
    "inner_radius":             (0.003,  0.03),    # 포트 반지름 [m]
    "fuel_length":              (0.1,    1.0),     # 연료 길이 [m]
    "oxidizer_tank_length":     (0.3,    2.0),     # 산화제 탱크 길이 [m]
    "oxidizer_tank_tickness":   (0.001,  0.01),    # 탱크/챔버 가공 두께 [m]
    "supercharge_pressure":     (2.3e6,  7.0e6),   # 헬륨 슈퍼차징 후 탱크 초기 압력 [Pa] (하한은 아래에서 N2O 포화압 + 여유로 다시 정함, 상한 < 0.97 × 임계압)
    "injector_oriffice_n":      (4,      64),      # 인젝터 오리피스 개수 (정수)
    "injector_oriffice_radius": (0.0003, 0.0015),  # 인젝터 오리피스 반지름 [m]
    "injector_oriffice_length": (0.003,  0.03),    # 인젝터 오리피스 길이 [m]
    "nozzle_throat_radius":     (0.004,  0.02),    # 노즐 목 반지름 [m]
    "nozzle_exit_radius":       (0.008,  0.05),    # 노즐 출구 반지름 [m]
}

# ══════════════════════════════════════════════════════════════════════════

import argparse
import contextlib
import csv
import hashlib
import json
import math
import multiprocessing
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

OPTIMIZER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR   = os.path.dirname(OPTIMIZER_DIR)
for _p in (OPTIMIZER_DIR, PROJECT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import constrain as cs
import cpu_gate
from managepipedb import load_pipes

FIXED_VALUE_PATH = os.path.join(OPTIMIZER_DIR, "fixedvalue.json")
RESULTS_DIR      = os.path.join(OPTIMIZER_DIR, "results")
# 대리모델로 평가한 실행은 따로 저장 (results/ 는 대리모델 학습 데이터라 예측값이 섞이면 안 된다)
SURROGATE_RESULTS_DIR = os.path.join(OPTIMIZER_DIR, "results_surrogate")

INT_VARS    = {"port_num", "injector_oriffice_n"}
MASS_VARS   = {"oxidizer_M"}
PRESSURE_VARS = {"supercharge_pressure"}
DESIGN_VARS = list(BOUNDS.keys())
DERIVED_VARS = ["combution_chamber_length", "pressurant_M"]
PIPE_VARS   = ["oxidizer_tank_radius", "combution_chamber_radius"]


# ──────────────────────────────────────────────────────────────────────────
# 설정 로드
# ──────────────────────────────────────────────────────────────────────────

def load_fixed_values(path=FIXED_VALUE_PATH):
    with open(path, "r", encoding="utf-8") as f:
        fixed = json.load(f)
    overlap = set(fixed) & (set(DESIGN_VARS) | set(DERIVED_VARS) | set(PIPE_VARS))
    if overlap:
        print(f"[경고] fixedvalue.json 의 {sorted(overlap)} 는 설계 변수라 무시됩니다.")
        for k in overlap:
            fixed.pop(k)
    if str(fixed.get("grain_type", "PORT")).upper() != "PORT":
        raise ValueError("포트는 원형 단일/다공만 지원합니다. fixedvalue.json 의 grain_type 을 PORT 로 두세요.")
    fixed["grain_type"] = "PORT"
    return fixed


# 결과 (candidates.csv · summary) 에 함께 남기는 고정값 — 대리모델이 입력으로 쓰는 운용 조건
RECORD_FIXED = ["temperature"]

# 슈퍼차징 하한 = 초기 온도에서의 N2O 포화압 + SUPERCHARGE_MIN_MARGIN (그 아래는 헬륨 0 g 이라 의미가 없다)
#   258 K (−15 °C): 2.07 MPa → 2.3 MPa,  293 K (20 °C): 5.04 MPa → 5.3 MPa
SUPERCHARGE_MIN_MARGIN = 0.25e6


def _supercharge_bounds():
    T = load_fixed_values()["temperature"]
    p_sat = cs._saturated("P", "T", T, "Q", 0, "NitrousOxide") * 1.001
    lo = math.ceil((p_sat + SUPERCHARGE_MIN_MARGIN) / 1e5) * 1e5
    hi = BOUNDS["supercharge_pressure"][1]
    if lo >= hi:
        raise ValueError(f"초기 온도 {T} K 에서 N2O 포화압 {p_sat / 1e6:.2f} MPa — 슈퍼차징 범위가 없습니다 (상한 {hi / 1e6} MPa).")
    return (lo, hi)


BOUNDS["supercharge_pressure"] = _supercharge_bounds()


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
    decimals = max(0, round(-math.log10(res)))
    return round(math.floor(x / res + 1e-6) * res, decimals)


def quantize(genome):
    g = dict(genome)
    for var in DESIGN_VARS:
        lo, hi = BOUNDS[var]
        x = min(max(g[var], lo), hi)
        if var in INT_VARS:
            g[var] = int(round(x))
        elif var in MASS_VARS:
            g[var] = _truncate(x, MASS_RESOLUTION)
        elif var in PRESSURE_VARS:
            g[var] = _truncate(x, PRESSURE_RESOLUTION)
        else:
            g[var] = _truncate(x, LENGTH_RESOLUTION)
    return g


def decode(genome, pipes, fixed):
    """genome → core Rocket config (fixedvalue + 설계 변수) 와 파이프"""
    pipe = pipes[genome["pipe"]]
    design = dict(fixed)
    for var in DESIGN_VARS:
        design[var] = genome[var]
    design["oxidizer_tank_radius"]    = pipe["inner_diameter"] / 2
    design["combution_chamber_radius"] = pipe["inner_diameter"] / 2
    if design["port_num"] == 1:
        design["port_center_radius"] = 0.0
    derive(design)
    return design, pipe


def trimmed_chamber_length(design):
    """연소 챔버 길이 = (연료 길이 + 2 × 연료 외경 반지름) × CHAMBER_LENGTH_FACTOR, 길이 해상도로 올림"""
    need = (design["fuel_length"] + 2 * design["outter_radius"]) * cs.CHAMBER_LENGTH_FACTOR
    decimals = max(0, round(-math.log10(LENGTH_RESOLUTION)))
    return round(math.ceil(need / LENGTH_RESOLUTION - 1e-6) * LENGTH_RESOLUTION, decimals)


def helium_mass(design, pressure=None):
    """
    탱크 초기 압력을 pressure (기본 design["supercharge_pressure"]) 로 만드는 헬륨 질량 [kg].
    core/rocket.py 와 같은 계산: P_tank = Psat(T) + n_He R T / V_gas, V_gas 는 N2O 액/기 분배로 정해지고 헬륨과 무관.
    충진이 성립하지 않으면 (전부 기체 / 임계 온도 초과) 0 — 그 설계는 OxtFill 제약에서 탈락한다.
    """
    d = design
    pressure = d["supercharge_pressure"] if pressure is None else pressure
    st = cs.oxidizer_tank_state(d["oxidizer"], d["oxidizer_MW"], d["oxidizer_M"], d["pressurant"], d["pressurant_MW"],
                                0.0, d["temperature"], math.pi * d["oxidizer_tank_radius"] ** 2 * d["oxidizer_tank_length"])
    if not st["valid_T"] or st["N_liquid"] <= 0 or st["V_gas"] <= 0:
        return 0.0
    n = max(pressure - st["P_sat"], 0.0) * st["V_gas"] / (cs.R * d["temperature"])
    decimals = max(0, round(-math.log10(HELIUM_RESOLUTION)))
    return round(round(n * d["pressurant_MW"] / HELIUM_RESOLUTION) * HELIUM_RESOLUTION, decimals)


def derive(design):
    """설계 변수로 정해지는 값 (DERIVED_VARS) 을 design 에 채운다 (제자리 수정, design 반환)"""
    design["combution_chamber_length"] = trimmed_chamber_length(design)
    design["pressurant_M"] = helium_mass(design)
    return design


def legacy_supercharge_pressure(design, pressurant_M=LEGACY_PRESSURANT_M):
    """예전 결과 (헬륨 질량 고정, 챔버 길이 자유) 의 탱크 초기 압력 = 그 설계의 supercharge_pressure 에 해당하는 값"""
    d = dict(design, pressurant=design.get("pressurant", "Helium"))
    st = cs.oxidizer_tank_state(d["oxidizer"], d["oxidizer_MW"], d["oxidizer_M"], d["pressurant"], d["pressurant_MW"],
                                pressurant_M, d["temperature"],
                                math.pi * d["oxidizer_tank_radius"] ** 2 * d["oxidizer_tank_length"])
    return st.get("P_tank", 0.0) if st["valid_T"] else 0.0


def design_key(design, pipe):
    return (pipe["pipe_id"],) + tuple(design[v] for v in DESIGN_VARS)


def design_id(design, pipe):
    """설계 하나를 가리키는 짧은 id (GUI 목록·상세 팝업용)"""
    return hashlib.sha1(repr(design_key(design, pipe)).encode()).hexdigest()[:12]


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
    g["oxidizer_M"]               = _uniform(rng, "oxidizer_M")
    g["supercharge_pressure"]     = _uniform(rng, "supercharge_pressure")

    g["outter_radius"] = _uniform(rng, "outter_radius", rcc)
    g["port_num"]      = rng.randint(*BOUNDS["port_num"])
    g["inner_radius"]  = _uniform(rng, "inner_radius",
                                  g["outter_radius"] * math.sqrt(cs.FUEL_MARGIN / g["port_num"]))
    g["port_center_radius"] = _uniform(rng, "port_center_radius",
                                       g["outter_radius"] - g["inner_radius"])
    # 챔버 길이 = (연료 길이 + 2R) × 1.3 이 파이프 길이를 넘지 않게
    g["fuel_length"] = _uniform(rng, "fuel_length",
                                pipe["length"] * cs.BULKHEAD_NOZZLE_MARGIN - 2 * g["outter_radius"])

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
        ok, _, _ = cs.check(cs.pre_sim_constraints(design, pipe)[0])
        if ok or tries > INIT_MAX_TRIES * size:
            population.append(g)
    if tries > INIT_MAX_TRIES * size:
        print("[경고] 형상 제약을 만족하는 초기 개체를 충분히 찾지 못했습니다. BOUNDS 를 확인하세요.")
    return population


# ──────────────────────────────────────────────────────────────────────────
# 시뮬레이션 (워커 프로세스에서 실행)
# ──────────────────────────────────────────────────────────────────────────

# 정상상태 구간: 추력이 최대의 SS_START_FRAC 에 처음 도달한 시점 ~ 급락(SS_DROP_WINDOW 초 사이 최대 추력의
# SS_DROP_FRAC 이상 감소, 액체 산화제 소진) 직전. 급락이 없으면 추력이 SS_END_FRAC 아래로 처음 떨어지기 전까지.
# (표시용 추가 지표 — MDO 의 목표 판정·Isp 는 기존 정의 그대로)
SS_START_FRAC  = 0.9
SS_DROP_WINDOW = 0.1     # [s]
SS_DROP_FRAC   = 0.2
SS_END_FRAC    = 0.5


def steady_state_metrics(thrust, mdot, dt):
    """
    thrust, mdot(산화제+연료 질량유량) 시계열 → 정상상태 구간 평균 추력과 평균 비추력
    반환: {ss_start, ss_end [s], ss_duration, thrust_avg_ss [N], isp_ss [s], impulse_ss [N·s]} 또는 None
    """
    thrust, mdot = np.asarray(thrust, dtype=float), np.asarray(mdot, dtype=float)
    n = min(len(thrust), len(mdot))
    if n < 3 or thrust.max() <= 0:
        return None
    thrust, mdot = thrust[:n], mdot[:n]
    f_max = thrust.max()
    i_peak = int(thrust.argmax())
    start = int(np.argmax(thrust >= SS_START_FRAC * f_max))
    k = max(1, int(round(SS_DROP_WINDOW / dt)))
    end = None
    if i_peak + k < n:
        drop = thrust[i_peak + k:] - thrust[i_peak:n - k]        # k 스텝 뒤와의 차이 (음수 = 감소)
        j = int(drop.argmin())
        if drop[j] <= -SS_DROP_FRAC * f_max:
            end = i_peak + j + 1                                  # 급락이 시작되기 직전까지
    if end is None:
        below = np.nonzero(thrust[i_peak:] < SS_END_FRAC * f_max)[0]
        end = i_peak + int(below[0]) if below.size else n
    end = max(end, start + 1)
    seg_f, seg_m = thrust[start:end], mdot[start:end]
    m_prop = float(seg_m.sum() * dt)
    impulse = float(seg_f.sum() * dt)
    return {
        "ss_start": start * dt, "ss_end": end * dt, "ss_duration": (end - start) * dt,
        "thrust_avg_ss": float(seg_f.mean()),
        "isp_ss": impulse / (m_prop * G0) if m_prop > 0 else None,
        "impulse_ss": impulse,
    }


SERIES_DT = 0.1          # 시계열 대리모델용 리샘플 간격 [s] (simulate(..., series=True))
SERIES_CHANNELS = ["thrust", "chamber_pressure", "tank_pressure", "mox_dot", "mf_dot"]


def _resample(x, k, n_out):
    """dt 간격 시계열 → k 개씩 평균 (길이 n_out, 모자라면 0)"""
    x = np.asarray(x, dtype=float)[:k * n_out]
    x = np.pad(x, (0, k * n_out - len(x)))
    return x.reshape(n_out, k).mean(1)


def simulate(config, series=False):
    """core 시뮬레이션 1회 — 시스템 CPU 사용률 관문 (cpu_gate) 을 통과한 뒤 실행"""
    with cpu_gate.slot():
        return _simulate(config, series)


def _simulate(config, series=False):
    """
    series=True: 결과에 "series" (SERIES_CHANNELS × simt/SERIES_DT, float32 리스트) 를 추가 — 시계열 대리모델 학습용
    core 시뮬레이션 1회. Rocket.ignite() 는 ../result.csv 를 덮어쓰므로 쓰지 않고
    engine.simulation() 을 직접 호출한다. core 의 스텝별 print 는 버린다.
    """
    started = time.time()
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            from core.rocket import Rocket
            rocket = Rocket(config)
            rocket.engine.simulation(config["dt"], config["simt"], config["thrust_coeff"])
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
        isp       = impulse / (m_prop * G0) if m_prop > 0 else 0.0

        f_max = float(thrust.max()) if thrust.size else 0.0
        burn = thrust >= BURN_THRESHOLD * f_max if f_max > 0 else np.zeros_like(thrust, dtype=bool)

        # 인젝터 차압 비율 (Pt - Pc)/Pc — 연소 구간의 최솟값/평균 (constrain.InjectorDP)
        p_tank = np.asarray(e.tank_pressure, dtype=float)
        p_cc   = np.asarray(e.chamber_pressure, dtype=float)
        n = min(len(p_tank), len(p_cc), len(burn))
        on = burn[:n] & (p_cc[:n] > 0)
        dp_ratio = (p_tank[:n][on] - p_cc[:n][on]) / p_cc[:n][on]
        burn_time  = float(np.count_nonzero(burn) * dt)
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
            "dp_ratio_min": float(dp_ratio.min()) if dp_ratio.size else None,
            "dp_ratio_avg": float(dp_ratio.mean()) if dp_ratio.size else None,
            **(steady_state_metrics(thrust, mox + mf, dt) or {}),
            **({"series": _series(e, config)} if series else {}),
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

def _series(e, config):
    k = max(1, int(round(SERIES_DT / config["dt"])))
    n_out = int(round(config["simt"] / SERIES_DT))
    return [_resample(getattr(e, c), k, n_out).astype(np.float32).tolist() for c in SERIES_CHANNELS]


def rank_key(ev):
    """작을수록 좋음. 합격(Isp 큰 순) < 시뮬 후 불합격(위반 작은 순) < 형상 불합격(위반 작은 순)"""
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
        "id": design_id(ev["design"], ev["pipe"]),
        "state": state,
        "pipe_id": ev["pipe"]["pipe_id"],
        "violation": ev["violation"],
        "failed": ev["failed"],
        "isp": sim.get("isp"),
        "impulse": sim.get("impulse"),
        "thrust_avg": sim.get("thrust_avg"),
        "burn_time": sim.get("burn_time"),
        "dp_ratio_min": cs.injector_dp_ratio(sim) if ev.get("sim") else None,
        "thrust_avg_ss": sim.get("thrust_avg_ss"),          # 정상상태 평균 (core 결과에만 있음)
        "isp_ss": sim.get("isp_ss"),
        "ss_start": sim.get("ss_start"),
        "ss_end": sim.get("ss_end"),
        "max_tank_pressure": sim.get("max_tank_pressure"),
        "max_chamber_pressure": sim.get("max_chamber_pressure"),
        "error": sim.get("error"),
        "design": {v: ev["design"][v] for v in DESIGN_VARS + DERIVED_VARS + RECORD_FIXED + PIPE_VARS},
    }


def evaluate(population, pipes, fixed, targets, cache, pool, counters,
             notify=None, should_stop=None, surrogate=None):
    """
    notify(i, state, ev): 후보 i 의 상태가 바뀔 때 호출 (state: simulating | 최종 상태)
    should_stop(): True 를 반환하면 StopRequested 발생
    surrogate: simulate_batch(designs, pipes) → sims 를 가진 객체 (rl.surrogate.SurrogateSimulator).
               주면 core 시뮬레이션 대신 대리모델 예측으로 평가한다.
    """
    notify = notify or (lambda i, state, ev: None)
    evaluations = [None] * len(population)
    keys = []
    waiting = {}   # key → 같은 결과를 기다리는 개체 인덱스 목록
    futures = {}   # future → (key, design, pipe, tank)

    for i, g in enumerate(population):
        design, pipe = decode(g, pipes, fixed)
        key = design_key(design, pipe)
        keys.append(key)
        if key in cache:
            notify(i, None, cache[key])
            continue
        if key in waiting:
            waiting[key].append(i)
            notify(i, "simulating", None)
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
        if surrogate is not None:
            futures[len(futures)] = (key, design, pipe, tank)
        else:
            futures[pool.submit(simulate, design)] = (key, design, pipe, tank)
        notify(i, "simulating", None)

    if surrogate is not None and futures:
        jobs = list(futures.values())
        for (key, design, pipe, tank), sim in zip(jobs, surrogate.simulate_batch([j[1] for j in jobs], [j[2] for j in jobs])):
            _record(cache, key, design, pipe, tank, sim, targets, counters, cs.SURROGATE_MARGIN)
            for i in waiting[key]:
                notify(i, None, cache[key])
        futures = {}

    remaining = set(futures)
    while remaining:
        if should_stop and should_stop():
            for fut in remaining:
                fut.cancel()
            raise StopRequested()
        done, remaining = wait(remaining, timeout=0.5, return_when=FIRST_COMPLETED)
        for fut in done:
            key, design, pipe, tank = futures[fut]
            _record(cache, key, design, pipe, tank, fut.result(), targets, counters)
            for i in waiting[key]:
                notify(i, None, cache[key])

    for i, g in enumerate(population):
        evaluations[i] = dict(cache[keys[i]], genome=g)
    return evaluations


def _record(cache, key, design, pipe, tank, sim, targets, counters, margin=None):
    """시뮬레이션(또는 예측) 결과로 결과 기반 제약을 검사해 캐시에 기록. 예측이면 margin 으로 엄격하게"""
    counters["simulated"] += 1
    post = cs.post_sim_constraints(design, pipe, sim, *targets, margin=margin)
    ok, violation, failed = cs.check(post)
    cache[key] = {"design": design, "pipe": pipe, "tank": tank, "sim": sim,
                  "feasible": ok, "violation": violation, "failed": failed}


def _diverse(ranked, k, exclude=(), min_dist=VERIFY_MIN_DIST):
    """
    rank_key 순 후보에서 서로 다른 설계 k 개 (exclude 와도 달라야 함).
    수렴한 개체군의 상위 후보는 거의 같은 설계라 예측 오차도 같이 움직인다 → 같이 불합격하는 것을 막는다.
    거리 = 설계 변수를 탐색 범위로 나눈 값의 최대 차이 (파이프가 다르면 무조건 다름)
    """
    def far(a, b):
        if a["oxidizer_tank_radius"] != b["oxidizer_tank_radius"]:
            return True
        return max(abs(a[v] - b[v]) / (BOUNDS[v][1] - BOUNDS[v][0]) for v in DESIGN_VARS) >= min_dist
    chosen = []
    for ev in ranked:
        if all(far(ev["design"], d) for d in list(exclude) + [c["design"] for c in chosen]):
            chosen.append(ev)
            if len(chosen) == k:
                break
    return chosen


def robust_order(evs, targets, band):
    """
    합격 형상을 '평탄 구간(최고 Isp − band 이내) 안에서 제약 여유 큰 순' → 나머지는 rank_key 순으로 정렬.
    반환: (정렬된 리스트, 평탄 구간 형상 수)
    """
    feas = [e for e in evs if e["feasible"]]
    if not feas:
        return sorted(evs, key=rank_key), 0
    top = max(e["sim"]["isp"] for e in feas)
    plateau = [e for e in feas if e["sim"]["isp"] >= top - band]
    score = lambda e: cs.robustness(e["design"], e["pipe"], e["sim"], *targets)[0]
    plateau.sort(key=lambda e: (-score(e), -e["sim"]["isp"]))
    ids = {id(e) for e in plateau}
    return plateau + sorted((e for e in evs if id(e) not in ids), key=rank_key), len(plateau)


def close_pool(pool, grace=5.0):
    """
    워커를 확실히 종료한다. shutdown(wait=False) 만으로는 쉬고 있는 워커가 남아 있다가
    서버를 재시작하면 고아 프로세스(개당 ~200 MB)로 쌓였다. grace 초 안에 안 끝나면 강제 종료
    (중단 요청이면 grace=0 → 돌던 시뮬레이션도 바로 끊는다).
    """
    procs = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    deadline = time.time() + grace
    for p in procs:
        p.join(max(0.0, deadline - time.time()))
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(1.0)


def verify_with_core(candidates, targets, pool, should_stop=None, on_result=None):
    """
    대리모델로 고른 후보를 진짜 core 시뮬레이션으로 다시 평가.
    on_result(rank, ev_core) 를 결과가 나오는 대로 호출. 반환: candidates 와 같은 순서의 ev 리스트
    """
    futures = {pool.submit(simulate, ev["design"]): rank for rank, ev in enumerate(candidates)}
    results = [None] * len(candidates)
    remaining = set(futures)
    while remaining:
        if should_stop and should_stop():
            for fut in remaining:
                fut.cancel()
            raise StopRequested()
        done, remaining = wait(remaining, timeout=0.5, return_when=FIRST_COMPLETED)
        for fut in done:
            rank = futures[fut]
            ev = candidates[rank]
            sim = fut.result()
            ok, violation, failed = cs.check(cs.post_sim_constraints(ev["design"], ev["pipe"], sim, *targets))
            results[rank] = dict(ev, sim=sim, feasible=ok, violation=violation, failed=failed)
            if on_result:
                on_result(rank, results[rank])
    return results


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
           progress=None, should_stop=None, strategy=None, surrogate=None, verify_top=None,
           feasible_parents=None, robust_pick=None, robust_band=None, tolerance=None):
    """
    progress(event, data): 진행 상황 콜백 (GUI 용). event 종류
        start            {pipes, pop_size, n_generations, n_workers, targets}
        generation_start {generation}
        candidate        {generation, index, state, summary}
        generation_end   {generation, history, ranking, best}
        strategy         {generation, action, name}   (strategy 를 쓸 때, generation = 다음 세대)
        done             {stopped, best, output_dir, elapsed, counters}
    should_stop(): True 를 반환하면 현재까지의 최선 결과로 종료
    strategy(rng, evaluations, pipes, history, ctx) → (다음 개체군, {"action", "name"})
        세대마다 다음 개체군을 만드는 방법을 바깥에서 정한다 (예: rl.policy.PPOStrategy).
        None 이면 아래 고정 하이퍼파라미터의 기본 GA.
    surrogate: 주면 core 시뮬레이션 대신 대리모델로 평가 (rl.surrogate.SurrogateSimulator).
        끝난 뒤 상위 verify_top 개 후보를 core 시뮬레이션으로 검증하고, 최종 결과는 검증된 값으로 정한다.
        이벤트 verify_start {n, round}, verify {id, rank, surrogate, core} 가 추가로 온다.
        탐색 중에도 새로 예측 합격한 상위 후보를 백그라운드로 core 검증한다
        → verify_submit {id}, verified_live {id, surrogate, core, best_core}.
    모든 모드: 새로 제약을 만족한 설계마다 feasible {id, generation, source, summary}.
        검증에서 합격이 없으면 core 와 예측의 차이로 편향을 보정하고 REFINE_GENERATIONS 세대 더 탐색한 뒤
        다시 검증한다 (최대 CORRECTION_ROUNDS 회) → 이벤트 correction {round, bias, n_generations}.
    """
    target_impulse    = TARGET_IMPULSE if target_impulse is None else target_impulse
    target_thrust_avg = TARGET_THRUST_AVG if target_thrust_avg is None else target_thrust_avg
    pop_size          = pop_size or POP_SIZE
    n_generations     = n_generations or N_GENERATIONS
    n_workers         = n_workers or N_WORKERS
    seed              = RANDOM_SEED if seed is None else seed
    verify_top        = VERIFY_TOP if verify_top is None else verify_top
    feasible_parents  = FEASIBLE_PARENTS if feasible_parents is None else feasible_parents
    robust_pick       = ROBUST_PICK if robust_pick is None else robust_pick
    robust_band       = ROBUST_ISP_BAND if robust_band is None else robust_band
    tolerance         = cs.TARGET_TOLERANCE if tolerance is None else tolerance
    # (목표 임펄스, 목표 평균 추력, 허용 오차) — cs.post_sim_constraints / cs.robustness 에 *targets 로 그대로 넘긴다
    targets = (target_impulse, target_thrust_avg, tolerance)
    emit = progress or (lambda event, data: None)

    rng    = random.Random(seed)
    fixed  = load_fixed_values()
    pipes  = load_candidate_pipes()
    cache  = {}
    counters = {"pre_rejected": 0, "simulated": 0}
    history = []
    best, stall = None, 0
    stopped = False
    surrogate_best, verified = None, None    # 대리모델 모드에서만 사용
    robust_info = None                        # 견고성 선택 결과
    parent_state = {"active": False}          # 합격 형상끼리만 교배가 시작됐는지
    started = time.time()

    # RocketCEA 가 첫 실행 시 ~/RocketCEA 를 만드는데, 여러 프로세스가 동시에 만들면 충돌
    os.makedirs(os.path.join(os.path.expanduser("~"), "RocketCEA"), exist_ok=True)

    if verbose:
        print(f"[GA] 목표 임펄스 {target_impulse:.1f} N·s, 목표 평균 추력 {target_thrust_avg:.1f} N (허용 ±{tolerance * 100:g}%)")
        print(f"[GA] 파이프 {len(pipes)}종, 개체 {pop_size}, 최대 세대 {n_generations}, 워커 {n_workers}")
    emit("start", {"pipes": [p["pipe_id"] for p in pipes], "pop_size": pop_size,
                   "n_generations": n_generations, "n_workers": n_workers,
                   "targets": {"impulse": target_impulse, "thrust_avg": target_thrust_avg, "tolerance": tolerance}})

    population = initial_population(rng, pipes, fixed, pop_size)

    # spawn: 워커를 새 인터프리터로 띄운다. fork 는 부모(Flask 요청 스레드, torch 스레드풀 등)가
    # 잡고 있던 락을 자식이 물려받아 영원히 멈출 수 있다 → pool.shutdown 에서 MDO 가 끝나지 않던 원인
    pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn"))
    total_gens = n_generations       # 대리모델 편향 보정으로 세대가 늘어날 수 있다
    feasible_seen = set()            # 이미 알린 합격 설계 (feasible 이벤트)
    live = {"futures": {}, "submitted": set(), "checked": [], "best_core": None}   # 대리모델 모드 실시간 검증

    def report_feasible(evaluations, gen):
        for ev in evaluations:
            if ev["feasible"]:
                k = design_key(ev["design"], ev["pipe"])
                if k not in feasible_seen:
                    feasible_seen.add(k)
                    emit("feasible", {"id": design_id(ev["design"], ev["pipe"]), "generation": gen,
                                      "source": "surrogate" if surrogate is not None else "core",
                                      "summary": candidate_summary(ev)})

    def check_live(ev_pred, sim):
        """실시간 검증 결과 반영 → 현재까지 core 로 확인된 최고 설계 갱신"""
        ok, violation, failed = cs.check(cs.post_sim_constraints(ev_pred["design"], ev_pred["pipe"], sim, *targets))
        ev_core = dict(ev_pred, sim=sim, feasible=ok, violation=violation, failed=failed)
        live["checked"].append((ev_pred, ev_core))
        if live["best_core"] is None or rank_key(ev_core) < rank_key(live["best_core"]):
            live["best_core"] = ev_core
        if verbose:
            print(f"  [실시간 검증] 예측 Isp {ev_pred['sim']['isp']:.2f} → core {sim.get('isp') or 0:.2f} s, "
                  f"{'합격' if ok else '불합격 ' + ','.join(failed)}")
        emit("verified_live", {"id": design_id(ev_pred["design"], ev_pred["pipe"]),
                               "surrogate": candidate_summary(ev_pred), "core": candidate_summary(ev_core),
                               "best_core": candidate_summary(live["best_core"])})

    def pump_live(gen=None, wait_all=False):
        """끝난 실시간 검증을 거두고, 새 예측 합격 후보를 제출 (워커 수 이내)"""
        while True:
            done = [f for f in live["futures"] if f.done()]
            for fut in done:
                ev_pred = live["futures"].pop(fut)
                try:
                    check_live(ev_pred, fut.result())
                except Exception:          # 취소된 작업
                    pass
            if not wait_all or not live["futures"]:
                break
            if should_stop and should_stop():
                raise StopRequested()
            wait(list(live["futures"]), timeout=0.5, return_when=FIRST_COMPLETED)
        if gen is None or len(live["futures"]) >= min(LIVE_VERIFY_MAX, n_workers):
            return
        ev = best                                    # 지금까지의 예측 최고
        if gen < LIVE_VERIFY_WARMUP or stall < LIVE_VERIFY_STABLE or not ev["feasible"]:
            return
        k = design_key(ev["design"], ev["pipe"])
        if k in live["submitted"]:
            return
        live["submitted"].add(k)
        live["futures"][pool.submit(simulate, ev["design"])] = ev
        emit("verify_submit", {"id": design_id(ev["design"], ev["pipe"])})

    def run_generations(gen_from, gen_to):
        """gen_from ~ gen_to-1 세대 진행. 중단 요청이면 False"""
        nonlocal population, best, stall
        for gen in range(gen_from, gen_to):
            # 대리모델 평가는 evaluate() 안에서 기다리는 구간이 없으므로 여기서도 중단 요청을 확인한다
            if should_stop and should_stop():
                return False
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
                                       notify, should_stop, surrogate)
            except StopRequested:
                return False

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
            report_feasible(evaluations, gen)
            if surrogate is not None and verify_top > 0:
                pump_live(gen)
            emit("generation_end", {"generation": gen, "history": history[-1],
                                    "ranking": order, "elite": order[:ELITE_COUNT],
                                    "best": candidate_summary(best)})

            if EARLY_STOP_PATIENCE and best["feasible"] and stall >= EARLY_STOP_PATIENCE:
                if verbose:
                    print(f"[GA] {EARLY_STOP_PATIENCE} 세대 동안 개선이 없어 조기 종료")
                return True
            if gen == gen_to - 1:
                return True

            # 부모 후보: 옵션이 켜져 있고 합격 형상이 충분하면 합격 형상만 (evaluations 는 rank_key 순이라 합격이 앞쪽)
            parents = evaluations
            if feasible_parents and n_feasible >= FEASIBLE_PARENT_MIN:
                parents = evaluations[:n_feasible]
                if not parent_state["active"]:
                    parent_state["active"] = True
                    emit("feasible_parents", {"generation": gen, "n": n_feasible})
                    if verbose:
                        print(f"[GA] 합격 형상 {n_feasible}개 → 세대 {gen + 2}부터 합격 형상끼리만 교배")

            if strategy is not None:
                ctx = {"gen": gen, "n_generations": total_gens, "simulated": counters["simulated"],
                       "budget": pop_size * total_gens, "stall": stall, "patience": STALL_NORM,
                       "n_pipes": len(pipes), "targets": targets[:2], "pop_size": pop_size,   # PPO 상태는 (임펄스, 추력) 만
                       "targets_full": targets, "surrogate": surrogate is not None}           # 직접 설계 에이전트용
                # PPO 가 보는 상태는 전체 개체군 기준, 다음 세대는 parents 에서 만든다
                population, chosen = strategy(rng, evaluations, pipes, history, ctx, parents=parents)
                if verbose:
                    print(f"[GA] 세대 {gen + 2} 전략: {chosen['name']}")
                emit("strategy", {"generation": gen + 1, **chosen})
                continue

            next_pop = [ev["genome"] for ev in evaluations[:ELITE_COUNT]]
            while len(next_pop) < pop_size:
                c1, c2 = crossover(rng, tournament(rng, parents), tournament(rng, parents))
                next_pop.append(mutate(rng, c1, len(pipes)))
                if len(next_pop) < pop_size:
                    next_pop.append(mutate(rng, c2, len(pipes)))
            population = next_pop
        return True

    try:
        stopped = not run_generations(0, n_generations)

        # 대리모델 결과 검증: 예측 상위 후보(서로 다른 설계)를 core 로 다시 돌려 진짜 최적 후보를 고른다.
        # 합격이 없으면 core 와 예측의 평균 차이(편향)로 대리모델을 보정하고 몇 세대 더 탐색 → 다시 검증.
        if surrogate is not None and not stopped and verify_top > 0 and cache:
            surrogate_best = best
            pump_live(wait_all=True)                 # 탐색 중 제출한 실시간 검증을 마저 받는다
            verified = [ev_core for _, ev_core in live["checked"]]
            for cycle in range(CORRECTION_ROUNDS + 1):
                pool_evs = [ev for ev in cache.values() if ev["sim"] is not None]
                # 견고성 선택이면 평탄 구간에서 제약 여유가 큰 후보부터 core 로 검증한다
                ranked = robust_order(pool_evs, targets, robust_band)[0] if robust_pick else sorted(pool_evs, key=rank_key)
                candidates = _diverse(ranked, verify_top, exclude=[ev["design"] for ev in verified])
                if not candidates:
                    break
                offset = len(verified)
                if verbose:
                    print(f"[GA] 대리모델 상위 후보 {len(candidates)}개를 core 시뮬레이션으로 검증 ({cycle + 1}회차)")
                emit("verify_start", {"n": offset + len(candidates), "round": cycle + 1})

                def on_result(rank, ev_core, candidates=candidates, offset=offset):
                    if verbose:
                        sur, core = candidates[rank]["sim"], ev_core["sim"] or {}
                        print(f"  검증 {offset + rank + 1}: 예측 Isp {sur['isp']:.2f} → core {core.get('isp') or 0:.2f} s, "
                              f"{'합격' if ev_core['feasible'] else '불합격 ' + ','.join(ev_core['failed'])}")
                    emit("verify", {"id": design_id(ev_core["design"], ev_core["pipe"]), "rank": offset + rank,
                                    "surrogate": candidate_summary(candidates[rank]), "core": candidate_summary(ev_core)})
                checked = verify_with_core(candidates, targets, pool, should_stop, on_result)
                verified += checked
                if any(ev["feasible"] for ev in verified) or cycle == CORRECTION_ROUNDS:
                    break

                # 편향 보정 후 재탐색
                bias = surrogate.update_bias(candidates, checked)
                if verbose:
                    print("[GA] 대리모델 편향 보정: " + ", ".join(f"{k} {v:+.4g}" for k, v in bias.items()))
                cache.clear()                       # 예측이 바뀌었으므로 다시 평가
                best, stall = None, 0
                # 조기 종료했으면 마지막으로 진행한 세대 바로 다음부터 이어서 번호를 매긴다 (중간이 비지 않게)
                gen_from = history[-1]["generation"] + 1 if history else total_gens
                total_gens = gen_from + REFINE_GENERATIONS
                emit("correction", {"round": cycle + 1, "bias": bias, "n_generations": total_gens})
                if not run_generations(gen_from, total_gens):
                    stopped = True
                    break
            if verified:
                best = min(verified, key=rank_key)
            counters["verified"] = len(verified)

        # 견고성 선택: Isp 가 사실상 같은(최고 − band 이내) 합격 형상 중 제약 여유가 가장 큰 것
        # (대리모델 모드는 core 로 검증된 형상들, core 모드는 평가한 모든 형상에서 고른다)
        if robust_pick and best is not None and best["feasible"] and not stopped:
            pool_evs = verified if surrogate is not None else [ev for ev in cache.values() if ev["sim"] is not None]
            ordered, n_plateau = robust_order(pool_evs, targets, robust_band)
            pick = ordered[0]
            s_pick, m_pick = cs.robustness(pick["design"], pick["pipe"], pick["sim"], *targets)
            s_best, _ = cs.robustness(best["design"], best["pipe"], best["sim"], *targets)
            if verbose:
                print(f"[GA] 견고성 선택: 평탄 구간 {n_plateau}개 (Isp ≥ {best['sim']['isp'] - robust_band:.2f} s) 중 "
                      f"Isp {pick['sim']['isp']:.2f} s, 제약 여유 {s_pick:.2f} (최고 Isp 형상 {best['sim']['isp']:.2f} s, 여유 {s_best:.2f})")
            emit("robust_pick", {"n_plateau": n_plateau, "band": robust_band,
                                 "top": candidate_summary(best), "top_margin": s_best,
                                 "pick": candidate_summary(pick), "pick_margin": s_pick, "pick_margins": m_pick})
            robust_info = {"n_plateau": n_plateau, "band": robust_band, "top_isp": best["sim"]["isp"],
                           "top_margin": s_best, "pick_isp": pick["sim"]["isp"], "pick_margin": s_pick,
                           "pick_margins": m_pick}
            best = pick
    except StopRequested:
        stopped = True
    finally:
        close_pool(pool, grace=0 if stopped else 5)

    result = {
        "targets": {"impulse": target_impulse, "thrust_avg": target_thrust_avg, "tolerance": tolerance},
        "evaluator": "surrogate" if surrogate is not None else "core",
        "best": best,
        "robust": robust_info,
        "stopped": stopped,
        "history": history,
        "counters": counters,
        "elapsed": time.time() - started,
        "all_evaluations": list(cache.values()),
    }
    if surrogate is not None:
        result["surrogate_best"] = surrogate_best or best
        result["verified"] = verified
    result["output_dir"] = save_result(result) if best is not None else None
    emit("done", {"stopped": stopped, "best": candidate_summary(best) if best else None,
                  "verified": bool(result.get("verified")),
                  "output_dir": result["output_dir"], "elapsed": result["elapsed"],
                  "counters": counters})
    return result


# ──────────────────────────────────────────────────────────────────────────
# 저장 / 출력
# ──────────────────────────────────────────────────────────────────────────

def best_rocket_json(best):
    """GUI/simulation.py 에서 바로 쓸 수 있는 rocket.json 형식"""
    return {"TAS-I": dict(best["design"])}


def save_result(result):
    base = SURROGATE_RESULTS_DIR if result.get("evaluator") == "surrogate" else RESULTS_DIR
    out_dir = os.path.join(base, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "best_rocket.json"), "w", encoding="utf-8") as f:
        json.dump(best_rocket_json(result["best"]), f, indent=4, ensure_ascii=False)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        best = result["best"]
        json.dump({
            "targets": result["targets"],
            "evaluator": result.get("evaluator", "core"),
            "robust_pick": result.get("robust"),
            # 대리모델 실행을 중간에 멈추면 core 검증 없이 예측값이 best 로 저장된다 → 그대로 쓰지 말 것
            "best_verified": result.get("evaluator") != "surrogate" or bool(result.get("verified")),
            "verified": [candidate_summary(ev) for ev in result["verified"]] if result.get("verified") else None,
            "feasible": best["feasible"],
            "failed_constraints": best["failed"],
            "pipe": best["pipe"],
            "performance": {k: v for k, v in (best["sim"] or {}).items() if k != "series"} if best["sim"] else None,
            "counters": result["counters"],
            "elapsed": result["elapsed"],
            "history": result["history"],
        }, f, indent=4, ensure_ascii=False)

    sim_cols = ["isp", "impulse", "thrust_avg", "burn_time", "max_tank_pressure",
                "max_chamber_pressure", "dp_ratio_min", "dp_ratio_avg", "thrust_avg_ss", "isp_ss",
                "termination_code", "error"]
    with open(os.path.join(out_dir, "candidates.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pipe_id"] + DESIGN_VARS + DERIVED_VARS + RECORD_FIXED + PIPE_VARS + ["feasible", "violation", "failed"] + sim_cols)
        for ev in sorted(result["all_evaluations"], key=rank_key):
            sim = ev["sim"] or {}
            writer.writerow(
                [ev["pipe"]["pipe_id"]]
                + [ev["design"][v] for v in DESIGN_VARS + DERIVED_VARS + RECORD_FIXED + PIPE_VARS]
                + [ev["feasible"], round(ev["violation"], 6), ";".join(ev["failed"])]
                + [sim.get(c) for c in sim_cols]
            )
    return out_dir


def print_generation(gen, evaluations, best, n_feasible, dt_gen):
    top = evaluations[0]
    if top["feasible"]:
        state = f"세대 최고 Isp {top['sim']['isp']:.2f} s"
    else:
        state = f"세대 최소 위반 {top['violation']:.4f} ({', '.join(top['failed'][:3])})"
    best_txt = f"{best['sim']['isp']:.2f} s" if best["feasible"] else "합격 해 없음"
    print(f"[GEN {gen:3d}] 합격 {n_feasible:3d}/{len(evaluations)} | {state} | "
          f"누적 최고 {best_txt} | {dt_gen:.1f}s")


def print_summary(result):
    best = result["best"]
    d, pipe, sim, tank = best["design"], best["pipe"], best["sim"], best["tank"]
    t_imp, t_thr = result["targets"]["impulse"], result["targets"]["thrust_avg"]
    mm = lambda x: f"{x * 1e3:9.2f} mm"
    line = "═" * 64

    print()
    print(line)
    print(" GA 최적화 결과 요약")
    print(line)
    status = "모든 제약 만족" if best["feasible"] else f"제약 미충족: {', '.join(best['failed'])}"
    print(f" 상태            : {status}")
    print(f" 소요 시간       : {result['elapsed']:.1f} s  "
          f"(시뮬레이션 {result['counters']['simulated']}회, "
          f"형상 제약 탈락 {result['counters']['pre_rejected']}회)")

    if sim and not sim.get("error"):
        print("─" * 64)
        print(" [성능]                     목표            결과")
        print(f"  총 임펄스       {t_imp:12.1f} N·s  {sim['impulse']:12.1f} N·s "
              f"({(sim['impulse'] / t_imp - 1) * 100:+.1f}%)")
        print(f"  평균 추력       {t_thr:12.1f} N    {sim['thrust_avg']:12.1f} N   "
              f"({(sim['thrust_avg'] / t_thr - 1) * 100:+.1f}%)")
        print(f"  비추력 Isp      {'':12s}       {sim['isp']:12.2f} s   (시간평균 {sim['isp_mean']:.2f} s)")
        if "thrust_max" in sim:      # core 결과 (대리모델 예측에는 시계열 기반 값이 없다)
            print(f"  최대 추력       {sim['thrust_max']:12.1f} N")
            if sim.get("thrust_avg_ss") is not None:
                print(f"  정상상태        {sim['ss_start']:5.2f}–{sim['ss_end']:5.2f} s   평균 추력 {sim['thrust_avg_ss']:.1f} N   "
                      f"평균 Isp {sim['isp_ss']:.2f} s")
            print(f"  연소 시간       {sim['burn_time']:12.2f} s    (시뮬 종료 {sim['sim_time']:.2f} s, "
                  f"{sim['termination_code']})")
            print(f"  소모 산화제     {sim['m_oxidizer_used']:12.3f} kg   소모 연료 {sim['m_fuel_used']:.3f} kg  "
                  f"평균 O/F {sim['of_avg']:.2f}")
        print(f"  최대 탱크 압력  {sim['max_tank_pressure'] / 1e6:12.3f} MPa  "
              f"최대 챔버 압력 {sim['max_chamber_pressure'] / 1e6:.3f} MPa")
        dp = cs.injector_dp_ratio(sim)
        if dp is not None:
            print(f"  인젝터 차압     {dp * 100:12.1f} %    (연소 중 최소 (Pt-Pc)/Pc, 하한 "
                  f"{cs.MIN_INJECTOR_DP_RATIO * 100:.0f}%)")
    elif sim:
        print(f"  시뮬레이션 오류: {sim['error']}")

    print("─" * 64)
    print(f" [파이프] {pipe['pipe_id']}  ({pipe['name']})")
    print(f"  구매 OD {pipe['outer_diameter'] * 1e3:.2f} mm / ID {pipe['inner_diameter'] * 1e3:.2f} mm / "
          f"t {pipe['wall_thickness'] * 1e3:.2f} mm / L {pipe['length'] * 1e3:.1f} mm")
    t = d["oxidizer_tank_tickness"]
    print(f"  가공 후 OD {(pipe['inner_diameter'] + 2 * t) * 1e3:.2f} mm  "
          f"(두께 {t * 1e3:.2f} mm, 외경 절삭량 {(pipe['wall_thickness'] - t) * 1e3:.2f} mm/side)")

    print("─" * 64)
    print(" [형상]")
    print(f"  산화제 질량                 {d['oxidizer_M']:9.2f} kg")
    print(f"  헬륨 슈퍼차징               {d['supercharge_pressure'] / 1e6:9.2f} MPa  (헬륨 {d['pressurant_M'] * 1e3:.2f} g)")
    if tank and tank.get("valid_T"):
        print(f"  탱크 초기 ullage            {tank['ullage'] * 100:9.1f} %   "
              f"초기 탱크압 {tank['P_tank'] / 1e6:.3f} MPa")
    print(f"  산화제 탱크 반지름 / 길이   {mm(d['oxidizer_tank_radius'])} / {mm(d['oxidizer_tank_length'])}")
    print(f"  연소 챔버 반지름 / 길이     {mm(d['combution_chamber_radius'])} / {mm(d['combution_chamber_length'])}"
          f"  (연료+2R ×{cs.CHAMBER_LENGTH_FACTOR})")
    print(f"  탱크/챔버 두께              {mm(t)}")
    print(f"  연료 외부 반지름 / 길이     {mm(d['outter_radius'])} / {mm(d['fuel_length'])}")
    port_txt = "단일 포트 (중앙)" if d["port_num"] == 1 else f"다공 {d['port_num']}개, 중심 배치 반지름 {d['port_center_radius'] * 1e3:.2f} mm"
    print(f"  포트                        {port_txt}")
    print(f"  포트 반지름                 {mm(d['inner_radius'])}")
    print(f"  인젝터 오리피스             {d['injector_oriffice_n']}개, 반지름 {d['injector_oriffice_radius'] * 1e3:.2f} mm, "
          f"길이 {d['injector_oriffice_length'] * 1e3:.2f} mm")
    eps = (d["nozzle_exit_radius"] / d["nozzle_throat_radius"]) ** 2
    print(f"  노즐 목 / 출구 반지름       {mm(d['nozzle_throat_radius'])} / {mm(d['nozzle_exit_radius'])}  (eps {eps:.2f})")
    print("─" * 64)
    print(f" 결과 저장: {result['output_dir']}")
    print("   best_rocket.json (rocket.json 형식), summary.json, candidates.csv")
    print(line)

def main(argv=None):
    parser = argparse.ArgumentParser(description="하이브리드 엔진 GA 최적화 (SI 단위)")
    parser.add_argument("--impulse", type=float, help=f"목표 총 임펄스 [N·s] (기본 {TARGET_IMPULSE})")
    parser.add_argument("--thrust", type=float, help=f"목표 평균 추력 [N] (기본 {TARGET_THRUST_AVG})")
    parser.add_argument("--surrogate", action="store_true", help="core 시뮬레이션 대신 대리모델(rl/surrogate.pt)로 평가")
    parser.add_argument("--gens", type=int, default=None, help="최대 세대 수")
    parser.add_argument("--tol", type=float, default=None, help="목표 임펄스·평균 추력 허용 오차 [%%] (기본 constrain.TARGET_TOLERANCE)")
    args = parser.parse_args(argv)

    surrogate = None
    if args.surrogate:
        from rl.surrogate import SurrogateSimulator
        surrogate = SurrogateSimulator()
    result = run_ga(args.impulse, args.thrust, n_generations=args.gens, surrogate=surrogate,
                    tolerance=args.tol / 100 if args.tol is not None else None)
    print_summary(result)
    return 0 if result["best"]["feasible"] else 1


if __name__ == "__main__":
    sys.exit(main())
