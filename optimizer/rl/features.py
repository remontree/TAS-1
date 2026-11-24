"""
대리모델 입력 특징 — 설계 변수 + 파생 값 (챔버 길이, 헬륨 질량) + 파이프
예전 데이터 (설계 변수에 슈퍼차징 압력이 없고 헬륨 질량 0.025 kg 고정, 챔버 길이 자유) 도 같은 특징으로 바꿔 함께 쓴다:
    supercharge_pressure = 그 설계의 초기 탱크압 (N2O 포화압 + 헬륨 0.025 kg 분압), pressurant_M = 0.025
초기 온도 (temperature) 도 입력 — 범용 온도 대리모델 (TEMP_RANGE). 온도 기록이 없는 데이터는 모두 258 K (−15 °C) 에서 만든 것
"""

import numpy as np

import GA

# core 시뮬레이션에 들어가는 파이프 값은 내경뿐 (두께는 설계 변수, 파이프 길이는 제약에만 쓰임).
# 길이·구매 두께를 입력에 넣으면 파이프 DB 가 바뀐 예전/새 데이터가 갈라져 엉뚱한 상관을 배운다.
PIPE_FEATURES = ["inner_diameter"]
DATA_PIPE_FEATURES = ["inner_diameter", "wall_thickness", "length"]     # rl/ts_data shard 에 저장된 파이프 열
FEATURE_VARS = GA.DESIGN_VARS + GA.DERIVED_VARS + ["temperature"]
PREV_FEATURE_VARS = GA.DESIGN_VARS + GA.DERIVED_VARS          # 온도 입력 전 형식 (2026-10-20 데이터)
LEGACY_TEMPERATURE = 258.0                                     # 온도 기록이 없는 데이터·결과의 초기 온도 [K]
TEMP_RANGE = (258.0, 308.0)                                    # 대리모델 학습 온도 범위 [K] (−15 ~ 35 °C)
N_FEATURES = len(FEATURE_VARS) + len(PIPE_FEATURES)
LEGACY_P_MAX = 15e6        # 예전 데이터의 환산 탱크압 상한 [Pa]


def design_features(design, pipe):
    return [design[v] for v in FEATURE_VARS] + [pipe[f] for f in PIPE_FEATURES]


def legacy_design(row, fixed, inner_diameter):
    """예전 형식의 설계 값 dict (combution_chamber_length 포함, supercharge_pressure 없음) → 새 형식 design"""
    d = dict(fixed, **row)
    d["temperature"] = row.get("temperature", LEGACY_TEMPERATURE)        # 기록이 없으면 그 데이터를 만든 온도
    d.setdefault("pressurant_M", GA.LEGACY_PRESSURANT_M)
    d["oxidizer_tank_radius"] = d["combution_chamber_radius"] = inner_diameter / 2
    if "supercharge_pressure" not in row:
        # 과충진 탱크는 기체 공간이 0 에 가까워 계산 압력이 발산 → 특징 정규화가 깨지지 않게 자른다 (core 도 실패하는 설계)
        d["supercharge_pressure"] = min(max(GA.legacy_supercharge_pressure(d, d["pressurant_M"]), 0.0), LEGACY_P_MAX)
    return d


def convert_X(X, var_names, fixed=None, pipe_names=None, temperature=LEGACY_TEMPERATURE):
    """
    저장된 입력 행렬 X (열 = var_names + pipe_names) → 현재 특징 (열 = FEATURE_VARS + PIPE_FEATURES)
    예전 형식 (슈퍼차징 압력 없음) 이면 rl.features.legacy_design 으로 변환.
    temperature: 온도 열이 없는 데이터를 만든 초기 온도
    """
    var_names = list(var_names)
    pipe_names = list(pipe_names or DATA_PIPE_FEATURES)
    nv = len(var_names)
    X = np.asarray(X, dtype=float)
    pcols = [nv + pipe_names.index(f) for f in PIPE_FEATURES]
    if var_names == FEATURE_VARS:
        return np.concatenate([X[:, :nv], X[:, pcols]], 1).astype(np.float32)
    if var_names == PREV_FEATURE_VARS:
        return np.concatenate([X[:, :nv], np.full((len(X), 1), temperature), X[:, pcols]], 1).astype(np.float32)
    fixed = dict(fixed or GA.load_fixed_values(), temperature=temperature)
    out = np.zeros((len(X), N_FEATURES), dtype=np.float32)
    for i, x in enumerate(X):
        row = {v: float(x[k]) for k, v in enumerate(var_names)}
        row.setdefault("temperature", temperature)
        pipe = {f: float(x[nv + k]) for k, f in enumerate(pipe_names)}
        d = legacy_design(row, fixed, pipe["inner_diameter"])
        out[i] = design_features(d, pipe)
    return out


def legacy_genome(row, pipe_index, fixed, pipe):
    """예전 candidates.csv 행 → 현재 genome (슈퍼차징 압력은 범위 안으로 잘림, 양자화)"""
    d = legacy_design({v: row[v] for v in GA.DESIGN_VARS + ["combution_chamber_length", "temperature"]
                       if v in row and row[v] == row[v]}, fixed, pipe["inner_diameter"])
    g = {"pipe": pipe_index, **{v: d[v] for v in GA.DESIGN_VARS}}
    g["port_num"], g["injector_oriffice_n"] = int(g["port_num"]), int(g["injector_oriffice_n"])
    return GA.quantize(g)


def run_compatible(run_dir):
    """학습 실행이 현재 설계 변수 구성 · 초기 온도 (fixedvalue) 로 학습됐는지 (config.json). 기록이 없으면 예전 구성 → False
    RL 은 한 온도에서만 학습하므로 온도가 다르면 쓰지 않는다 (온도 기록이 없는 실행은 258 K)"""
    import json
    import os
    path = os.path.join(run_dir, "config.json")
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return False
    names = cfg.get("design_vars") or cfg.get("vars") or []
    same_T = abs(float(cfg.get("temperature", LEGACY_TEMPERATURE)) - float(GA.load_fixed_values()["temperature"])) < 0.5
    return list(names[:len(GA.DESIGN_VARS)]) == GA.DESIGN_VARS and same_T
