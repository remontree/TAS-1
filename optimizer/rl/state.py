"""
개체군과 기록 → 35개 숫자 벡터 (AI 가 보는 상태)
(제약이 추가되면 길이가 바뀌어 예전에 학습한 PPO 모델은 쓸 수 없다 → rl/policy.py 가 걸러냄)
모든 값은 0~1 사이로 맞춘다.
"""

import math
from collections import Counter

import numpy as np

import GA

CONSTRAINT_NAMES = [
    "ChamberThickness", "ChamberEquality", "ChamberLength", "OxtFill", "InjectorArea", "MaxOri",
    "FuelArea", "FuelPort", "PortWeb", "PortResolution", "FuelLenght", "FuelRadius",
    "NozzleThroatSize", "NozzleExpansion", "NozzleExitSize", "Toxt", "OxtVolume",
    "SimCompleted", "Tcc", "Impulse", "ThrustAvg", "InjectorDP",
]
STATE_DIM = 3 + 6 + 2 + len(CONSTRAINT_NAMES) + 2


def _clip01(x):
    return float(min(max(x, 0.0), 1.0))


def extract_state(evaluations, history, ctx):
    """
    evaluations : 이번 세대 ev 리스트
    history     : 세대별 기록 dict 리스트 (마지막이 이번 세대)
    ctx         : gen, n_generations, simulated, budget, stall, patience, n_pipes, targets
    """
    n = len(evaluations)
    now = history[-1]

    # 진행도 3
    progress = [
        ctx["gen"] / max(1, ctx["n_generations"] - 1),
        _clip01(ctx["simulated"] / ctx["budget"]),
        _clip01(ctx["stall"] / max(1, ctx["patience"])),
    ]

    # 품질 6
    n_feasible = sum(ev["feasible"] for ev in evaluations)
    n_rejected = sum(ev["sim"] is None for ev in evaluations)
    has_feasible = now["best_isp"] is not None
    isp_now = now["best_isp"] or 0.0
    past = history[-4] if len(history) >= 4 else history[0]
    improvement = isp_now - past["best_isp"] if has_feasible and past["best_isp"] is not None else 0.0
    quality = [
        n_feasible / n,
        n_rejected / n,
        float(has_feasible),
        _clip01((isp_now - 150.0) / 100.0) if has_feasible else 0.0,
        _clip01(improvement / 5.0),
        0.0 if has_feasible else _clip01(math.log1p(now["best_violation"]) / 3.0),
    ]

    # 다양성 2
    spreads = []
    for var in GA.DESIGN_VARS:
        lo, hi = GA.BOUNDS[var]
        values = [ev["genome"][var] for ev in evaluations]
        spreads.append(float(np.std(values)) / (hi - lo))
    diversity = [
        _clip01(sum(spreads) / len(spreads) * 2),
        len({ev["genome"]["pipe"] for ev in evaluations}) / ctx["n_pipes"],
    ]

    # 제약 실패 빈도 21
    counts = Counter(name for ev in evaluations for name in ev["failed"])
    failures = [counts[name] / n for name in CONSTRAINT_NAMES]

    # 목표 2
    impulse, thrust = ctx["targets"]
    targets = [_clip01(impulse / 20000.0), _clip01(thrust / 2000.0)]

    return np.array(progress + quality + diversity + failures + targets, dtype=np.float32)
