"""
대리모델 학습용 데이터셋 생성
────────────────────────────────────────────────────────────────────────────
GA 실행 기록(optimizer/results/*/candidates.csv)의 설계를 기준점으로 삼아
설계 변수와 운전 조건을 무작위로 흔든 입력을 만들고, core 시뮬레이션을 병렬로 돌려
시계열 대신 스칼라 성능 지표(평균/최대 추력, 평균/최대 압력, 비추력 등)를 저장한다.

실행 (TAS-1 폴더에서)
    .venv/bin/python surrogate/build_dataset.py --n 6000
"""
import argparse, contextlib, glob, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT_CSV = os.path.join(ROOT, "surrogate", "dataset.csv")
G0 = 9.80665
BURN_THRESHOLD = 0.05

# 대리모델 입력 변수와 샘플링 범위
DESIGN_VARS = {
    "oxidizer_M":               (1.0, 10.0),
    "oxidizer_tank_radius":     (0.04, 0.065),
    "oxidizer_tank_length":     (0.5, 2.0),
    "oxidizer_tank_tickness":   (0.001, 0.01),
    "combution_chamber_radius": (0.04, 0.065),
    "combution_chamber_length": (0.3, 1.5),
    "outter_radius":            (0.02, 0.065),
    "inner_radius":             (0.003, 0.03),
    "port_num":                 (1, 8),
    "port_center_radius":       (0.0, 0.05),
    "fuel_length":              (0.1, 1.0),
    "injector_oriffice_n":      (4, 64),
    "injector_oriffice_radius": (0.0003, 0.0015),
    "injector_oriffice_length": (0.003, 0.03),
    "nozzle_throat_radius":     (0.004, 0.02),
    "nozzle_exit_radius":       (0.008, 0.05),
}
CONDITION_VARS = {
    "temperature":   (250.0, 295.0),
    "pressurant_M":  (0.0, 0.05),
    "injector_cd":   (0.45, 0.8),
    "cstar_eta":     (0.80, 0.97),
    "thrust_coeff":  (0.80, 0.98),
}
INT_VARS = {"port_num", "injector_oriffice_n"}


def load_fixed():
    with open(os.path.join(ROOT, "optimizer", "fixedvalue.json"), encoding="utf-8") as f:
        return json.load(f)


def base_designs():
    pipes = pd.read_csv(os.path.join(ROOT, "optimizer", "PIPEDB.csv")).set_index("pipe_id")
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, "optimizer", "results", "*", "candidates.csv"))])
    d = d[d.isp.notna()].copy()
    d["oxidizer_tank_radius"] = d.pipe_id.map(lambda p: pipes.loc[p, "inner_diameter"] / 2)
    d["combution_chamber_radius"] = d["oxidizer_tank_radius"]
    return d[list(DESIGN_VARS)].to_dict("records")


def sample(rng, bases, fixed):
    b = bases[rng.integers(len(bases))]
    cfg = dict(fixed)
    scale = rng.choice([0.03, 0.08, 0.15])
    for v, (lo, hi) in DESIGN_VARS.items():
        x = b[v] + rng.normal(0, scale * (hi - lo))
        x = min(max(x, lo), hi)
        cfg[v] = int(round(x)) if v in INT_VARS else float(x)
    # 같은 파이프 가정이 기본이지만 일부는 챔버/탱크 반지름을 따로 둔다
    if rng.random() < 0.6:
        cfg["combution_chamber_radius"] = cfg["oxidizer_tank_radius"]
    cfg["outter_radius"] = min(cfg["outter_radius"], cfg["combution_chamber_radius"] * 0.98)
    if cfg["port_num"] == 1:
        cfg["port_center_radius"] = 0.0
    cfg["nozzle_exit_radius"] = max(cfg["nozzle_exit_radius"], cfg["nozzle_throat_radius"] * 1.05)
    for v, (lo, hi) in CONDITION_VARS.items():
        # 절반은 기본 운전 조건 그대로, 절반은 범위 안에서 무작위
        cfg[v] = float(rng.uniform(lo, hi)) if rng.random() < 0.5 else float(min(max(fixed[v], lo), hi))
    return cfg


def simulate(cfg):
    started = time.time()
    row = {k: cfg[k] for k in list(DESIGN_VARS) + list(CONDITION_VARS)}
    try:
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            from core.rocket import Rocket
            r = Rocket(cfg)
            r.engine.simulation(cfg["dt"], cfg["simt"], cfg["thrust_coeff"])
        e, dt = r.engine, cfg["dt"]
        F = np.asarray(e.thrust[:-1], float)
        Pc = np.asarray(e.chamber_pressure[:-1], float)
        Pt = np.asarray(e.tank_pressure[:-1], float)
        mox = np.asarray(e.mox_dot[:-1], float)
        mf = np.asarray(e.mf_dot[:-1], float)
        isp_t = np.asarray(e.Isp[:-1], float)
        of = np.asarray(e.of[:-1], float)
        cstar = np.asarray(e.cstar[:-1], float)
        fmax = float(F.max()) if F.size else 0.0
        burn = F >= BURN_THRESHOLD * fmax if fmax > 0 else np.zeros_like(F, bool)
        tb = float(burn.sum() * dt)
        m_ox, m_f = float(mox.sum() * dt), float(mf.sum() * dt)
        I = float(e.impulse)
        row.update({
            "ok": 1, "termination_code": e.termination_code, "error": "",
            "impulse": I,
            "isp": I / ((m_ox + m_f) * G0) if m_ox + m_f > 0 else 0.0,
            "isp_mean": float(isp_t[burn].mean()) if burn.any() else 0.0,
            "thrust_avg": float(F[burn].sum() * dt / tb) if tb > 0 else 0.0,
            "thrust_max": fmax,
            "burn_time": tb,
            "chamber_pressure_avg": float(Pc[burn].mean()) if burn.any() else 0.0,
            "chamber_pressure_max": float(Pc.max()),
            "tank_pressure_avg": float(Pt[burn].mean()) if burn.any() else 0.0,
            "tank_pressure_max": float(Pt.max()),
            "of_avg": m_ox / m_f if m_f > 0 else 0.0,
            "cstar_avg": float(cstar[burn].mean()) if burn.any() else 0.0,
            "m_oxidizer_used": m_ox,
            "m_fuel_used": m_f,
            "mox_dot_avg": float(mox[burn].mean()) if burn.any() else 0.0,
        })
    except Exception as exc:
        row.update({"ok": 0, "termination_code": "", "error": f"{type(exc).__name__}: {exc}"[:120]})
    row["wall_time"] = time.time() - started
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    fixed, bases = load_fixed(), base_designs()
    cfgs = [sample(rng, bases, fixed) for _ in range(args.n)]
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(simulate, c) for c in cfgs]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 200 == 0 or i == len(futs):
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                print(f"{i}/{len(futs)}  {time.time() - t0:.0f}s  ok={np.mean([r['ok'] for r in rows]):.3f}", flush=True)


if __name__ == "__main__":
    main()
