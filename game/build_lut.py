"""
TAS-1 Launch League — 엔진 결과표(LUT) 생성
게임 노브 6개 (엔진 5개 × 9단계 + 연료 포트 수 1~8 8단계 = 472,392개 설계) 를 TAS-1 시계열 대리모델(rl/ts_surrogate.pt)로 한 번에 예측하고
constrain.py 의 제약으로 판정해 game/data/engine_lut.jsonl.gz 로 저장한다.
형식: 첫 줄 meta JSON, 이후 "키<TAB>행 JSON" 한 줄씩 (서버·node 가 행 단위로 읽어 메모리를 아낀다).
키 = 노브 인덱스 6자리 (ox, p_MPa, n, fuel_m, eps, ports)

실행 (TAS-1 .venv, torch 필요):
    cd "<TAS-1>/game" && ../.venv/bin/python build_lut.py
"""

import gzip
import itertools
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.join(os.path.dirname(HERE), "optimizer")
sys.path.insert(0, OPT)
os.chdir(OPT)  # GA/constrain 이 상대 경로로 fixedvalue.json·PIPEDB.csv 를 읽는다

import GA  # noqa: E402
import constrain as cs  # noqa: E402
from rl import features as ft  # noqa: E402
from rl import ts_surrogate as ts  # noqa: E402

CFG = json.load(open(os.path.join(HERE, "game_config.json"), encoding="utf-8"))
OUT = os.path.join(HERE, "data", "engine_lut.jsonl.gz")
CURVE_DT = 0.2  # 저장 곡선 간격 [s]


def base_design():
    with open(os.path.join(OPT, CFG["base_design"]), encoding="utf-8") as f:
        d = next(iter(json.load(f).values()))
    d["temperature"] = CFG["temperature_K"]
    d["oxidizer_tank_tickness"] = CFG.get("wall_t", d["oxidizer_tank_tickness"])
    return d


def pick_pipe(inner_d):
    pipes = GA.load_candidate_pipes()
    return min(pipes, key=lambda p: abs(p["inner_diameter"] - inner_d))


def port_layout(d, n):
    """포트 n 개 배치: 포트 반지름은 기본 설계 그대로, 이웃 포트 사이 웹이 MIN_PORT_WEB × 1.25 이상 되도록 중심 반지름을 넓힌다"""
    ri = d["inner_radius"]
    if n == 1:
        return 0.0   # 단일 포트는 중앙 (GA.derive 와 같음)
    need = (cs.MIN_PORT_WEB * 1.25 + 2 * ri) / (2 * math.sin(math.pi / n))
    rc = round(max(d["port_center_radius"], need), 5)
    assert rc + ri < d["outter_radius"], f"port {n}: {rc} + {ri} >= {d['outter_radius']}"
    return rc


def tank_length_for(d, ox, ullage):
    """초기 기체 공간 비율이 ullage 가 되는 탱크 길이 (이분법, GA 길이 해상도로 올림)"""
    r = d["oxidizer_tank_radius"]
    lo, hi = 0.05, 4.0
    for _ in range(50):
        mid = (lo + hi) / 2
        st = cs.oxidizer_tank_state(d["oxidizer"], d["oxidizer_MW"], ox, d["pressurant"], d["pressurant_MW"], 0.0,
                                    d["temperature"], math.pi * r * r * mid)
        if st["valid_T"] and st["N_liquid"] > 0 and st["ullage"] >= ullage:
            hi = mid
        else:
            lo = mid
    return math.ceil(hi / GA.LENGTH_RESOLUTION) * GA.LENGTH_RESOLUTION


def main():
    t_start = time.time()
    K = CFG["knobs"]
    base = base_design()
    pipe = pick_pipe(base["oxidizer_tank_radius"] * 2)
    base["oxidizer_tank_radius"] = base["combution_chamber_radius"] = pipe["inner_diameter"] / 2
    throat = base["nozzle_throat_radius"]
    ullage = CFG["tank_ullage"]
    print(f"pipe {pipe['pipe_id']}  ID {pipe['inner_diameter'] * 1000:.0f} mm  T {base['temperature']} K")

    lo_t, hi_t = CFG.get("tank_len_range", [0.3, 2.0])  # 서로게이트 학습 범위 안으로 (짧으면 기체 공간이 늘어날 뿐)
    tank_len = {i: min(max(tank_length_for(base, ox, ullage), lo_t), hi_t) for i, ox in enumerate(K["ox"])}
    NL = [len(K[k]) for k in ("ox", "p_MPa", "n", "fuel_m", "eps", "ports")]
    port_c = {i: port_layout(base, n) for i, n in enumerate(K["ports"])}
    print("port center radius [mm]", {K["ports"][i]: round(v * 1000, 2) for i, v in port_c.items()})
    combos = list(itertools.product(*[range(n) for n in NL]))
    designs, pre_fail, keys = [], [], []
    helium = {}
    for c in combos:
        i_ox, i_p, i_n, i_l, i_e, i_g = c
        d = dict(base)
        d["oxidizer_M"] = K["ox"][i_ox]
        d["oxidizer_tank_length"] = tank_len[i_ox]
        d["supercharge_pressure"] = K["p_MPa"][i_p] * 1e6
        d["injector_oriffice_n"] = int(K["n"][i_n])
        d["fuel_length"] = K["fuel_m"][i_l]
        d["port_num"] = int(K["ports"][i_g])
        d["port_center_radius"] = port_c[i_g]
        d["nozzle_exit_radius"] = round(throat * math.sqrt(K["eps"][i_e]), 5)
        d["combution_chamber_length"] = GA.trimmed_chamber_length(d)
        hk = (i_ox, i_p)
        if hk not in helium:
            helium[hk] = GA.helium_mass(d)
        d["pressurant_M"] = helium[hk]
        cons, _ = cs.pre_sim_constraints(d, pipe)
        _, _, failed = cs.check(cons)
        designs.append(d)
        pre_fail.append(failed)
        keys.append("".join(map(str, c)))
    print(f"designs {len(designs)}  ({time.time() - t_start:.1f}s)")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ts.TSSurrogate.load(ts.MODEL_PATH, device)
    simt = ts.surrogate_simt(40.0)
    X = np.array([ft.design_features(d, pipe) for d in designs], dtype=np.float32)
    dt = ts.DT
    sigma = pipe["yield_strength"]
    stats = {"ok": 0, "n": 0}
    t_inf = 0.0
    CHUNK = 8192   # 41만 설계를 한 번에 예측하면 시계열만 4 GB → 나눠서 예측하고 바로 기록
    tmp = OUT + ".rows.tmp"
    with open(tmp, "w", encoding="utf-8") as fo:
        for s0 in range(0, len(X), CHUNK):
            t1 = time.time()
            outs = [model.predict(X[s:s + 2048]) for s in range(s0, min(s0 + CHUNK, len(X)), 2048)]
            if device == "cuda":
                torch.cuda.synchronize()
            t_inf += time.time() - t1
            out = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
            sims = ts.sims_from_output(out, simt, model.has_t_end, keep_series=False)
            series = out["series"]  # (N, C, 400) thrust, Pc, Pt, mox_dot, mf_dot, dp
            for j, sim in enumerate(sims):
                i = s0 + j
                d, k = designs[i], keys[i]
                th = series[j, 0]
                mox, mf = series[j, 3], series[j, 4]
                on = np.nonzero(th >= GA.BURN_THRESHOLD * max(th.max(), 1e-9))[0]
                t_end = (on[-1] + 1) * dt if len(on) else 0.0
                n_keep = min(len(th), int(math.ceil((t_end + 0.4) / dt)) + 1)
                step = int(round(CURVE_DT / dt))
                idx = np.arange(0, n_keep, step)
                codes = []
                if "OxtPressure" in pre_fail[i] or "OxtFill" in pre_fail[i]:
                    codes.append("ERR_TANK_OVERPRESSURE")
                tank_ok = cs.Toxt(sigma, d["oxidizer_tank_radius"], sim["max_tank_pressure"], cs.SAFETY_FACTOR,
                                  d["oxidizer_tank_tickness"]).validate()
                cc_ok = cs.Tcc(sigma, d["combution_chamber_radius"], sim["max_chamber_pressure"], cs.SAFETY_FACTOR,
                               d["oxidizer_tank_tickness"]).validate()
                if not tank_ok and "ERR_TANK_OVERPRESSURE" not in codes:
                    codes.append("ERR_TANK_OVERPRESSURE")
                if not cc_ok:
                    codes.append("ERR_CHAMBER_OVERPRESSURE")
                if cs.injector_dp_ratio(sim) < cs.MIN_INJECTOR_DP_RATIO:
                    codes.append("ERR_INJECTOR_DP")
                if sim["termination_code"] == "TIME_LIMIT_REACHED":
                    codes.append("ERR_TIMEOUT")
                if not codes:
                    stats["ok"] += 1
                isp = sim["isp"]
                row = {
                    "isp": round(isp, 1),
                    "isp_std": round(sim["isp_std"], 2),
                    "impulse": round(sim["impulse"], 0),
                    "thrust_avg": round(sim["thrust_avg"], 0),
                    "burn_time": round(sim["burn_time"], 2),
                    "t_end": round(t_end, 2),
                    "max_tank_MPa": round(sim["max_tank_pressure"] / 1e6, 2),
                    "max_chamber_MPa": round(sim["max_chamber_pressure"] / 1e6, 2),
                    "dp_ratio_min": round(sim["dp_ratio_min"], 3),
                    "p_ok": round(sim["p_completed"], 3),
                    "term": sim["termination_code"],
                    "codes": codes,
                    "ox_used": round(float(mox[:n_keep].sum() * dt), 3),
                    "fuel_used": round(float(mf[:n_keep].sum() * dt), 3),
                    "geom": {
                        "tank_len": d["oxidizer_tank_length"],
                        "chamber_len": d["combution_chamber_length"],
                        "fuel_len": d["fuel_length"],
                        "outer_r": d["outter_radius"],
                        "port_r": d["inner_radius"],
                        "port_n": d["port_num"],
                        "port_c": d["port_center_radius"],
                        "throat_r": d["nozzle_throat_radius"],
                        "exit_r": d["nozzle_exit_radius"],
                        "inj_n": d["injector_oriffice_n"],
                        "inj_r": d["injector_oriffice_radius"],
                        "helium_g": round(d["pressurant_M"] * 1000, 1),
                    },
                    "wall_t": base["oxidizer_tank_tickness"],
                    "dt": CURVE_DT,
                    "thrust": [round(float(v)) for v in th[idx]],
                    "mdot": [round(float(v), 4) for v in (mox + mf)[idx]],
                    "pc": [round(float(v) / 1e6, 2) for v in series[j, 1][idx]],
                }
                fo.write(k + "\t" + json.dumps(row, separators=(",", ":")) + "\n")
                stats["n"] += 1
            print(f"  {min(s0 + CHUNK, len(X))}/{len(X)}  ({time.time() - t_start:.0f}s)", flush=True)
    print(f"surrogate {device}: {t_inf:.2f}s for {len(X)} designs")
    meta = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": os.path.basename(ts.MODEL_PATH),
        "device": device,
        "inference_s": round(t_inf, 3),
        "n": stats["n"],
        "n_ok": stats["ok"],
        "pipe": pipe["pipe_id"],
        "pipe_inner_d": pipe["inner_diameter"],
        "wall_t": base["oxidizer_tank_tickness"],
        "temperature_K": base["temperature"],
        "knobs": K,
        "base": CFG["base_design"],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8", compresslevel=6) as f:
        f.write(json.dumps(meta, separators=(",", ":")) + "\n")
        with open(tmp, encoding="utf-8") as fi:
            for line in fi:
                f.write(line)
    os.remove(tmp)
    print(json.dumps(meta, ensure_ascii=False))
    print(f"saved {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB, total {time.time() - t_start:.1f}s)")


if __name__ == "__main__":
    main()
