"""
대리모델 학습
────────────────────────────────────────────────────────────────────────────
surrogate/dataset.csv (build_dataset.py 결과) 로
  - 회귀 MLP 앙상블 : 입력 → 스칼라 성능 지표 14개
  - 분류 MLP 앙상블 : 시뮬레이션 정상 수행 / 연소 완료(simt 이전 종료) 확률
을 학습하고, 브라우저에서 바로 추론할 수 있도록 static/surrogate_model.json 으로 내보낸다.

실행 (TAS-1 폴더에서)
    .venv/bin/python surrogate/train.py
"""
import json, os, sys, time

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "surrogate"))
from build_dataset import DESIGN_VARS, CONDITION_VARS, INT_VARS, load_fixed  # noqa: E402

DATA = os.path.join(ROOT, "surrogate", "dataset.csv")
OUT = os.path.join(ROOT, "static", "surrogate_model.json")
R_GAS = 8.3143

INPUTS = list(DESIGN_VARS) + list(CONDITION_VARS)
OUTPUTS = ["isp", "isp_mean", "impulse", "thrust_avg", "thrust_max", "burn_time",
           "chamber_pressure_avg", "chamber_pressure_max", "tank_pressure_avg", "tank_pressure_max",
           "of_avg", "cstar_avg", "m_oxidizer_used", "m_fuel_used"]
N_REG, N_CLF, HIDDEN = 5, 3, 96

# N2O 포화 물성 표 (250–300 K, 0.5 K 간격) — 브라우저에서 선형 보간
T_GRID = np.arange(250.0, 300.01, 0.5)
N2O_TABLE = {
    "T": T_GRID.tolist(),
    "Psat": [PropsSI("P", "T", t, "Q", 0, "NitrousOxide") * 1.001 for t in T_GRID],
    "v_hat": [1 / PropsSI("Dmolar", "T", t, "Q", 0, "NitrousOxide") for t in T_GRID],
}


def engineered(df, fixed):
    """물리 기반 파생 특성 (static/surrogate.html 의 features() 와 같은 계산)"""
    T = df.temperature.values
    Psat = np.interp(T, N2O_TABLE["T"], N2O_TABLE["Psat"])
    vh = np.interp(T, N2O_TABLE["T"], N2O_TABLE["v_hat"])
    V = np.pi * df.oxidizer_tank_radius ** 2 * df.oxidizer_tank_length
    N = df.oxidizer_M / fixed["oxidizer_MW"]
    NL = (N * R_GAS * T - Psat * V) / (R_GAS * T - Psat * vh)
    Vg = V - NL * vh
    Pp = df.pressurant_M / fixed["pressurant_MW"] * R_GAS * T / np.maximum(Vg, 1e-9)
    f = pd.DataFrame({
        "ullage": np.clip(Vg / V, -1, 1),
        "log_p_tank0": np.log(np.clip(Psat + Pp, 1e5, 1e9)),
        "log_inj_area": np.log(df.injector_oriffice_n * np.pi * df.injector_oriffice_radius ** 2),
        "log_throat_area": np.log(np.pi * df.nozzle_throat_radius ** 2),
        "log_eps": np.log((df.nozzle_exit_radius / df.nozzle_throat_radius) ** 2),
        "log_port_area": np.log(df.port_num * np.pi * df.inner_radius ** 2),
        "log_burn_area": np.log(df.port_num * 2 * np.pi * df.inner_radius * df.fuel_length),
        "fuel_mass": fixed["fuel_rho"] * df.fuel_length * np.pi * (df.outter_radius ** 2 - df.port_num * df.inner_radius ** 2),
    })
    return f


def init(sizes, rng):
    return [[rng.normal(0, np.sqrt(2 / (a + b)), (a, b)), np.zeros(b)] for a, b in zip(sizes[:-1], sizes[1:])]


def forward(P, x):
    hs = [x]
    for i, (W, b) in enumerate(P):
        z = hs[-1] @ W + b
        hs.append(np.tanh(z) if i < len(P) - 1 else z)
    return hs


def fit(X, T, kind, seed, epochs, hidden, weight=None):
    """Adam + 계단식 학습률, 부트스트랩 표본 (앙상블 다양성)"""
    rng = np.random.default_rng(seed)
    P = init([X.shape[1], hidden, hidden, T.shape[1]], rng)
    m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in P]
    v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in P]
    bi = rng.integers(0, len(X), len(X))
    Xb, Tb = X[bi], T[bi]
    Wb = weight[bi] if weight is not None else np.ones((len(X), 1))
    lr, t, bs = 2e-3, 0, 128
    for ep in range(epochs):
        if ep == int(epochs * 0.6): lr = 6e-4
        if ep == int(epochs * 0.85): lr = 1.5e-4
        perm = rng.permutation(len(Xb))
        for k in range(0, len(Xb), bs):
            j = perm[k:k + bs]
            hs = forward(P, Xb[j]); out = hs[-1]
            if kind == "reg":
                g = 2 * (out - Tb[j]) * Wb[j] / len(j)
            else:
                g = (1 / (1 + np.exp(-out)) - Tb[j]) * Wb[j] / len(j)
            t += 1
            for i in range(len(P) - 1, -1, -1):
                W = P[i][0]
                grads = (hs[i].T @ g + 1e-5 * W, g.sum(0))
                if i > 0:
                    g = (g @ W.T) * (1 - hs[i] ** 2)
                for q, gr in enumerate(grads):
                    m[i][q] = 0.9 * m[i][q] + 0.1 * gr
                    v[i][q] = 0.999 * v[i][q] + 0.001 * gr * gr
                    P[i][q] -= lr * (m[i][q] / (1 - 0.9 ** t)) / (np.sqrt(v[i][q] / (1 - 0.999 ** t)) + 1e-8)
    return P


def export(P):
    r = lambda a: [float(f"{x:.6g}") for x in a]
    return [{"W": [r(row) for row in W], "b": r(b)} for W, b in P]


def main():
    fixed = load_fixed()
    d = pd.read_csv(DATA)
    d["error"] = d["error"].fillna("")
    d["burnout"] = ((d.ok == 1) & (d.termination_code != "TIME_LIMIT_REACHED")).astype(float)
    feats = pd.concat([d[INPUTS].reset_index(drop=True), engineered(d, fixed).reset_index(drop=True)], axis=1)
    FEAT = list(feats.columns)
    X = feats.values.astype(float)

    rng = np.random.default_rng(7)
    idx = rng.permutation(len(d)); n_te = int(len(d) * 0.15)
    te, tr = idx[:n_te], idx[n_te:]
    xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-12
    Xn = (X - xm) / xs

    # 회귀: 정상 수행 + 연소 완료 + 양수 출력
    good = (d.burnout.values == 1) & (d[OUTPUTS].values > 0).all(1) & (d.isp.values > 50)
    Y = np.log(d[OUTPUTS].values.clip(1e-12))
    tr_r, te_r = tr[good[tr]], te[good[te]]
    ym, ys = Y[tr_r].mean(0), Y[tr_r].std(0) + 1e-12
    Yn = (Y - ym) / ys

    t0 = time.time()
    regs = [fit(Xn[tr_r], Yn[tr_r], "reg", s, 500, HIDDEN) for s in range(N_REG)]
    print(f"regression trained {time.time() - t0:.0f}s  (train {len(tr_r)}, test {len(te_r)})")

    C = d[["ok", "burnout"]].values.astype(float)
    w = np.where(C[:, :1] < 0.5, 3.0, 1.0) * np.where(C[:, 1:] < 0.5, 2.0, 1.0)
    clfs = [fit(Xn[tr], C[tr], "clf", 100 + s, 150, 48, w[tr]) for s in range(N_CLF)]
    print(f"classifier trained {time.time() - t0:.0f}s")

    # 검증
    pred = np.exp(np.mean([forward(P, Xn[te_r])[-1] for P in regs], 0) * ys + ym)
    true = d[OUTPUTS].values[te_r]
    metrics, parity = {}, {}
    pi = rng.choice(len(te_r), min(300, len(te_r)), replace=False)
    for k, o in enumerate(OUTPUTS):
        yt, yp = true[:, k], pred[:, k]
        metrics[o] = {
            "r2": float(1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)),
            "mae": float(np.mean(np.abs(yt - yp))),
            "mape": float(np.mean(np.abs(yt - yp) / np.abs(yt)) * 100),
        }
        parity[o] = [[float(f"{yt[i]:.5g}"), float(f"{yp[i]:.5g}")] for i in pi]
        print(f"{o:22s} R2 {metrics[o]['r2']:.4f}  MAPE {metrics[o]['mape']:.2f}%")
    pc = np.mean([1 / (1 + np.exp(-forward(P, Xn[te])[-1])) for P in clfs], 0)
    for k, name in enumerate(["ok", "burnout"]):
        metrics[f"acc_{name}"] = float(np.mean((pc[:, k] > 0.5) == (C[te, k] > 0.5)))
        print(f"acc_{name} {metrics[f'acc_{name}']:.4f}  (base {C[te, k].mean():.3f})")

    ok = d[d.ok == 1]
    model = {
        "inputs": INPUTS, "features": FEAT, "outputs": OUTPUTS,
        "xm": xm.tolist(), "xs": xs.tolist(), "ym": ym.tolist(), "ys": ys.tolist(),
        "regs": [export(P) for P in regs], "clfs": [export(P) for P in clfs],
        "metrics": metrics, "parity": parity,
        "ranges": {v: [float(d[v].min()), float(d[v].max())] for v in INPUTS},
        "int_vars": sorted(INT_VARS),
        "fixed": fixed, "n2o": N2O_TABLE,
        "n_samples": int(len(d)), "n_train": int(len(tr_r)), "n_test": int(len(te_r)),
        "sim_wall_mean": float(ok.wall_time.mean()),
        "hidden": HIDDEN,
    }
    with open(OUT, "w") as f:
        json.dump(model, f, separators=(",", ":"))
    print(f"saved {OUT} ({os.path.getsize(OUT) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
