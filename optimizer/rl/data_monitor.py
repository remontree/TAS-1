"""
시계열 데이터 생성 (rl/ts_data.py) 진행 상황을 웹 학습 모니터(/training, kind=data) 에 기록
    rl/runs/<id>/status.json · live.jsonl · rl/runs/<id>.log

ts_data.main 이 DataMonitor 를 직접 쓰고, 이미 돌고 있는 생성 작업은 붙어서 감시할 수 있다:
    python -m rl.data_monitor --dir rl/ts_data/<실행> --log rl/ts_data_he.log --pid <ts_data pid>
      (로그의 진행 줄 + 저장된 shard 로 통계 — shard 는 SHARD 개마다 저장되므로 통계는 그만큼 늦다)
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np

import GA
import constrain as cs
import cpu_gate
from rl import features as ft

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")
P_BINS = np.linspace(2.0e6, 7.0e6, 11)                 # 온도별 포화압 (−15 °C 2.07 ~ 35 °C 7.0 MPa) 을 모두 덮는 범위
T_BINS = np.linspace(258.0, 308.0, 11)                  # 초기 온도 [K] (−15 ~ 35 °C)
ISP_BINS = np.arange(0, 270, 10)
P_IDX = GA.DESIGN_VARS.index("supercharge_pressure")
T_IDX = ft.FEATURE_VARS.index("temperature")


class _Bins:
    """구간별 개수 · 정상 종료 · 차압 통과 · 평균 최소 차압 · 평균 Isp"""

    def __init__(self, edges):
        self.edges = edges
        k = len(edges) - 1
        self.n, self.ok, self.dp_pass = np.zeros(k, int), np.zeros(k, int), np.zeros(k, int)
        self.dp_sum, self.isp_sum = np.zeros(k), np.zeros(k)

    def add(self, v, ok, dp, isp):
        b = int(np.clip(np.searchsorted(self.edges, v, side="right") - 1, 0, len(self.edges) - 2))
        self.n[b] += 1
        if ok:
            self.ok[b] += 1
            if dp is not None:
                self.dp_sum[b] += min(dp, 1.0)
                self.dp_pass[b] += dp >= cs.MIN_INJECTOR_DP_RATIO
            if isp is not None:
                self.isp_sum[b] += isp

    def to_json(self):
        ok = np.maximum(self.ok, 1)
        return [{"lo": float(self.edges[i]), "hi": float(self.edges[i + 1]), "n": int(self.n[i]), "ok": int(self.ok[i]),
                 "dp_mean": float(self.dp_sum[i] / ok[i]) if self.ok[i] else None,
                 "dp_pass": float(self.dp_pass[i] / ok[i]) if self.ok[i] else None,
                 "isp_mean": float(self.isp_sum[i] / ok[i]) if self.ok[i] else None}
                for i in range(len(self.edges) - 1)]


class Stats:
    """생성된 시뮬레이션 통계 (슈퍼차징 압력 · 초기 온도 구간별 정상 종료율·차압, Isp 분포, 종류별 개수, 최근 결과)"""

    def __init__(self):
        self.p = _Bins(P_BINS)
        self.t = _Bins(T_BINS)
        self.isp_hist = np.zeros(len(ISP_BINS) - 1, int)
        self.kinds = {}
        self.terms = {}
        self.recent = []

    @property
    def n_ok(self):
        return int(self.p.ok.sum())

    def add(self, x, kind, sim):
        """x: 입력 행 (rl.ts_data.data_row 순서), sim: scalars dict + error / termination_code"""
        p = float(x[P_IDX])
        T = float(x[T_IDX]) if len(x) == len(ft.FEATURE_VARS) + len(ft.DATA_PIPE_FEATURES) else ft.LEGACY_TEMPERATURE
        ok = not sim.get("error")
        k = self.kinds.setdefault(kind, [0, 0])
        k[0] += 1
        k[1] += ok
        term = sim.get("termination_code") or ("ERROR" if sim.get("error") else "—")
        self.terms[term] = self.terms.get(term, 0) + 1
        isp = sim.get("isp")
        dp = sim.get("dp_ratio_min")
        self.p.add(p, ok, dp, isp)
        self.t.add(T, ok, dp, isp)
        if ok and isp is not None:
            self.isp_hist[int(np.clip(np.searchsorted(ISP_BINS, isp, side="right") - 1, 0, len(ISP_BINS) - 2))] += 1
        self.recent.append({"kind": kind, "p": p, "T": T, "he": float(x[len(GA.DESIGN_VARS) + 1]), "ok": ok,
                            "term": term, "isp": isp, "dp": dp, "impulse": sim.get("impulse"),
                            "thrust_avg": sim.get("thrust_avg"), "burn_time": sim.get("burn_time")})
        self.recent = self.recent[-12:]

    def to_json(self):
        return {"p_bins": self.p.to_json(), "t_bins": self.t.to_json(),
                "isp_hist": {"edges": ISP_BINS.tolist(), "counts": self.isp_hist.tolist()},
                "kinds": self.kinds, "terms": self.terms, "recent": self.recent[::-1]}


class DataMonitor:
    def __init__(self, run_id, data_dir, n_target, pid, config):
        self.dir = os.path.join(RUNS_DIR, run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.status = {"id": run_id, "name": run_id, "kind": "data", "state": "running", "pid": pid,
                       "started": time.time(), "step": 0, "total_steps": n_target,
                       "data_dir": os.path.relpath(data_dir, HERE), "config": config}
        self.live = open(os.path.join(self.dir, "live.jsonl"), "a", encoding="utf-8")
        self.log = open(os.path.join(RUNS_DIR, run_id + ".log"), "a", encoding="utf-8")
        self.stats = Stats()
        self.t_last = 0.0

    def set_started(self, t):
        self.status["started"] = t

    def save(self):
        self.status["stats"] = self.stats.to_json()
        tmp = os.path.join(self.dir, "status.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.status, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(self.dir, "status.json"))

    def progress(self, done, n_ok, rate, kinds_done=None, force=False, step=None):
        """진행 기록 (몇 초에 한 번만 파일에 쓴다). step: 진행 막대 기준 (기본 = 정상 종료 개수)"""
        now = time.time()
        if not force and now - self.t_last < 5:
            return
        self.t_last = now
        n_err = done - n_ok
        step = n_ok if step is None else step
        self.status.update(step=step, done=done, n_ok=n_ok, rate=rate, kinds_done=kinds_done)
        self.status["cpu"] = cpu_gate.status()
        row = {"step": step, "t": round(now, 1), "data/done": done, "data/rate": rate,
               "data/err_rate": n_err / max(done, 1)}
        st = self.stats
        n_okt = st.n_ok
        if n_okt:
            row["data/dp_pass"] = float(st.p.dp_pass.sum() / n_okt)
            row["data/isp_mean"] = float(st.p.isp_sum.sum() / n_okt)
        self.live.write(json.dumps(row) + "\n")
        self.live.flush()
        self.save()

    def write_log(self, line):
        self.log.write(line.rstrip() + "\n")
        self.log.flush()

    def finish(self, state="done"):
        self.status.update(state=state, finished=time.time())
        self.save()
        self.live.close()
        self.log.close()


# ──────────────────────────────────────────────────────────────────────────
# 이미 돌고 있는 생성 작업 감시
# ──────────────────────────────────────────────────────────────────────────

LINE = re.compile(r"정상 (\d+)/(\d+) \(전체 (\d+)\)\s+([\d.]+)개/s.*?(\{.*\})")


def _alive(pid):
    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as f:
            return b"rl.ts_data" in f.read()
    except (OSError, ValueError, TypeError):
        return False


def attach(data_dir, log_path, pid, run_id=None, poll=10.0):
    data_dir = os.path.abspath(data_dir)
    with open(os.path.join(data_dir, "info.json"), encoding="utf-8") as f:
        info = json.load(f)
    run_id = run_id or "tsdata_" + os.path.basename(data_dir)
    mon = DataMonitor(run_id, data_dir, info.get("n_target"), pid,
                      {"simt": info.get("simt"), "mix": info.get("mix"), "n_target": info.get("n_target"),
                       "workers": info.get("workers", GA.N_WORKERS),
                       "pressure_bounds": GA.BOUNDS["supercharge_pressure"], "data_dir": os.path.relpath(data_dir, HERE)})
    mon.set_started(os.path.getmtime(os.path.join(data_dir, "info.json")))     # 폴더 ctime 은 shard 가 생길 때마다 바뀐다
    seen_shards, log_pos, last = set(), 0, None
    while True:
        # 저장된 shard → 통계
        for path in sorted(glob.glob(os.path.join(data_dir, "shard_*.npz"))):
            if path in seen_shards:
                continue
            try:
                d = np.load(path)
                X, meta = d["X"], json.loads(str(d["meta"]))
            except Exception:
                continue                       # 쓰는 중
            seen_shards.add(path)
            for x, m in zip(X, meta):
                mon.stats.add(x, m["kind"], dict(m["scalars"], error=m["error"], termination_code=m["termination_code"]))
        mon.status["shards"] = len(seen_shards)
        # 로그 → 진행
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                f.seek(log_pos)
                chunk = f.read()
                log_pos = f.tell()
        except OSError:
            chunk = ""
        for line in chunk.splitlines():
            m = LINE.search(line)
            if m:
                last = m
                mon.write_log(line.strip())
            elif line.startswith(("시계열 데이터", "완료")):
                mon.write_log(line.strip())
        if last:
            n_ok, _, done, rate, kinds = last.groups()
            try:
                kinds = json.loads(kinds.replace("'", '"'))
            except ValueError:
                kinds = None
            mon.progress(int(done), int(n_ok), float(rate), kinds, force=True)
        else:
            mon.save()
        if not _alive(pid):
            finished = "완료" in chunk or (last and int(last.group(1)) >= int(last.group(2)))
            mon.finish("done" if finished else "stopped")
            return
        time.sleep(poll)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--id", default=None)
    a = ap.parse_args()
    attach(a.dir, a.log, a.pid, a.id)
