"""
core 시뮬레이션 동시 실행 개수를 시스템 전체 CPU 사용률에 맞춰 조절하는 관문 (프로세스·작업 공통)

GA.simulate 가 시작 전에 slot() 을 잡는다. MDO · 데이터 생성 · RL core 검증 등 어떤 프로세스 풀에서
부르든 같은 상태 파일 (/dev/shm) 을 공유하므로, 동시에 돌아가는 작업 전체의 시뮬레이션 수가 함께 제한된다.

    - CPU_INTERVAL 마다 /proc/stat 으로 시스템 전체 사용률 (다른 프로그램 포함) 을 잰다
    - 허용 개수 = 지금 돌고 있는 개수 + (목표 − 사용률) × 코어 수   (1 ~ 코어 수 − 1)
    - 허용 개수가 차 있으면 자리가 날 때까지 기다린다 (기다리는 동안 CPU 거의 안 씀)
    - 죽은 프로세스의 자리는 pid 확인으로 자동 회수

환경 변수 TAS_CPU_TARGET (기본 0.75 — 상한 80% 아래로 여유), TAS_CPU_GATE=0 이면 끔
"""

import contextlib
import fcntl
import json
import os
import time

CPU_TARGET = float(os.environ.get("TAS_CPU_TARGET", 0.75))
CPU_INTERVAL = 2.0
ENABLED = os.environ.get("TAS_CPU_GATE", "1") != "0"
STATE_PATH = "/dev/shm/tas1_cpu_gate.json" if os.path.isdir("/dev/shm") else "/tmp/tas1_cpu_gate.json"
LOCK_PATH = STATE_PATH + ".lock"
N_CPU = os.cpu_count() or 2


def _cpu_times():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle = v[3] + (v[4] if len(v) > 4 else 0)
    return sum(v), idle


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextlib.contextmanager
def _locked():
    with open(LOCK_PATH, "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            try:
                with open(STATE_PATH) as f:
                    st = json.load(f)
            except (OSError, ValueError):
                st = {}
            yield st
            tmp = STATE_PATH + f".{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(st, f)
            os.replace(tmp, STATE_PATH)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _update(st, now):
    """사용률 측정 · 허용 개수 갱신 (CPU_INTERVAL 마다)"""
    st["active"] = [p for p in st.get("active", []) if _alive(p)]
    if now - st.get("t", 0) < CPU_INTERVAL and "limit" in st:
        return
    total, idle = _cpu_times()
    if "total" in st and total > st["total"]:
        util = 1.0 - (idle - st["idle"]) / (total - st["total"])
        st["util"] = round(util, 3)
        free = (CPU_TARGET - util) * N_CPU
        # 늘릴 때는 천천히 (새 시뮬레이션이 CPU 를 다 쓰기까지 시간이 걸림), 줄일 때는 바로
        step = max(1, int(free * 0.6)) if free >= 1 else (0 if free > -0.5 else int(free) - 1)
        st["limit"] = max(1, min(N_CPU - 1, len(st["active"]) + step))
    else:
        st.setdefault("limit", max(1, N_CPU // 2))
    st.update(total=total, idle=idle, t=now)


def status():
    """웹 표시용: 시스템 CPU 사용률, 허용 개수, 실행 중 개수"""
    try:
        with open(STATE_PATH) as f:
            st = json.load(f)
        return {"util": st.get("util"), "limit": st.get("limit"), "active": len(st.get("active", [])),
                "waiting": st.get("waiting", 0), "target": CPU_TARGET}
    except (OSError, ValueError):
        return None


@contextlib.contextmanager
def slot():
    """시뮬레이션 1회 자리. 허용 개수가 찰 때까지 기다린다"""
    if not ENABLED:
        yield
        return
    pid = os.getpid()
    waiting = False
    while True:
        with _locked() as st:
            _update(st, time.time())
            if len(st["active"]) < st["limit"]:
                st["active"].append(pid)
                if waiting:
                    st["waiting"] = max(0, st.get("waiting", 0) - 1)
                break
            if not waiting:
                st["waiting"] = st.get("waiting", 0) + 1
                waiting = True
        time.sleep(0.5)
    try:
        yield
    finally:
        with _locked() as st:
            act = st.get("active", [])
            if pid in act:
                act.remove(pid)
            st["active"] = act
