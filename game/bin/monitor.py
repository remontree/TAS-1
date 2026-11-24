"""TAS-1 Launch League 서버 모니터 (터미널).  q 종료 · r 재시작(두 번) · l 로그 전환"""
import curses
import json
import os
import ssl
import subprocess
import time
import urllib.request

GAME = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CFG = json.load(open(os.path.join(GAME, "server_config.json"), encoding="utf-8"))
PORT = CFG["port"]
SVC = "tas1-game.service"
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def sh(*a):
    try:
        return subprocess.run(a, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def svc():
    out = sh("systemctl", "--user", "show", SVC, "-p", "ActiveState,SubState,MainPID,NRestarts,ActiveEnterTimestamp")
    return dict(l.split("=", 1) for l in out.splitlines() if "=" in l)


def health():
    t0 = time.time()
    try:
        with urllib.request.urlopen(f"https://127.0.0.1:{PORT}/healthz", context=CTX, timeout=4) as r:
            return json.loads(r.read()), (time.time() - t0) * 1000
    except Exception as e:
        return {"error": str(e)[:60]}, None


def tail(path, n):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 16000))
            return f.read().decode("utf-8", "replace").splitlines()[-n:]
    except OSError:
        return []


def cert_end():
    out = sh("openssl", "x509", "-enddate", "-noout", "-in", os.path.join(GAME, "certs", "server.crt"))
    return out.replace("notAfter=", "")


def main(scr):
    curses.curs_set(0)
    curses.use_default_colors()
    for i, c in enumerate([curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_RED, curses.COLOR_CYAN], 1):
        curses.init_pair(i, c, -1)
    G, Y, R, CY = (curses.color_pair(i) for i in range(1, 5))
    scr.timeout(2000)
    logmode, armed, hist, cend = "app", 0, [], cert_end()
    while True:
        s = svc()
        h, lat = health()
        hist.append(h.get("req_per_min", 0) if lat else 0)
        hist = hist[-60:]
        scr.erase()
        H, W = scr.getmaxyx()

        def put(y, x, t, a=0):
            if 0 <= y < H - 1:
                scr.addnstr(y, x, t, max(0, W - x - 1), a)

        put(0, 0, " TAS-1 LAUNCH LEAGUE · SERVER MONITOR ", curses.A_REVERSE | curses.A_BOLD)
        put(0, 42, time.strftime("%Y-%m-%d %H:%M:%S"))
        st = s.get("ActiveState", "?")
        put(2, 0, "서비스   ")
        put(2, 9, f"{st}/{s.get('SubState', '?')}", G if st == "active" else R)
        put(2, 30, f"PID {s.get('MainPID', '-')}  재시작 {s.get('NRestarts', '0')}회  시작 {s.get('ActiveEnterTimestamp', '-')[:24]}")
        put(3, 0, "HTTPS    ")
        if lat is not None:
            put(3, 9, f"OK {lat:.0f} ms  :{PORT}", G if lat < 300 else Y)
            up = h.get("uptime", 0)
            put(3, 30, f"가동 {up // 3600}h {up // 60 % 60}m  메모리 {h.get('rss_mb', '?')} MB  비행계산 {h.get('node_ms') or '-'} ms")
        else:
            put(3, 9, f"응답 없음 {h.get('error', '')}", R)
        wd = tail(os.path.join(GAME, "data", "logs", "watchdog.log"), 1)
        put(4, 0, "감시     ")
        put(4, 9, (sh("systemctl", "--user", "is-active", "tas1-game-watchdog.timer") + " · 20초마다 점검, 3회 실패 시 재시작"), CY)
        put(5, 9, ("마지막 감시 기록: " + wd[0]) if wd else "감시 실패 기록 없음", Y if wd else 0)
        put(6, 0, "인증서   ")
        put(6, 9, cend or "-")
        if lat is not None:
            stt = h.get("stats", {})
            put(8, 0, "참가자", curses.A_BOLD)
            put(8, 9, f"{h.get('players', 0)}명 (5분 내 활동 {h.get('active_5m', 0)})   발사 {h.get('launches', 0)}회 (5분 {h.get('launches_5m', 0)})")
            put(9, 0, "요청", curses.A_BOLD)
            put(9, 9, f"{h.get('req_per_min', 0)}/분  총 {stt.get('requests', 0)}  5xx {stt.get('5xx', 0)}  404 {stt.get('404', 0)}  CSRF거부 {stt.get('csrf_reject', 0)}")
            put(10, 0, "보안", curses.A_BOLD)
            bans = h.get("bans", {})
            put(10, 9, f"로그인 실패 {stt.get('login_fail', 0)}  관리자 실패 {stt.get('admin_fail', 0)}  차단 중 IP {len(bans)}  누적 차단 {stt.get('bans', 0)}",
                R if bans or stt.get("admin_fail") else 0)
            for i, (ip, b) in enumerate(list(bans.items())[:3]):
                put(11 + i, 9, f"  {ip}  {b['reason']}  ~{time.strftime('%H:%M', time.localtime(b['until']))}", R)
            mx = max(hist + [1])
            bars = "".join(" ▁▂▃▄▅▆▇█"[min(8, int(v / mx * 8))] for v in hist)
            put(14, 0, "요청/분", curses.A_BOLD)
            put(14, 9, bars + f"  (최대 {mx})", CY)
        put(16, 0, f"── {'app.log' if logmode == 'app' else 'access.log'} " + "─" * 40, curses.A_DIM)
        lines = tail(os.path.join(GAME, "data", "logs", "app.log" if logmode == "app" else "access.log"), max(1, H - 19))
        for i, l in enumerate(lines):
            a = R if ("ERROR" in l or "BAN" in l or "FAIL" in l) else (Y if "WARN" in l else 0)
            put(17 + i, 0, l, a)
        put(H - 1, 0, (" r 한 번 더 누르면 재시작 " if armed else " q 종료 · r 재시작 · l 로그 전환 "), curses.A_REVERSE)
        scr.refresh()
        k = scr.getch()
        if k in (ord("q"), 27):
            break
        if k == ord("l"):
            logmode = "access" if logmode == "app" else "app"
        if k == ord("r"):
            if armed:
                subprocess.run(["systemctl", "--user", "restart", SVC])
                armed = 0
            else:
                armed = 1
        elif k != -1:
            armed = 0


if __name__ == "__main__":
    curses.wrapper(main)
