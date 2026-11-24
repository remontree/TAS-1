"""
TAS-1 Launch League — 게임 서버 (Flask, gunicorn 1 worker × N threads 로 실행)
엔진: data/engine_lut.jsonl.gz (TAS-1 시계열 대리모델 사전계산) · 기체/비행: static/js/flight.js (node 로 서버 재계산)
"""

import collections
import csv
import gzip
import hashlib
import hmac
import io
import json
import logging
import logging.handlers
import mimetypes
import os
import random
import re
import secrets
import sqlite3
import subprocess
import threading
import time
from functools import wraps

import segno
from flask import Flask, Response, abort, g, jsonify, request, send_file, session
from werkzeug.security import check_password_hash

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
STATIC = os.path.join(BASE, "static")
CFG = json.load(open(os.path.join(BASE, "server_config.json"), encoding="utf-8"))
DB_PATH = os.path.join(DATA, "game.db")
os.makedirs(os.path.join(DATA, "logs"), exist_ok=True)

log = logging.getLogger("tas1")
log.setLevel(logging.INFO)
_h = logging.handlers.RotatingFileHandler(os.path.join(DATA, "logs", "app.log"), maxBytes=5_000_000, backupCount=5)
_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log.addHandler(_h)

# ── 결과표 ───────────────────────────────────────────────────────────────
# 첫 줄 meta, 이후 "키<TAB>행 JSON". 행은 JSON 문자열 그대로 보관 (41만 행을 파이썬 객체로 들면 수 GB) — 응답·node 입력에 그대로 쓴다
LUT = {}
with gzip.open(os.path.join(DATA, "engine_lut.jsonl.gz"), "rt", encoding="utf-8") as f:
    LUT_META = json.loads(f.readline())
    for _line in f:
        _k, _row = _line.rstrip("\n").split("\t", 1)
        LUT[_k] = _row
KNOBS = LUT_META["knobs"]
ENGINE_KNOBS = ("ox", "p_MPa", "n", "fuel_m", "eps", "ports")   # 엔진 키 자리 순서
ENGINE_LEVELS = [len(KNOBS[k]) for k in ENGINE_KNOBS]
# 기체 단계 수는 flight.js 가 기준 (한 곳에서만 정의)
AIR_LEVELS = json.loads(subprocess.run(
    ["node", "-e", "const F=require(process.argv[1]);console.log(JSON.stringify({body:F.AIR.body.vals.length,fin:F.AIR.fin.vals.length,nose:F.NOSES.length}))",
     os.path.join(BASE, "static", "js", "flight.js")], capture_output=True, text=True, check=True).stdout)
AI_PATH = os.path.join(DATA, "ai_baseline.json")
START = time.time()

app = Flask(__name__, static_folder=None)
app.secret_key = open(os.path.join(DATA, "secret.key"), "rb").read()
app.config.update(
    SESSION_COOKIE_NAME="__Host-tas1",
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=60 * 60 * 24 * 30,
    MAX_CONTENT_LENGTH=32 * 1024,
)


# ── DB ──────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS players(
  id INTEGER PRIMARY KEY, student_id TEXT UNIQUE NOT NULL, name TEXT NOT NULL, dept TEXT NOT NULL,
  phone TEXT NOT NULL, created_at REAL NOT NULL, last_seen REAL, hidden INTEGER DEFAULT 0,
  token_ver INTEGER DEFAULT 1, reg_ip TEXT, consent_at REAL);
CREATE TABLE IF NOT EXISTS launches(
  id INTEGER PRIMARY KEY, player_id INTEGER NOT NULL, created_at REAL NOT NULL, official INTEGER NOT NULL,
  engine TEXT NOT NULL, air TEXT NOT NULL, outcome TEXT NOT NULL, apo INTEGER, score_target INTEGER,
  score_max INTEGER, margin REAL, max_v REAL, impact_v REAL, land_d REAL, wind TEXT, traj TEXT);
CREATE INDEX IF NOT EXISTS ix_l_player ON launches(player_id, official);
CREATE INDEX IF NOT EXISTS ix_l_time ON launches(created_at);
"""


def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, timeout=10)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA busy_timeout=5000")
    return g.db


@app.teardown_appcontext
def _close(_exc):
    c = g.pop("db", None)
    if c is not None:
        c.close()


def init_db():
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript(SCHEMA)
    # 포트 수 노브 추가 전 기록 (5자리 키) = 포트 2개 → 6번째 자리 인덱스를 붙인다
    c.execute("UPDATE launches SET engine = engine || ? WHERE length(engine) = 5", (str(KNOBS["ports"].index(2)),))
    c.commit()
    c.close()
    os.chmod(DB_PATH, 0o600)


init_db()


# ── 보안: 레이트 리밋 · 차단 · 로그인 잠금 ─────────────────────────────────
class Limiter:
    def __init__(self):
        self.lock = threading.Lock()
        self.hits = collections.defaultdict(collections.deque)
        self.bans = {}          # ip -> (until, reason)
        self.fails = {}         # key -> [count, first, locked_until]

    def allow(self, key, limit, window):
        now = time.time()
        with self.lock:
            q = self.hits[key]
            while q and q[0] < now - window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True

    def banned(self, ip):
        b = self.bans.get(ip)
        if b and b[0] > time.time():
            return b
        if b:
            self.bans.pop(ip, None)
        return None

    def ban(self, ip, seconds, reason):
        self.bans[ip] = (time.time() + seconds, reason)
        log.warning("BAN %s %ss %s", ip, seconds, reason)
        STATS["bans"] += 1

    def locked(self, key):
        f = self.fails.get(key)
        return f and f[2] > time.time()

    def fail(self, key, limit, window, lock_s):
        now = time.time()
        with self.lock:
            f = self.fails.get(key)
            if not f or f[1] < now - window:
                f = [0, now, 0]
            f[0] += 1
            if f[0] >= limit:
                f[2] = now + lock_s
                log.warning("LOCK %s for %ss after %s failures", key, lock_s, f[0])
            self.fails[key] = f
            return f[0]

    def clear(self, key):
        self.fails.pop(key, None)


LIM = Limiter()
STATS = collections.Counter()
REQ_TIMES = collections.deque(maxlen=20000)
NODE_MS = collections.deque(maxlen=200)
SEC = CFG["security"]


LOOPBACK = ("127.0.0.1", "::1")


def client_ip():
    """터널(cloudflared · Tailscale Funnel)은 같은 머신에서 들어오므로, 루프백에서 온 요청만 전달 헤더를 믿는다.
    인터넷에서 28443 으로 바로 온 요청은 헤더를 무시해 IP 위조로 차단을 피할 수 없다."""
    ra = request.remote_addr or "?"
    if ra in LOOPBACK:
        fwd = request.headers.get("CF-Connecting-IP") or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if fwd:
            return fwd
    return ra


def err(msg, status=400, **extra):
    return jsonify(dict(error=msg, **extra)), status


@app.before_request
def _guard():
    ip = client_ip()
    g.ip = ip
    STATS["requests"] += 1
    REQ_TIMES.append(time.time())
    b = LIM.banned(ip)
    if b:
        STATS["blocked"] += 1
        return err("잠시 차단되었습니다. 나중에 다시 시도하세요.", 429, retry=int(b[0] - time.time()))
    if not LIM.allow("ip:" + ip, SEC["ip_req_per_min"], 60):
        LIM.ban(ip, SEC["ban_seconds"], "flood")
        return err("요청이 너무 많습니다.", 429)
    if request.method == "POST":
        # CSRF: 같은 출처 JS 만 보낼 수 있는 헤더 + JSON 본문 요구
        if request.headers.get("X-TAS1") != "1" or not request.is_json:
            STATS["csrf_reject"] += 1
            return err("잘못된 요청입니다.", 400)


@app.after_request
def _headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "same-origin"
    resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
        "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
    if CFG.get("hsts"):
        resp.headers["Strict-Transport-Security"] = "max-age=86400"
    if resp.status_code == 404:
        STATS["404"] += 1
        if not LIM.allow("404:" + g.get("ip", "?"), SEC["max_404_per_min"], 60):
            LIM.ban(g.get("ip", "?"), SEC["ban_seconds"], "scan")
    if resp.status_code >= 500:
        STATS["5xx"] += 1
    if request.path.startswith("/api/"):
        resp.headers.setdefault("Cache-Control", "no-store")
        if (resp.mimetype == "application/json" and not resp.direct_passthrough and "Content-Encoding" not in resp.headers
                and "gzip" in request.headers.get("Accept-Encoding", "")):
            body = resp.get_data()
            if len(body) > 800:
                resp.set_data(gzip.compress(body, 6))
                resp.headers["Content-Encoding"] = "gzip"
                resp.headers["Vary"] = "Accept-Encoding"
    return resp


@app.errorhandler(404)
def _404(_e):
    return err("없는 경로입니다.", 404)


@app.errorhandler(413)
def _413(_e):
    return err("요청이 너무 큽니다.", 413)


@app.errorhandler(500)
def _500(e):
    log.exception("500: %s", e)
    return err("서버 오류가 발생했습니다.", 500)


# ── 공용 ────────────────────────────────────────────────────────────────
def mask_name(n):
    n = n.strip()
    if len(n) <= 1:
        return "*"
    if len(n) == 2:
        return n[0] + "*"
    return n[0] + "*" * (len(n) - 2) + n[-1]


def mask_phone(p):
    return p[:3] + "-****-" + p[-4:] if len(p) >= 10 else "****"


def current_wind():
    hour = time.strftime("%Y%m%d%H")
    seed = int(hashlib.sha256((hour + str(CFG.get("wind_seed", "tas1"))).encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    w = CFG["wind"]
    nxt = (int(time.time()) // 3600 + 1) * 3600
    return {"spd": round(rng.uniform(w["min"], w["max"]), 1), "dir": rng.randrange(0, 360, 5), "next_change": nxt}


def player_required(fn):
    @wraps(fn)
    def wrap(*a, **kw):
        pid = session.get("pid")
        if not pid:
            return err("로그인이 필요합니다.", 401)
        p = db().execute("SELECT * FROM players WHERE id=?", (pid,)).fetchone()
        if not p or p["token_ver"] != session.get("tv"):
            session.clear()
            return err("로그인이 필요합니다.", 401)
        g.player = p
        db().execute("UPDATE players SET last_seen=? WHERE id=?", (time.time(), pid))
        db().commit()
        return fn(*a, **kw)
    return wrap


def admin_required(fn):
    @wraps(fn)
    def wrap(*a, **kw):
        if session.get("admin_until", 0) < time.time():
            return err("관리자 로그인이 필요합니다.", 401)
        return fn(*a, **kw)
    return wrap


def official_used(pid):
    return db().execute("SELECT COUNT(*) FROM launches WHERE player_id=? AND official=1", (pid,)).fetchone()[0]


def board(mode, limit=20):
    col = "score_target" if mode == "target" else "score_max"
    rows = db().execute(f"""
        SELECT p.id pid, p.name, p.dept, l.apo, l.{col} score, l.created_at
        FROM launches l JOIN players p ON p.id=l.player_id
        WHERE l.official=1 AND p.hidden=0 AND l.{col}>0
          AND l.id = (SELECT l2.id FROM launches l2 WHERE l2.player_id=l.player_id AND l2.official=1
                      ORDER BY l2.{col} DESC, l2.created_at ASC LIMIT 1)
        ORDER BY score DESC, l.created_at ASC LIMIT ?""", (limit,)).fetchall()
    return [{"rank": i + 1, "pid": r["pid"], "name": mask_name(r["name"]), "dept": r["dept"][:14],
             "apo": r["apo"], "score": r["score"]} for i, r in enumerate(rows)]


def my_rank(pid, mode):
    col = "score_target" if mode == "target" else "score_max"
    mine = db().execute(f"SELECT MAX({col}) FROM launches WHERE player_id=? AND official=1", (pid,)).fetchone()[0]
    if not mine:
        return None, None
    better = db().execute(f"""SELECT COUNT(*) FROM (SELECT player_id, MAX({col}) s FROM launches l JOIN players p
        ON p.id=l.player_id WHERE official=1 AND p.hidden=0 GROUP BY player_id HAVING s>?)""", (mine,)).fetchone()[0]
    return better + 1, mine


def total_ranked():
    return db().execute("""SELECT COUNT(DISTINCT player_id) FROM launches l JOIN players p ON p.id=l.player_id
        WHERE official=1 AND p.hidden=0""").fetchone()[0]


def ai_baseline():
    try:
        return json.load(open(AI_PATH, encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ── 페이지 · 정적 파일 (gzip 미리 압축해 두고 전송 — Funnel 경유 시 전송량이 체감 속도를 좌우) ──
COMPRESSIBLE = (".js", ".css", ".html", ".svg", ".json")


def precompress():
    for root, _dirs, files in os.walk(STATIC):
        for fn in files:
            if not fn.endswith(COMPRESSIBLE):
                continue
            src = os.path.join(root, fn)
            dst = src + ".gz"
            if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
                with open(src, "rb") as f, gzip.open(dst, "wb", compresslevel=9) as o:
                    o.write(f.read())


precompress()


def serve_static(p, max_age):
    full = os.path.realpath(os.path.join(STATIC, p))
    if not full.startswith(os.path.realpath(STATIC) + os.sep) or not os.path.isfile(full):
        abort(404)
    if "gzip" in request.headers.get("Accept-Encoding", "") and os.path.isfile(full + ".gz"):
        resp = send_file(full + ".gz", mimetype=mimetypes.guess_type(full)[0] or "application/octet-stream",
                         conditional=True, etag=True)
        resp.headers["Content-Encoding"] = "gzip"
    else:
        resp = send_file(full, conditional=True, etag=True)
    resp.headers["Vary"] = "Accept-Encoding"
    resp.headers["Cache-Control"] = f"public, max-age={max_age}"
    return resp


@app.route("/")
def index():
    return serve_static("index.html", 0)


@app.route("/wall")
def wall():
    return serve_static("wall.html", 0)


@app.route(CFG.get("admin_path", "/admin"))
def admin_page():
    return serve_static("admin.html", 0)


@app.route("/static/<path:p>")
def static_files(p):
    if p.endswith(".gz"):
        abort(404)
    # 라이브러리는 바뀌지 않으므로 오래 캐시, 앱 코드는 짧게 (ETag 로 바뀐 것만 다시 받음)
    return serve_static(p, 7 * 86400 if p.startswith("vendor/") else 300)


@app.route("/healthz")
def healthz():
    out = {"ok": True, "uptime": round(time.time() - START)}
    if g.ip in LOOPBACK:
        now = time.time()
        c = db()
        out.update({
            "pid": os.getpid(),
            "rss_mb": round(int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6, 1),
            "req_per_min": sum(1 for t in REQ_TIMES if t > now - 60),
            "stats": dict(STATS),
            "bans": {ip: {"until": int(b[0]), "reason": b[1]} for ip, b in list(LIM.bans.items()) if b[0] > now},
            "players": c.execute("SELECT COUNT(*) FROM players").fetchone()[0],
            "active_5m": c.execute("SELECT COUNT(*) FROM players WHERE last_seen>?", (now - 300,)).fetchone()[0],
            "launches": c.execute("SELECT COUNT(*) FROM launches").fetchone()[0],
            "launches_5m": c.execute("SELECT COUNT(*) FROM launches WHERE created_at>?", (now - 300,)).fetchone()[0],
            "node_ms": round(sum(NODE_MS) / len(NODE_MS), 1) if NODE_MS else None,
        })
    return jsonify(out)


# ── 공개 API ────────────────────────────────────────────────────────────
@app.route("/api/meta")
def api_meta():
    return jsonify({
        "event": CFG["event_name"], "target": CFG["target_alt"], "official_limit": CFG["official_launches"],
        "knobs": KNOBS, "wind": current_wind(),
        "lut": {k: LUT_META[k] for k in ("built_at", "model", "device", "inference_s", "n", "n_ok", "temperature_K", "pipe")},
        "ai": ai_baseline(),
    })


ID_RE = re.compile(r"^\d{6,12}$")
PHONE_RE = re.compile(r"^01[016789]\d{7,8}$")


def norm_phone(s):
    return re.sub(r"\D", "", str(s or ""))


@app.route("/api/register", methods=["POST"])
def api_register():
    ip = g.ip
    if not LIM.allow("reg:" + ip, SEC["register_per_ip_10min"], 600):
        return err("등록 요청이 너무 많습니다. 잠시 후 다시 시도하세요.", 429)
    d = request.get_json(silent=True) or {}
    name = str(d.get("name", "")).strip()
    sid = str(d.get("student_id", "")).strip()
    dept = str(d.get("dept", "")).strip()
    phone = norm_phone(d.get("phone"))
    if not d.get("consent"):
        return err("개인정보 수집·이용에 동의해야 참가할 수 있습니다.")
    if not (1 <= len(name) <= 20) or re.search(r"[<>\"'&]", name):
        return err("이름은 1–20자로 입력하세요.")
    if not ID_RE.match(sid):
        return err("학번은 숫자 6–12자리입니다.")
    if not (1 <= len(dept) <= 30) or re.search(r"[<>\"'&]", dept):
        return err("학과를 입력하세요.")
    if not PHONE_RE.match(phone):
        return err("전화번호는 010으로 시작하는 숫자로 입력하세요.")
    c = db()
    if c.execute("SELECT 1 FROM players WHERE student_id=?", (sid,)).fetchone():
        return err("이미 등록된 학번입니다. '다시 접속'에서 학번과 전화번호로 들어오세요.", 409, code="exists")
    now = time.time()
    cur = c.execute("INSERT INTO players(student_id,name,dept,phone,created_at,last_seen,reg_ip,consent_at) "
                    "VALUES(?,?,?,?,?,?,?,?)", (sid, name, dept, phone, now, now, ip, now))
    c.commit()
    session.clear()
    session.permanent = True
    session["pid"], session["tv"] = cur.lastrowid, 1
    log.info("REGISTER pid=%s ip=%s", cur.lastrowid, ip)
    STATS["registers"] += 1
    return jsonify({"ok": True})


@app.route("/api/login", methods=["POST"])
def api_login():
    ip = g.ip
    d = request.get_json(silent=True) or {}
    sid = str(d.get("student_id", "")).strip()
    phone = norm_phone(d.get("phone"))
    if LIM.locked("lip:" + ip) or LIM.locked("lsid:" + sid):
        return err("로그인 시도가 너무 많아 잠시 잠겼습니다. 15분 뒤 다시 시도하거나 부스에 문의하세요.", 429)
    p = db().execute("SELECT * FROM players WHERE student_id=?", (sid,)).fetchone() if ID_RE.match(sid) else None
    if not p or not hmac.compare_digest(p["phone"], phone):
        LIM.fail("lsid:" + sid, SEC["login_fail_per_id"], 900, SEC["login_lock_seconds"])
        n = LIM.fail("lip:" + ip, SEC["login_fail_per_ip"], 900, SEC["login_lock_seconds"])
        STATS["login_fail"] += 1
        log.info("LOGIN FAIL sid=%s ip=%s n=%s", sid[:4] + "***", ip, n)
        time.sleep(0.4)
        return err("학번 또는 전화번호가 맞지 않습니다.", 401)
    LIM.clear("lsid:" + sid)
    session.clear()
    session.permanent = True
    session["pid"], session["tv"] = p["id"], p["token_ver"]
    STATS["logins"] += 1
    return jsonify({"ok": True})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/me")
@player_required
def api_me():
    p = g.player
    used = official_used(p["id"])
    hist = db().execute("""SELECT id, created_at, official, engine, air, outcome, apo, score_target, score_max
        FROM launches WHERE player_id=? ORDER BY id DESC LIMIT 30""", (p["id"],)).fetchall()
    rt, st = my_rank(p["id"], "target")
    rm, sm = my_rank(p["id"], "max")
    return jsonify({
        "name": p["name"], "masked": mask_name(p["name"]), "dept": p["dept"], "student_id": p["student_id"],
        "phone": mask_phone(p["phone"]), "official_left": max(0, CFG["official_launches"] - used),
        "rank": {"target": rt, "target_score": st, "max": rm, "max_score": sm, "of": total_ranked()},
        "history": [dict(h) for h in hist],
    })


def valid_engine(key):
    return isinstance(key, str) and len(key) == len(ENGINE_LEVELS) and key.isdigit() and all(int(c) < n for c, n in zip(key, ENGINE_LEVELS))


@app.route("/api/engine/<key>")
@player_required
def api_engine(key):
    if not valid_engine(key):
        abort(404)
    if not LIM.allow("eng:%s" % g.player["id"], SEC["engine_per_min"], 60):
        return err("너무 빠르게 요청하고 있습니다.", 429)
    resp = Response(LUT[key], mimetype="application/json")
    resp.headers["Cache-Control"] = "private, max-age=3600"
    resp.headers["ETag"] = LUT_META["built_at"].replace(" ", "T") + "-" + key
    return resp


def run_flight(engine, air, wind):
    rest = json.dumps({"ox": KNOBS["ox"][int(engine[0])], "air": air, "wind": wind, "target": CFG["target_alt"]})
    payload = '{"row":' + LUT[engine] + "," + rest[1:]
    t0 = time.time()
    out = subprocess.run(["node", os.path.join(BASE, "server", "score.js")], input=payload.encode(),
                         capture_output=True, timeout=20, check=True)
    NODE_MS.append((time.time() - t0) * 1000)
    return json.loads(out.stdout)


_launch_lock = threading.Lock()


@app.route("/api/launch", methods=["POST"])
@player_required
def api_launch():
    p = g.player
    d = request.get_json(silent=True) or {}
    engine = str(d.get("engine", ""))
    air = d.get("air") or {}
    official = bool(d.get("official"))
    if not valid_engine(engine):
        return err("엔진 설계가 올바르지 않습니다.")
    try:
        air = {"body": int(air["body"]), "fin": int(air["fin"]), "nose": int(air["nose"])}
    except (KeyError, TypeError, ValueError):
        return err("기체 설계가 올바르지 않습니다.")
    if not all(0 <= air[k] < AIR_LEVELS[k] for k in ("body", "fin", "nose")):
        return err("기체 설계가 올바르지 않습니다.")
    if not LIM.allow("launch:%s" % p["id"], SEC["launch_per_min"], 60) or not LIM.allow("launch2:%s" % p["id"], 1, 2):
        return err("발사 간격이 너무 짧습니다. 잠시 후 다시 시도하세요.", 429)
    wind = current_wind()
    try:
        res = run_flight(engine, air, {"spd": wind["spd"], "dir": wind["dir"]})
    except (subprocess.SubprocessError, ValueError) as e:
        log.exception("flight failed: %s", e)
        return err("비행 계산에 실패했습니다. 다시 시도하세요.", 500)
    if res.get("outcome") == "NO_ASSEMBLY":
        return err("엔진이 지관에 들어가지 않아 조립할 수 없습니다. 지관을 늘리세요.")
    c = db()
    with _launch_lock:
        if official and official_used(p["id"]) >= CFG["official_launches"]:
            return err("공식 발사를 모두 사용했습니다. 연습 발사는 계속할 수 있어요.", 403)
        fl = res["fl"]
        sc = res["score"]
        cur = c.execute("""INSERT INTO launches(player_id,created_at,official,engine,air,outcome,apo,score_target,
            score_max,margin,max_v,impact_v,land_d,wind,traj) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (p["id"], time.time(), int(official), engine, json.dumps(air), fl["outcome"], fl["apo"],
                         sc["target"], sc["max"], res["margin"], fl["maxV"], fl.get("impactV", 0),
                         (fl["land"][0] ** 2 + fl["land"][1] ** 2) ** 0.5 if fl.get("land") else None,
                         json.dumps(wind), json.dumps(fl["tr"], separators=(",", ":"))))
        c.commit()
    STATS["launches"] += 1
    rt, _ = my_rank(p["id"], "target")
    rm, _ = my_rank(p["id"], "max")
    log.info("LAUNCH pid=%s off=%s eng=%s air=%s -> %s %sm", p["id"], int(official), engine, air, fl["outcome"], fl["apo"])
    return jsonify({"id": cur.lastrowid, "official": official, "wind": wind, "fl": fl, "score": sc,
                    "margin": res["margin"], "rank": {"target": rt, "max": rm, "of": total_ranked()},
                    "official_left": max(0, CFG["official_launches"] - official_used(p["id"]))})


@app.route("/api/board")
def api_board():
    mode = "max" if request.args.get("mode") == "max" else "target"
    return jsonify({"mode": mode, "rows": board(mode, 20), "ai": ai_baseline(), "target": CFG["target_alt"]})


def _thin(tr, n=80):
    asc = [r for r in tr if not r.get("chute")]
    step = max(1, len(asc) // n)
    return [[round((r["p"][0] ** 2 + r["p"][1] ** 2) ** 0.5), round(r["h"])] for r in asc[::step]] + \
        ([[round((asc[-1]["p"][0] ** 2 + asc[-1]["p"][1] ** 2) ** 0.5), round(asc[-1]["h"])]] if asc else [])


@app.route("/api/feed")
def api_feed():
    c = db()
    rows = c.execute("""SELECT l.id, l.created_at, l.official, l.outcome, l.apo, l.score_target, l.impact_v, l.traj,
        p.name, p.dept FROM launches l JOIN players p ON p.id=l.player_id WHERE p.hidden=0
        ORDER BY l.id DESC LIMIT 12""").fetchall()
    feed = []
    for i, r in enumerate(rows):
        item = {"id": r["id"], "t": r["created_at"], "official": r["official"], "outcome": r["outcome"],
                "apo": r["apo"], "score": r["score_target"], "name": mask_name(r["name"]), "dept": r["dept"][:12]}
        if i < 6 and r["traj"]:
            item["path"] = _thin(json.loads(r["traj"]))
        feed.append(item)
    rud = c.execute("""SELECT l.outcome, l.impact_v, l.apo, l.air, p.name FROM launches l JOIN players p
        ON p.id=l.player_id WHERE p.hidden=0 AND l.outcome IN ('CRASH','BURST')
        ORDER BY l.impact_v DESC LIMIT 5""").fetchall()
    now = time.time()
    return jsonify({
        "feed": feed,
        "rud": [{"name": mask_name(r["name"]), "outcome": r["outcome"], "impact_v": round(r["impact_v"] or 0),
                 "apo": r["apo"], "fin": json.loads(r["air"])["fin"]} for r in rud],
        "stats": {
            "players": c.execute("SELECT COUNT(*) FROM players WHERE hidden=0").fetchone()[0],
            "launches": c.execute("SELECT COUNT(*) FROM launches").fetchone()[0],
            "crashes": c.execute("SELECT COUNT(*) FROM launches WHERE outcome IN ('CRASH','BURST')").fetchone()[0],
            "launches_10m": c.execute("SELECT COUNT(*) FROM launches WHERE created_at>?", (now - 600,)).fetchone()[0],
        },
        "wind": current_wind(), "target": CFG["target_alt"],
    })


@app.route("/api/replay/<int:lid>")
def api_replay(lid):
    r = db().execute("""SELECT l.engine, l.air, l.traj, l.outcome, l.apo, l.wind, p.name FROM launches l
        JOIN players p ON p.id=l.player_id WHERE l.id=? AND p.hidden=0""", (lid,)).fetchone()
    if not r:
        abort(404)
    return jsonify({"engine": r["engine"], "air": json.loads(r["air"]), "tr": json.loads(r["traj"]),
                    "outcome": r["outcome"], "apo": r["apo"], "wind": json.loads(r["wind"]),
                    "name": mask_name(r["name"]), "row": json.loads(LUT[r["engine"]]),
                    "ox": KNOBS["ox"][int(r["engine"][0])]})


_tunnel = {"t": 0, "url": None}


def public_url():
    """터널 주소(data/tunnel_url, 정식 인증서)가 있으면 우선, 없으면 server_config 의 public_url"""
    now = time.time()
    if now - _tunnel["t"] > 10:
        try:
            u = open(os.path.join(DATA, "tunnel_url")).read().strip()
            _tunnel["url"] = u if u.startswith("https://") else None
        except OSError:
            _tunnel["url"] = None
        _tunnel["t"] = now
    return (_tunnel["url"] or CFG.get("public_url") or request.host_url).rstrip("/")


@app.route("/api/qr.svg")
def api_qr():
    q = segno.make(public_url() + "/", error="m")
    buf = io.BytesIO()
    q.save(buf, kind="svg", scale=8, border=2, dark="#000000", light="#ffffff")
    resp = Response(buf.getvalue(), mimetype="image/svg+xml")
    resp.headers["Cache-Control"] = "public, max-age=600"
    return resp


@app.route("/api/url")
def api_url():
    return jsonify({"url": public_url() + "/"})


# ── 관리자 ──────────────────────────────────────────────────────────────
def _admin_hash():
    try:
        return json.load(open(os.path.join(DATA, "admin.json")))["hash"]
    except (OSError, ValueError, KeyError):
        return None


@app.route("/api/admin/login", methods=["POST"])
def api_admin_login():
    ip = g.ip
    if LIM.locked("adm:" + ip) or LIM.locked("adm:all"):
        return err("관리자 로그인이 잠겼습니다. 잠시 후 다시 시도하세요.", 429)
    pw = str((request.get_json(silent=True) or {}).get("password", ""))
    h = _admin_hash()
    if not h or not check_password_hash(h, pw):
        LIM.fail("adm:" + ip, SEC["admin_fail_per_ip"], 1800, 1800)
        LIM.fail("adm:all", 20, 3600, 3600)
        STATS["admin_fail"] += 1
        log.warning("ADMIN LOGIN FAIL ip=%s", ip)
        time.sleep(1.0)
        return err("비밀번호가 맞지 않습니다.", 401)
    LIM.clear("adm:" + ip)
    session["admin_until"] = time.time() + 8 * 3600
    log.info("ADMIN LOGIN ip=%s", ip)
    return jsonify({"ok": True})


@app.route("/api/admin/logout", methods=["POST"])
def api_admin_logout():
    session.pop("admin_until", None)
    return jsonify({"ok": True})


@app.route("/api/admin/players")
@admin_required
def api_admin_players():
    rows = db().execute("""SELECT p.*, (SELECT COUNT(*) FROM launches l WHERE l.player_id=p.id AND official=1) off,
        (SELECT COUNT(*) FROM launches l WHERE l.player_id=p.id) total,
        (SELECT MAX(score_target) FROM launches l WHERE l.player_id=p.id AND official=1) best_t,
        (SELECT MAX(score_max) FROM launches l WHERE l.player_id=p.id AND official=1) best_m
        FROM players p ORDER BY p.id DESC""").fetchall()
    return jsonify([{k: r[k] for k in ("id", "student_id", "name", "dept", "phone", "created_at", "last_seen",
                                       "hidden", "off", "total", "best_t", "best_m")} for r in rows])


@app.route("/api/admin/player/<int:pid>", methods=["POST"])
@admin_required
def api_admin_player(pid):
    d = request.get_json(silent=True) or {}
    c = db()
    act = d.get("action")
    if act == "hide":
        c.execute("UPDATE players SET hidden=1-hidden WHERE id=?", (pid,))
    elif act == "reset_official":
        c.execute("UPDATE launches SET official=0 WHERE player_id=?", (pid,))
    elif act == "logout":
        c.execute("UPDATE players SET token_ver=token_ver+1 WHERE id=?", (pid,))
    else:
        return err("알 수 없는 작업입니다.")
    c.commit()
    log.info("ADMIN %s pid=%s", act, pid)
    return jsonify({"ok": True})


@app.route("/api/admin/stats")
@admin_required
def api_admin_stats():
    now = time.time()
    return jsonify({
        "uptime": round(now - START), "req_per_min": sum(1 for t in REQ_TIMES if t > now - 60), "stats": dict(STATS),
        "bans": [{"ip": ip, "until": int(b[0]), "reason": b[1]} for ip, b in list(LIM.bans.items()) if b[0] > now],
        "node_ms": round(sum(NODE_MS) / len(NODE_MS), 1) if NODE_MS else None, "wind": current_wind(),
        "lut": LUT_META, "target": CFG["target_alt"],
    })


@app.route("/api/admin/unban", methods=["POST"])
@admin_required
def api_admin_unban():
    ip = str((request.get_json(silent=True) or {}).get("ip", ""))
    LIM.bans.pop(ip, None)
    return jsonify({"ok": True})


@app.route("/api/admin/export.csv")
@admin_required
def api_admin_export():
    rows = db().execute("""SELECT p.id, p.student_id, p.name, p.dept, p.phone, datetime(p.created_at,'unixepoch','localtime') reg,
        (SELECT MAX(score_target) FROM launches l WHERE l.player_id=p.id AND official=1) best_target,
        (SELECT MAX(score_max) FROM launches l WHERE l.player_id=p.id AND official=1) best_max
        FROM players p ORDER BY best_target DESC""").fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "학번", "이름", "학과", "전화번호", "등록", "목표고도 최고점", "최고고도"])
    for r in rows:
        w.writerow(list(r))
    log.info("ADMIN export ip=%s", g.ip)
    return Response("﻿" + buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=tas1_players.csv"})
