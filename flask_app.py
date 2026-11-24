import os
import sys
import json
import time
import uuid
import copy
import hashlib
import threading
import zipfile
import io
import pandas as pd
from datetime import timedelta
from functools import wraps
from concurrent.futures import ProcessPoolExecutor, as_completed
from flask import (Flask, request, jsonify, send_from_directory,
                   send_file, session, redirect, url_for, render_template_string)
from flask_cors import CORS

from core.rocket import Rocket

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
JSON_PATH   = os.path.join(BASE_DIR, "rocket.json")
RESULT_CSV  = os.path.join(BASE_DIR, "result.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

# ── 보안 설정 ────────────────────────────────────────────────────────────────
app.secret_key = os.environ.get('SECRET_KEY', 'change-me-in-production-use-env-var')
app.permanent_session_lifetime = timedelta(hours=12)
# 배치 실험에서 다수의 케이스 JSON을 한 번에 POST하므로 넉넉하게 설정
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024  # 256 MB

# 로컬 실행(start_local.sh) 시 LOCAL_MODE=1 → 로그인 생략
LOCAL_MODE = os.environ.get('LOCAL_MODE') == '1'

PASSWORD_HASH = os.environ.get('PASSWORD_HASH', '')

LOGIN_HTML = '''<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>TUSI — Access</title>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <style>
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    html,body{height:100%;font-family:'IBM Plex Sans',sans-serif;background:#161616;color:#f4f4f4;-webkit-font-smoothing:antialiased}
    body{display:flex;flex-direction:column;align-items:center;justify-content:center}
    .login-wrap{width:100%;max-width:400px;padding:0 24px;animation:fadeUp .35s cubic-bezier(.25,.46,.45,.94) forwards;opacity:0}
    @keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
    .login-logo{display:flex;align-items:center;gap:10px;margin-bottom:40px}
    .login-logo__mark{height:32px;width:auto;opacity:0.9}
    .login-logo__name{font-size:16px;font-weight:600;letter-spacing:.16px}
    .login-logo__sub{font-size:11px;color:#a8a8a8;letter-spacing:.32px;margin-top:2px;font-family:"IBM Plex Mono",monospace}
    .login-title{font-size:28px;font-weight:300;margin-bottom:6px;letter-spacing:-.24px}
    .login-subtitle{font-size:13px;color:#a8a8a8;margin-bottom:32px;letter-spacing:.16px;line-height:1.6}
    .form-item{margin-bottom:20px}
    label{display:block;font-size:12px;color:#c6c6c6;letter-spacing:.32px;margin-bottom:8px}
    input[type="password"]{width:100%;height:48px;background:#262626;border:none;border-bottom:1px solid #6f6f6f;color:#f4f4f4;font-family:"IBM Plex Mono",monospace;font-size:14px;padding:0 16px;outline:none;letter-spacing:.32px;transition:border-color .15s,box-shadow .15s}
    input[type="password"]:focus{border-bottom-color:#ffffff;box-shadow:inset 0 -2px 0 0 #ffffff}
    input[type="password"]::placeholder{color:#a8a8a8}
    .error-msg{display:flex;align-items:center;gap:8px;background:#262626;border-left:3px solid #fa4d56;padding:12px 16px;font-size:13px;color:#fa4d56;margin-bottom:20px}
    button[type="submit"]{width:100%;height:48px;background:#0f62fe;color:#fff;border:none;font-family:"IBM Plex Sans",sans-serif;font-size:14px;font-weight:500;letter-spacing:.16px;cursor:pointer;transition:background .15s;margin-top:8px;display:flex;align-items:center;justify-content:center;gap:8px}
    button[type="submit"]:hover{background:#0050e6}
    button[type="submit"]:active{background:#002d9c}
    .login-footer{margin-top:40px;font-size:11px;color:#a8a8a8;letter-spacing:.32px;text-align:center;line-height:1.7}
    .login-footer strong{color:#f4f4f4}
  </style>
</head>
<body>
  <div class="login-wrap">
    <div class="login-logo">
      <img src="/static/logo.png" alt="" class="login-logo__mark"/>
      <div><div class="login-logo__name">TUSI</div><div class="login-logo__sub">HYBRID ROCKET ENGINE SIMULATOR</div></div>
    </div>
    <div class="login-title">Access Required</div>
    <div class="login-subtitle">이 시스템은 인가된 사용자만 접근할 수 있습니다.<br>액세스 패스워드를 입력하세요.</div>
    {% if error %}
    <div class="error-msg">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="#fa4d56"><path d="M8 1C4.1 1 1 4.1 1 8s3.1 7 7 7 7-3.1 7-7-3.1-7-7-7zm-.75 3h1.5v5h-1.5V4zm.75 7a1 1 0 110-2 1 1 0 010 2z"/></svg>
      {{ error }}
    </div>
    {% endif %}
    <form method="POST" action="/login">
      <div class="form-item">
        <label for="password">ACCESS PASSWORD</label>
        <input type="password" id="password" name="password" placeholder="••••••••" autofocus autocomplete="current-password"/>
      </div>
      <button type="submit">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M11 2a2 2 0 012 2v1h1v9H2V5h1V4a2 2 0 012-2h6zm1 5H4v6h8V7zm-4 1a1.5 1.5 0 110 3 1.5 1.5 0 010-3zM9 4H7v1h2V4z"/></svg>
        Authenticate
      </button>
    </form>
    <div class="login-footer">
      <strong>전략 물자 관련 법률에 따라 비밀번호를 걸어두었습니다.<br>사용을 희망하시는 분은 이종원에게 개별 문의 주시기 바랍니다.</strong>
    </div>
  </div>
</body>
</html>'''


# ── 인증 데코레이터 ───────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not LOCAL_MODE and not session.get('authenticated'):
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ── 인증 라우트 ───────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if LOCAL_MODE or session.get('authenticated'):
        return redirect('/')
    error = None
    if request.method == 'POST':
        pw      = request.form.get('password', '')
        pw_hash = hashlib.sha256(pw.encode('utf-8')).hexdigest()
        if pw_hash == PASSWORD_HASH:
            session.permanent = True
            session['authenticated'] = True
            return redirect('/')
        time.sleep(0.3)
        error = '패스워드가 올바르지 않습니다.'
    return render_template_string(LOGIN_HTML, error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ── 기존 단일 시뮬레이션 ──────────────────────────────────────────────────────
DP_COLUMN = 'Injector dP/Pc (%)'


def _add_injector_dp(df):
    """
    인젝터 차압 비율 (Pt - Pc)/Pc [%] 열을 추가하고 연소 구간 요약을 반환.
    연소 구간 = 추력이 최대의 5% 이상 (optimizer/GA.py BURN_THRESHOLD 와 같음). 하한은 constrain.MIN_INJECTOR_DP_RATIO.
    """
    try:
        _mdo_modules()
        import constrain as cs
        pt, pc, thrust = df['Tank Pressure (Pa)'], df['Chamber Pressure (Pa)'], df['Thrust (N)']
        burn = (thrust >= 0.05 * thrust.max()) & (pc > 0)
        df[DP_COLUMN] = ((pt - pc) / pc * 100).where(burn)
        ratio = df[DP_COLUMN].dropna() / 100
        if ratio.empty:
            return None
        return {'min_ratio': float(ratio.min()), 'avg_ratio': float(ratio.mean()),
                'min_time': float(df.loc[ratio.idxmin(), 'Time (s)']),
                'limit': cs.MIN_INJECTOR_DP_RATIO, 'ok': bool(ratio.min() >= cs.MIN_INJECTOR_DP_RATIO)}
    except (KeyError, ImportError) as exc:
        print(f"[injector dp] skipped: {exc}")
        return None


def _steady_state(df, dt):
    """정상상태 구간 평균 추력·평균 비추력 (optimizer/GA.py steady_state_metrics 와 같은 정의)"""
    try:
        GA, _, _ = _mdo_modules()
        return GA.steady_state_metrics(df['Thrust (N)'].to_numpy(),
                                       (df['mox_dot (kg/s)'] + df['mf_dot (kg/s)']).to_numpy(), dt)
    except (KeyError, ImportError) as exc:
        print(f"[steady state] skipped: {exc}")
        return None


@app.route('/api/simulate', methods=['POST'])
@login_required
def simulate():
    start_total = time.time()
    try:
        config = request.get_json()
        if not config:
            return jsonify({'error': 'No configuration data provided'}), 400

        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            rocket_config = json.load(f)

        sim_start  = time.time()
        target_key = "TAS-I" if "TAS-I" in rocket_config else list(rocket_config.keys())[0]
        inner_cfg  = rocket_config[target_key]

        rocket1 = Rocket(inner_cfg)
        df      = rocket1.ignite()
        dp_info = _add_injector_dp(df)
        ss_info = _steady_state(df, float(inner_cfg.get('dt', 0.01)))
        df.to_csv(RESULT_CSV, index=False)
        sim_end = time.time()

        grain_data = {'snapshots': [], 'extent': float(inner_cfg.get('outter_radius', 0.05)), 'port_centers': []}
        try:
            fuel = getattr(rocket1, 'fuel', None)
            if fuel is not None:
                if callable(getattr(fuel, 'get_snapshots', None)):
                    grain_data['snapshots'] = fuel.get_snapshots()
                if callable(getattr(fuel, 'get_grain_meta', None)):
                    meta = fuel.get_grain_meta()
                    grain_data['extent']       = meta.get('extent', grain_data['extent'])
                    grain_data['port_centers'] = meta.get('port_centers', [])
        except Exception as grain_err:
            print(f"[grain] snapshot collection skipped: {grain_err}")

        data       = df.to_dict(orient='list')
        if DP_COLUMN in data:   # 연소 구간 밖은 NaN → JSON 에 NaN 을 못 넣으므로 null (차트에서 끊김)
            data[DP_COLUMN] = [None if v != v else v for v in data[DP_COLUMN]]
        numeric_df = df.select_dtypes(include=['number'])
        stats      = {}
        if not numeric_df.empty:
            summary = numeric_df.describe()
            for col in numeric_df.columns:
                if col == 'Time (s)':
                    continue
                stats[col] = {
                    'min': round(float(summary.at['min',  col]), 6),
                    'max': round(float(summary.at['max',  col]), 6),
                    'avg': round(float(summary.at['mean', col]), 6),
                }

        end_total = time.time()
        return jsonify({
            'success':   True,
            'timing': {
                'total':           f"{round(end_total  - start_total, 3)}s",
                'simulation_only': f"{round(sim_end    - sim_start,   3)}s",
                'data_processing': f"{round(end_total  - sim_end,     3)}s",
            },
            'columns':      list(df.columns),
            'data':         data,
            'stats':        stats,
            'row_count':    len(df),
            'grain':        grain_data,
            'total_impulse': round(float(getattr(rocket1.engine, 'impulse', 0.0)), 6),
            'injector_dp':  dp_info,
            'steady_state': ss_info,
            'termination': {
                'code':    getattr(rocket1, 'termination_code', None),
                'message': getattr(rocket1, 'termination_message', None),
            },
        })
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(error_msg)
        return jsonify({'error': str(e), 'details': error_msg}), 500


@app.route('/')
@login_required
def index():
    return send_from_directory(os.path.join(BASE_DIR, 'static'), 'index.html')


@app.route('/api/default-config')
@login_required
def default_config():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({}), 404


@app.route('/api/download-csv', methods=['GET'])
@login_required
def download_csv():
    if os.path.exists(RESULT_CSV):
        return send_file(RESULT_CSV, as_attachment=True, download_name='simulation_result.csv')
    return jsonify({'error': 'No result CSV available'}), 404


# ═══════════════════════════════════════════════════════════════════════════════
#  배치 실험 (Experiment Runner)
#
#  [병렬화 설계]
#  ThreadPoolExecutor → Python GIL 로 인해 CPU-바운드 코드는 사실상 직렬 실행.
#  ProcessPoolExecutor → 별도 OS 프로세스이므로 GIL 없이 진정한 병렬 실행 가능.
#
#  [상태 업데이트 방식]
#  프로세스 간 메모리 공유 불가 → 워커는 결과 dict 만 반환.
#  as_completed() 로 완료 순서대로 메인 스레드의 _jobs 딕셔너리를 갱신한다.
#  케이스 제출 즉시 "running" 으로 표시해 UI 에 실시간 상태 반영.
# ═══════════════════════════════════════════════════════════════════════════════

_jobs: dict = {}
_jobs_lock  = threading.Lock()


# ── 모듈 최상위 워커 함수 (ProcessPoolExecutor pickle 요건) ───────────────────
def _worker_process(case_name: str, case_config: dict, results_dir: str) -> dict:
    """
    별도 OS 프로세스에서 실행되는 시뮬레이션 워커.

    - GIL 없이 병렬 실행 → 워커 수만큼 진짜 속도 향상.
    - 결과 DataFrame은 CSV 로만 저장, 반환 dict 에는 메타데이터만 포함.
    - CSV 내용은 /api/experiment/download/<filename> 명시적 요청 시에만 전송.
    """
    started = time.time()
    try:
        # 자식 프로세스에서 안전하게 임포트
        from core.rocket import Rocket as _Rocket
        rocket    = _Rocket(case_config)
        df        = rocket.ignite()
        safe      = "".join(c if c.isalnum() or c in '-_.' else '_' for c in case_name)
        csv_fn    = f"{safe}.csv"
        df.to_csv(os.path.join(results_dir, csv_fn), index=False)
        row_count = len(df)
        del df   # 메모리 즉시 해제 — 부모 프로세스로 전송하지 않음
        return {
            'name':        case_name,
            'status':      'done',
            'started_at':  started,
            'finished_at': time.time(),
            'row_count':   row_count,
            'csv_name':    csv_fn,
            'error':       None,
        }
    except Exception as exc:
        return {
            'name':        case_name,
            'status':      'error',
            'started_at':  started,
            'finished_at': time.time(),
            'row_count':   0,
            'csv_name':    None,
            'error':       str(exc)[:400],
        }


def _warmup_rocketcea():
    """
    병렬 프로세스 시작 전 ~/RocketCEA/ 디렉터리를 미리 생성한다.
    RocketCEA 는 첫 실행 시 이 디렉터리에 바이너리를 복사하는데,
    여러 프로세스가 동시에 시도하면 [Errno 17] File exists 가 발생한다.
    부모 프로세스에서 미리 만들어두면 자식들은 이미 존재하는 것을 보고 건너뛴다.
    """
    try:
        cea_dir = os.path.join(os.path.expanduser('~'), 'RocketCEA')
        os.makedirs(cea_dir, exist_ok=True)
        print(f"[warmup] RocketCEA dir ready: {cea_dir}")
    except Exception as e:
        print(f"[warmup] warning: {e}")


def _cleanup_old_csvs():
    """3일 이상 지난 결과 CSV 자동 삭제."""
    cutoff = time.time() - 3 * 24 * 3600
    try:
        for fname in os.listdir(RESULTS_DIR):
            if fname.endswith('.csv'):
                fpath = os.path.join(RESULTS_DIR, fname)
                try:
                    if os.path.getmtime(fpath) < cutoff:
                        os.remove(fpath)
                        print(f"[cleanup] deleted old result: {fname}")
                except Exception:
                    pass
    except Exception:
        pass


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in '-_.' else '_' for c in name)


def _run_job(job_id: str, cases: list, num_workers: int):
    """
    백그라운드 스레드에서 실행.
    ProcessPoolExecutor 로 케이스를 OS 프로세스 단위로 병렬 실행하고,
    as_completed() 로 완료 순서대로 _jobs 딕셔너리를 갱신한다.
    """
    with _jobs_lock:
        _jobs[job_id]['status']     = 'running'
        _jobs[job_id]['started_at'] = time.time()

    # RocketCEA 디렉터리를 먼저 생성 (자식 프로세스 Errno 17 방지)
    _warmup_rocketcea()

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        future_map: dict = {}

        # 모든 케이스를 한꺼번에 제출 (num_workers 만큼 즉시 병렬 시작)
        for name, cfg in cases:
            fut = pool.submit(_worker_process, name, cfg, RESULTS_DIR)
            future_map[fut] = name
            # 제출 즉시 "running" 으로 표시 → UI 에서 실시간 확인 가능
            with _jobs_lock:
                _jobs[job_id]['cases'][name].update({
                    'status':     'running',
                    'started_at': time.time(),
                })

        # 완료되는 순서대로 결과 반영
        for fut in as_completed(future_map):
            try:
                result = fut.result()
            except Exception as exc:
                # 워커 프로세스 자체가 크래시한 경우
                name = future_map[fut]
                result = {
                    'name':        name,
                    'status':      'error',
                    'started_at':  None,
                    'finished_at': time.time(),
                    'row_count':   0,
                    'csv_name':    None,
                    'error':       f"Worker crash: {str(exc)[:300]}",
                }
            with _jobs_lock:
                _jobs[job_id]['cases'][result['name']].update({
                    'status':      result['status'],
                    'finished_at': result['finished_at'],
                    'row_count':   result['row_count'],
                    'csv_name':    result['csv_name'],
                    'error':       result['error'],
                })

    with _jobs_lock:
        _jobs[job_id]['status']      = 'done'
        _jobs[job_id]['finished_at'] = time.time()


def _build_job_summary(job: dict) -> dict:
    """잡 목록 API용 — 케이스별 상세 없이 전체 요약만 반환."""
    cases  = job.get('cases', {})
    counts = {'pending': 0, 'running': 0, 'done': 0, 'error': 0}
    for c in cases.values():
        st = c.get('status', 'pending')
        counts[st] = counts.get(st, 0) + 1
    return {
        'id':          job['id'],
        'status':      job['status'],
        'created_at':  job['created_at'],
        'started_at':  job['started_at'],
        'finished_at': job['finished_at'],
        'num_workers': job['num_workers'],
        'total':       job['total'],
        'counts':      counts,
        'case_names':  list(cases.keys()),
    }


def _build_job_status(job: dict) -> dict:
    """
    폴링 API용 — 케이스별 완료 여부 + 최소 메타데이터만 반환.
    실제 시뮬레이션 데이터(CSV 내용)는 절대 포함하지 않는다.
    """
    cases_out = {}
    for name, c in job.get('cases', {}).items():
        cases_out[name] = {
            'status':      c.get('status'),
            'started_at':  c.get('started_at'),
            'finished_at': c.get('finished_at'),
            'row_count':   c.get('row_count', 0),
            'csv_name':    c.get('csv_name'),
            'error':       c.get('error'),
        }
    return {
        'id':          job['id'],
        'status':      job['status'],
        'created_at':  job['created_at'],
        'started_at':  job['started_at'],
        'finished_at': job['finished_at'],
        'num_workers': job['num_workers'],
        'total':       job['total'],
        'cases':       cases_out,
    }


# 서버 시작 시 오래된 파일 정리
# (GA 워커는 spawn 으로 뜨면서 이 파일을 __mp_main__ 으로 다시 import 하므로 그때는 건너뛴다)
if __name__ != '__mp_main__':
    _cleanup_old_csvs()


# ── Experiment 페이지 ─────────────────────────────────────────────────────────
@app.route('/experiment')
@login_required
def experiment_page():
    return send_from_directory(os.path.join(BASE_DIR, 'static'), 'experiment.html')


# ── Experiment API ────────────────────────────────────────────────────────────

@app.route('/api/experiment/jobs')
@login_required
def experiment_jobs_list():
    """서버의 모든 잡 요약 목록 — 탭 재접속 시 잡 복구에 사용."""
    with _jobs_lock:
        jobs_copy = copy.deepcopy(_jobs)
    summaries = [_build_job_summary(j) for j in jobs_copy.values()]
    summaries.sort(key=lambda j: j['created_at'], reverse=True)
    return jsonify({'jobs': summaries})


@app.route('/api/experiment/submit', methods=['POST'])
@login_required
def experiment_submit():
    _cleanup_old_csvs()
    data = request.get_json(silent=True)
    if not data or 'cases' not in data:
        return jsonify({'error': 'cases 필드가 필요합니다'}), 400

    cases_dict  = data['cases']
    num_workers = max(1, min(int(data.get('num_workers', 4)), 32))
    if not cases_dict:
        return jsonify({'error': '케이스가 비어 있습니다'}), 400

    job_id     = str(uuid.uuid4())
    cases_list = list(cases_dict.items())

    with _jobs_lock:
        _jobs[job_id] = {
            'id':          job_id,
            'status':      'pending',
            'created_at':  time.time(),
            'started_at':  None,
            'finished_at': None,
            'num_workers': num_workers,
            'total':       len(cases_list),
            'cases': {
                name: {
                    'status':      'pending',
                    'started_at':  None,
                    'finished_at': None,
                    'row_count':   0,
                    'csv_name':    None,
                    'error':       None,
                }
                for name in cases_dict
            }
        }

    threading.Thread(target=_run_job, args=(job_id, cases_list, num_workers), daemon=True).start()
    return jsonify({'job_id': job_id, 'total': len(cases_list)})


@app.route('/api/experiment/status/<job_id>')
@login_required
def experiment_status(job_id):
    """케이스별 완료 여부 + 메타데이터만 반환. CSV 내용 미포함."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(_build_job_status(copy.deepcopy(job)))


@app.route('/api/experiment/results')
@login_required
def experiment_results():
    files = []
    try:
        for fname in sorted(os.listdir(RESULTS_DIR)):
            if fname.endswith('.csv'):
                fpath = os.path.join(RESULTS_DIR, fname)
                stat  = os.stat(fpath)
                files.append({'name': fname, 'size': stat.st_size, 'mtime': stat.st_mtime})
    except Exception:
        pass
    return jsonify({'files': files})


@app.route('/api/experiment/download/<path:filename>')
@login_required
def experiment_download(filename):
    """CSV 파일 본체는 명시적 다운로드 요청 시에만 전송."""
    safe  = os.path.basename(filename)
    fpath = os.path.join(RESULTS_DIR, safe)
    if not os.path.isfile(fpath):
        return jsonify({'error': 'File not found'}), 404
    return send_file(fpath, as_attachment=True, download_name=safe)


@app.route('/api/experiment/download-all')
@login_required
def experiment_download_all():
    buf = io.BytesIO()
    cnt = 0
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(RESULTS_DIR)):
            if fname.endswith('.csv'):
                zf.write(os.path.join(RESULTS_DIR, fname), fname)
                cnt += 1
    if cnt == 0:
        return jsonify({'error': '다운로드할 결과 파일이 없습니다'}), 404
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name='tusi_experiment_results.zip',
                     mimetype='application/zip')


@app.route('/api/experiment/delete/<path:filename>', methods=['DELETE'])
@login_required
def experiment_delete(filename):
    safe  = os.path.basename(filename)
    fpath = os.path.join(RESULTS_DIR, safe)
    if not os.path.isfile(fpath):
        return jsonify({'error': 'File not found'}), 404
    os.remove(fpath)
    return jsonify({'deleted': safe})


# ═══════════════════════════════════════════════════════════════════════════════
#  MDO (유전 알고리즘 형상 최적화)
#  optimizer/GA.py 를 백그라운드 스레드에서 실행하고, 진행 이벤트를 _mdo_job 에
#  누적해 /api/mdo/status 폴링으로 GUI 에 전달한다. 동시에 1개 잡만 실행.
# ═══════════════════════════════════════════════════════════════════════════════

OPTIMIZER_DIR = os.path.join(BASE_DIR, "optimizer")

_mdo_lock = threading.Lock()
_mdo_job  = None
_mdo_stop = threading.Event()

MDO_LOG_LIMIT = 300
# 목적 공간 점 상한. 대리모델 모드는 세대가 많아 점이 수만 개 쌓인다 (88k 점 = 상태 응답 10 MB → 웹 렉).
# 넘으면 절반으로 솎고 points_epoch 를 올려 브라우저가 처음부터 다시 받게 한다.
MDO_POINTS_LIMIT = 3000
# 상태 응답에 보내는 세대 기록 수 상한 (그래프·세대 막대용으로 고르게 추림, 마지막 세대는 항상 포함)
MDO_HISTORY_LIMIT = 400
# 제약 만족 형상 목록 상한 (대리모델 모드는 예측 합격이 수천 개 → core 검증 안 된 하위 예측부터 버림)
MDO_FEASIBLE_LIMIT = 400
VERIFY_RANK = {"pass": 0, "pending": 1, "none": 2, "fail": 3}


def _feasible_entry(job, fid, summary, source, generation):
    entry = job["feasible"].get(fid)
    if entry is None:
        entry = {"id": fid, "generation": generation, "source": source,
                 "verify": "pass" if source == "core" else "none", "pred": summary, "core": None}
        job["feasible"][fid] = entry
        if len(job["feasible"]) > MDO_FEASIBLE_LIMIT:
            # 검증 안 된 예측 중 Isp 가 가장 낮은 것부터 버린다
            drop = sorted((e for e in job["feasible"].values() if e["verify"] == "none" and e["id"] != fid),
                          key=lambda e: e["pred"].get("isp") or 0)
            for e in drop[:len(job["feasible"]) - MDO_FEASIBLE_LIMIT]:
                del job["feasible"][e["id"]]
    return entry


def _feasible_row(e):
    """목록용 요약 (상세는 /api/mdo/design/<id>)"""
    s = e["core"] if e["core"] is not None else e["pred"]
    d = s.get("design") or {}
    return {"id": e["id"], "generation": e["generation"], "source": e["source"], "verify": e["verify"],
            "pipe_id": s.get("pipe_id"), "isp": s.get("isp"), "impulse": s.get("impulse"),
            "thrust_avg": s.get("thrust_avg"), "dp_ratio_min": s.get("dp_ratio_min"), "burn_time": s.get("burn_time"),
            "pred_isp": e["pred"].get("isp"), "failed": e["core"]["failed"] if e["core"] else [],
            # 갤러리 그림용 치수 [m]
            "geom": {k: d.get(k) for k in ("oxidizer_tank_radius", "combution_chamber_radius", "oxidizer_tank_length",
                                           "oxidizer_tank_tickness", "combution_chamber_length", "fuel_length",
                                           "outter_radius", "inner_radius", "port_num", "port_center_radius",
                                           "nozzle_throat_radius", "nozzle_exit_radius", "oxidizer_M")}}


def _mdo_modules(reload=False):
    """optimizer 모듈 로드. reload=True 면 GA.py/constrain.py 수정 사항을 다시 읽는다."""
    import importlib
    if OPTIMIZER_DIR not in sys.path:
        sys.path.insert(0, OPTIMIZER_DIR)
    import constrain, managepipedb, GA
    if reload:
        constrain = importlib.reload(constrain)
        managepipedb = importlib.reload(managepipedb)
        GA = importlib.reload(GA)
        # rl 모듈도 의존 순서대로 다시 읽는다 (GA 만 새로 읽으면 새 GA 가 예전 rl 클래스를 부르는 일이 생김)
        for name in ("rl.state", "rl.actions", "rl.operators", "rl.surrogate", "rl.policy",
                     "rl.ts_surrogate", "rl.direct", "rl.curve_agent"):
            if name in sys.modules:
                importlib.reload(sys.modules[name])
    return GA, constrain, managepipedb


def _mdo_log(job, msg):
    job["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    if len(job["log"]) > MDO_LOG_LIMIT:
        del job["log"][:-MDO_LOG_LIMIT]


def _mdo_on_progress(job, event, data):
    with _mdo_lock:
        if event == "start":
            job.update(pop_size=data["pop_size"], n_generations=data["n_generations"],
                       n_workers=data["n_workers"], pipes=data["pipes"])
            _mdo_log(job, f"GA 시작 — 파이프 {len(data['pipes'])}종, 개체 {data['pop_size']}, "
                          f"최대 {data['n_generations']}세대, 워커 {data['n_workers']}")

        elif event == "generation_start":
            g = data["generation"]
            job["generation"] = g
            job["population"] = [{"index": i, "state": "pending"} for i in range(job["pop_size"])]
            _mdo_log(job, f"세대 {g + 1} 평가 시작")

        elif event == "candidate":
            i, s = data["index"], data["summary"]
            cell = {"index": i, "state": data["state"]}
            if s:
                cell.update({k: s[k] for k in ("pipe_id", "isp", "impulse", "thrust_avg", "dp_ratio_min",
                                                "violation", "failed", "error")})
                key = json.dumps(s["design"], sort_keys=True) + s["pipe_id"]
                if key not in job["_seen"]:
                    job["_seen"].add(key)
                    for name in s["failed"]:
                        job["fail_counts"][name] = job["fail_counts"].get(name, 0) + 1
                    if s["impulse"] is not None:
                        job["points"].append({
                            "gen": data["generation"], "impulse": s["impulse"],
                            "thrust_avg": s["thrust_avg"], "isp": s["isp"],
                            "feasible": s["state"] == "feasible",
                        })
                        if len(job["points"]) > MDO_POINTS_LIMIT:
                            job["points"] = job["points"][::2]
                            job["points_epoch"] += 1
            if i < len(job["population"]):
                job["population"][i] = cell

        elif event == "generation_end":
            h = data["history"]
            job["history"].append(h)
            for rank, i in enumerate(data["ranking"]):
                job["population"][i]["rank"] = rank
            for i in data["elite"]:
                job["population"][i]["elite"] = True
            prev = job.get("best")
            job["best"] = data["best"]
            if data["best"]["state"] == "feasible" and (
                    not prev or prev.get("isp") is None or prev["state"] != "feasible"
                    or data["best"]["isp"] > prev["isp"]):
                job["best_improved_at"] = h["generation"]
            best_txt = (f"최고 Isp {h['best_isp']:.2f} s" if h["best_isp"] is not None
                        else f"합격 해 없음 (최소 위반 {h['best_violation']:.3f})")
            _mdo_log(job, f"세대 {h['generation'] + 1} 완료 — 합격 {h['n_feasible']}/{job['pop_size']}, "
                          f"{best_txt}, {h['duration']:.1f}s")

        elif event == "strategy" and "curve_agent" in data:
            d = data["curve_agent"]
            desc = data["name"].split(": ", 1)[-1]
            job["curve_log"].append([data["generation"], desc, d["counts"]])
            if len(job["curve_log"]) > 400:
                job["curve_log"] = job["curve_log"][-400:]
            job["curve_last"] = dict(d, generation=data["generation"], desc=desc)
            if data["generation"] % 10 == 1 or data["generation"] <= 1:
                top = sorted(d["counts"].items(), key=lambda kv: -kv[1])[:3]
                _mdo_log(job, f"세대 {data['generation'] + 1} 판단 — 최고 개체: {desc} · 개체군 선택 상위: "
                              + ", ".join(f"{k} {v}" for k, v in top))

        elif event == "strategy" and "direct" in data:
            # 직접 설계 에이전트: 세대마다 스텝 크기·파이프 변경·변수별 이동량, 최고 개체의 예측 연소 곡선
            d = data["direct"]
            job["direct_steps"].append([data["generation"], round(d["step_mean"], 5), round(d["step_min"], 5),
                                        round(d["step_max"], 5), d["pipe_changes"]])
            job["direct_last"] = dict(d, generation=data["generation"])
            if data["generation"] % 10 == 1 or data["generation"] <= 1:
                _mdo_log(job, f"세대 {data['generation'] + 1} 에이전트 이동: 평균 스텝 {d['step_mean'] * 100:.2f}% "
                              f"(최소 {d['step_min'] * 100:.2f}%, 최대 {d['step_max'] * 100:.2f}%), 파이프 변경 {d['pipe_changes']}개")

        elif event == "strategy":
            job["strategies"].append({"generation": data["generation"], "action": data["action"],
                                      "name": data["name"]})
            _mdo_log(job, f"세대 {data['generation'] + 1} 전략 (PPO): {data['name']}")

        elif event == "feasible":
            _feasible_entry(job, data["id"], data["summary"], data["source"], data["generation"])

        elif event == "verify_submit":
            e = job["feasible"].get(data["id"])
            if e and e["verify"] == "none":
                e["verify"] = "pending"

        elif event == "verified_live":
            e = _feasible_entry(job, data["id"], data["surrogate"], "surrogate", job["generation"])
            e["core"] = data["core"]
            e["verify"] = "pass" if data["core"]["state"] == "feasible" else "fail"
            prev = job.get("best_core")
            job["best_core"] = data["best_core"]
            c = data["core"]
            if c["state"] == "feasible" and (not prev or prev.get("state") != "feasible" or c["isp"] > prev["isp"]):
                _mdo_log(job, f"core 검증 최고 갱신: Isp {c['isp']:.2f} s (예측 {data['surrogate']['isp']:.2f} s) · {c['pipe_id']}")

        elif event == "feasible_parents":
            job["parents_feasible_from"] = data["generation"] + 1
            _mdo_log(job, f"합격 형상 {data['n']}개 확보 → 세대 {data['generation'] + 2}부터 합격 형상끼리만 교배")

        elif event == "robust_pick":
            job["robust"] = {k: data[k] for k in ("n_plateau", "band", "top_margin", "pick_margin", "pick_margins")}
            job["robust"].update(top_isp=data["top"]["isp"], pick_isp=data["pick"]["isp"], same=data["top"]["id"] == data["pick"]["id"])
            p = data["pick"]
            if data["top"]["id"] != p["id"]:
                e = _feasible_entry(job, p["id"], p, "core", job["generation"])
            _mdo_log(job, f"견고성 선택: Isp {data['top']['isp'] - data['band']:.2f} s 이상 합격 형상 {data['n_plateau']}개 중 "
                          f"제약 여유 최대 → Isp {p['isp']:.2f} s (여유 {data['pick_margin']:.2f}) · "
                          f"최고 Isp 형상 {data['top']['isp']:.2f} s (여유 {data['top_margin']:.2f})")

        elif event == "trim":
            job["trim"] = {k: data[k] for k in ("accepted", "old_length", "new_length")}
            job["trim"].update(isp_before=data["before"]["isp"], isp_after=data["after"]["isp"],
                               failed=data["after"]["failed"])
            a = data["after"]
            if data["accepted"]:
                job["best_core"] = a
                e = _feasible_entry(job, a["id"], a, "core", job["generation"])
                e["trimmed"] = True
            _mdo_log(job, f"연소 챔버 길이 정리: {data['old_length'] * 1e3:.1f} → {data['new_length'] * 1e3:.1f} mm "
                          f"(연료+2R ×1.3) · core Isp {data['before']['isp']:.2f} → {(a['isp'] or 0):.2f} s · "
                          f"{'채택' if data['accepted'] else '불합격(' + ','.join(a['failed']) + ') → 원래 길이 유지'}")

        elif event == "verify_start":
            job["verify_n"] = data["n"]
            _mdo_log(job, f"대리모델 상위 후보를 core 시뮬레이션으로 검증 시작 ({data.get('round', 1)}회차, 누적 {data['n']}개)")

        elif event == "correction":
            job["corrections"].append({"round": data["round"], "bias": data["bias"]})
            job["n_generations"] = data["n_generations"]
            b = data["bias"]
            _mdo_log(job, f"검증 합격 없음 → core 결과로 대리모델 편향 보정 (평균 추력 {b.get('thrust_avg', 0):+.1f} N, "
                          f"임펄스 {b.get('impulse', 0):+.0f} N·s, 차압 {b.get('dp_ratio_min', 0) * 100:+.1f}%p) "
                          f"후 {data['n_generations']}세대까지 재탐색")

        elif event == "verify":
            job["verified"].append({"rank": data["rank"], "surrogate": data["surrogate"], "core": data["core"]})
            e = job["feasible"].get(data["id"])
            if e is None and data["core"]["state"] == "feasible":      # 예측은 불합격이었지만 core 는 합격
                e = _feasible_entry(job, data["id"], data["surrogate"], "surrogate", job["generation"])
            if e is not None:
                e["core"] = data["core"]
                e["verify"] = "pass" if data["core"]["state"] == "feasible" else "fail"
            c = data["core"]
            if c["state"] == "feasible" and (not job.get("best_core") or job["best_core"].get("state") != "feasible"
                                              or c["isp"] > job["best_core"]["isp"]):
                job["best_core"] = c
            c = data["core"]
            _mdo_log(job, f"검증 {data['rank'] + 1}/{job['verify_n']}: 예측 Isp {data['surrogate']['isp']:.2f} → "
                          f"core {c['isp'] or 0:.2f} s, {'합격' if c['state'] == 'feasible' else '불합격 ' + ','.join(c['failed'])}")

        elif event == "done":
            job["_seen"].clear()          # 중복 판별용 설계 키 (수만 개) — 끝나면 필요 없음
            job["status"] = "stopped" if data["stopped"] else "done"
            if data.get("verified"):
                job["best_source"] = "core"
            job["output_dir"] = data["output_dir"]
            if data["best"]:
                job["best"] = data["best"]
            job["finished_at"] = time.time()
            _mdo_log(job, "사용자 요청으로 중단" if data["stopped"] else "최적화 완료")


def _mdo_run(job):
    try:
        GA, _, _ = _mdo_modules(reload=True)
        strategy, surrogate = None, None
        if job["evaluator"] == "surrogate" and job["mode"] in ("direct", "curve"):
            from rl.ts_surrogate import TSSurrogateSimulator
            surrogate = TSSurrogateSimulator()
            with _mdo_lock:
                _mdo_log(job, "평가: 시계열 대리모델 (rl/ts_surrogate.pt, 연소 전 과정 예측) — 끝난 뒤 상위 후보를 core 시뮬레이션으로 검증")
        elif job["evaluator"] == "surrogate":
            from rl.surrogate import SurrogateSimulator
            surrogate = SurrogateSimulator()
            with _mdo_lock:
                _mdo_log(job, "평가: 대리모델 (rl/surrogate.pt) — 끝난 뒤 상위 후보를 core 시뮬레이션으로 검증")
        if job["mode"] == "ppo":
            from rl.policy import PPOStrategy, resolve_model
            # argmax: 가장 확률 높은 전략 / 확률적(기본): 정책 분포에서 뽑는다 (seed 가 있으면 재현 가능)
            strategy = PPOStrategy(resolve_model(job["model"]), deterministic=bool(job["argmax"]),
                                   seed=job["settings"].get("seed"))
            with _mdo_lock:
                _mdo_log(job, f"PPO 정책 로드: {job['model']} — 전략 선택 {'argmax' if job['argmax'] else '확률적'}")
        elif job["mode"] == "direct":
            from rl.policy import resolve_model
            from rl.direct import DirectStrategy
            strategy = DirectStrategy(resolve_model(job["model"]), deterministic=bool(job["argmax"]),
                                      seed=job["settings"].get("seed"))
            with _mdo_lock:
                _mdo_log(job, f"직접 설계 에이전트 로드: {job['model']} — 개체마다 에이전트가 설계 변수를 직접 조정 "
                              f"(교배·선택 없음, 행동 {'평균값' if job['argmax'] else '확률적 샘플'}) · 관측: 시계열 대리모델의 연소 곡선 경향성")
        elif job["mode"] == "curve":
            from rl.policy import resolve_model
            from rl.curve_agent import CurveStrategy
            strategy = CurveStrategy(resolve_model(job["model"]), deterministic=bool(job["argmax"]),
                                     seed=job["settings"].get("seed"))
            with _mdo_lock:
                _mdo_log(job, f"곡선 판단 에이전트 로드: {job['model']} — 개체마다 예측 연소 곡선(추력·압력·Isp(t)·O/F)을 보고 "
                              f"어떤 변수를 얼마나 바꿀지 판단 (이동 {'가장 확률 높은 것' if job['argmax'] else '확률적 샘플'})")
        GA.run_ga(job["targets"]["impulse"], job["targets"]["thrust_avg"],
                  pop_size=job["settings"].get("pop_size"),
                  n_generations=job["settings"].get("n_generations"),
                  seed=job["settings"].get("seed"),
                  progress=lambda e, d: _mdo_on_progress(job, e, d),
                  should_stop=_mdo_stop.is_set,
                  strategy=strategy,
                  surrogate=surrogate,
                  verify_top=job["settings"].get("verify_top"),
                  feasible_parents=job["settings"].get("feasible_parents", False),
                  robust_pick=job["settings"].get("robust_pick", False),
                  robust_band=job["settings"].get("robust_band"),
                  tolerance=job["settings"].get("tolerance"))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        with _mdo_lock:
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
            job["finished_at"] = time.time()
            _mdo_log(job, f"오류: {job['error']}")


@app.route('/mdo')
@login_required
def mdo_page():
    return send_from_directory(os.path.join(BASE_DIR, 'static'), 'mdo.html')


@app.route('/surrogate')
@login_required
def surrogate_page():
    return send_from_directory(os.path.join(BASE_DIR, 'static'), 'surrogate.html')


@app.route('/drawing')
@login_required
def drawing_page():
    return send_from_directory(os.path.join(BASE_DIR, 'static'), 'drawing.html')


@app.route('/report')
@login_required
def report_page():
    return send_from_directory(os.path.join(BASE_DIR, 'static'), 'report.html')


@app.route('/api/mdo/meta')
@login_required
def mdo_meta():
    """GUI 초기값: GA 기본 하이퍼파라미터, 파이프 DB, 고정값"""
    try:
        GA, cs, pdb = _mdo_modules(reload=True)
        from rl.policy import list_models
        from rl.actions import ACTIONS
        from rl import direct, curve_agent
        return jsonify({
            'ppo_models': list_models(),
            'direct_models': direct.list_models(),
            'curve_models': curve_agent.list_models(),
            'design_vars': GA.DESIGN_VARS,
            'ts_surrogate_available': os.path.exists(os.path.join(OPTIMIZER_DIR, 'rl', 'ts_surrogate.pt')),
            'actions': [a['name'] for a in ACTIONS],
            'defaults': {
                'target_impulse':    GA.TARGET_IMPULSE,
                'target_thrust_avg': GA.TARGET_THRUST_AVG,
                'pop_size':          GA.POP_SIZE,
                'n_generations':     GA.N_GENERATIONS,
                'elite_count':       GA.ELITE_COUNT,
                'n_workers':         GA.N_WORKERS,
                'seed':              GA.RANDOM_SEED,
                'tolerance':         cs.TARGET_TOLERANCE,
                'verify_top':        GA.VERIFY_TOP,
                'min_dp_ratio':      cs.MIN_INJECTOR_DP_RATIO,
                'feasible_parent_min': GA.FEASIBLE_PARENT_MIN,
                'robust_band':       GA.ROBUST_ISP_BAND,
                'surrogate_margin':  cs.SURROGATE_MARGIN,
            },
            'surrogate_available': os.path.exists(os.path.join(OPTIMIZER_DIR, 'rl', 'surrogate.pt')),
            'bounds': GA.BOUNDS,
            'pipes':  pdb.load_pipes(),
            'fixed':  GA.load_fixed_values(),
        })
    except Exception as exc:
        return jsonify({'error': f"{type(exc).__name__}: {exc}"}), 500


@app.route('/api/mdo/start', methods=['POST'])
@login_required
def mdo_start():
    global _mdo_job
    data = request.get_json(silent=True) or {}
    try:
        impulse = float(data['target_impulse'])
        thrust  = float(data['target_thrust_avg'])
        settings = {}
        settings['feasible_parents'] = bool(data.get('feasible_parents'))
        settings['robust_pick'] = bool(data.get('robust_pick'))
        if data.get('tolerance') not in (None, ''):
            settings['tolerance'] = float(data['tolerance']) / 100        # 화면은 %, 내부는 비율
        if data.get('robust_band') not in (None, ''):
            settings['robust_band'] = float(data['robust_band'])
        for key in ('pop_size', 'n_generations', 'seed', 'verify_top'):
            if data.get(key) not in (None, ''):
                settings[key] = int(data[key])
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': '목표 임펄스와 평균 추력을 숫자로 입력하세요.'}), 400
    if impulse <= 0 or thrust <= 0:
        return jsonify({'error': '목표 임펄스와 평균 추력은 양수여야 합니다.'}), 400
    if settings.get('pop_size', 4) < 4 or settings.get('n_generations', 1) < 1:
        return jsonify({'error': '개체 수는 4 이상, 세대 수는 1 이상이어야 합니다.'}), 400
    mode = data.get('mode') or 'ga'
    model = data.get('model') or 'rl/ppo_ga.zip'
    evaluator = data.get('evaluator') or 'core'
    if not 0 < settings.get('tolerance', 0.05) <= 0.5:
        return jsonify({'error': '허용 오차는 0 보다 크고 50% 이하여야 합니다.'}), 400
    if mode not in ('ga', 'ppo', 'direct', 'curve'):
        return jsonify({'error': f'알 수 없는 최적화 모드: {mode}'}), 400
    if evaluator not in ('core', 'surrogate'):
        return jsonify({'error': f'알 수 없는 평가 방식: {evaluator}'}), 400
    if evaluator == 'surrogate' and not os.path.exists(os.path.join(OPTIMIZER_DIR, 'rl', 'surrogate.pt')):
        return jsonify({'error': '대리모델이 없습니다. optimizer 폴더에서 python -m rl.surrogate 로 학습하세요.'}), 400
    if mode in ('direct', 'curve') and evaluator == 'surrogate' and not os.path.exists(os.path.join(OPTIMIZER_DIR, 'rl', 'ts_surrogate.pt')):
        return jsonify({'error': '시계열 대리모델이 없습니다. python -m rl.ts_surrogate 로 학습하세요.'}), 400
    if mode in ('ppo', 'direct', 'curve'):
        _mdo_modules()
        from rl.policy import resolve_model
        try:
            resolve_model(model)
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400

    with _mdo_lock:
        if _mdo_job and _mdo_job['status'] == 'running':
            return jsonify({'error': '이미 실행 중인 MDO 가 있습니다.'}), 409
        _mdo_stop.clear()
        _mdo_job = {
            'id': str(uuid.uuid4()),
            'status': 'running',
            'error': None,
            'targets': {'impulse': impulse, 'thrust_avg': thrust},
            'settings': settings,
            'mode': mode,
            'model': model if mode in ('ppo', 'direct', 'curve') else None,
            'argmax': bool(data.get('argmax')) if mode in ('ppo', 'direct', 'curve') else None,   # True = 가장 확률 높은 전략(평균 행동), False = 확률적
            'strategies': [],
            'direct_steps': [],     # 직접 설계: [세대, 평균 스텝, 최소, 최대, 파이프 변경 수]
            'direct_last': None,    # 직접 설계: 최근 세대 이동 통계 + 최고 개체 예측 연소 곡선
            'curve_log': [],        # 곡선 판단: [세대, 최고 개체의 이동 설명, {변수: 선택 수}]
            'curve_last': None,     # 곡선 판단: 최근 세대 판단 (최고 개체 이동·주의 시간대·예측 곡선)
            'evaluator': evaluator,
            'best_source': evaluator,   # surrogate 모드에서 검증이 끝나면 core
            'verify_n': 0,
            'verified': [],
            'feasible': {},         # id → 제약 만족 형상 (예측 / core 검증 상태)
            'best_core': None,      # 대리모델 모드: 지금까지 core 로 확인된 최고 형상
            'trim': None,           # 마지막 연소 챔버 길이 정리 결과
            'robust': None,         # 견고성 선택 결과
            'corrections': [],
            'started_at': time.time(),
            'finished_at': None,
            'pop_size': settings.get('pop_size', 0),
            'n_generations': settings.get('n_generations', 0),
            'n_workers': 0,
            'pipes': [],
            'generation': -1,
            'population': [],
            'history': [],
            'points': [],
            'fail_counts': {},
            'best': None,
            'best_improved_at': None,
            'output_dir': None,
            'log': [],
            '_seen': set(),
            'points_epoch': 0,
        }
        job = _mdo_job
    threading.Thread(target=_mdo_run, args=(job,), daemon=True).start()
    return jsonify({'job_id': job['id']})


@app.route('/api/mdo/feasible')
@login_required
def mdo_feasible():
    """
    제약 만족 형상 목록. filter=verified 면 core 검증 합격만. Isp 내림차순 (검증 합격 우선).
    group=1 (기본): 설계 변수 차이가 탐색 범위의 1% 미만인 형상은 하나로 묶고 similar 에 개수
    """
    only_verified = request.args.get('filter') == 'verified'
    group = request.args.get('group', '1') != '0'
    with _mdo_lock:
        if not _mdo_job:
            return jsonify({'rows': []})
        entries = list(_mdo_job['feasible'].values())
        rows = [_feasible_row(e) for e in entries]
        designs = {e['id']: (e['core'] or e['pred']).get('design') or {} for e in entries}
    if only_verified:
        rows = [r for r in rows if r['verify'] == 'pass']
    rows.sort(key=lambda r: (VERIFY_RANK[r['verify']] if r['source'] == 'surrogate' else 0, -(r['isp'] or 0)))
    if group and rows:
        GA, _, _ = _mdo_modules()
        span = {v: (GA.BOUNDS[v][1] - GA.BOUNDS[v][0]) for v in GA.DESIGN_VARS}

        def near(a, b):
            if a.get('oxidizer_tank_radius') != b.get('oxidizer_tank_radius'):
                return False
            return all(abs((a.get(v) or 0) - (b.get(v) or 0)) < 0.01 * span[v] for v in GA.DESIGN_VARS)
        kept = []
        for r in rows:
            d = designs[r['id']]
            rep_row = next((k for k in kept if k['verify'] == r['verify'] and near(designs[k['id']], d)), None)
            if rep_row is None:
                kept.append(dict(r, similar=0))
            else:
                rep_row['similar'] += 1
        rows = kept
    return jsonify({'rows': rows})


@app.route('/api/mdo/design/<design_id>')
@login_required
def mdo_design(design_id):
    """형상 하나의 상세 (예측 / core 결과, 설계 변수 전체)"""
    with _mdo_lock:
        if not _mdo_job or design_id not in _mdo_job['feasible']:
            return jsonify({'error': '형상을 찾을 수 없습니다 (목록에서 밀려났거나 다른 실행).'}), 404
        e = copy.deepcopy(_mdo_job['feasible'][design_id])
        targets = dict(_mdo_job['targets'])
    return jsonify({'design': e, 'targets': targets})


@app.route('/api/mdo/stop', methods=['POST'])
@login_required
def mdo_stop():
    with _mdo_lock:
        if not _mdo_job or _mdo_job['status'] != 'running':
            return jsonify({'error': '실행 중인 MDO 가 없습니다.'}), 404
        _mdo_stop.set()
        _mdo_log(_mdo_job, "중단 요청 — 진행 중인 세대를 정리하는 중")
    return jsonify({'stopping': True})


@app.route('/api/mdo/status')
@login_required
def mdo_status():
    """현재(또는 마지막) MDO 잡 상태. points_from 이후의 목적 공간 점만 보낸다."""
    points_from = request.args.get('points_from', 0, type=int)
    points_epoch = request.args.get('points_epoch', -1, type=int)
    with _mdo_lock:
        if not _mdo_job:
            return jsonify({'job': None})
        heavy = ('_seen', 'points', 'history', 'strategies', 'feasible', 'direct_steps', 'curve_log')
        job = {k: copy.deepcopy(v) for k, v in _mdo_job.items() if k not in heavy}
        if points_epoch != _mdo_job['points_epoch']:     # 점을 솎았으면 처음부터 다시
            points_from = 0
        job['points'] = copy.deepcopy(_mdo_job['points'][points_from:])
        job['points_total'] = len(_mdo_job['points'])
        hist = _mdo_job['history']
        job['history_total'] = len(hist)
        if len(hist) > MDO_HISTORY_LIMIT:
            step = len(hist) / MDO_HISTORY_LIMIT
            hist = [hist[int(i * step)] for i in range(MDO_HISTORY_LIMIT - 1)] + [hist[-1]]
        job['history'] = copy.deepcopy(hist)
        # 전략 기록은 [세대, 전략 번호] 로 압축 (이름은 /api/mdo/meta 의 actions)
        job['strategies'] = [[s['generation'], s['action']] for s in _mdo_job['strategies']]
        job['last_strategy'] = copy.deepcopy(_mdo_job['strategies'][-1]) if _mdo_job['strategies'] else None
        job['curve_log'] = copy.deepcopy((_mdo_job.get('curve_log') or [])[-60:])
        ds = _mdo_job.get('direct_steps') or []
        if len(ds) > MDO_HISTORY_LIMIT:
            step = len(ds) / MDO_HISTORY_LIMIT
            ds = [ds[int(i * step)] for i in range(MDO_HISTORY_LIMIT - 1)] + [ds[-1]]
        job['direct_steps'] = copy.deepcopy(ds)
        rows = [_feasible_row(e) for e in _mdo_job['feasible'].values()]
        job['feasible_counts'] = {k: sum(r['verify'] == k for r in rows) for k in VERIFY_RANK}
        job['feasible_total'] = len(rows)
        job['stop_requested'] = _mdo_stop.is_set()
    end = job['finished_at'] or time.time()
    job['elapsed'] = end - job['started_at']
    return jsonify({'job': job})


# ── Surrogate (optimizer/rl/surrogate.py 의 PyTorch 앙상블) ────────────────────
_surrogate_lock  = threading.Lock()
_surrogate_cache = {'mtime': None, 'model': None}


def _surrogate_model():
    """surrogate.pt 를 읽어 둔다. 다시 학습해 파일이 바뀌면 자동으로 새로 읽는다."""
    _mdo_modules()
    from rl import surrogate as sg
    with _surrogate_lock:
        mtime = os.path.getmtime(sg.MODEL_PATH)
        if _surrogate_cache['mtime'] != mtime:
            _surrogate_cache['model'] = sg.SurrogateEnsemble.load(sg.MODEL_PATH)
            _surrogate_cache['mtime'] = mtime
        return sg, _surrogate_cache['model']


@app.route('/api/surrogate/meta')
@login_required
def surrogate_meta():
    """모델 정보(검증 지표·학습 범위) + 파이프 DB + 고정값"""
    try:
        GA, _, _ = _mdo_modules()
        from rl import surrogate as sg
        if not os.path.exists(sg.META_PATH):
            return jsonify({'error': '대리모델이 없습니다. optimizer 폴더에서 python -m rl.surrogate 로 학습하세요.'}), 404
        with open(sg.META_PATH, encoding='utf-8') as f:
            meta = json.load(f)
        import constrain as cs
        meta.update(pipes=GA.load_candidate_pipes(), fixed=GA.load_fixed_values(),
                    min_dp_ratio=cs.MIN_INJECTOR_DP_RATIO)
        return jsonify(meta)
    except Exception as exc:
        return jsonify({'error': f"{type(exc).__name__}: {exc}"}), 500


@app.route('/api/surrogate/predict', methods=['POST'])
@login_required
def surrogate_predict():
    """
    요청: {designs: [{pipe_id, 설계변수...}, ...], check: true, targets: {impulse, thrust_avg} (선택)}
    응답: {results: [{p_ok, mean{}, std{}}], constraints: {...} (check 일 때 첫 설계만), ms}
    """
    import torch
    data = request.get_json(silent=True) or {}
    designs = data.get('designs') or []
    if not designs or len(designs) > 2000:
        return jsonify({'error': 'designs 는 1~2000개여야 합니다.'}), 400
    try:
        t0 = time.perf_counter()
        GA, cs, _ = _mdo_modules()
        sg, model = _surrogate_model()
        pipes = GA.load_candidate_pipes()
        by_id = {p['pipe_id']: i for i, p in enumerate(pipes)}
        fixed = GA.load_fixed_values()
        X, decoded, pipe0 = [], [], None
        for d in designs:
            # GA.decode 로 파생 값 (챔버 길이 = (연료+2R)×1.3, 헬륨 질량) 까지 채운 뒤 대리모델 입력으로
            genome = {'pipe': by_id[d['pipe_id']], **{v: float(d[v]) for v in GA.DESIGN_VARS}}
            # 초기 온도: 요청 값 (범용 온도 대리모델) 이 없으면 fixedvalue
            design, pipe = GA.decode(genome, pipes, dict(fixed, temperature=float(d.get('temperature') or fixed['temperature'])))
            decoded.append(design)
            pipe0 = pipe0 or pipe
            X.append(sg.features(design, pipe))
        p_ok, mean, std = model.predict_stats(torch.tensor(X, dtype=torch.float32))
        results = [{'p_ok': p, 'mean': dict(zip(sg.TARGETS, m)), 'std': dict(zip(sg.TARGETS, sd)),
                    'derived': {v: dd[v] for v in GA.DERIVED_VARS}}
                   for p, m, sd, dd in zip(p_ok.tolist(), mean.tolist(), std.tolist(), decoded)]
        for r in results:
            if 'dp_ratio_min' not in r['mean']:   # 차압 출력이 없는 예전 모델 → 최대 압력 비로 근사
                r['mean']['dp_ratio_min'] = cs.injector_dp_ratio(r['mean'])
                r['std']['dp_ratio_min'] = 0.0
                r['dp_estimated'] = True
        out = {'results': results}

        if data.get('check'):
            # 첫 설계에 MDO 와 같은 제약 검사 (시뮬레이션 결과 자리에 대리모델 예측을 넣는다)
            genome = {'pipe': by_id[designs[0]['pipe_id']], **{v: designs[0][v] for v in GA.DESIGN_VARS}}
            design, pipe = decoded[0], pipe0
            pre, _ = cs.pre_sim_constraints(design, pipe)
            sim = dict(results[0]['mean'], error=None,
                       termination_code='SURROGATE' if results[0]['p_ok'] > 0.5 else 'TIME_LIMIT_REACHED')
            targets = data.get('targets') or {}
            has_targets = all(float(targets.get(k) or 0) > 0 for k in ('impulse', 'thrust_avg'))
            post = cs.post_sim_constraints(design, pipe, sim,
                                           float(targets['impulse']) if has_targets else 1.0,
                                           float(targets['thrust_avg']) if has_targets else 1.0)
            if not has_targets:
                post = [c for c in post if c.name() not in ('Impulse', 'ThrustAvg')]
            items = [{'name': c.name(), 'stage': stage, 'violation': c.violation()}
                     for stage, cons in (('pre', pre), ('post', post)) for c in cons]
            out['constraints'] = {'items': items, 'feasible': all(i['violation'] <= 0 for i in items),
                                  'with_targets': has_targets}
        out['ms'] = (time.perf_counter() - t0) * 1000
        return jsonify(out)
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({'error': f'입력 오류: {exc}'}), 400
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"{type(exc).__name__}: {exc}"}), 500


# ── RL 학습 모니터 (optimizer/rl/train_ppo.py 가 남기는 status.json / live.jsonl / progress.csv) ─────
RL_RUNS_DIR = os.path.join(OPTIMIZER_DIR, "rl", "runs")


def _rl_run_dir(run_id):
    if not run_id or "/" in run_id or run_id.startswith(".") or "\\" in run_id:
        return None
    path = os.path.join(RL_RUNS_DIR, run_id)
    return path if os.path.isdir(path) else None


RL_LIVE_STATES = ("running", "starting", "baseline", "teacher", "pretrain", "verify")


def _pid_is_training(pid):
    """pid 가 살아 있는 train_ppo 프로세스인지 (/proc 의 명령줄로 확인)"""
    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as f:
            cmd = f.read()
            return any(m in cmd for m in (b"rl.train_ppo", b"rl.train_direct", b"rl.train_curve", b"rl.ts_surrogate",
                                          b"rl.ts_data", b"rl.ts_active"))
    except (OSError, ValueError, TypeError):
        return False


def _rl_status(run_id, path):
    """status.json 이 있으면 그대로, 예전 실행이면 로그·평가 기록으로 추정. 프로세스가 죽었는데 running 이면 crashed"""
    st = {}
    if os.path.exists(os.path.join(path, "status.json")):
        try:
            with open(os.path.join(path, "status.json"), encoding="utf-8") as f:
                st = json.load(f)
        except (OSError, ValueError):
            st = {}
    if not st:
        cfg_path = os.path.join(path, "config.json")
        cfg = json.load(open(cfg_path, encoding="utf-8")) if os.path.exists(cfg_path) else {}
        st = {"name": run_id, "state": "done", "config": cfg, "total_steps": cfg.get("steps"),
              "started": os.path.getctime(path), "legacy": True}
        prog = os.path.join(path, "progress.csv")
        if os.path.exists(prog):
            rows = list(pd.read_csv(prog).to_dict("records"))
            if rows:
                st["step"] = int(rows[-1]["step"])
    if st.get("state") in RL_LIVE_STATES and not _pid_is_training(st.get("pid")):
        st["state"] = "crashed"
    st["id"] = run_id
    return st


@app.route('/training')
@login_required
def training_page():
    return send_from_directory(os.path.join(BASE_DIR, 'static'), 'training.html')


@app.route('/api/rl/runs')
@login_required
def rl_runs():
    if not os.path.isdir(RL_RUNS_DIR):
        return jsonify({'runs': []})
    runs = []
    for name in os.listdir(RL_RUNS_DIR):
        path = _rl_run_dir(name)
        if path and (os.path.exists(os.path.join(path, "progress.csv")) or os.path.exists(os.path.join(path, "status.json"))):
            st = _rl_status(name, path)
            runs.append({k: st.get(k) for k in ("id", "state", "step", "total_steps", "started", "finished",
                                                 "best", "baseline", "legacy", "kind", "phase")})
    runs.sort(key=lambda r: r.get("started") or 0, reverse=True)
    return jsonify({'runs': runs})


@app.route('/api/rl/runs/<run_id>')
@login_required
def rl_run(run_id):
    """since: 이미 받은 live.jsonl 줄 수 → 그 뒤만 보낸다"""
    path = _rl_run_dir(run_id)
    if not path:
        return jsonify({'error': '실행을 찾을 수 없습니다.'}), 404
    since = request.args.get('since', 0, type=int)
    live, total = [], 0
    live_path = os.path.join(path, "live.jsonl")
    if os.path.exists(live_path):
        with open(live_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                total = i + 1
                if i >= since and line.strip():
                    try:
                        live.append(json.loads(line))
                    except ValueError:
                        total = i            # 쓰는 중인 마지막 줄 → 다음에 다시
                        break
    evals = []
    prog = os.path.join(path, "progress.csv")
    if os.path.exists(prog):
        df = pd.read_csv(prog)
        evals = df.astype(object).where(df.notna(), None).to_dict("records")   # NaN → null (JSON)
    log_tail = []
    log_path = os.path.join(RL_RUNS_DIR, run_id + ".log")
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as f:
            log_tail = [l.rstrip() for l in f.readlines()[-40:] if "━" not in l]
    st = _rl_status(run_id, path)
    if st.get("kind") == "surrogate":      # 비교 기준: 이전 시계열 대리모델 v2 의 같은 테스트 지표
        ref = os.path.join(OPTIMIZER_DIR, "rl", "ts_surrogate_v2_mdo.json")
        if os.path.exists(ref):
            with open(ref, encoding="utf-8") as f:
                st["reference"] = json.load(f)
    return jsonify({'status': st, 'live': live, 'live_total': total,
                    'evals': evals, 'log': log_tail})


@app.route('/api/rl/runs/<run_id>/stop', methods=['POST'])
@login_required
def rl_run_stop(run_id):
    """학습 중단 (SIGINT → train_ppo 가 지금까지의 정책을 저장하고 끝냄)"""
    import signal
    path = _rl_run_dir(run_id)
    if not path:
        return jsonify({'error': '실행을 찾을 수 없습니다.'}), 404
    st = _rl_status(run_id, path)
    if st.get("state") not in RL_LIVE_STATES or not _pid_is_training(st.get("pid")):
        return jsonify({'error': '실행 중인 학습이 아닙니다.'}), 409
    os.kill(int(st["pid"]), signal.SIGINT)
    return jsonify({'stopping': True})


# ── vendor 정적 라이브러리 (인증 불필요) ──────────────────────────────────────
@app.route('/vendor/<path:filename>')
def serve_vendor(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'static', 'vendor'), filename)


# ── 정적 파일 / SPA 폴백 ──────────────────────────────────────────────────────
@app.route('/<path:filename>')
@login_required
def serve_static(filename):
    static_dir = os.path.join(BASE_DIR, 'static')
    filepath   = os.path.join(static_dir, filename)
    if os.path.isfile(filepath):
        return send_from_directory(static_dir, filename)
    return send_from_directory(static_dir, 'index.html')


def _terminate_children(signum, frame):
    """서버 종료(stop_local.sh 의 kill) 시 GA 워커 등 자식 프로세스도 같이 끝낸다 (고아로 남지 않게)"""
    import multiprocessing
    _mdo_stop.set()
    for child in multiprocessing.active_children():
        child.terminate()
    sys.exit(0)


if __name__ == '__main__':
    import signal
    signal.signal(signal.SIGTERM, _terminate_children)
    app.run(host=os.environ.get('HOST', '0.0.0.0'),
            port=int(os.environ.get('PORT', 10000)), threaded=True)
