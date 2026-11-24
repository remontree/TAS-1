#!/usr/bin/env bash
# TAS-1 로컬 실행: 백엔드(Flask)가 꺼져 있으면 켜고, 준비되면 브라우저를 엽니다.
# 바탕화면의 "TAS-1 Simulator" 아이콘이 이 스크립트를 실행합니다.

APP_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
PORT="${PORT:-10000}"
URL="http://127.0.0.1:$PORT"
VENV="$APP_DIR/.venv"
LOG="$APP_DIR/server.log"
PID_FILE="$APP_DIR/.server.pid"

fail() {
  zenity --error --title="TAS-1" --width=420 --text="$1" 2>/dev/null \
    || notify-send "TAS-1" "$1" 2>/dev/null
  echo "ERROR: $1" >&2
  exit 1
}

open_browser() {
  if command -v google-chrome >/dev/null; then
    google-chrome "$URL" >/dev/null 2>&1 &
  else
    xdg-open "$URL" >/dev/null 2>&1 &
  fi
}

# 이미 실행 중이면 브라우저만 연다
if curl -s -o /dev/null "$URL/login"; then
  open_browser
  exit 0
fi

# 가상환경 / 의존성 확인 (최초 1회 설치)
if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import flask, flask_cors, CoolProp, rocketcea, skimage" 2>/dev/null; then
  command -v uv >/dev/null || fail "uv가 설치되어 있지 않습니다."
  [ -x "$VENV/bin/python" ] || uv venv --python 3.11 "$VENV" >>"$LOG" 2>&1 \
    || fail "Python 3.11 가상환경 생성 실패. $LOG 를 확인하세요."
  uv pip install --python "$VENV/bin/python" -r "$APP_DIR/requirements.txt" >>"$LOG" 2>&1 \
    || fail "패키지 설치 실패 (rocketcea는 gfortran 필요: sudo apt install gfortran). $LOG 를 확인하세요."
fi

# 백엔드 실행 (로컬 전용: 127.0.0.1, 로그인 생략)
cd "$APP_DIR" || exit 1
echo "--- $(date) start" >>"$LOG"
LOCAL_MODE=1 HOST=127.0.0.1 PORT="$PORT" \
  nohup "$VENV/bin/python" flask_app.py >>"$LOG" 2>&1 &
echo $! >"$PID_FILE"

# 서버 준비 대기 (최대 30초)
for _ in $(seq 1 60); do
  curl -s -o /dev/null "$URL/login" && { open_browser; exit 0; }
  kill -0 "$(cat "$PID_FILE")" 2>/dev/null || fail "서버가 시작 직후 종료되었습니다. $LOG 를 확인하세요."
  sleep 0.5
done
fail "서버 응답 시간 초과. $LOG 를 확인하세요."
