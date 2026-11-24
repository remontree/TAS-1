#!/usr/bin/env bash
# TAS-1 로컬 백엔드 종료
APP_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
PID_FILE="$APP_DIR/.server.pid"
if [ -f "$PID_FILE" ] && kill "$(cat "$PID_FILE")" 2>/dev/null; then
  notify-send "TAS-1" "서버를 종료했습니다." 2>/dev/null
else
  notify-send "TAS-1" "실행 중인 서버가 없습니다." 2>/dev/null
fi
rm -f "$PID_FILE"
