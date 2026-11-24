#!/usr/bin/env bash
# cloudflared 임시 터널(trycloudflare.com, 계정 불필요) — 정식 HTTPS 인증서로 28443 서버를 공개한다.
# 터널 주소는 data/tunnel_url 에 기록 → 서버 QR·대형 화면이 자동으로 새 주소를 쓴다.
GAME="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
PORT="$("$GAME/.venv/bin/python" -c "import json;print(json.load(open('$GAME/server_config.json'))['port'])")"
CF="$(command -v cloudflared || echo "$HOME/.local/bin/cloudflared")"
[ -x "$CF" ] || { echo "cloudflared 가 없습니다" >&2; exit 1; }
rm -f "$GAME/data/tunnel_url"
"$CF" tunnel --no-autoupdate --url "https://127.0.0.1:$PORT" --no-tls-verify 2>&1 | while IFS= read -r line; do
  echo "$line"
  if [[ $line =~ (https://[a-z0-9-]+\.trycloudflare\.com) ]]; then
    echo "${BASH_REMATCH[1]}" > "$GAME/data/tunnel_url"
    echo "TUNNEL URL ${BASH_REMATCH[1]}"
  fi
done
