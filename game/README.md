# TAS-1 Launch League

TAS-1의 시계열 대리모델로 사전 계산한 엔진 성능과 JavaScript 비행 계산을 사용하는 별도 체험 게임입니다. 메인 시뮬레이터 실행에는 필요하지 않습니다.

## 준비

메인 프로젝트의 Python 환경 외에 Linux, `uv`, Node.js, OpenSSL 및 systemd 사용자 서비스가 필요합니다.

```bash
cp game/server_config.example.json game/server_config.json
mkdir -p game/data
cd game
../.venv/bin/python build_lut.py
```

`build_lut.py`는 많은 설계를 예측하여 `data/engine_lut.jsonl.gz`를 생성하므로 시간과 메모리가 필요합니다. `game_config.json`이 참조하는 기준 설계 JSON은 저장소에 포함되어 있습니다. 기본 게임 온도는 258K이며 현재 배포 모델과의 적용 범위는 사용 목적에 맞게 검토해야 합니다.

`server_config.json`의 공개 주소·포트·관리자 경로를 설정한 뒤 다음 명령으로 서비스를 설치합니다.

```bash
./bin/tas1-game install
./bin/tas1-game status
```

설치 과정에서 게임용 가상환경, 세션 키, 관리자 비밀번호, 자체 서명 인증서를 생성합니다. 자체 서명 HTTPS 인증서는 브라우저에서 신뢰 설정이 필요합니다. 참가자 화면은 `/`, 전광판은 `/wall`, 관리자 화면은 설정한 `admin_path`입니다.

관리 명령: `start`, `stop`, `restart`, `monitor`, `logs`, `admin-password`, `backup`, `purge`, `uninstall`. `purge`는 참가자 기록을 삭제합니다. 선택적인 터널 기능은 별도 Tailscale 설정이 필요합니다.

운영 설정, 참가자 DB, 인증키, 인증서, 로그, 백업, 생성된 LUT는 Git에 포함하지 않습니다. `server_config.example.json`을 복사하여 각 환경에서 설정하세요.
