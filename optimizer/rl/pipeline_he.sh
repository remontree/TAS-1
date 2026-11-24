#!/usr/bin/env bash
# 헬륨 슈퍼차징 변수 추가 후 재학습 파이프라인 (optimizer 폴더 기준, 로그 rl/pipeline_he.log)
#   1. 데이터 생성 (rl.ts_data, pid $1) 종료 대기
#   2. 시계열 대리모델 v4a (차압 채널 · 새 입력) 학습
#   3. 능동 학습 데이터: v4a 위에서 에이전트식으로 간 설계를 core 로 (rl.ts_active)
#   4. 시계열 대리모델 v4 (능동 학습 데이터 포함) → rl/ts_surrogate.pt
#   5. 스칼라 대리모델 (GA Surrogate 평가 · Surrogate 페이지) → rl/surrogate.pt
#   6. 곡선 판단 에이전트 새로 학습 (앞 보기 → 증류 → PPO, core 검증으로 best_model)
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
DATA_PID=$1
TS="--epochs 300 --models 5 --d 192 --layers 4 --heads 6 --ff 512 --lr 7e-4"
log() { echo "[$(date '+%m-%d %T')] $*"; }

if [ -n "$DATA_PID" ]; then
  while kill -0 "$DATA_PID" 2>/dev/null; do sleep 30; done
fi
log "데이터 생성 종료"

N=tssur_v4a_$(date +%Y%m%d_%H%M)
log "2. 시계열 대리모델 v4a 학습 ($N)"
$PY -u -m rl.ts_surrogate $TS --out rl/ts_surrogate_v4a.pt --run $N > rl/runs/$N.log 2>&1 || { log "v4a 학습 실패"; exit 1; }

log "3. 능동 학습 데이터 (v4a, 2500개)"
$PY -u -m rl.ts_active --n 2500 --ts-model rl/ts_surrogate_v4a.pt > rl/ts_active.log 2>&1 || { log "능동 학습 데이터 실패"; exit 1; }

N=tssur_v4_$(date +%Y%m%d_%H%M)
log "4. 시계열 대리모델 v4 학습 ($N)"
$PY -u -m rl.ts_surrogate $TS --out rl/ts_surrogate_v4.pt --run $N > rl/runs/$N.log 2>&1 || { log "v4 학습 실패"; exit 1; }
cp rl/ts_surrogate_v4.pt rl/ts_surrogate.pt; cp rl/ts_surrogate_v4_meta.json rl/ts_surrogate_meta.json
log "   rl/ts_surrogate.pt ← v4"

log "5. 스칼라 대리모델"
cp rl/surrogate.pt rl/surrogate_old17.pt; cp rl/surrogate_meta.json rl/surrogate_old17_meta.json
$PY -u -m rl.surrogate > rl/surrogate_train.log 2>&1 || log "스칼라 대리모델 실패 (계속 진행)"

N=curve_he_$(date +%Y%m%d_%H%M)
log "6. 곡선 판단 에이전트 ($N)"
$PY -u -m rl.train_curve --steps 4000000 --core-freq 500000 --name $N > rl/runs/$N.log 2>&1 || { log "곡선 에이전트 실패"; exit 1; }
log "완료 ($N)"
