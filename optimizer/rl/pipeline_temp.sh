#!/usr/bin/env bash
# 범용 온도 대리모델 (−15 ~ 35 °C) + 30 °C RL 재학습 파이프라인 (optimizer 폴더 기준, 로그 rl/pipeline_temp.log)
#   0. 지금 모델 (258 K) 백업
#   1. 데이터 생성: 설계마다 초기 온도 258~308 K (rl.ts_data) — 예전 258 K 데이터는 온도 = 258 로 함께 학습
#   2. 시계열 대리모델 v5a (입력에 온도)
#   3. 능동 학습 데이터: fixedvalue 온도 (303 K) 에서 v5a 위 에이전트식 이동 → core
#   4. 시계열 대리모델 v5 → rl/ts_surrogate.pt
#   5. 스칼라 대리모델 → rl/surrogate.pt
#   6. 곡선 판단 에이전트 (303 K)
# core 시뮬레이션은 모두 cpu_gate 로 시스템 CPU 80% 아래에서 동시 실행 개수 자동 조절
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
TS="--epochs 300 --models 5 --d 192 --layers 4 --heads 6 --ff 512 --lr 7e-4"
log() { echo "[$(date '+%m-%d %T')] $*"; }

log "0. 258 K 모델 백업"
for f in ts_surrogate.pt ts_surrogate_meta.json surrogate.pt surrogate_meta.json; do
  [ -f rl/$f ] && cp rl/$f rl/${f%.*}_258K.${f##*.}
done

log "1. 데이터 생성 (온도 258~308 K, 10000개)"
$PY -u -m rl.ts_data --n 10000 --seed 21 > rl/ts_data_temp.log 2>&1 || { log "데이터 생성 실패"; exit 1; }

N=tssur_v5a_$(date +%Y%m%d_%H%M)
log "2. 시계열 대리모델 v5a 학습 ($N)"
$PY -u -m rl.ts_surrogate $TS --out rl/ts_surrogate_v5a.pt --meta rl/ts_surrogate_v5a_meta.json --run $N > rl/runs/$N.log 2>&1 || { log "v5a 학습 실패"; exit 1; }

log "3. 능동 학습 데이터 (v5a, 303 K, 2500개)"
$PY -u -m rl.ts_active --n 2500 --ts-model rl/ts_surrogate_v5a.pt > rl/ts_active_temp.log 2>&1 || { log "능동 학습 데이터 실패"; exit 1; }

N=tssur_v5_$(date +%Y%m%d_%H%M)
log "4. 시계열 대리모델 v5 학습 ($N)"
$PY -u -m rl.ts_surrogate $TS --out rl/ts_surrogate_v5.pt --meta rl/ts_surrogate_v5_meta.json --run $N > rl/runs/$N.log 2>&1 || { log "v5 학습 실패"; exit 1; }
cp rl/ts_surrogate_v5.pt rl/ts_surrogate.pt; cp rl/ts_surrogate_v5_meta.json rl/ts_surrogate_meta.json
log "   rl/ts_surrogate.pt ← v5"

log "5. 스칼라 대리모델"
$PY -u -m rl.surrogate > rl/surrogate_train_temp.log 2>&1 || log "스칼라 대리모델 실패 (계속 진행)"

N=curve_T303_$(date +%Y%m%d_%H%M)
log "6. 곡선 판단 에이전트 303 K ($N)"
$PY -u -m rl.train_curve --steps 4000000 --core-freq 500000 --name $N > rl/runs/$N.log 2>&1 || { log "곡선 에이전트 실패"; exit 1; }
log "완료 ($N)"
