#!/usr/bin/env bash
# 시계열 데이터 생성이 끝나면: 대리모델 재학습 → v2 정책에서 이어서 RL v3 학습 (optimizer 폴더에서 실행)
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
while pgrep -f "m rl.ts_data" >/dev/null; do sleep 30; done
echo "[$(date +%T)] 데이터 생성 종료 → 시계열 대리모델 재학습"
cp rl/ts_surrogate.pt rl/ts_surrogate_v1.pt; cp rl/ts_surrogate_meta.json rl/ts_surrogate_v1_meta.json
$PY -u -m rl.ts_surrogate --epochs 200 --models 3 > rl/ts_surrogate_train.log 2>&1 || { echo "대리모델 학습 실패"; exit 1; }
echo "[$(date +%T)] 대리모델 완료 → RL v3"
N=direct_v3_$(date +%Y%m%d_%H%M)
while pgrep -f "m rl.train_direct" >/dev/null; do sleep 30; done
$PY -u -m rl.train_direct --steps 8000000 --name $N --init rl/runs/direct_v2_20261019_1908/best_model.zip > rl/runs/$N.log 2>&1
echo "[$(date +%T)] RL v3 종료 ($N)"
