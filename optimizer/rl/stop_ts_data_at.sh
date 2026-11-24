#!/usr/bin/env bash
# 정상 시뮬레이션이 TARGET 개를 넘으면, 다음 shard 가 저장된 직후 rl.ts_data 를 (워커까지) 종료
cd "$(dirname "$0")/.."
TARGET=${1:-10000}
LOG=rl/ts_data_gen.log
DIR=$(ls -d rl/ts_data/*/ | tail -1)
MAIN=$(pgrep -f "m rl.ts_data" | head -1)
until [ "$(grep -o '정상 [0-9]*' $LOG | tail -1 | grep -o '[0-9]*')" -ge "$TARGET" ]; do sleep 20; done
n=$(ls $DIR/shard_*.npz | wc -l)
until [ "$(ls $DIR/shard_*.npz | wc -l)" -gt "$n" ]; do sleep 2; done
sleep 3                                   # 압축 저장이 끝나도록
KIDS=$(pgrep -P "$MAIN")
kill "$MAIN" $KIDS 2>/dev/null
sleep 5; kill -9 $KIDS 2>/dev/null
echo "[$(date +%T)] 정상 $(grep -o '정상 [0-9]*' $LOG | tail -1) 에서 종료, shard $(ls $DIR/shard_*.npz | wc -l)개"
