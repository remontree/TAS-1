# 배포 모델

| 파일 | 역할 |
|---|---|
| `static/surrogate_model.json` | 브라우저에서 실행하는 대리모델 |
| `optimizer/rl/surrogate.pt` | 성능 요약값 예측 앙상블 |
| `optimizer/rl/surrogate_meta.json` | 입력 특성 및 검증 메타데이터 |
| `optimizer/rl/ts_surrogate.pt` | 연소 시계열 예측 앙상블 |
| `optimizer/rl/ts_surrogate_meta.json` | 시계열 모델 메타데이터 |
| `optimizer/rl/runs/curve_T303_20261021_0759/best_model.zip` | 곡선 기반 PPO 설계 정책 |
| 같은 디렉터리의 `config.json` | 모델 검색 및 온도·설계 변수 호환성 확인 |

현재 `fixedvalue.json`과 호환되는 곡선 정책의 best 가중치만 배포한다. 이전 온도·관측 규격의 정책, 중간 체크포인트, 중복 final 가중치는 제외한다. GA 전략 선택 및 direct 정책은 현재 설정에 호환되는 학습 모델이 없어 포함하지 않는다.

학습 원본, teacher 상태, TensorBoard 및 실험 CSV는 포함하지 않는다. 추론과 기존 정책 실행에는 필요하지 않으며, 재학습에는 별도로 데이터를 생성해야 한다. `surrogate/build_dataset.py`는 GA 실행 기록을 입력으로 사용하므로 새 저장소에서는 먼저 GA 결과를 생성해야 한다.

`optimizer/results/20261019_031737/best_rocket.json`은 게임 결과표 생성에 필요한 기준 설계이므로 예외적으로 포함한다. 게임 LUT는 `game/build_lut.py`로 재생성한다.
