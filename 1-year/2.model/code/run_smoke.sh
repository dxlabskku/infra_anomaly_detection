#!/bin/bash
# 스모크 테스트: 시설별·분할별 처음 16장(분할당 64장)으로 처음부터 끝까지. 출력은 smoke/ 에만.
#   검출기 epochs=2 batch=8, 부트스트랩 50회 (config.yaml smoke.overrides). test 평가 경로까지 확인한다.
#   bash run_smoke.sh [--overwrite]
set -euo pipefail
CODE="$(cd "$(dirname "$0")" && pwd)"
CFG="${CFG:-$CODE/config.yaml}"
EXPS="${EXPS:-base_gray3 G GCL}"
EXTRA=("$@")
cd "$CODE"
source "$CODE/_pyenv.sh"

WORK="$("$PY" common.py work-dir --config "$CFG" --smoke)"
mkdir -p "$WORK/logs"
TS="$(date +%Y%m%d_%H%M%S)"
TIMING="$WORK/logs/smoke_timing_${TS}.tsv"
echo "스모크 작업 폴더: $WORK  파이썬: $PY  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  실험: $EXPS"

S=(--config "$CFG" --smoke ${EXTRA[@]+"${EXTRA[@]}"})
step s1_prepare "$PY" s1_prepare.py "${S[@]}"
for e in $EXPS; do step "s2_$e" "$PY" s2_train.py --exp "$e" "${S[@]}"; done
step s3_predict  "$PY" s3_predict.py  --exp $EXPS "${S[@]}"
step s4_evaluate "$PY" s4_evaluate.py --exp $EXPS --test "${S[@]}"
timing_summary
echo "스모크 테스트 통과. 보고서: $WORK/eval/report.md"
