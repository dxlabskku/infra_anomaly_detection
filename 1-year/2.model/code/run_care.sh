#!/bin/bash
# CARE-YOLO 전체 실행: s1 입력 준비 -> s2 학습(실험별) -> s3 예측(val·test 박스) -> s4 평가(val 만)
#   CUDA_VISIBLE_DEVICES=7 bash run_care.sh               실험: EXPS (기본 "base_gray3 G GCL")
#   PARALLEL=1 이면 s2 실험들을 같은 GPU 에서 동시에 학습한다 (기본 1)
#   SKIP_S1=1 이면 입력 준비를 건너뛴다 (prepared/ 가 이미 있을 때)
#   TEST=1 (기본) 이면 s4 에서 test 지표도 계산한다. KPI 대상은 config experiments_main (GCL) 으로 미리 정해 둔다
#   추가 인자는 모든 단계에 넘긴다 (예: --overwrite)
set -euo pipefail
CODE="$(cd "$(dirname "$0")" && pwd)"
CFG="${CFG:-$CODE/config.yaml}"
EXPS="${EXPS:-base_gray3 G GCL}"
PARALLEL="${PARALLEL:-1}"
EXTRA=("$@")
cd "$CODE"
source "$CODE/_pyenv.sh"

WORK="$("$PY" common.py work-dir --config "$CFG")"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORK/logs"
TIMING="$WORK/logs/run_care_${TS}_timing.tsv"
echo "작업 폴더: $WORK  파이썬: $PY  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  실험: $EXPS  단계별 시간: $TIMING"
"$PY" common.py env --out "$WORK/logs/env_run_care_${TS}.txt" >/dev/null

train_all() {
  if [[ "$PARALLEL" == "1" ]]; then
    local pids=() rc=0
    for e in $EXPS; do
      "$PY" s2_train.py --config "$CFG" --exp "$e" ${EXTRA[@]+"${EXTRA[@]}"} > "$WORK/logs/s2_${e}_console_${TS}.log" 2>&1 &
      pids+=($!)
      echo "  s2 $e 시작 (pid $!, 콘솔 로그 $WORK/logs/s2_${e}_console_${TS}.log)"
    done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    return $rc
  fi
  for e in $EXPS; do "$PY" s2_train.py --config "$CFG" --exp "$e" ${EXTRA[@]+"${EXTRA[@]}"}; done
}

[[ "${SKIP_S1:-0}" == "1" ]] || step s1_prepare "$PY" s1_prepare.py --config "$CFG" ${EXTRA[@]+"${EXTRA[@]}"}
step s2_train    train_all
step s3_predict  "$PY" s3_predict.py  --config "$CFG" --exp $EXPS ${EXTRA[@]+"${EXTRA[@]}"}
T=(); [[ "${TEST:-1}" == "1" ]] && T=(--test)
step s4_evaluate "$PY" s4_evaluate.py --config "$CFG" --exp $EXPS ${T[@]+"${T[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
timing_summary
echo "완료. 보고서: $WORK/eval/report.md"
