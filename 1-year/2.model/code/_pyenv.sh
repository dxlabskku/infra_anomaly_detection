# run_care.sh, run_smoke.sh 가 source 한다. 파이썬: PY 환경 변수 > ../.venv > PATH 의 python3
WORK_ROOT="$(dirname "$CODE")"
if [[ -z "${PY:-}" ]]; then
  if [[ -x "$WORK_ROOT/.venv/bin/python" ]]; then
    PY="$WORK_ROOT/.venv/bin/python"
  else
    PY="$(command -v python3 || true)"
  fi
fi
[[ -x "$PY" ]] || { echo "파이썬을 찾지 못했습니다: $PY (PY=... 로 지정하거나 README 4절대로 .venv 를 만드세요)"; exit 1; }
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')"
  export CUDA_VISIBLE_DEVICES
fi
step() {
  local name="$1"; shift
  local t0; t0=$(date +%s)
  echo "=== [$name] $(date '+%F %T')"
  "$@"
  local dt=$(( $(date +%s) - t0 ))
  printf "%s\t%s\n" "$name" "$dt" >> "$TIMING"
  echo "=== [$name] 끝 $(date '+%F %T')  ${dt}초"
}
timing_summary() {
  echo "단계별 시간 (초):"
  awk -F'\t' '{printf "  %-16s %8d초  (%.1f분)\n", $1, $2, $2/60; s += $2} END {printf "  %-16s %8d초  (%.1f분)\n", "합계", s, s/60}' "$TIMING"
}
