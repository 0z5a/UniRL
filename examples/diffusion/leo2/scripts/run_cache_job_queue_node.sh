#!/usr/bin/env bash
# Atomically claim and execute sharded single-node Leo2 benchmark jobs.
set -euo pipefail

QUEUE="${1:?expected queue TSV as argument 1}"
STATE_ROOT="${2:?expected shared queue state directory as argument 2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_ID="$(hostname)-$$"

[[ -f "${QUEUE}" ]] || { echo "queue not found: ${QUEUE}" >&2; exit 2; }
mkdir -p "${STATE_ROOT}/claims" "${STATE_ROOT}/done" "${STATE_ROOT}/failed" "${STATE_ROOT}/logs"

failures=0
claimed=0
while IFS=$'\t' read -r job_id output_root cases_csv prompts_csv steps estimated_seconds; do
  [[ "${job_id}" != "job_id" ]] || continue
  [[ -n "${job_id}" && -n "${output_root}" && -n "${cases_csv}" && -n "${prompts_csv}" ]] || {
    echo "malformed queue row for job_id=${job_id@Q}" >&2
    exit 2
  }
  claim="${STATE_ROOT}/claims/${job_id}"
  if ! mkdir "${claim}" 2>/dev/null; then
    continue
  fi
  claimed=$((claimed + 1))
  {
    echo "worker_id=${WORKER_ID}"
    echo "hostname=$(hostname)"
    echo "pid=$$"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "estimated_seconds=${estimated_seconds}"
  } >"${claim}/metadata.env"
  echo "[${WORKER_ID}] starting ${job_id} (estimate ${estimated_seconds}s)"
  set +e
  "${SCRIPT_DIR}/launch_cache_benchmark_node.sh" \
    "${output_root}" "${cases_csv}" "${prompts_csv}" "${steps}" \
    >"${STATE_ROOT}/logs/${job_id}.log" 2>&1
  status=$?
  set -e
  {
    echo "worker_id=${WORKER_ID}"
    echo "ended_at=$(date --iso-8601=seconds)"
    echo "exit_code=${status}"
  } >>"${claim}/metadata.env"
  if [[ "${status}" -eq 0 ]]; then
    mv "${claim}" "${STATE_ROOT}/done/${job_id}"
    echo "[${WORKER_ID}] completed ${job_id}"
  else
    mv "${claim}" "${STATE_ROOT}/failed/${job_id}"
    echo "[${WORKER_ID}] failed ${job_id} with exit ${status}" >&2
    failures=$((failures + 1))
  fi
done <"${QUEUE}"

echo "[${WORKER_ID}] queue exhausted: claimed=${claimed}, failures=${failures}"
((failures == 0))
