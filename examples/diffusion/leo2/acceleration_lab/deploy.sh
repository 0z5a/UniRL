#!/usr/bin/env bash
# Deploy and restart the read-only Leo2 Acceleration Lab service.
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${LEO2_ACCELERATION_LAB_ROOT:-/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/leo2_acceleration_lab}"
MANIFEST="${1:-/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/leo2_acceleration_benchmark/releases/current.json}"
BIND="${LEO2_ACCELERATION_LAB_BIND:-127.0.0.1}"
PORT="${LEO2_ACCELERATION_LAB_PORT:-16023}"
RUN_DIR="${TARGET_DIR}/run"
PID_FILE="${RUN_DIR}/server.pid"
LOG_FILE="${RUN_DIR}/server.log"

[[ -f "${MANIFEST}" ]] || {
  echo "expected a published release manifest, got ${MANIFEST}" >&2
  exit 2
}
[[ "${PORT}" =~ ^[1-9][0-9]*$ ]] && ((PORT <= 65535)) || {
  echo "expected port in [1,65535], got ${PORT@Q}" >&2
  exit 2
}

mkdir -p "${TARGET_DIR}" "${RUN_DIR}"
rsync -a --delete --exclude=run "${SOURCE_DIR}/" "${TARGET_DIR}/"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}")"
  if [[ "${old_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    command_line="$(ps -p "${old_pid}" -o args=)"
    [[ "${command_line}" == *"${TARGET_DIR}/server.py"* ]] || {
      echo "refusing to stop PID ${old_pid}; command is ${command_line@Q}" >&2
      exit 1
    }
    kill -TERM "${old_pid}"
    for _ in $(seq 1 30); do
      kill -0 "${old_pid}" 2>/dev/null || break
      sleep 1
    done
    kill -0 "${old_pid}" 2>/dev/null && {
      echo "old Lab server PID ${old_pid} did not stop within 30 seconds" >&2
      exit 1
    }
  fi
fi

nohup python3 "${TARGET_DIR}/server.py" \
  --bind "${BIND}" --port "${PORT}" --manifest "${MANIFEST}" \
  >>"${LOG_FILE}" 2>&1 </dev/null &
pid=$!
echo "${pid}" >"${PID_FILE}"

for _ in $(seq 1 30); do
  if python3 - "${BIND}" "${PORT}" <<'PY'
import json
import sys
import urllib.error
import urllib.request

host = "127.0.0.1" if sys.argv[1] in {"0.0.0.0", "::"} else sys.argv[1]
try:
    with urllib.request.urlopen(f"http://{host}:{sys.argv[2]}/api/health", timeout=1) as response:
        payload = json.load(response)
except urllib.error.URLError:
    raise SystemExit(1)
if payload.get("status") != "ok":
    raise RuntimeError(f"expected healthy Lab response, got {payload!r}")
PY
  then
    echo "Leo2 Acceleration Lab ready: pid=${pid} bind=${BIND}:${PORT}"
    exit 0
  fi
  kill -0 "${pid}" 2>/dev/null || {
    echo "Lab server exited during startup; inspect ${LOG_FILE}" >&2
    exit 1
  }
  sleep 1
done
echo "Lab server did not become healthy within 30 seconds; inspect ${LOG_FILE}" >&2
exit 1
