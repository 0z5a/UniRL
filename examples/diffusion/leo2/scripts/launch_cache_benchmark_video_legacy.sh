#!/usr/bin/env bash
# Preserve the already-started 2026-09-07 visual-only benchmark contract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LEO2_BENCH_BOT_TASK=video
exec "${SCRIPT_DIR}/launch_cache_benchmark_node.sh" "$@"
