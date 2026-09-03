#!/usr/bin/env python3
"""Summarise ``[leo2 perf]`` and ``[leo2 mem]`` lines from a UniRL smoke log."""

import re
import statistics
import sys

PERF = re.compile(
    r"\[leo2 perf\] fwd#(\d+) grad=(\w+) tokens=(\d+) dt=([\d.]+)s "
    r"before=([\d.]+)GB after=([\d.]+)GB peak=([\d.]+)GB"
)
PRE = re.compile(r"\[leo2 mem\] pre-rollout alloc=([\d.]+)GB.*local GPU param bytes=([\d.]+)GB")
REPEAT = re.compile(r"repeated (\d+)x")


def main(path: str) -> None:
    rows = []
    pre = None
    with open(path, "rb") as fh:
        for raw in fh:
            line = raw.decode("utf-8", "replace").replace("\r", "\n")
            for sub in line.split("\n"):
                m = PERF.search(sub)
                if m:
                    rep = REPEAT.search(sub)
                    rows.append(
                        {
                            "idx": int(m[1]),
                            "grad": m[2] == "True",
                            "tokens": int(m[3]),
                            "dt": float(m[4]),
                            "before": float(m[5]),
                            "after": float(m[6]),
                            "peak": float(m[7]),
                            "rep": int(rep[1]) if rep else 1,
                        }
                    )
                p = PRE.search(sub)
                if p:
                    pre = (float(p[1]), float(p[2]))
    if pre:
        print(f"pre-rollout resident: {pre[0]:.1f} GB (local sharded params {pre[1]:.1f} GB)")
    for grad in (False, True):
        sel = [r for r in rows if r["grad"] == grad]
        if not sel:
            continue
        dts = [r["dt"] for r in sel]
        print(
            f"{'train (grad)' if grad else 'rollout/replay (no_grad)':>26}: "
            f"n_lines={len(sel)} (x{sum(r['rep'] for r in sel)} incl. dedup) "
            f"tokens~{statistics.median(r['tokens'] for r in sel):.0f} "
            f"dt median={statistics.median(dts):.2f}s min={min(dts):.2f}s max={max(dts):.2f}s "
            f"peak max={max(r['peak'] for r in sel):.1f}GB "
            f"resident after={max(r['after'] for r in sel):.1f}GB"
        )


if __name__ == "__main__":
    main(sys.argv[1])
