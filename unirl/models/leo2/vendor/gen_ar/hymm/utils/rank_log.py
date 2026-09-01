"""Rank-prefixed stdout helpers for distributed inference logging."""
from __future__ import annotations

import sys

import torch
from transformers import TextStreamer


def dist_rank() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


_RANK0_ONLY = False


def set_rank0_only(enabled: bool) -> None:
    """If enabled, rank_print / RankPrefixedTextStreamer only emit on rank 0."""
    global _RANK0_ONLY
    _RANK0_ONLY = bool(enabled)


def is_rank0_only() -> bool:
    return _RANK0_ONLY


def _should_emit() -> bool:
    return (not _RANK0_ONLY) or dist_rank() == 0


def rank_print(*args, **kwargs) -> None:
    if not _should_emit():
        return
    print(f"[rank {dist_rank()}]", *args, **kwargs)


def rank_print_multiline(text: str) -> None:
    if not _should_emit():
        return
    for line in text.splitlines():
        rank_print(line)


class RankPrefixedTextStreamer(TextStreamer):
    """TextStreamer with [rank N] prefix, line-buffered to avoid interleaving.

    HF's TextStreamer flushes per decoded chunk (often per character for CJK).
    With multiple ranks sharing the same stdout, char-level flushes from
    different ranks get interleaved into garbled lines. We buffer per rank
    until a newline (or stream end) and then emit the whole line in a single
    atomic write so that pdsh / log collectors can prefix it cleanly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefix = f"[rank {dist_rank()}] "
        self._buffer = ""

    def _emit_line(self, line: str) -> None:
        # One sys.stdout.write per complete line: writes <= PIPE_BUF (typically
        # 4096 bytes on Linux) are atomic, so other ranks' lines won't cut in.
        sys.stdout.write(f"{self._prefix}{line}\n")

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        if not _should_emit():
            return
        if text:
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._emit_line(line)
        if stream_end and self._buffer:
            self._emit_line(self._buffer)
            self._buffer = ""
        sys.stdout.flush()
