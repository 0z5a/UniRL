"""Fixed-count micro-batching — the default planner."""

from __future__ import annotations

from unirl.algorithms.base import StageAlgorithm
from unirl.train.stack.planner.types import Plan, _build_micro_batch_slices, _positive_int, _update_ranges
from unirl.types.sample import Part


def _count_plan(*, total: int, num_updates: int, micro_batch_size: int) -> Plan:
    """Fixed-count plan: contiguous equal updates, each split into ``micro_batch_size`` micros."""
    plan: Plan = []
    for u_start, u_end in _update_ranges(total_size=total, num_updates=num_updates):
        plan.append(
            [
                (u_start + ms, u_start + me)
                for ms, me in _build_micro_batch_slices(total_size=u_end - u_start, micro_batch_size=micro_batch_size)
            ]
        )
    return plan


def _forward_pack_plan(part: Part, *, num_updates: int, micro_batch_size: int) -> Plan | None:
    """Build updates from exact rollout-forward packs when provenance is present."""
    provenance = part.forward_pack_sample_ids
    if not provenance:
        return None
    updates = _positive_int(name="num_updates_per_batch", value=num_updates)
    max_micro = _positive_int(name="micro_batch_size", value=micro_batch_size)
    total = int(part.batch_size)
    packs = []
    seen = set()
    cursor = 0
    while cursor < total:
        pack = provenance[cursor]
        if not isinstance(pack, tuple) or not pack or not all(isinstance(sample_id, str) for sample_id in pack):
            raise ValueError("forward_pack_sample_ids entries must be non-empty tuples of sample ids")
        end = cursor + len(pack)
        if end > total:
            raise ValueError("training DP shard ends inside a recorded rollout forward pack")
        if tuple(part.sample_ids[cursor:end]) != pack or any(value != pack for value in provenance[cursor:end]):
            raise ValueError("training samples changed content, order, or boundary within a rollout forward pack")
        if pack in seen:
            raise ValueError("a rollout forward pack occurs more than once in the training batch")
        if len(pack) > max_micro:
            raise ValueError(
                f"rollout forward pack size {len(pack)} exceeds micro_batch_size={max_micro}; "
                "replay cannot split a recorded pack"
            )
        seen.add(pack)
        packs.append((cursor, end))
        cursor = end

    if len(packs) % updates:
        raise ValueError(
            f"cannot preserve {len(packs)} forward packs across {updates} optimizer updates; "
            "the forward-pack count must be divisible by num_updates_per_batch"
        )
    packs_per_update = len(packs) // updates
    return [packs[index : index + packs_per_update] for index in range(0, len(packs), packs_per_update)]


class CountPlanner:
    """Fixed-count micro-batches that preserve recorded rollout packs when present."""

    def arrange(self, part: Part, *, num_updates: int, micro_batch_size: int) -> tuple[Part, Plan]:
        plan = _forward_pack_plan(part, num_updates=num_updates, micro_batch_size=micro_batch_size)
        if plan is None:
            plan = _count_plan(
                total=int(part.batch_size),
                num_updates=num_updates,
                micro_batch_size=micro_batch_size,
            )
        return part, plan

    def validate(self, algorithm: StageAlgorithm) -> None:
        return None
