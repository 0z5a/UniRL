#!/usr/bin/env python3
"""Check one- or two-node DeepEP dispatch/combine correctness."""

from __future__ import annotations

import os
import time

import deep_ep
import torch
import torch.distributed as dist
from deep_ep.version import __version__, __version_suffix__


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    world_size = int(os.environ["WORLD_SIZE"])
    if local_world_size != 8 or world_size not in (8, 16):
        raise RuntimeError(
            "expected 8 local ranks and either 8 or 16 total ranks, "
            f"got local_world_size={local_world_size}, world_size={world_size}"
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    group = dist.new_group(list(range(world_size)), backend="nccl")
    buffer = None
    started = time.perf_counter()
    try:
        deep_ep.Buffer.set_num_sms(8)
        buffer = deep_ep.Buffer(
            group,
            num_nvl_bytes=256 * 1024 * 1024,
            num_rdma_bytes=(256 * 1024 * 1024 if world_size > local_world_size else 0),
            explicitly_destroy=True,
        )

        tokens = 64
        hidden = 128
        num_experts = 64
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20260903 + rank)
        x = torch.randn(
            (tokens, hidden),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        topk_idx = (torch.arange(tokens, device="cuda") % num_experts).view(-1, 1)
        topk_idx = topk_idx.to(deep_ep.topk_idx_t)
        topk_weights = torch.ones((tokens, 1), device="cuda", dtype=torch.float32)

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,
        ) = buffer.get_dispatch_layout(topk_idx, num_experts)
        if world_size == local_world_size and num_tokens_per_rdma_rank is not None:
            raise RuntimeError("expected an intranode layout with no RDMA-rank counts")
        if world_size > local_world_size and num_tokens_per_rdma_rank is None:
            raise RuntimeError(
                f"expected RDMA-rank counts for world_size={world_size}, local_world_size={local_world_size}"
            )

        recv_x, _, _, recv_counts, handle, _ = buffer.dispatch(
            x=x,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            config=buffer.get_dispatch_config(world_size),
        )
        combined_x, _, _ = buffer.combine(
            x=recv_x,
            handle=handle,
            config=buffer.get_combine_config(world_size),
        )
        recv_cached, _, _, _, _, _ = buffer.dispatch(
            x=x,
            handle=handle,
            config=buffer.get_dispatch_config(world_size),
        )
        combined_cached, _, _ = buffer.combine(
            x=recv_cached,
            handle=handle,
            config=buffer.get_combine_config(world_size),
        )
        torch.cuda.synchronize()

        errors = torch.stack(
            (
                (combined_x.float() - x.float()).abs().max(),
                (combined_cached.float() - x.float()).abs().max(),
            )
        )
        dist.all_reduce(errors, op=dist.ReduceOp.MAX, group=group)
        if errors.max().item() != 0.0:
            raise RuntimeError(f"dispatch/combine mismatch: {errors.tolist()}")
        if len(recv_counts) != num_experts // world_size:
            raise RuntimeError(f"unexpected local expert counts: {recv_counts}")

        dist.barrier(group=group)
        if rank == 0:
            print(
                {
                    "status": f"DEEPEP_{world_size}RANK_ROUNDTRIP_OK",
                    "deep_ep": f"{__version__}+{__version_suffix__}",
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "world_size": world_size,
                    "local_world_size": local_world_size,
                    "tokens_per_rank": tokens,
                    "hidden": hidden,
                    "num_experts": num_experts,
                    "num_sms": deep_ep.Buffer.num_sms,
                    "nvl_buffer_mib_per_rank": 256,
                    "max_abs_error_first": errors[0].item(),
                    "max_abs_error_cached": errors[1].item(),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                },
                flush=True,
            )
    finally:
        if buffer is not None:
            buffer.destroy()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
