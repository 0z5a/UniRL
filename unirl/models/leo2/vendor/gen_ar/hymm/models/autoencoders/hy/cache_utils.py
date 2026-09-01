import torch

from hy_parallelism.utils import is_recomputing
from hy_parallelism.training.cast_device import cast_to_device, CopyWork
from hy_parallelism.training.checkpointing import forward_with_checkpointing

VAE_OFFLOAD_POOL = 'vae_pool'

global_offloaded_tensor_pool = {}


def write_feat_cache(feat_cache, idx, val):
    if not is_recomputing():
        feat_cache[idx] = val
    else:
        if isinstance(val, torch.Tensor):
            return
        feat_cache[idx] = val


def offload_cache_key(tensor: torch.Tensor) -> tuple:
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tensor.dtype,
        tuple(tensor.stride()),
    )


def get_offloaded_tensor(tensor):
    assert isinstance(tensor, torch.Tensor)
    key = offload_cache_key(tensor)
    if key in global_offloaded_tensor_pool:
        return global_offloaded_tensor_pool[key]
    ret = cast_to_device(
        tensor, 'cpu', pool=VAE_OFFLOAD_POOL,
        use_side_stream_for_tensor_copies=True, async_op=True,
    )
    global_offloaded_tensor_pool[key] = ret
    return ret


def checkpoint_cache_state(feat_idx, feat_cache, module, snapshot_cache=True):
    if feat_cache is None:
        return
    if not torch.is_grad_enabled():
        return
    state = module.__dict__.setdefault("_feat_ac_state", {"idx": 0, "snaps": []})
    if is_recomputing():
        feat_idx[0] = state["idx"]
        if snapshot_cache:
            snap = state["snaps"].pop() if state["snaps"] else None
            if snap:
                for i, v in snap.items():
                    if isinstance(v, CopyWork):
                        v = v.wait()
                    feat_cache[i] = v
    else:
        state["idx"] = feat_idx[0]
        if snapshot_cache and torch.is_grad_enabled():
            i0 = feat_idx[0]
            state["snaps"].append({
                i: (
                    get_offloaded_tensor(feat_cache[i])
                    if isinstance(feat_cache[i], torch.Tensor) else feat_cache[i]
                )
                for i in range(i0, len(feat_cache))
            })


def move_cache(feat_cache, device):
    if feat_cache is None:
        return
    for i in range(len(feat_cache)):
        if isinstance(feat_cache[i], torch.Tensor):
            feat_cache[i] = feat_cache[i].to(device)


def vae_checkpoint_no_split_modules():
    from hymm.models.autoencoders.flux import flux2
    from hymm.models.autoencoders.hy import (
        autoencoder_kl_causal_rmsnorm_3d as hy3d,
        autoencoder_kl_causal_rmsnorm_3d_v3 as hy_v3,
        autoencoder_kl_causal_rmsnorm_3d_v3_3 as hy_v3_3,
    )

    return (
        hy3d.AttnBlock,
        hy3d.ResnetBlock,
        flux2.AttnBlock,
        flux2.ResnetBlock,

        hy_v3.Up_ResidualBlock,
        hy_v3.AttentionBlock,

        hy_v3_3.Up_ResidualBlock,
        hy_v3_3.AttentionBlock,
    )


def setup_vae_checkpointing(vae, args):
    if not getattr(args, "use_vae_decoder", False):
        return
    if vae is None:
        return

    from hy_parallelism.distributed.fsdp_util import apply_fsdp_checkpointing

    apply_fsdp_checkpointing(
        vae,
        no_split_modules=vae_checkpoint_no_split_modules(),
        p=1,
        use_reentrant=False,
        activation_offloading=True,
        activation_offload_list=[0],
    )
