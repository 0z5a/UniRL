from pathlib import Path


def patch_megatron_for_ptm_v2():
    """ Patch Megatron-LM to support PTM v2 training. """
    from angelptm.toolkits.patch import apply_patches

    ptm_v2_megatron_patchers_root = Path(__file__).parents[2] / "deps/AngelPTM/angelptm/megatron"
    apply_patches(str(ptm_v2_megatron_patchers_root), "angelptm")

    hymm_megatron_patchers_root = Path(__file__).parent / "megatron"
    apply_patches(str(hymm_megatron_patchers_root), "hymm.ptm_v2")

    import megatron
    # To solve the segment fault when saving ckpt
    megatron.core.energy_monitor.has_nvml = False
