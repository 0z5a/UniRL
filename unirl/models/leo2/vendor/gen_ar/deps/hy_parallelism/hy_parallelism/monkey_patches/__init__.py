import os
import importlib
import loguru



all_patches = []

def has_torch_titan():
    return False # Never try to import torchtitan. torchtitan lead to segfault at exit.
    if importlib.util.find_spec('torchtitan') is None:
        return False
    try:
        from torchtitan.models.moe import MoE
    except:
        return False

TORCH_TITAN_INSTALLED = has_torch_titan()

if TORCH_TITAN_INSTALLED:
    all_patches.append(".torchtitan.patch_torchtitan_quantization")
    all_patches.append(".moe.patch_moe_reset_parameters") # requires patching `patch_torchtitan_quantization` first

# Order matters.
all_patches.extend([
    ".torch.patch_pipeline_schedule",
    ".torch.patch_torch_nn_utils_grad_norm",
    '.torch.patch_pick_load_for_old_python',
    '.torch.patch_checkpoint_wrapper',
    '.torch.patch_dcp_error_pickle_code_object_for_python313',
    # Must run after patch_dcp_error_pickle_code_object_for_python313 so
    # reduce_scatter can pick up the patched _wrap_exception at call time.
    '.torch.patch_dcp_dist_wrapper',
    # '.torch.patch_all_gather_into_tensor_for_distributed_debug',
    '.fsdp.patch_reshard_after_forward',
    # ".torch.patch_torch_grouped_mm",
    '.fsdp.patch_skipped_unshard_in_dual_stream_ac',
    '.fsdp.patch_record_post_forward',

    # 如果不需要创建多个 device mesh, 可以不 patch 这个
    # TODO: 这个在新版本的 pytorch 会报错, `_flatten_mesh_list` attribute is not found
    # ".torch.patch_torch_device_mesh_hash_collision",

    '.transformers.patch_transformers_is_torch_greater_or_equal',
])

if os.environ.get("HY_PARALLELISM_PATCH_TORCH_LOAD", "0") == "1":
    all_patches.append(".fast_access.patch_torch_load")

if os.environ.get('HY_PARALLELISM_ENABLE_JVP', '0') == '1' and TORCH_TITAN_INSTALLED:
    all_patches.append(".moe.patch_for_jvp")

# if os.environ.get('HY_PARALLELISM_PACK_CHECKPOINT_FOR_INFERENCE', '0') == '1':

for patch in all_patches:
    if os.environ.get('RANK', '0') == '0':
        loguru.logger.debug(f'Applying patch: {patch}')
    module_name, function_name = patch.rsplit('.', 1)
    module = importlib.import_module(f'{module_name}', package=__package__)
    patch_function = getattr(module, function_name)
    try:
        patch_function()
    except Exception as e:
        loguru.logger.warning(f"Fail to apply patch: {patch}. Cause: {e}")
