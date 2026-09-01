import os
import torch
from hy_parallelism import bing_utils
from torch.distributed import checkpoint as dcp


_ORIGINAL_TORCH_LOAD = torch.load
_ORIGINAL_DCP_LOAD = dcp.load


def patch_torch_load():
    def patched_load(
        f,
        map_location=None,
        pickle_module=None,
        *,
        weights_only=None,
        mmap=None,
        **pickle_load_args,
    ):
        path = bing_utils.get_nonlocal_file(f)
        return _ORIGINAL_TORCH_LOAD(
            path,
            map_location=map_location,
            pickle_module=pickle_module,
            weights_only=weights_only,
            mmap=mmap,
            **pickle_load_args,
        )

    torch.load = patched_load

    def dcp_load(
        state_dict,
        *,
        checkpoint_id=None,
        storage_reader=None,
        planner=None,
        process_group=None,
        no_dist=False,
    ) -> None:
        checkpoint_id = bing_utils.get_nonlocal_file(checkpoint_id)
        return _ORIGINAL_DCP_LOAD(
            state_dict,
            checkpoint_id=checkpoint_id,
            storage_reader=storage_reader,
            planner=planner,
            process_group=process_group,
            no_dist=no_dist,
        )
    dcp.load = dcp_load


