from functools import wraps

from angelptm.toolkits.patch import PatchesManager

from hymm.config import preprocess_args
from hymm.core.global_vars import set_args as set_hymm_args
from hymm.core.parallel_states import ParallelState
from hymm.config import preprocess_args


def set_global_vars_wrapper(fn):
    @wraps(fn)
    def wrapper(*_args, **_kwargs):
        """ Set global vars of hymm """
        output = fn(*_args, **_kwargs)

        # Set global vars: parallel states
        ParallelState.from_megatron()
        # Set global vars: args
        from megatron.training import get_args as get_megatron_args

        args = get_megatron_args()
        args = preprocess_args(args)
        set_hymm_args(args)

        return output

    return wrapper


PatchesManager.register_patch(
    "megatron.training.initialize.initialize_megatron", set_global_vars_wrapper
)
