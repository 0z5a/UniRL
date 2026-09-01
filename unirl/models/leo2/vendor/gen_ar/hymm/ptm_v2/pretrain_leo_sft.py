"""SFT trainer for Leo (AngelPTM v2).

This trainer reuses the entire ``pretrain_leo`` pipeline (model / dataset /
forward / Megatron ``pretrain`` plumbing) and differs in exactly one aspect:

    The per-dataset data-iteration progress saved in the checkpoint
    (``epoch_consumed_samples`` / ``consumed_epoch`` / ...) is NOT restored.
    Instead the scalar state is built from scratch, so data iteration always
    starts from sample 0.

Why a dedicated trainer:
    During SFT the dataset usually differs from (and is often much smaller than)
    the pretraining dataset. Restoring a stale ``epoch_consumed_samples`` would
    drive the ``DistributedSampler.start_index`` past the end of the new dataset,
    making the sampler length negative (``ValueError: __len__() should return >= 0``).
    Starting the data iteration from scratch avoids this entirely.

Note:
    Model weights are still loaded from the checkpoint by Megatron's normal load
    path; only the *data-iteration bookkeeping* is reset. The regular
    ``pretrain_leo`` trainer is left completely untouched, so true resume runs are
    unaffected.

Usage:
    Select this trainer via the entry point, e.g.
        --trainer pretrain_leo_sft.launch
"""

from . import pretrain_leo


def build_scalar_state_from_scratch():
    """Build a fresh scalar state with ``epoch_consumed_samples`` starting at 0.

    Mirrors ``pretrain_leo.build_or_load_scalar_state`` but never reads the
    checkpoint's ``training_states.pt``. ``build_scalar_state(None)`` creates a
    default ``MultiModalScalarStates`` (all consumed counters at 0) and registers
    it as the global scalar state.
    """
    from hymm.core.extra_model_provider import build_scalar_state

    scalar_state = build_scalar_state(None)
    scalar_state.current_run_update_steps = 0
    scalar_state.current_forward_times = 0
    pretrain_leo.print_rank_0(
        f"[SFT] build scalar state from scratch (epoch_consumed_samples reset): {scalar_state}"
    )


def launch(frozen_args):
    # ``build_extra_model`` (in pretrain_leo) resolves ``build_or_load_scalar_state``
    # from its module globals at call time, so swapping the attribute here makes the
    # whole run use the from-scratch builder without forking build_extra_model or
    # DatasetsProvider.
    pretrain_leo.build_or_load_scalar_state = build_scalar_state_from_scratch
    return pretrain_leo.launch(frozen_args)
