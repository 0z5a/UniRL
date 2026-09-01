from packaging import version

import torch
from torch.distributed.pipelining import *
from torch.distributed.pipelining.schedules import PipelineScheduleSingle

def patch_pipeline_schedule():
    """
    Removing raise for GPipe

    https://github.com/pytorch/pytorch/issues/171312
    """

    def new_pipeline_schedule_single_init(
        self,
        stage,
        n_microbatches,
        loss_fn=None,
        args_chunk_spec=None,
        kwargs_chunk_spec=None,
        output_merge_spec=None,
        scale_grads=True,
    ):
        # Init parent - use explicit super() call to work with monkey-patching
        super(PipelineScheduleSingle, self).__init__(
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
        )
        # Self attributes
        self._stage = stage
        self._num_stages = stage.num_stages
        self._stage_initialized = False

        self.pipeline_order = (
            self._get_pipeline_order()
        )

    # old version works fine
    if version.parse(torch.__version__) > version.parse('2.7.0') and version.parse(torch.__version__) <= version.parse('2.9.1'):
        PipelineScheduleSingle.__init__ = new_pipeline_schedule_single_init