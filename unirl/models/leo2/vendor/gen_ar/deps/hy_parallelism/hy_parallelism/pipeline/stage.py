from torch.distributed.pipelining.stage import PipelineStage
import loguru
import logging
logger = logging.getLogger(__name__)


class NoShapeInferenceStage(PipelineStage):
    r"""Pipeline stage that supports custom shape inference.
    
    This is implemented by providing a :attr:`_is_mock_stage` to the engine.
    Allowing custom shape inference based on this state.

    .. note::
        This stage should only be used when there are many possible batch sizes. For cases
        with fewer batch size variations, manual shape inference may be sufficient and this
        stage may not be necessary.

    .. seealso::
        :class:`~torch.distributed.pipelining.PipelineStage` for the base pipeline stage class.
    """

    def __init__(self, parallel_engine, *args, **kwargs):
        from hy_parallelism.engines.parallel_engine import BaseParallelEngine
        assert isinstance(parallel_engine, BaseParallelEngine)
        self.parallel_engine = parallel_engine
        super().__init__(*args, **kwargs)

    # def mock_forward(self, *args, **kwargs):
    #     from hy_parallelism.engines.parallel_engine import is_implemented
    #     if is_implemented(self.parallel_engine.mock_pp_forward):
    #         return self.parallel_engine.mock_pp_forward(*args, **kwargs)
    #     else:
    #         return self._original_submod(*args, **kwargs)

    def _shape_inference(self, args, kwargs):
        self.parallel_engine._is_mock_stage = True
        try:
            return super()._shape_inference(args, kwargs)
        finally:
            self.parallel_engine._is_mock_stage = False
