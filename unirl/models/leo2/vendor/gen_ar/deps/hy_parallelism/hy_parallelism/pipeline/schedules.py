# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

from torch.distributed.pipelining import ScheduleInterleaved1F1B
from torch.distributed.pipelining.schedules import _ComputationType


# WIP
class InferencePipelineScheduleMulti(ScheduleInterleaved1F1B):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for rank in range(self.pp_group_size):
            new_rank_ops = []
            for op in self.pipeline_order[rank]:
                if op is None:
                    new_rank_ops.append(op)
                    continue
                if op.computation_type in [_ComputationType.FULL_BACKWARD, _ComputationType.BACKWARD_WEIGHT, _ComputationType.BACKWARD_INPUT]:
                    ...
                    # new_rank_ops.append(None)
                else:
                    new_rank_ops.append(op)
            self.pipeline_order[rank] = new_rank_ops
        # raise NotImplementedError