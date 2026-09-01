# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

import math
# from torch.distributed.pipelining import ScheduleZBVZeroBubble
from typing import Optional

import loguru
import torch
from typing import *

from .parallel_engine import BaseParallelEngine

class AriesParallel(BaseParallelEngine):
    def pipeline_manual_split(self):
        DOUBLE_BLOCK_PREFIX="double_blocks."
        SINGLE_BLOCK_PREFIX="single_blocks."

        layer_prefix = DOUBLE_BLOCK_PREFIX
        assert isinstance(self.recursive_get_attr(self.model, layer_prefix), nn.ModuleList)

        if self.pp_size == 2:
            splits = [
                'single_blocks.0',
            ]
        elif self.pp_size == 4:
            # 11d 9d+4s 18s 18s
            splits = [
                'double_blocks.11',
                'single_blocks.4',
                'single_blocks.22',
            ]
        elif self.pp_size == 8:
            # 5d 7d 8d+3s 7s 8s 7s 8s
            # splits = [
            #     'double_blocks.5',
            #     'double_blocks.12',
            #     'single_blocks.3',
            #     'single_blocks.10',
            #     'single_blocks.18',
            #     'single_blocks.25',
            #     'single_blocks.33',
            # ]
            # 5d 5d 5d 5d 10s 10s 10s 10s
            #
            splits = [
                'double_blocks.4',
                'double_blocks.9',
                'double_blocks.14',
                'double_blocks.19',
                'single_blocks.10',
                'single_blocks.20',
                'single_blocks.30',
            ]


        num_stages = len(splits) + 1

        stages = []
        models = []

        for stage_idx in self.stage_ids_this_rank(num_stages):

            def remove_module_fn(model, stage_idx, layer_prefix):
                if stage_idx != 0:
                    model.img_in = None
                    model.txt_in = None
                    model.time_in = None
                    model.vector_in = None
                if stage_idx != num_stages - 1:
                    model.final_layer = None

                start_layer = splits[stage_idx - 1] if stage_idx > 0 else None
                stop_layer = splits[stage_idx] if stage_idx < num_stages - 1 else None

                drop = start_layer is not None
                for i in range(len(model.double_blocks) + len(model.single_blocks)):
                    if i < len(model.double_blocks):
                        layer_prefix = DOUBLE_BLOCK_PREFIX
                        real_idx = i
                    else:
                        layer_prefix = SINGLE_BLOCK_PREFIX
                        real_idx = i - len(model.double_blocks)

                    if f'{layer_prefix}{real_idx}' == start_layer:
                        drop = False
                    if f'{layer_prefix}{real_idx}' == stop_layer:
                        drop = True

                    if drop:
                        setattr(self.recursive_get_attr(model, layer_prefix), str(real_idx), None)


            from functools import partial
            stage, model_chunk = self.build_stage(
                splits,
                layer_prefix,
                stage_idx,
                num_stages,
                None,
                None,
                is_first=stage_idx == 0,
                is_last=stage_idx == num_stages - 1,
                remove_module_fn=partial(remove_module_fn, stage_idx=stage_idx, layer_prefix=layer_prefix),
            )

            stages.append(stage)
            models.append(model_chunk)


        return stages, models

    def config_forward_args(self):
        self.set_static_kwargs_keys(['hidden_states', 'guidance', 'encoder_attention_mask', 'timestep', 'text_states'])
        self.set_n_pp_args(7)

