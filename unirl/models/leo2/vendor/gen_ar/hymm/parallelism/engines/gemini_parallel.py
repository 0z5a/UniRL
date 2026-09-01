
import math
# from torch.distributed.pipelining import ScheduleZBVZeroBubble
from typing import Optional
from torch.distributed._tensor import DeviceMesh, DTensor, Shard, Replicate
from torch import distributed as dist
from torch import nn

import loguru
import torch
from typing import *

from hymm.parallelism.pipeline import hy_get_schedule_class as get_schedule_class
from .parallel_engine import BaseParallelEngine
from ..parallel_states import get_parallel_state



class GeminiParallelEngine(BaseParallelEngine):
    def __init__(self, model, ds_config=None, *args, **kwargs):
        model.is_expert = lambda fqn: 'experts' in fqn
        super().__init__(model, *args, **kwargs)


        if ds_config:
            import deepspeed
            from ...utils.monitor import MonitorMaster
            monitor_config = deepspeed.runtime.config.get_monitor_config(ds_config)
            self._monitor = MonitorMaster(monitor_config)
        else:
            self._monitor = None

        assert len(self.fsdp_models)
        assert len(self.fsdp_models) == 1
        self.set_kv_cache = self.fsdp_models[-1].set_kv_cache
        self.clear_kv_cache = self.fsdp_models[-1].clear_kv_cache
        self.set_rope_cache = self.fsdp_models[-1].set_rope_cache
        self.clear_rope_cache = self.fsdp_models[-1].clear_rope_cache
        self.config = self.fsdp_models[-1].config

    def to_kvcache_mode(self):
        from hymm.parallelism.utils import hook_safe_forward
        self.kvcache_context = hook_safe_forward(self.fsdp_models[-1], 'infer_forward', refresh_fn=self.force_refresh_pipeline_scheduler)
        self.kvcache_context.__enter__()

    def to_normal_mode(self):
        if hasattr(self, 'kvcache_context'):
            self.kvcache_context.__exit__(None, None, None)

    def clear_states_after_auto_benchmark(self):
        self.clear_kv_cache()

    def infer_forward(self, *args, **kwargs):
        from hymm.parallelism.utils import hook_safe_forward
        with hook_safe_forward(self.fsdp_models[-1], 'infer_forward', refresh_fn=self.force_refresh_pipeline_scheduler), hook_safe_forward(self.fsdp_models[-1].language_model, 'infer_forward', refresh_fn=None):
            return self(*args, **kwargs)

    def pre_process_input(self, *args, **kwargs):
        if 'rope_image_info' in kwargs:
            rope_image_info = kwargs['rope_image_info']
            from hymm.parallelism.utils import obj_to_tensor, stack_tensor, tensor_to_obj, batch_obj_to_tensor
            kwargs['rope_image_info'] = batch_obj_to_tensor(rope_image_info)
        kwargs['real_seqlen'] = kwargs['idx'].shape[1]
        return args, kwargs



    @property
    def monitor(self):
        return self._monitor

    def pipeline_manual_split(self):
        LAYER_PREFIX = "language_model.transformer.h."

        schedule_class = get_schedule_class(self.pipeline_parallel_schedule)
        from torch.distributed.pipelining.schedules import PipelineScheduleMulti
        if issubclass(schedule_class, PipelineScheduleMulti):
            mul = 2
            raise NotImplementedError('提前按非interleave切分过了')
        else:
            mul = 1
        n_layers = len(self.model.language_model.transformer.h)
        splits = [n_layers // (self.pp_size * mul) * i for i in range(1, self.pp_size * mul)]
        splits = [f'{LAYER_PREFIX}{i}' for i in splits]
        # splits = self.model.pp_splits[int(math.log(self.pp_size, 2)) - 1 + index_bias]


        num_stages = len(splits) + 1

        stages = []
        models = []
        for stage_idx in self.stage_ids_this_rank(num_stages):

            def remove_module_fn(model, stage_idx, layer_prefix):

                start_layer = splits[stage_idx - 1] if stage_idx > 0 else None
                stop_layer = splits[stage_idx] if stage_idx < num_stages - 1 else None

                drop = start_layer is not None
                for i in range(len(model.language_model.transformer.h)):
                    layer_prefix = LAYER_PREFIX
                    real_idx = i

                    if f'{layer_prefix}{real_idx}' == start_layer:
                        drop = False
                    if f'{layer_prefix}{real_idx}' == stop_layer:
                        drop = True

                    # 已经提前切过
                    # if drop:
                    #     setattr(self.recursive_get_attr(model, layer_prefix), str(real_idx), None)


                assert hasattr(model, 'pre_process') and hasattr(model, 'post_process')
                if stage_idx == 0:
                    if not (model.pre_process and not model.post_process):
                        loguru.logger.debug(f'{model.pre_process=} {model.post_process=} {stage_idx=} {num_stages=}')
                        raise ValueError

                    # model.language_model.transformer.ln_f = None

                    # assert model.pre_process and not model.post_process
                if stage_idx == num_stages - 1:
                    if not (not model.pre_process and model.post_process):
                        loguru.logger.debug(f'{model.pre_process=} {model.post_process=} {stage_idx=} {num_stages=}')
                        raise ValueError

                    # model.language_model.transformer.wte = None
                    # assert not model.pre_process and model.post_process


            from functools import partial
            stage, model_chunk = self.build_stage(
                stage_idx,
                num_stages,
                is_first=stage_idx == 0,
                is_last=stage_idx == num_stages - 1,
                remove_module_fn=partial(remove_module_fn, stage_idx=stage_idx, layer_prefix=LAYER_PREFIX),
            )

            stages.append(stage)
            models.append(model_chunk)

        return stages, models

    def apply_ep(self, model):
        return
        assert self.parallel_dims.ep_enabled
        # ptm moe don't use apply_ep

        def set_module_from_path(model: nn.Module, path: str, path_new: str, value: any):
            attrs = path.split(".")
            attrs_new = path_new.split(".")
            if len(attrs) == 1:
                setattr(model, attrs_new[0], value)
                if attrs_new[0] != attrs[0]:
                    delattr(model, attrs[0])
            else:
                next_obj = getattr(model, attrs[0])
                set_module_from_path(next_obj, ".".join(attrs[1:]), ".".join(attrs_new[1:]), value)


        ep_fqn_list = []
        ep_fqn_ep_list = []
        ep_local_chunk_list = []
        for fqn, param in model.named_parameters():
            if model.is_expert(fqn):
                from torch.distributed.tensor import distribute_tensor
                dtensor = distribute_tensor(
                    param.data,
                    self.parallel_dims.ep_mesh,
                    placements=[Shard(0)],
                )

                local_chunk = torch.nn.Parameter(dtensor.to_local(), requires_grad=param.requires_grad)
                new_fqn = fqn

                ep_fqn_list.append(fqn)
                ep_fqn_ep_list.append(new_fqn)
                ep_local_chunk_list.append(local_chunk)

        for fqn, fqn_ep, local_chunk in zip(ep_fqn_list, ep_fqn_ep_list, ep_local_chunk_list):
            set_module_from_path(model, fqn, fqn_ep, local_chunk)

    def apply_fsdp(self, model):
        from hymm.parallelism.fsdp_util import apply_fsdp2

        fsdp_kwargs = dict(
            param_dtype=self.weight_prec,
            reduce_dtype=self.weight_prec,
        )
        parallel_dims = self.parallel_dims
        loguru.logger.info(f'{fsdp_kwargs=}')
        return apply_fsdp2(
            model=model,
            # blocks=list(model.language_model.transformer.h) + ([model.vision_model_so] if hasattr(model, 'vision_model_so') else []),
            blocks=list(model.language_model.transformer.h),
            # dp_mesh=ep_mesh if parallel_dims.ep_enabled else world_mesh[('dp',)],
            default_fsdp_mesh=parallel_dims.default_fsdp_mesh,
            expert_fsdp_mesh=parallel_dims.expert_fsdp_mesh,
            **fsdp_kwargs,
            pp_enabled=self.enable_pp,
            ep_enabled=self.parallel_dims.ep_enabled,
            cpu_offload=self.cpu_offload,
            reshard_after_forward_policy="always",
        )

    def get_resolution_key(self, idx, *args, **kwargs):
        ret = tuple(idx.shape)
        if 'first_step' in kwargs:
            ret += (kwargs['first_step'],)
        return ret

    def get_batch_size(self, idx, *args, **kwargs):
        return idx.shape[0]

    # def get_eval_ret(self):
    #     assert len(self.fsdp_models) == 1
    #     return self.fsdp_models[0]._eval_ret

    def reshard(self):
        for m in self.fsdp_models:
            for block in m.language_model.transformer.h:
                if block is not None:
                    block.reshard()
            m.reshard() # after validation, we need to reshard the parameters

    def config_forward_args(self):
        self.set_static_kwargs_keys([
            'x_t',
            't',
            'target',
            'diffusion_loss_fn',
            'src_x',
            'src_t',
            'src_image_mask',
            'input_pos',
            'iw_ih_scatter_index',
            'iw_ih_scatter_src',
            'timestep_scatter_index',
            'timestep_scatter_src',
            'text_mask',
            'image_mask',
            'image_loss_weight',
            'attention_mask',
            'freqs_cos',
            'freqs_sin',
            'data_type',
            'und_images',
            'und_image_masks',
            'src_face_embedding',
            'rope_image_info',
            'vision_encoder_kwargs',
            'sample_offsets',
            'n_samples',
            'return_loss',
            'first_step',
            'guidance',
            'guidance_scatter_index',
            'timestep_r_scatter_index',
            'r'
        ])
        self.set_n_pp_args(0)

    def enable_kv_cache(self):
        self.set_high_prio_m_microbaches(1)

    @classmethod
    def get_inference_pipeline(cls, model, ckpt_path=None, kv_cache_enabled=False):
        engine_kwargs = dict(
            model=model,
            load_ckpt_path=ckpt_path,
            micro_batch_size=1,
            pp_enable_autocast=True,
            # pp_enable_autocast=False,
            autocast_prec='bf16',
            weight_prec='bf16',
            # cpu_offload=True,
        )

        if kv_cache_enabled:
            if 'micro_batch_size' in engine_kwargs:
                del engine_kwargs['micro_batch_size']
            engine_kwargs['m_microbatch'] = 1

        return cls(**engine_kwargs)
