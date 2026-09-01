
import re
import os
import math
# from torch.distributed.pipelining import ScheduleZBVZeroBubble
from typing import Optional
from torch.distributed._tensor import DeviceMesh, DTensor, Shard, Replicate
from torch import distributed as dist
from torch import nn

import loguru
import torch
from typing import *

from hy_parallelism.pipeline import hy_get_schedule_class as get_schedule_class
from hy_parallelism.engines.parallel_engine import BaseParallelEngine
from hy_parallelism.parallel_states import get_parallel_state


class GeminiParallelEngine(BaseParallelEngine):

    def is_expert(self, fqn):
        return 'experts' in fqn

    def is_moe_router(self, module_full_name, module):
        return module_full_name.endswith('mlp.gate')

    def __init__(self, model, ds_config=None, weight_prec='bf16', *args, **kwargs):
        from hy_parallelism.common.constants import TORCH_DTYPE_MAP
        self.weight_prec = TORCH_DTYPE_MAP[weight_prec]

        super().__init__(model, *args, **kwargs)


        if ds_config:
            import deepspeed
            from hymm.utils.monitor import MonitorMaster
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
        from hy_parallelism.utils import hook_safe_forward
        self.kvcache_context = hook_safe_forward(self.fsdp_models[-1], 'infer_forward', refresh_fn=self.force_refresh_pipeline_scheduler)
        self.kvcache_context.__enter__()

    def to_normal_mode(self):
        if hasattr(self, 'kvcache_context'):
            self.kvcache_context.__exit__(None, None, None)

    def clear_states_after_auto_benchmark(self):
        self.clear_kv_cache()

    def infer_forward(self, *args, **kwargs):
        from hy_parallelism.utils import hook_safe_forward
        with hook_safe_forward(self.fsdp_models[-1], 'infer_forward', refresh_fn=self.force_refresh_pipeline_scheduler), hook_safe_forward(self.fsdp_models[-1].language_model, 'infer_forward', refresh_fn=None):
            return self(*args, **kwargs)

    def pre_process_input(self, *args, **kwargs):
        if 'rope_image_info' in kwargs:
            rope_image_info = kwargs['rope_image_info']
            from hy_parallelism.utils import obj_to_tensor, stack_tensor, tensor_to_obj, batch_obj_to_tensor
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

    def apply_ep_after_init(self, model):
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
        from hy_parallelism.distributed.fsdp_util import apply_fsdp2

        fsdp_kwargs = dict(
            param_dtype=self.weight_prec,
            reduce_dtype=self.weight_prec,
        )
        parallel_dims = self.parallel_dims
        loguru.logger.info(f'{fsdp_kwargs=}')
        # TODO: CHECK THIS dtype
        model.vision_model_so = model.vision_model_so.to(self.weight_prec)
        model.vision_aligner_so = model.vision_aligner_so.to(self.weight_prec)
        return apply_fsdp2(
            model=model,
            # blocks=list(model.language_model.transformer.h) + ([model.vision_model_so] if hasattr(model, 'vision_model_so') else []),
            blocks=list(model.language_model.transformer.h),
            # dp_mesh=ep_mesh if parallel_dims.ep_enabled else world_mesh[('dp',)],
            default_fsdp_mesh=parallel_dims.default_fsdp_mesh,
            expert_fsdp_mesh=parallel_dims.expert_fsdp_mesh,
            **fsdp_kwargs,
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
        self.set_replacable_kwargs(['idx'])
        # self.set_static_kwargs_keys([
        #     'x_t',
        #     't',
        #     'target',
        #     'diffusion_loss_fn',
        #     'src_x',
        #     'src_t',
        #     'src_image_mask',
        #     'input_pos',
        #     'iw_ih_scatter_index',
        #     'iw_ih_scatter_src',
        #     'timestep_scatter_index',
        #     'timestep_scatter_src',
        #     'text_mask',
        #     'image_mask',
        #     'image_loss_weight',
        #     'attention_mask',
        #     'freqs_cos',
        #     'freqs_sin',
        #     'data_type',
        #     'und_images',
        #     'und_image_masks',
        #     'src_face_embedding',
        #     'rope_image_info',
        #     'vision_encoder_kwargs',
        #     'sample_offsets',
        #     'n_samples',
        #     'return_loss',
        #     'first_step',
        #     'guidance',
        #     'guidance_scatter_index'
        # ])
        self.set_n_pp_args(0)

    def enable_kv_cache(self):
        self.set_high_prio_m_microbaches(1)

    @classmethod
    def get_inference_pipeline(cls, model, ckpt_path=None, kv_cache_enabled=False):
        engine_kwargs = dict(
            model=model,
            load_ckpt_path=ckpt_path,
            micro_batch_size=1,
            enable_autocast=True,
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

    def clip_grad_norm_(
        self,
        parameters,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach=None,
    ):
        if self.enable_pp or self.enable_ep:
            # This original implementation could lead to cross mesh computation.
            # return clip_grad_norm_(
            #     parameters, max_norm, norm_type, error_if_nonfinite, foreach, pp_mesh=self.parallel_dims.pp_mesh
            # )
            if self.enable_ep:
                from hy_parallelism.utils import clip_grad_norm_by_mesh_old_ep_

                self._tag_is_expert_info_to_param_and_grad()
                return clip_grad_norm_by_mesh_old_ep_(
                    self.optimizer_container.param_groups_by_mesh, max_norm, norm_type, error_if_nonfinite, foreach, pp_mesh=self.parallel_dims.pp_mesh,
                )
            else:
                from hy_parallelism.utils import clip_grad_norm_by_mesh_
                return clip_grad_norm_by_mesh_(
                    self.optimizer_container.param_groups_by_mesh, max_norm, norm_type, error_if_nonfinite, foreach, pp_mesh=self.parallel_dims.pp_mesh,
                )
        else:
            grad_norm = nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite, foreach)
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor()
            return grad_norm

    @torch.no_grad()
    def get_global_grad_norm(self) -> float:
        from hy_parallelism.utils import get_global_grad_norm_by_mesh_old_ep
        if self.enable_ep:
            self._tag_is_expert_info_to_param_and_grad()
            return get_global_grad_norm_by_mesh_old_ep(
                self.optimizer_container.param_groups_by_mesh,
                norm_type=2.0,
                error_if_nonfinite=False,
                foreach=None,
                pp_mesh=self.pp_mesh
            ).item()
        else:
            return super().get_global_grad_norm()


    # NOTE(kevinkhwu): 因为 hymm/parallelism 的 engine 视乎被改过（已经观察到关于ckpt的读写行为与之前不一致）
    #     为了确保兼容性，直接复用旧的ckpt读写实现（即 hymm/parallelism 里的）
    def create_checkpoint_manager(self, ckpt_dir, keep_latest_k=-1):
        from hy_parallelism.checkpoint.checkpoint_manager import CheckpointManager, Checkpoint
        self.MODEL_FOLDER = 'weights'
        self.OPTIMIZER_FOLDER = 'optimizers'
        self.TRAINING_STATES_FOLDER = 'states'

        self.model_checkpoint_manager = CheckpointManager(
            {},
            Checkpoint(
                dump_folder=ckpt_dir,
                folder=self.MODEL_FOLDER,
                enable_checkpoint=True,
                keep_latest_k=keep_latest_k,
            ),
            model_parts=self.fsdp_models,
        )
        # optimizer 和 weights 分开存，方便后面储存不够时只删 optimizer
        if hasattr(self, 'optimizer_container') and hasattr(self, 'lr_scheduler_container'):
            self.optimizer_checkpoint_manager = CheckpointManager(
                {},
                Checkpoint(
                    dump_folder=ckpt_dir,
                    folder=self.OPTIMIZER_FOLDER,
                    enable_checkpoint=True,
                    keep_latest_k=keep_latest_k,
                ),
                model_parts=None,
                optimizers=self.optimizer_container,
                lr_schedulers=self.lr_scheduler_container,
            )

        # keys that not appears in self.training_states will not be loaded
        self.training_states_checkpoint_manager = CheckpointManager(
            {'states': self.training_states},
            Checkpoint(
                dump_folder=ckpt_dir,
                folder=self.TRAINING_STATES_FOLDER,
                enable_checkpoint=True,
                keep_latest_k=keep_latest_k,
            ),
            model_parts=None,
        )


    # NOTE(kevinkhwu): 因为 hymm/parallelism 的 engine 视乎被改过（已经观察到关于ckpt的读写行为与之前不一致）
    #     为了确保兼容性，直接复用旧的ckpt读写实现（即 hymm/parallelism 里的）
    def load_checkpoint(self,
                        load_dir,
                        tag=None,  # unused
                        load_module_strict=True,  # unused
                        load_optimizer_states=None, # True False None
                        load_lr_scheduler_states=True,
                        load_module_only=False,  # unused
                        custom_load_fn=None,  # unused
                        ):
        """
        2 种输入:
            1. torch ckpt, 先转dcp再读，适应已经切分好的模型
            2. dcp 目录, 下面必须要有 'weights', 'optimizer' 这种文件夹
                如果下面没有这个，而只有distcp, 则建议用 checkpoint_manager.load_ckpt
        """
        from pathlib import Path
        path = Path(load_dir)
        if tag is not None:
            path = path / tag
        if path.is_dir():
            # self.load_dcp_state_dict(os.path.join(load_dir, self.MODEL_FOLDER))

            weight_path = Path(path) / self.MODEL_FOLDER
            if not weight_path.exists():
                assert not load_optimizer_states
                weight_path = Path(path)
                assert len(list(weight_path.glob('*.distcp'))) > 0
            self.model_checkpoint_manager.load_from_path(str(weight_path))



            if os.path.isdir(os.path.join(path, self.OPTIMIZER_FOLDER)) and load_optimizer_states is not False:
                try:
                    self.optimizer_checkpoint_manager.load_from_path(os.path.join(path, self.OPTIMIZER_FOLDER)) # TODO: strict=False to allow loading old checkpoints.
                except:
                    if load_optimizer_states is True:
                        msg = (
                            'The old `hymm/parallelism` optimizer state_dict implementation is not supported any more. '
                            'Loading old optimizer states can cause error. Loading the new one works fine '
                            'Here we temporarily skip the error.'
                        )
                        loguru.logger.warning(msg)
                        # raise
            else:
                if load_optimizer_states is True:
                    msg = f'The user specifies load_optimizer_states=True, but {os.path.join(path, self.OPTIMIZER_FOLDER)} does not exist'
                    loguru.logger.error(msg)
                    raise FileNotFoundError(msg)
            if os.path.isdir(os.path.join(path, self.TRAINING_STATES_FOLDER)):
                self.training_states_checkpoint_manager.load_from_path(os.path.join(path, self.TRAINING_STATES_FOLDER), strict=False) # TODO: strict=False to allow loading old checkpoints.
            # ---------------------------
            # Best-effort restore of non-tensor training states (e.g., scalar_state/update_steps)
            # Torch distributed checkpoint (DCP) is tensor-centric; python objects in client_state may be
            # dropped or not reliably persisted depending on torch version/backends.
            # We therefore also support a lightweight sidecar file saved by save_checkpoint().
            # ---------------------------
            sidecar_path = os.path.join(path, "client_state.pt")
            if os.path.isfile(sidecar_path):
                try:
                    sidecar_state = torch.load(sidecar_path, map_location="cpu", mmap=True)
                    if isinstance(sidecar_state, dict):
                        # NOTE: we never restore/override `config` from checkpoint sidecar.
                        for k, v in sidecar_state.items():
                            if k == "config":
                                continue
                            self.training_states[k] = v
                except Exception as e:
                    loguru.logger.warning(f"Failed to load sidecar client_state from {sidecar_path}: {type(e)} {e}")

            # If still missing scalar_state (legacy checkpoints), try to infer step from directory name.
            if "scalar_state" not in self.training_states:
                inferred_step = None
                base = os.path.basename(str(path))
                # pure-torch trainers often use zero-padded numeric tags, e.g. 0000250
                if base.isdigit():
                    inferred_step = int(base)
                else:
                    # also support step-123 style folders
                    m = re.search(r"(?:^|/)step-(\d+)(?:/|$)", str(path))
                    if m:
                        inferred_step = int(m.group(1))
                if inferred_step is not None:
                    self.training_states["scalar_state"] = [{"update_steps": inferred_step}]
        else:
            self.load_full_sd_by_cvt_to_dcp(path)
            assert not load_optimizer_states
        return path, self.training_states

    # NOTE(kevinkhwu): 因为 hymm/parallelism 的 engine 视乎被改过（已经观察到关于ckpt的读写行为与之前不一致）
    #     为了确保兼容性，直接复用旧的ckpt读写实现（即 hymm/parallelism 里的）
    def save_checkpoint(
            self, save_dir=None, tag=None,  client_state={},
            # unused
            save_latest=True, exclude_frozen_parameters=False,
            save_optimizer_states=True,
    ):
        if save_dir is not None:
            if self.ckpt_dir is not None:
                model_old_folder = self.model_checkpoint_manager.folder
                if hasattr(self, 'optimizer_checkpoint_manager'):
                    optimizer_old_folder = self.optimizer_checkpoint_manager.folder
                states_old_folder = self.training_states_checkpoint_manager.folder
            self.model_checkpoint_manager.folder = save_dir
            if hasattr(self, 'optimizer_checkpoint_manager'):
                self.optimizer_checkpoint_manager.folder = save_dir
            self.training_states_checkpoint_manager.folder = save_dir



        if tag is None:
            tag = 'latest'


        self.reshard()
        self.model_checkpoint_manager.save(step_or_tag=tag)
        if save_optimizer_states:
            if hasattr(self, 'optimizer_checkpoint_manager'):
                self.optimizer_checkpoint_manager.save(step_or_tag=tag)
        if isinstance(client_state, dict):
            old_state = self.training_states.copy()
            self.training_states.update(client_state)
        else:
            raise NotImplementedError
        self.training_states_checkpoint_manager.save(step_or_tag=tag)
        if isinstance(client_state, dict):
            self.training_states.clear()
            self.training_states.update(old_state)

        # ---------------------------
        # Sidecar save for non-tensor client_state (e.g., scalar_state with python ints/dicts/lists).
        # This makes resuming step/update_steps robust regardless of DCP's non-tensor support.
        # Only rank0 writes; then barrier to ensure visibility for all ranks on shared FS.
        # ---------------------------
        try:
            # resolve tag directory where weights/optimizers/states are saved
            if save_dir is not None:
                tag_dir = os.path.join(save_dir, str(tag))
            else:
                tag_dir = os.path.join(self.model_checkpoint_manager.folder, str(tag))
            if dist.is_available() and dist.is_initialized():
                _rank = dist.get_rank()
            else:
                _rank = 0
            if _rank == 0:
                os.makedirs(tag_dir, exist_ok=True)
                torch.save(client_state, os.path.join(tag_dir, "client_state.pt"))
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception as e:
            loguru.logger.warning(f"Failed to save sidecar client_state.pt for tag={tag}: {type(e)} {e}")


        if save_dir is not None and self.ckpt_dir is not None:
            self.model_checkpoint_manager.folder = model_old_folder
            if hasattr(self, 'optimizer_checkpoint_manager'):
                self.optimizer_checkpoint_manager.folder = optimizer_old_folder
            self.training_states_checkpoint_manager.folder = states_old_folder