import os
from dataclasses import dataclass
from functools import cached_property
from functools import partial
from torch import nn

import einops
import loguru
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from ..models.autoregressive.transfusion import Transfusion
from ..models.autoregressive.multimodal_transfusion import MultiModalTransfusion
from ..utils.fsdp_wrapper import FSDPEngine
from ..utils.helpers import get_obj_from_str
from ..utils.torch_utils import (
    build_optimizer,
)
from ..constants import LAUNCHER
from ..core.global_vars import get_nccl_timeout


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    sp: int
    tp: int
    pp: int
    world_size: int

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, sp, tp, pp = (
            self.dp_replicate,
            self.dp_shard,
            self.sp,
            self.tp,
            self.pp,
        )
        for d in (dp_replicate, sp, tp, pp):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"

        assert dp_shard == -1 or dp_shard >= 1, " dp_shard must -1 or >=1."
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (dp_replicate * sp * tp * pp)
        assert dp_shard >= 1

        assert dp_replicate * dp_shard * sp * tp * pp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"sp({sp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

    def build_mesh(self, device_type):
        from torch.distributed.device_mesh import init_device_mesh

        dims = []
        names = []
        for d, name in zip(
                [self.pp, self.dp_replicate, self.dp_shard, self.sp, self.tp],
                ["pp", "dp_replicate", "dp_shard", "sp", "tp"],
        ):
            if d <= 1:
                # if d <= 1 and name not in ['tp', 'sp']:
                continue
            dims.append(d)
            names.append(name)

        # logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        # Create all the submesh here to ensure all required process groups are
        # initialized:
        # Mesh for data loading (no communication on this mesh)
        dp_mesh_dim_names = []
        # Mesh for param sharding
        dp_shard_sp_mesh_dim_names = []
        # Mesh for loss all-reduce
        dp_sp_mesh_dim_names = []

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
            dp_sp_mesh_dim_names.append("dp_replicate")
        if self.dp_shard_enabled:
            dp_mesh_dim_names.append("dp_shard")
            dp_shard_sp_mesh_dim_names.append("dp_shard")
            dp_sp_mesh_dim_names.append("dp_shard")
        if self.sp_enabled:
            dp_shard_sp_mesh_dim_names.append("sp")
            dp_sp_mesh_dim_names.append("sp")

        if dp_mesh_dim_names != []:
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        if dp_shard_sp_mesh_dim_names != []:
            mesh[tuple(dp_shard_sp_mesh_dim_names)]._flatten(
                mesh_dim_name="dp_shard_sp"
            )
        if dp_sp_mesh_dim_names != []:
            mesh[tuple(dp_sp_mesh_dim_names)]._flatten(mesh_dim_name="dp_sp")

        return mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def sp_enabled(self):
        return self.sp > 1

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @cached_property
    def non_data_parallel_size(self):
        return self.sp * self.tp * self.pp


def destroy_parallel_group():
    """Destroy the parallel group."""
    dist.destroy_process_group()

def qkv_weight_conversion(weight, config):
    weight = einops.rearrange(
        weight,
        '(n_query_groups total_qkv head_size) in_ch -> n_query_groups total_qkv  head_size in_ch',
        n_query_groups=config.n_query_groups, head_size=config.head_size
    )

    weight = weight.split((config.n_head // config.n_query_groups, 1, 1), dim=1)

    transform = lambda x: x.reshape(-1, x.shape[-1])

    q_weight = transform(weight[0])
    k_weight = transform(weight[1])
    v_weight = transform(weight[2])
    return q_weight, k_weight, v_weight


def convert_tp_friendly_qkv(model):
    for layer_id, block in enumerate(model.transformer.h):
        attn = block.attn
        if not attn.tp_friendly_qkv:
            config = attn.config
            factory_kwargs = {'device': next(attn.parameters()).device, 'dtype': next(attn.parameters()).dtype}

            if config.attention_bias:
                raise NotImplementedError('Checkpoint conversion with bias is not implemented for TP-friendly qkv yet.')

            attn.attn_q = nn.Linear(config.n_embd, config.n_head * config.head_size, bias=config.attention_bias, **factory_kwargs)
            attn.attn_k = nn.Linear(config.n_embd, config.n_query_groups * config.head_size, bias=config.attention_bias, **factory_kwargs)
            attn.attn_v = nn.Linear(config.n_embd, config.n_query_groups * config.head_size, bias=config.attention_bias, **factory_kwargs)

            q_weight, k_weight, v_weight = qkv_weight_conversion(attn.attn.weight, config)
            attn.attn_q.weight.data.copy_(q_weight)
            attn.attn_k.weight.data.copy_(k_weight)
            attn.attn_v.weight.data.copy_(v_weight)

            delattr(block.attn, 'attn')
            block.attn.tp_friendly_qkv = True

def apply_tp(
        model, tp_mesh
):
    n_head_changed = False
    convert_tp_friendly_qkv(model)
    for layer_id, block in enumerate(model.transformer.h):
        layer_plan = {
            "attn.attn_q": ColwiseParallel(),
            "attn.attn_k": ColwiseParallel(),
            "attn.attn_v": ColwiseParallel(),
            "attn.proj": RowwiseParallel(),
            # "attn.proj": ColwiseParallel(input_layouts=Shard(-1), output_layouts=Replicate()),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(),
            # "mlp.down_proj": ColwiseParallel(input_layouts=Shard(-1), output_layouts=Replicate()),
        }

        if not n_head_changed:
            # self.logger.info(f'old n_head: {block.config.n_head}  new n_head: {block.config.n_head // tp_mesh.size()}')
            if block.config.n_head < tp_mesh.size() or block.config.n_query_groups < tp_mesh.size():
                raise ValueError(f'TP size is to large for current model. n_head: {block.config.n_head}, n_query_groups: {block.config.n_query_groups}.')
            block.config.n_head = block.config.n_head // tp_mesh.size()
            block.config.n_query_groups = block.config.n_query_groups // tp_mesh.size()
            n_head_changed = True

        parallelize_module(
            module=block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )


def tp_sp_decorator(trainer_class=None, tp_size=8, sp_size=1, pp_size=1, dp_size=1, dp_replicate=1, dp_shard=-1):

    def wrapper(trainer_class):    
        loguru.logger.info(f'Applying TP, FSDP on {trainer_class.__name__}')

        class TPSPTrainer(trainer_class):
            def __init__(self, args):
                # assert args.launcher != 'deepspeed'
                args.launcher = 'torch'
                self.use_dcp = False
                super().__init__(args)

                if self.use_dcp:
                    from hymm.utils.checkpoint_manager import CheckpointManager
                    self.checkpoint_manager = CheckpointManager(
                        model_parts=[self.model_engine],
                        optimizers=[None],
                        dump_folder=self.ckpt_dir,
                        folder='',
                    )

                    if not self._old_no_load_pretrained and self.args.pretrained_ckpt:
                        self.checkpoint_manager.load_from_path(self.args.pretrained_ckpt)
                        self.args.no_load_pretrained = self._old_no_load_pretrained

            def save_checkpoint(self):
                if self.use_dcp:
                    self.checkpoint_manager.save(self.ss.update_steps, force=True)
                else:
                    return super().save_checkpoint()

            def init_distributed_env(self):
                super().init_distributed_env()

                self.local_rank = int(os.environ["LOCAL_RANK"])
                self.global_rank = int(os.environ["RANK"])
                self.world_size = int(os.environ["WORLD_SIZE"])


                self.parallel_dims = ParallelDims(
                    dp_shard=dp_shard,
                    dp_replicate=dp_replicate,
                    sp=sp_size,
                    tp=tp_size,
                    pp=pp_size,
                    world_size=self.world_size,
                )

                torch.cuda.set_device(self.local_rank)
                if not dist.is_initialized():
                    dist.init_process_group("nccl", timeout=get_nccl_timeout())
                self.device = torch.cuda.current_device()
                self.device_mesh = self.parallel_dims.build_mesh(device_type="cuda")
                if self.parallel_dims.dp_enabled:
                    dp_mesh = self.device_mesh["dp"]
                    dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
                else:
                    dp_degree, dp_rank = 1, 0

                self.dp_size = dp_degree
                self.dp_rank = dp_rank

                if sp_size > 1:
                    self.sp_group = self.device_mesh.get_group(mesh_dim="sp")
                    self.sp_rank = self.device_mesh.get_local_rank(mesh_dim="sp")
                    self.sp_size = sp_size
                else:
                    self.sp_group = None
                    self.sp_rank = 0
                    self.sp_size = 1

                if tp_size > 1:
                    self.tp_group = self.device_mesh.get_group(mesh_dim="tp")
                    self.tp_rank = self.device_mesh.get_local_rank(mesh_dim="tp")
                    self.tp_size = tp_size
                else:
                    self.tp_rank = 0
                    self.tp_size = 1

                if dp_replicate > 1:
                    self.dp_replicate_rank = self.device_mesh.get_local_rank(mesh_dim="dp_replicate")
                else:
                    self.dp_replicate_rank = 0


                # self.logger.warning('Overwriting `Trainer.rank` could lead to unexpected behavior. \n'
                #                       'Please make sure that `Trainer.rank` is only used in a proper way for dataloader sampling.\n'
                #                       'Another hack may be overwriting data preparation function, which may sacrifice generalization.')
                # self.rank = self.dp_rank
                # self.world_size = self.dp_size

            def apply_fsdp(self, deepspeed_config, lr_scheduler):
                if self.parallel_dims.dp_shard_enabled or self.parallel_dims.sp_enabled:
                    if self.parallel_dims.dp_replicate_enabled:
                        dp_mesh_dim_names = ("dp_replicate", "dp_shard_sp")
                        sharding_strategy = ShardingStrategy.HYBRID_SHARD
                    else:
                        dp_mesh_dim_names = ("dp_shard_sp",)
                        sharding_strategy = ShardingStrategy.FULL_SHARD
                else:
                    raise NotImplemented

                device_mesh = self.device_mesh[tuple(dp_mesh_dim_names)]

                # Initialize FSDP
                self.logger.info("Using FSDP")
                args = self.args
                self.get_trainable_params(self.model, args.training_parts),
                mp_policy = MixedPrecision(
                    # param_dtype=PRECISION_TO_TYPE[args.param_dtype],
                    # reduce_dtype=PRECISION_TO_TYPE[args.reduce_dtype],
                    # buffer_dtype=PRECISION_TO_TYPE[args.buffer_dtype],
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )
                self.logger.debug(f'{args.param_dtype=} {args.reduce_dtype=} {args.buffer_dtype=}')
                wrap_module = get_obj_from_str(args.wrap_module)
                wrap_policy = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={
                        wrap_module,
                    },
                )
                optimizer_cls = build_optimizer(args)
                self.model_engine = FSDPEngine(
                    args,
                    deepspeed_config,
                    self.model,
                    optimizer_cls,
                    lr_scheduler,
                    logger=self.logger,
                    mixed_precision=mp_policy,
                    sharding_strategy=sharding_strategy,
                    auto_wrap_policy=wrap_policy,
                    use_orig_params=True,
                    device_mesh=device_mesh,
                    device_id=torch.cuda.current_device(), # tp DTensor is on gpu, others may be on cpu
                    sync_module_states=True,
                )
                self.optimizer = self.model_engine.optimizer
                self.lr_scheduler = self.model_engine.lr_scheduler

            def model_init_pre_hook(self):
                def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
                    def model_attention_type(sd):
                        for k in sd.keys():
                            if 'attn.attn.' in k:
                                return 'old'
                            if 'attn.attn_q.' in k:
                                return 'new'
                        raise ValueError('Unknown attention type')

                    ckpt_attn_type = model_attention_type(state_dict)
                    model_attn_type = model_attention_type(self.state_dict())

                    if ckpt_attn_type == model_attn_type:
                        return super(type(self), self).load_state_dict(state_dict, strict=strict, assign=assign)

                    if ckpt_attn_type == 'old' and model_attn_type == 'new':
                        new_sd = {}
                        for k, weight in state_dict.items():
                            if 'attn.attn.' in k:
                                try:
                                    q_weight, k_weight, v_weight = qkv_weight_conversion(weight, self.config)
                                    new_sd[k.replace('attn.attn', 'attn.attn_q')] = q_weight
                                    new_sd[k.replace('attn.attn', 'attn.attn_k')] = k_weight
                                    new_sd[k.replace('attn.attn', 'attn.attn_v')] = v_weight
                                except:
                                    self.logger.warning(f'failed to convert {k}')
                                    raise
                            else:
                                new_sd[k] = weight
                        return super(type(self), self).load_state_dict(new_sd, strict=strict, assign=assign)
                    else: # ckpt_attn_type == 'new' and model_attn_type == 'old':
                        convert_tp_friendly_qkv(self)
                        return load_state_dict(self, state_dict, strict=strict, assign=assign)
                        # raise NotImplementedError('You are loading a NEW-IMPLEMENTED tp-friendly checkpoint, while your current model is not tp-friendly. '
                        #                           'Try to set tp_friendly_qkv=True in `attn_layers.py`.')

                from hymm.models.autoregressive.transfusion import Transfusion
                from hymm.models.autoregressive.multimodal_transfusion import MultiModalTransfusion
                Transfusion.load_state_dict = load_state_dict
                MultiModalTransfusion.load_state_dict = load_state_dict
                if self.use_dcp:
                    # should load state dict after FSDP
                    self._old_no_load_pretrained = self.args.no_load_pretrained
                    self.args.no_load_pretrained = True

            def model_init_post_hook(self):
                assert isinstance(self.model, (Transfusion, MultiModalTransfusion, ))
                if tp_size > 1:
                    if isinstance(self.model, MultiModalTransfusion):
                        apply_tp(self.model.language_model, self.device_mesh["tp"])
                    else:
                        apply_tp(self.model, self.device_mesh["tp"])
                    self.logger.info(f"TP: Initialized with tp size {self.tp_size}.")

        return TPSPTrainer

    print(f"Launcher info: ============{LAUNCHER}============")
    if trainer_class is not None:
        if LAUNCHER == "deepspeed":
            loguru.logger.info(f'Using DeepSpeed for {trainer_class.__name__}, skip dp and tp!!!!!!!!')
            return trainer_class
        return wrapper(trainer_class)
    else:
        return wrapper
