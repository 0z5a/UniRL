from hy_parallelism.utils import get_parallel_state
import loguru
import torch
from torch import nn
# from hymm.models.modules.Leo import *
from typing import Optional, Dict, Any, Union
from hy_parallelism.distributed.fsdp_util import apply_fsdp2
from hy_parallelism.engines.parallel_engine import BaseParallelEngine

class LeoEngine(BaseParallelEngine):

    def pre_process_input(self, *args, **kwargs):
        if not self.enable_pp:
            return args, kwargs
        if 'extra_kwargs' in kwargs:
            # dict_keys(['byt5_text_states', 'byt5_text_mask', 'is_token_replace'])

            assert kwargs['extra_kwargs'].keys() == {'byt5_text_states', 'byt5_text_mask', 'is_token_replace'}

            byt5_text_states = kwargs['extra_kwargs'].pop('byt5_text_states')
            byt5_text_mask = kwargs['extra_kwargs'].pop('byt5_text_mask')
            is_token_replace = kwargs['extra_kwargs'].pop('is_token_replace')
            del kwargs['extra_kwargs']
            kwargs['byt5_text_states'] = byt5_text_states
            kwargs['byt5_text_mask'] = byt5_text_mask
            kwargs['is_token_replace'] = is_token_replace
        return args, kwargs


    def __init__(self, model, *args, **kwargs):

        old_forward = model.forward
        from functools import wraps

        def pp_friendly_forward(
            self,

            pp_x=None,
            pp_img=None, pp_txt=None, pp_vec=None, pp_text_mask=None,
            pp_loss_moe_0=None, pp_loss_moe_1=None, 
            pp_token_replace_vec=None, # optional, put in the end

            hidden_states: torch.Tensor=None,
            timestep: torch.LongTensor=None,
            text_states: torch.Tensor=None,
            text_states_2: torch.Tensor=None,
            encoder_attention_mask: torch.Tensor=None,
            vision_states: torch.Tensor=None,
            output_features=False,
            output_features_stride=8,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            freqs_cos: Optional[torch.Tensor] = None,
            freqs_sin: Optional[torch.Tensor] = None,
            return_dict: bool = False,
            guidance=None,
            mask_type="t2v",

            # Replace extra_kwargs
            byt5_text_states=None,
            byt5_text_mask=None,
            is_token_replace=False,
            ut=None,
        ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
            extra_kwargs = dict(
                byt5_text_states=byt5_text_states,
                byt5_text_mask=byt5_text_mask,
                is_token_replace=is_token_replace,
            )

            return old_forward(
                hidden_states=hidden_states,
                timestep=timestep,
                text_states=text_states,
                text_states_2=text_states_2,
                encoder_attention_mask=encoder_attention_mask,
                vision_states=vision_states,
                output_features=output_features,
                output_features_stride=8,
                attention_kwargs=attention_kwargs,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                return_dict=return_dict,
                guidance=guidance,
                mask_type=mask_type,
                extra_kwargs=extra_kwargs,
                ut=ut,

                pp_x=pp_x,
                pp_img=pp_img,
                pp_txt=pp_txt,
                pp_vec=pp_vec,
                pp_text_mask=pp_text_mask,
                pp_loss_moe_0=pp_loss_moe_0,
                pp_loss_moe_1=pp_loss_moe_1,
                pp_token_replace_vec=pp_token_replace_vec,
            )

        if get_parallel_state().pp_enabled:
            model.forward = pp_friendly_forward.__get__(model)



        super().__init__(model, *args, **kwargs)

    def apply_fsdp(self, model):
        default_fsdp_kwargs = self.default_fsdp_kwargs.copy()

        # PTM MOE implementation requires param_dtype to be float32
        default_fsdp_kwargs['param_dtype'] = torch.float32
        default_fsdp_kwargs['reduce_dtype'] = torch.float32

        # default_fsdp_kwargs['reshard_after_forward_policy'] = 'always'

        apply_fsdp2(
            model, 
            blocks=list(model.double_blocks) + list(model.single_blocks),
            **default_fsdp_kwargs,
        )
        return model

    def apply_ac(self, model):
        from hy_parallelism.distributed.fsdp_util import apply_fsdp_checkpointing
        # When applying pp, double_blocks[0] could be None
        # apply_fsdp_checkpointing(model, no_split_modules=type(model.double_blocks[0]), p=1)
        
        # Find block types from both double and single blocks
        no_split_module_type = None
        for block in model.double_blocks:
            if block is not None:
                no_split_module_type = type(block)
                break
        if no_split_module_type is not None:
            apply_fsdp_checkpointing(model, no_split_modules=no_split_module_type, p=1, use_reentrant=True)


        for block in model.single_blocks:
            if block is not None:
                no_split_module_type = type(block)
                break
        if no_split_module_type is not None:
            apply_fsdp_checkpointing(model, no_split_modules=no_split_module_type, p=1, use_reentrant=True)

    def fsdp_blocks(self):
        for m in self.fsdp_models:
            for block in list(m.double_blocks) + list(m.single_blocks):
                if block is None:
                    continue
                for fqn, module in block.named_modules():
                    if self.is_expert(fqn):
                        yield module
                yield block
            yield m

    def get_resolution_key(self, idx, *args, **kwargs):
        ret = tuple(idx.shape)
        return ret

    def get_batch_size(self, idx, *args, **kwargs):
        return idx.shape[0]

    # def get_eval_ret(self):
    #     assert len(self.fsdp_models) == 1
    #     return self.fsdp_models[0]._eval_ret

    def config_forward_args(self):
        self.set_n_pp_args(8)
        self.set_replacable_kwargs([])


    def apply_ep(self, model):
        from torch.distributed.tensor.placement_types import Shard
        assert self.parallel_dims.ep_enabled
        # ptm moe don't use apply_ep
        assert model.moe_config is not None, 'Enabling ep but no moe config'
        if model.moe_config.use_ptm_moe:
            return

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
                    # self.parallel_dims.world_mesh['ep'],
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
