import loguru
from dataclasses import dataclass

from hy_parallelism.parallel_states import get_parallel_state

@dataclass
class MOEConfig():
    # TODO(kevinkhwu):
    #   1. hidden_act is not supported for ptm MOE
    #   2. mlp_bias is not supported for ptm MOE
    
    def __init__(
        self,
        num_experts=1,
        use_mixed_mlp_moe=False,
        num_shared_expert=1,
        moe_topk=1,
        intermediate_size=None,
        hidden_size=None, # input size, and output size if output_hidden_size is not None
        output_hidden_size=None, # output size if not None, otherwise hidden_size
        threshold_train=0.,
        threshold_eval=0.,
        capacity_factor_train=1.25,
        capacity_factor_eval=1.25,
        straight_through_dispatch_tensor=True,
        differentiable_topk=False,
        differentiable_topk_fused=True,
        balance_loss_coef=1e-2,
        router_z_loss_coef=1e-3,
        fused_experts=False,
        use_ptm_moe=True,
        moe_use_deepep=False,
        ptm_moe_pad_seqlen=0,
        enable_moe_token_mask=False,
        use_grouped_gemm=True,
        use_fused_moe=True,
        disable_fused_topk_gating=True,
        **kwargs,
    ):
        self.num_experts = num_experts
        self.use_mixed_mlp_moe = use_mixed_mlp_moe
        self.num_shared_expert = num_shared_expert
        self.moe_topk = moe_topk
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        if output_hidden_size is None:
            self.output_hidden_size = hidden_size
        else:
            self.output_hidden_size = output_hidden_size

        self.threshold_train = threshold_train
        self.threshold_eval = threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.straight_through_dispatch_tensor = straight_through_dispatch_tensor
        self.differentiable_topk = differentiable_topk
        self.differentiable_topk_fused = differentiable_topk_fused
        self.balance_loss_coef = balance_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        self.fused_experts = fused_experts
        self.use_ptm_moe = use_ptm_moe
        self.moe_use_deepep = moe_use_deepep
        self.ptm_moe_pad_seqlen = ptm_moe_pad_seqlen
        self.enable_moe_token_mask = enable_moe_token_mask
        self.use_grouped_gemm = use_grouped_gemm
        self.use_fused_moe = use_fused_moe

        self.disable_fused_topk_gating = disable_fused_topk_gating

        for k, v in kwargs.items():
            loguru.logger.info(f'MOEConfig:Setting extra entry {k} to {v}')
            setattr(self, k, v)
        
        # self._check_config()

    def to_dict(self):
        return self.__dict__

    def _check_config(self):
        assert self.intermediate_size is not None
        assert self.hidden_size is not None

        using_fused_topk_gating = not self.disable_fused_topk_gating and self.use_fused_moe

        if not self.disable_fused_topk_gating:
            msg = (
                'There is an existing bug in PTM fused_topk_gating backward (_topk_softmax_softmax_bwd). '
                'youngfyang recommends setting disable_fused_topk_gating=True since loss/gradient misalignment still exists after fixing the backward kernel. '
                'However, kevinkhwu found a potential bug with aux loss calculation when used_token is given and fixed this issue in v0.2.3a1.'
                'Therefore, disabling fused topk gating may be no longer necessary(?), as disabling this will also disable token dropping, which is not desired.' # 
            )
            loguru.logger.warning(msg)
        else:
            # will make deepep dropless, warning later
            ...

        if self.output_hidden_size != self.hidden_size:
            assert not self.use_grouped_gemm, 'Grouped GEMM is not supported for non-equal hidden_size and output_hidden_size'



        if not self.use_grouped_gemm:
            raise NotImplementedError('kevinkhwu: Non-grouped GEMM is not implemented.')

        if self.moe_use_deepep:
            assert self.ptm_moe_pad_seqlen == 0, 'DeepSeek MOE does not require padding.'
            assert get_parallel_state().ep > 1, 'DeepSeek MOE only supports ep > 1.'


            if not using_fused_topk_gating:
                loguru.logger.error(
                    'Fused topk is disabled. Using Naive topkgating making DeepEP Dropless (PTM MOE works fine).'
                )

            if self.use_fused_moe:
                if using_fused_topk_gating:
                    pass
                    # loguru.logger.warning(
                    #     'When using deepep, Fused permute implementation ignores gate.drop_tokens. `drop_and_pad` is only effective self.gate.drop_tokens and not self.use_fused_moe.\n'
                    #     'Token is dropped, but `drop_and_pad` is not effective.' # what would happen is unknown.
                    #     # 'When drop_and_pad=True, in routing_map, the number of non-zeros in each column equals to '
                    #     # 'expert capacity. This function exploits this feature to use ops that support cuda graph.'
                    # )
            else:
                loguru.logger.warning(
                    'youngfyang hardcodes use_fused_moe to True in Leo pp_ep_ptm.' 
                    ' (https://git.woa.com/Foundation_AI_Infra/HunyuanVideo_pureTorch/blob/dev/kevinkhwu/pp_ep_ptm/hymm/models/modules/moe/ptm_moe/ptm_hunyuan_moe.py#L559).' 
                    # 'use_fused_moe is set to False by default in AngelPTM.'
                    'Setting this to False will disable fused gating, fused permute and token droping.'
                    'However, for unknown reason, use_fused_moe is set to False by default in AngelPTM.'
                )

            # assert not self.use_fused_moe, 'DeepSeek MOE does not support fused MOE.'
        else:
            if self.use_ptm_moe:
                assert self.ptm_moe_pad_seqlen > 0, 'Non-DeepSeek MOE (PTM MOE) requires padding.'

        # assert self.use_fused_moe,         

    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)