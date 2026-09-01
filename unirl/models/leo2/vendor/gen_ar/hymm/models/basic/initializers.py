# This file contains custom weight initialization functions for neural network layers.

import math
import os
import torch.nn.init as init


def proj_reset_parameters(_self):
    init.xavier_uniform_(_self.weight.view(_self.weight.size(0), -1))
    if _self.bias is not None:
        init.zeros_(_self.bias)


def zero_reset_parameters(_self):
    r"""Reset module parameters to zeros by default.

    本函数默认行为应该是0初始化，但因为0初始化可能会导致隐藏精度问题，
    不利于CI测试。因此需要使用随机初始化，但由于此函数调用位置过多，
    不便于在调用处修改，因此只能patch此函数或者在内部hack逻辑。
    TODO: 目前暂时选择内部hack逻辑，未来考虑重构。
    """
    try:
        from hymm.core.global_vars import get_args
        args = get_args()
        fake_zero_init = getattr(args, 'fake_zero_init', False)
    except Exception:
        fake_zero_init = False
    if fake_zero_init:
        init.kaiming_normal_(_self.weight)
        if hasattr(_self, 'bias') and _self.bias is not None:
            init.uniform_(_self.bias, -0.05, 0.05)
    else:
        init.zeros_(_self.weight)
        if hasattr(_self, 'bias') and _self.bias is not None:
            init.zeros_(_self.bias)


def normal_weight_reset_parameters(std=0.02, bias_type="default"):

    def _wrap_fn(_self):
        init.normal_(_self.weight, std=std)
        if hasattr(_self, "bias") and _self.bias is not None:
            if bias_type == "default":
                fan_in, _ = init._calculate_fan_in_and_fan_out(_self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(_self.bias, -bound, bound)
            elif bias_type == "zeros":
                init.zeros_(_self.bias)
            else:
                raise ValueError(f"Unsupported bias_init_type: {bias_type}")

    return _wrap_fn
