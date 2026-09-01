from collections import OrderedDict
from copy import deepcopy

import torch

DEFAULT_DECAY = {
    "fp32": 0.9999,
    "fp16": 0.988,  # 0.92%
    "bf16": 0.91,  # 0.99%
}
DEFAULT_POWER = {
    "fp32": 2 / 3,
    "fp16": 0.408774,
    "bf16": 0.209151,
}

# power = log(1/(1-decay)) / log(step)
# example: decay=0.998, step=5000  ==>  power=0.729654


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def use_default(value, default):
    return value if value is not None else default


class EMA(object):
    def __init__(
        self,
        module,
        dtype,
        decay=None,
        warmup=False,
        power=None,
        min_decay=0.0,
        inv_gamma=1.0,
        update_after_step=0,
        logger=None,
    ):
        self.module = module
        self.dtype = dtype
        self.warmup = warmup
        self.power = use_default(power, DEFAULT_POWER[dtype])
        self.decay = use_default(decay, DEFAULT_DECAY[dtype])
        self.min_decay = min_decay
        self.inv_gamma = inv_gamma
        self.update_after_step = update_after_step
        self.decay_steps = 0
        self.logger = logger

        self.build_ema_model(module)

        if logger is not None:
            logger.info(f"Using EMA with date type {dtype} (decay={decay}, warmup={warmup}, warmup_power={power}).")

    def build_ema_model(self, module):
        self.ema_model = deepcopy(module)
        if self.dtype == "fp16":
            self.ema_model = self.ema_model.half()
        elif self.dtype == "fp32":
            self.ema_model = self.ema_model.float()
        elif self.dtype == "bf16":
            self.ema_model = self.ema_model.to(torch.bfloat16)
        else:
            raise ValueError(f"Unknown EMA dtype {self.dtype}.")

        requires_grad(self.ema_model, False)

    def set_decay_steps(self, decay_steps):
        self.decay_steps = decay_steps
        if self.logger is not None:
            self.logger.info(f"Set EMA decay steps to {decay_steps}.")

    def get_decay(self):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).

        @jarvizhang's notes on EMA max_value when enabling FP16:
            If using FP16 for EMA, max_value=0.988 is better (Don't larger than 0.988, unless you know
            what you are doing). This is because FP16 has less precision than FP32, so the EMA value can
            be pushed out of the range of FP16.

            gamma=1, power=0.446249 are good values for models (reaches decay factor 0.985 at 30K steps,
            0.988 at 50K steps).

        @ckczzjzhang's notes on EMA max_value when enabling BF16:
            If using BF16 for EMA, max_value=0.94 is better. This is because BF16 has less precision than
            FP16, so the EMA value can be pushed out of the range of BF16.

            gamma=1, power=0.209151 are good values for models (reaches decay factor 0.90 at 60K steps,
            0.91 for 100k steps).
        """
        if self.warmup:
            step = max(0, self.decay_steps - self.update_after_step - 1)
            value = 1 - (1 + step / self.inv_gamma) ** -self.power
            return max(self.min_decay, min(value, self.decay))
        else:
            return self.decay

    @torch.no_grad()
    def update(self, module):
        """
        Step the EMA model towards the current model.
        """
        decay = self.get_decay()

        ema_params = OrderedDict(self.ema_model.named_parameters())
        model_params = OrderedDict(module.named_parameters())
        for name, param in model_params.items():
            if self.dtype == 'fp16':
                ema_params[name].mul_(decay).add_(param.data.half(), alpha=1 - decay)
            elif self.dtype == 'fp32':
                ema_params[name].mul_(decay).add_(param.data.float(), alpha=1 - decay)
            elif self.dtype == 'bf16':
                ema_params[name].mul_(decay).add_(param.data.to(torch.bfloat16), alpha=1 - decay)
            else:
                raise ValueError(f"Unknown EMA dtype {self.dtype}.")

        self.decay_steps += 1

    def state_dict(self, *args, **kwargs):
        return self.ema_model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.ema_model.load_state_dict(*args, **kwargs)

    @property
    def config(self):
        return {
            "dtype": self.dtype,
            "warmup": self.warmup,
            "power": self.power,
            "decay": self.decay,
            "min_decay": self.min_decay,
            "inv_gamma": self.inv_gamma,
            "update_after_step": self.update_after_step,
            "decay_steps": self.decay_steps,
        }


class DistributedEMA(EMA):
    def __init__(
        self,
        world_size,
        rank,
        module,
        dtype,
        decay=None,
        warmup=False,
        power=None,
        min_decay=0.0,
        inv_gamma=1.0,
        update_after_step=0,
        logger=None,
    ):
        self.world_size = world_size
        self.rank = rank
        super().__init__(module, dtype, decay, warmup, power, min_decay, inv_gamma, update_after_step, logger)

    @torch.no_grad()
    def build_ema_model(self, module):
        num_params = sum([param.numel() for param in list(module.parameters())])

        num_params_per_rank = (num_params + self.world_size - 1) // self.world_size  # ceil
        start_index = self.rank * num_params_per_rank
        end_index = min(start_index + num_params_per_rank, num_params) - 1  # 闭区间

        self.name_list = []
        self.index_list = []
        self.ema_param_list = []

        cur_param_start_index = 0
        for name, param in module.named_parameters():
            numel = param.numel()

            cur_param_end_index = cur_param_start_index + numel - 1  # 闭区间

            max_start_index = max(start_index, cur_param_start_index)
            min_end_index = min(end_index, cur_param_end_index)
            if max_start_index <= min_end_index:
                self.name_list.append(name)
                index = (max_start_index - cur_param_start_index, min_end_index - cur_param_start_index)
                self.index_list.append(index)
                # view will not create new tensor
                if self.dtype == "fp16":
                    self.ema_param_list.append(param.view(-1)[index[0]: index[1] + 1].clone().detach().half())
                elif self.dtype == "fp32":
                    self.ema_param_list.append(param.view(-1)[index[0]: index[1] + 1].clone().detach().float())
                elif self.dtype == "bf16":
                    self.ema_param_list.append(param.view(-1)[index[0]: index[1] + 1].clone().detach().to(torch.bfloat16))
                else:
                    raise ValueError(f"Unknown EMA dtype {self.dtype}.")

            cur_param_start_index += numel

    @torch.no_grad()
    def update(self, module):
        """
        Step the EMA model towards the current model.
        """
        decay = self.get_decay()

        state_dict = OrderedDict(module.named_parameters())
        for ema_param, name, index in zip(self.ema_param_list, self.name_list, self.index_list):
            param = state_dict[name].view(-1)
            if self.dtype == 'fp16':
                ema_param.mul_(decay).add_(param[index[0]: index[1] + 1].half(), alpha=1.0 - decay)
            elif self.dtype == 'fp32':
                ema_param.mul_(decay).add_(param[index[0]: index[1] + 1].float(), alpha=1.0 - decay)
            elif self.dtype == 'bf16':
                ema_param.mul_(decay).add_(param[index[0]: index[1] + 1].to(torch.bfloat16), alpha=1.0 - decay)
            else:
                raise ValueError(f"Unknown EMA dtype {self.dtype}.")

        self.decay_steps += 1

    def state_dict(self):
        return [self.config, self.name_list, self.index_list, self.ema_param_list]

    def load_state_dict(self, ema_ckpt):
        ckpt_config, ckpt_name_list, ckpt_index_list, ckpt_param_list = ema_ckpt
        assert len(self.name_list) == len(ckpt_name_list), f"load ema ckpt error: len(self.name_list):{len(self.name_list)}, len(ckpt_name_list):{len(ckpt_name_list)}"
        for ema_param, name, index, ckpt_param, ckpt_name, ckpt_index in zip(
            self.ema_param_list, self.name_list, self.index_list,
            ckpt_param_list, ckpt_name_list, ckpt_index_list
        ):
            assert name == ckpt_name and index == ckpt_index, f"load ema ckpt error: name:{name}, ckpt_name:{ckpt_name}, index:{index}, ckpt_index:{ckpt_index}"
            if self.dtype == 'fp16':
                ema_param.copy_(ckpt_param.half())
            elif self.dtype == 'fp32':
                ema_param.copy_(ckpt_param.float())
            elif self.dtype == 'bf16':
                ema_param.copy_(ckpt_param.to(torch.bfloat16))
            else:
                raise ValueError(f"Unknown EMA dtype {self.dtype}.")

    # used by ptm
    def save_ckpt(self):
        """
            save all ema weights to {checkpoint_name}/global_step{engine.global_steps}/dist_ema/, offline validation need convert ema ckpt
        """
        return [self.config, self.name_list, self.index_list, self.ema_param_list]

    def load_ckpt(self, ema_ckpt):
        """
            every GPU loads own checkpoint
        """
        from megatron import print_rank_0
        ema_config, ckpt_name_list, ckpt_index_list, ckpt_param_list = ema_ckpt
        # set decay steps from ckpt
        print_rank_0(f"get ema_config from ema ckpt, but only use decay_steps, ema_config:{ema_config}")
        self.set_decay_steps(ema_config['decay_steps'])
        # check ema ckpt before load
        assert len(ckpt_name_list) == len(
            self.ema_param_list), f"rank: {self.rank} load ema ckpt error: len(self.ema_param_list):{len(self.ema_param_list)}, len(ckpt_name_list):{len(ckpt_name_list)}"
        for ema_param, name, index, ckpt_param, ckpt_name, ckpt_index in zip(self.ema_param_list, self.name_list,
                                                                             self.index_list, ckpt_param_list,
                                                                             ckpt_name_list, ckpt_index_list):
            assert name == ckpt_name and index == ckpt_index, f"rank: {self.rank}, load ema ckpt error: name:{name}, ckpt_name:{ckpt_name}, index:{index}, ckpt_index:{ckpt_index}"
            if self.dtype == 'fp16':
                ema_param.copy_(ckpt_param.half())
            elif self.dtype == 'fp32':
                ema_param.copy_(ckpt_param.float())
            elif self.dtype == 'bf16':
                ema_param.copy_(ckpt_param.to(torch.bfloat16))
            else:
                raise ValueError(f"Unknown EMA dtype {self.dtype}.")


class TPDistributedEMA(EMA):
    """
    --ptm-v2 enable TPDistributedEMA
    """

    def __init__(self,
                 global_world_size,
                 global_rank,
                 dp_world_size,
                 dp_rank,
                 module,
                 dtype,
                 decay=None,
                 warmup=False,
                 power=None,
                 min_decay=0.0,
                 inv_gamma=1.0,
                 update_after_step=0,
                 parallelism_weights=None,
                 ):
        # for example, 16 gpus and tp=2, global_world_size is 16, dp_world_size is 8
        self.global_world_size = global_world_size
        self.global_rank = global_rank
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.parallelism_weights = parallelism_weights or []
        super().__init__(module, dtype, decay, warmup, power, min_decay, inv_gamma, update_after_step)

    @torch.no_grad()
    def build_ema_model(self, module):
        """
        the weights in PARALLELISM_WEIGHTS are divided by self.dp_world_size and saved in self.dp_ema_param_list
        the weights not in PARALLELISM_WEIGHTS are divided by self.global_world_size and saved in self.global_ema_param_list
        """
        # self.global_ema_param_list save no parallel params
        self.global_name_list = []
        self.global_index_list = []
        self.global_ema_param_list = []
        # self.dp_ema_param_list save parallel params
        self.dp_name_list = []
        self.dp_index_list = []
        self.dp_ema_param_list = []

        parallel_params_set = set()
        no_parallel_params_set = set()
        parallel_params_num = 0
        no_parallel_params_num = 0
        for name, param in module.named_parameters():
            numel = param.numel()
            if any(substring in name for substring in self.parallelism_weights):
                parallel_params_num += numel
                parallel_params_set.add(name)
            else:
                no_parallel_params_num += numel
                no_parallel_params_set.add(name)

        def build_param_list_from_module(world_size, rank, params_set, num_params, name_list, index_list,
                                         ema_param_list):
            num_params_per_rank = (num_params + world_size - 1) // world_size  # ceil
            # [start_index, end_index] is the range coverd by the current gpu rank
            start_index = rank * num_params_per_rank
            end_index = min(start_index + num_params_per_rank, num_params) - 1  # 闭区间
            cur_param_start_index = 0
            for _name, _param in module.named_parameters():
                # only copy params in params_set
                if _name not in params_set:
                    continue
                _numel = _param.numel()
                # [cur_param_start_index, cur_param_end_index] is the range covered by the _param
                cur_param_end_index = cur_param_start_index + _numel - 1  # 闭区间
                # [max_start_index, min_end_index] is the intersection of the current gpu rank range and the current _param range
                max_start_index = max(start_index, cur_param_start_index)
                min_end_index = min(end_index, cur_param_end_index)
                if max_start_index <= min_end_index:
                    name_list.append(_name)
                    index = (max_start_index - cur_param_start_index, min_end_index - cur_param_start_index)
                    index_list.append(index)
                    # view will not create new tensor
                    if self.dtype == 'fp16':
                        ema_param_list.append(_param.view(-1)[index[0]: index[1] + 1].clone().detach().half())
                    elif self.dtype == 'fp32':
                        ema_param_list.append(_param.view(-1)[index[0]: index[1] + 1].clone().detach().float())
                    elif self.dtype == 'bf16':
                        ema_param_list.append(
                            _param.view(-1)[index[0]: index[1] + 1].clone().detach().to(torch.bfloat16))
                    else:
                        raise ValueError(f"Unknown EMA dtype {self.dtype}.")
                cur_param_start_index += _numel

        # build self.global_ema_param_list
        build_param_list_from_module(self.global_world_size, self.global_rank, no_parallel_params_set,
                                     no_parallel_params_num, self.global_name_list, self.global_index_list,
                                     self.global_ema_param_list)
        # build self.dp_ema_param_list
        build_param_list_from_module(self.dp_world_size, self.dp_rank, parallel_params_set, parallel_params_num,
                                     self.dp_name_list, self.dp_index_list, self.dp_ema_param_list)

    @torch.no_grad()
    def update(self, module):
        """
        Step the EMA model towards the current model.
        """
        decay = self.get_decay()

        state_dict = OrderedDict(module.named_parameters())
        # update self.global_ema_param_list
        for ema_param, name, index in zip(self.global_ema_param_list, self.global_name_list, self.global_index_list):
            param = state_dict[name].view(-1)
            if self.dtype == 'fp16':
                ema_param.mul_(decay).add_(param[index[0]: index[1] + 1].half(), alpha=1.0 - decay)
            elif self.dtype == 'fp32':
                ema_param.mul_(decay).add_(param[index[0]: index[1] + 1].float(), alpha=1.0 - decay)
            elif self.dtype == 'bf16':
                ema_param.mul_(decay).add_(param[index[0]: index[1] + 1].to(torch.bfloat16), alpha=1.0 - decay)
            else:
                raise ValueError(f"Unknown EMA dtype {self.dtype}.")

        # update self.dp_ema_param_list
        for ema_param, name, index in zip(self.dp_ema_param_list, self.dp_name_list, self.dp_index_list):
            param = state_dict[name].view(-1)
            if self.dtype == 'fp16':
                ema_param.mul_(decay).add_(param[index[0]: index[1] + 1].half(), alpha=1.0 - decay)
            elif self.dtype == 'fp32':
                ema_param.mul_(decay).add_(param[index[0]: index[1] + 1].float(), alpha=1.0 - decay)
            elif self.dtype == 'bf16':
                ema_param.mul_(decay).add_(param[index[0]: index[1] + 1].to(torch.bfloat16), alpha=1.0 - decay)
            else:
                raise ValueError(f"Unknown EMA dtype {self.dtype}.")

        self.decay_steps += 1

    def save_ckpt(self):
        """
        save all ema weights to {checkpoint_name}/global_step{engine.global_steps}/dist_ema/, offline validation need convert ema ckpt
        """
        return {
            'config': self.config,
            'global_name_list': self.global_name_list,
            'global_index_list': self.global_index_list,
            'global_ema_param_list': self.global_ema_param_list,
            'dp_name_list': self.dp_name_list,
            'dp_index_list': self.dp_index_list,
            'dp_ema_param_list': self.dp_ema_param_list
        }

    def load_ckpt(self, ema_ckpt, print_fn=None):
        """
        every GPU loads own checkpoint
        """
        print_fn = print_fn or print

        ema_config = ema_ckpt['config']
        # set decay steps from ckpt
        print_fn(f"get ema_config from ema ckpt, but only use decay_steps, ema_config:{ema_config}")
        self.set_decay_steps(ema_config['decay_steps'])

        def load_param(ema_param_list, ema_name_list, ema_index_list, ckpt_param_list, ckpt_name_list, ckpt_index_list):
            assert len(ema_name_list) == len(
                ckpt_name_list), f"global_rank: {self.global_rank} load ema ckpt error: len(ema_name_list):{len(ema_name_list)}, len(ckpt_name_list):{len(ckpt_name_list)}"
            for ema_param, name, index, ckpt_param, ckpt_name, ckpt_index in zip(ema_param_list, ema_name_list,
                                                                                 ema_index_list, ckpt_param_list,
                                                                                 ckpt_name_list, ckpt_index_list):
                assert name == ckpt_name and index == ckpt_index, f"global_rank: {self.global_rank}, load ema ckpt error: name:{name}, ckpt_name:{ckpt_name}, index:{index}, ckpt_index:{ckpt_index}"
                if self.dtype == 'fp16':
                    ema_param.copy_(ckpt_param.half())
                elif self.dtype == 'fp32':
                    ema_param.copy_(ckpt_param.float())
                elif self.dtype == 'bf16':
                    ema_param.copy_(ckpt_param.to(torch.bfloat16))
                else:
                    raise ValueError(f"Unknown EMA dtype {self.dtype}.")

        # load self.global_ema_param_list
        load_param(self.global_ema_param_list, self.global_name_list, self.global_index_list,
                   ema_ckpt['global_ema_param_list'], ema_ckpt['global_name_list'], ema_ckpt['global_index_list'])
        # load self.dp_ema_param_list
        load_param(self.dp_ema_param_list, self.dp_name_list, self.dp_index_list, ema_ckpt['dp_ema_param_list'],
                   ema_ckpt['dp_name_list'], ema_ckpt['dp_index_list'])
