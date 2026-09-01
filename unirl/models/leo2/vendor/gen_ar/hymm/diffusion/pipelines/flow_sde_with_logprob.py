# Copied from https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import BaseOutput, is_scipy_available, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput, FlowMatchEulerDiscreteScheduler


def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    determistic: bool = False,
    win_velocity: Optional[torch.FloatTensor] = None,
    lose_velocity: Optional[torch.FloatTensor] = None,
    sde_noise_scale: float = 0.7,
) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    device = model_output.device
    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]
    sigma = self.sigmas[step_index].view(-1, 1, 1, 1).to(device)
    sigma_prev = self.sigmas[prev_step_index].view(-1, 1, 1, 1).to(device)
    sigma_max = self.sigmas[1].item()
    dt = sigma_prev - sigma

    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * sde_noise_scale
    
    # our sde
    prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
    
    if win_velocity is not None or lose_velocity is not None:
        if determistic:
            if lose_velocity is None:
                # 添加一定噪声
                noise = torch.randn_like(model_output)
                lose_velocity = model_output + noise * 0.3 # 0.1
            win_log_prob = - (win_velocity.detach() - model_output) ** 2
            lose_log_prob = - (lose_velocity.detach() - model_output) ** 2
            win_log_prob = win_log_prob.mean(dim=tuple(range(1, win_log_prob.ndim)))
            lose_log_prob = lose_log_prob.mean(dim=tuple(range(1, lose_log_prob.ndim)))
            return win_log_prob, lose_log_prob, prev_sample_mean, std_dev_t * torch.sqrt(-1*dt)
        else:
            win_log_prob = None
            lose_log_prob = None
            if win_velocity is not None:
                win_sample = sample + dt * win_velocity
                # win_log_prob = (
                #     -((win_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
                #     - torch.log(std_dev_t * torch.sqrt(-1*dt))
                #     - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
                # )
                win_log_prob = - (win_sample.detach() - prev_sample_mean) ** 2
                # mean along all but batch dimension
                win_log_prob = win_log_prob.mean(dim=tuple(range(1, win_log_prob.ndim)))

            if lose_velocity is not None:
                lose_sample = sample + dt * lose_velocity
            else:
                lose_sample = sample + dt * model_output
            
            # lose_log_prob = (
            #     -((lose_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
            #     - torch.log(std_dev_t * torch.sqrt(-1*dt))
            #     - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            # )
            lose_log_prob = - (lose_sample.detach() - prev_sample_mean) ** 2
            # mean along all but batch dimension
            lose_log_prob = lose_log_prob.mean(dim=tuple(range(1, lose_log_prob.ndim)))
            
            return win_log_prob, lose_log_prob, prev_sample_mean, std_dev_t * torch.sqrt(-1*dt)
        
    else:
        if prev_sample is not None and generator is not None:
            raise ValueError(
                "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
                " `prev_sample` stays `None`."
            )

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise

        # No noise is added during evaluation
        if determistic:
            prev_sample = sample + dt * model_output

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
            - torch.log(std_dev_t * torch.sqrt(-1*dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        
        return prev_sample, log_prob, prev_sample_mean, std_dev_t * torch.sqrt(-1*dt)
