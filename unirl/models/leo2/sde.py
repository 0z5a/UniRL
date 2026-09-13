"""Leo2-native FlowGRPO transition math."""

from __future__ import annotations

import math
from typing import Optional

import torch

from unirl.sde.kernels import FlowSDEStrategy, GeneratorLike


class Leo2FlowSDEStrategy(FlowSDEStrategy):
    """Match Hymm's FlowMatchDiscreteScheduler operation order exactly."""

    def denoise(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        *,
        eta: float = 1.0,
        prev_sample: Optional[torch.Tensor] = None,
        generator: GeneratorLike = None,
        sigma_max: float = 0.99,
        step_index: int = 0,
    ):
        """Run one native-ordered Leo2 FlowGRPO transition."""
        del step_index
        from diffusers.utils.torch_utils import randn_tensor

        sigma_values = sigma.reshape(-1)
        sigma_next_values = sigma_next.reshape(-1)
        if sigma_values.numel() > 1:
            if sigma_values.numel() != sample.shape[0] or sigma_next_values.numel() != sample.shape[0]:
                raise ValueError(
                    "Leo2 batched SDE requires one sigma pair per sample; "
                    f"got sigma={tuple(sigma.shape)}, sigma_next={tuple(sigma_next.shape)}, batch={sample.shape[0]}"
                )
            generators = generator if isinstance(generator, list) else [generator] * sample.shape[0]
            if len(generators) != sample.shape[0]:
                raise ValueError(
                    f"Leo2 batched SDE requires {sample.shape[0]} generators, got {len(generators)}"
                )
            rows = [
                self.denoise(
                    noise_pred=noise_pred[index : index + 1],
                    sample=sample[index : index + 1],
                    sigma=sigma_values[index],
                    sigma_next=sigma_next_values[index],
                    eta=eta,
                    prev_sample=None if prev_sample is None else prev_sample[index : index + 1],
                    generator=generators[index],
                    sigma_max=sigma_max,
                )
                for index in range(sample.shape[0])
            ]
            previous, log_probs, means = zip(*rows)
            return (
                torch.cat(previous),
                None if log_probs[0] is None else torch.cat(log_probs),
                torch.cat(means),
            )

        model_output = noise_pred.float()
        sample = sample.float()
        sigma = sigma.float().reshape(())
        sigma_next = sigma_next.float().reshape(())
        dt = sigma_next - sigma
        sigma_for_std = sigma.to(device=model_output.device)
        sigma_next_for_std = sigma_next.to(device=model_output.device)
        sigma_max_for_std = torch.as_tensor(
            sigma_max,
            device=model_output.device,
            dtype=torch.float32,
        )
        denom_sigma = torch.where(sigma_for_std == 1, sigma_max_for_std, sigma_for_std)
        base_std = torch.sqrt(sigma_for_std / (1 - denom_sigma)) * eta
        std_dev_t = base_std * torch.sqrt(-(sigma_next_for_std - sigma_for_std))
        pred_original_sample = sample - sigma * model_output.to(torch.float32)
        score_estimate = -(sample - pred_original_sample * (1 - sigma)) / (sigma**2)
        prev_sample_mean = (
            sample
            + dt * model_output.to(torch.float32)
            + 0.5 * (std_dev_t**2) * score_estimate
        )
        if eta < 1e-7:
            return sample + dt * model_output, None, prev_sample_mean
        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + variance_noise * std_dev_t
        std_dev_t = torch.clamp(std_dev_t, min=1e-6)
        log_prob = (
            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2)
            / (2 * (std_dev_t**2))
            - torch.log(std_dev_t)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, log_prob, prev_sample_mean


__all__ = ["Leo2FlowSDEStrategy"]
