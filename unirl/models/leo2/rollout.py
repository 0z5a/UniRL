"""Dedicated Leo2 sampler with independent CP/EP and verified LoRA publication."""

from __future__ import annotations

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.weight_sync.transfer.checksum import fingerprint_tensor
from unirl.rollout.engine.trainside.engine import TrainsideRolloutEngine
from unirl.sde.kernels import StepStrategy
from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig
from unirl.train.backend.fsdp import FSDPBackend
from unirl.train.backend.sharded_state import load_model_state_dict
from unirl.train.configs import FSDPConfig, LoraConfig

from .bundle import Leo2Bundle
from .config import Leo2PipelineConfig
from .pipeline import Leo2Pipeline


class Leo2RolloutEngine(TrainsideRolloutEngine):
    """Own a sharded native pipeline in independent workers; see README.md."""

    _component_name = "leo2"

    def __init__(
        self,
        *,
        bundle_config: Leo2PipelineConfig,
        fsdp_cfg: FSDPConfig,
        lora_cfg: LoraConfig,
        strategy: StepStrategy,
        stage_attrs: tuple[str, ...] = ("diffusion",),
        forward_batch_size: int = 1,
        offload_on_sleep: bool = False,
    ) -> None:
        if bundle_config.preprocessing_cache_mode != "readonly":
            raise ValueError("Leo2RolloutEngine requires readonly preprocessing conditions.")
        if (
            fsdp_cfg.sp_size != bundle_config.context_parallel_size
            or fsdp_cfg.ep_size != bundle_config.expert_parallel_size
        ):
            raise ValueError("Leo2 rollout FSDP and bundle CP/EP settings must agree.")
        if tuple(stage_attrs) != ("diffusion",) or forward_batch_size != 1:
            raise ValueError("Leo2 rollout requires the diffusion stage and forward_batch_size=1.")
        if fsdp_cfg.activation_checkpointing:
            raise ValueError("Leo2 rollout does not use activation checkpointing.")
        if lora_cfg.dropout or lora_cfg.bias != "none" or lora_cfg.frozen_adapters:
            raise ValueError("Leo2 rollout supports a single dropout-free, bias-free LoRA adapter.")
        bundle = Leo2Bundle.from_config(bundle_config)
        self._backend = FSDPBackend(
            bundle=bundle,
            block_class_names=("LeoLayer", "LeoDualLayer", "LeoTripleLayer"),
            trainable_attr="model",
            fsdp_cfg=fsdp_cfg,
            lora_cfg=lora_cfg,
            optimizer_cfg=OptimizerConfig(
                learning_rate=0.0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, weight_decay=0.0
            ),
            scheduler_cfg=LrSchedulerConfig(type="constant", warmup_steps=0, total_steps=1),
        )
        self._lora_cfg = lora_cfg
        pipeline = Leo2Pipeline.from_bundle(bundle, config=bundle_config, strategy=strategy)
        super().__init__(pipeline=pipeline, forward_batch_size=1)
        self._offload_on_sleep = bool(offload_on_sleep)
        self._offloaded = False

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        """Release sampler weights when training shares its physical GPUs."""
        with self._generate_lock:
            if self._offload_on_sleep and not self._offloaded:
                self._backend.offload()
                self._offloaded = True

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        """Restore sampler weights before publication or generation."""
        with self._generate_lock:
            if self._offloaded:
                self._offloaded = False
                try:
                    self._backend.onload()
                except Exception:
                    self._backend.offload()
                    self._offloaded = True
                    raise

    @property
    def is_offloaded(self) -> bool:
        """Report whether the sampler weights currently reside on CPU."""
        return self._offloaded

    def _adapter_parameters(self) -> dict[str, tuple[str, torch.Tensor]]:
        """Map canonical wire names to the existing default adapter parameters."""
        result = {}
        for name, parameter in self._backend.model.named_parameters():
            for marker in (".lora_A.default.", ".lora_B.default."):
                if marker in name:
                    canonical = name.replace(marker, marker.replace("default.", ""))
                    result[canonical] = (name, parameter)
                    break
        return result

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: dict[str, torch.Tensor],
        *,
        peft_config: dict | None = None,
    ) -> None:
        """Load a complete canonical adapter, resharding into the rollout model."""
        if adapter_name != "default" or peft_config is None:
            raise ValueError("Leo2 rollout requires the default adapter and its PEFT configuration.")
        if peft_config.get("r") != self._lora_cfg.rank or peft_config.get("lora_alpha") != self._lora_cfg.alpha:
            raise ValueError("Leo2 rollout and trainer LoRA rank/alpha must match.")
        with self._generate_lock:
            parameters = self._adapter_parameters()
            if not parameters or parameters.keys() != lora_tensors.keys():
                missing = sorted(parameters.keys() - lora_tensors.keys())
                extra = sorted(lora_tensors.keys() - parameters.keys())
                raise ValueError(f"Leo2 adapter tensor mismatch: missing={missing[:4]}, extra={extra[:4]}")
            state = {}
            for key, (name, parameter) in parameters.items():
                tensor = lora_tensors[key]
                if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                    raise ValueError(f"Leo2 adapter shape/dtype mismatch for {key}")
                state[name] = tensor
            load_model_state_dict(self._backend.model, state, strict=False, broadcast_from_rank0=True)

    def tp_per_stage(self) -> dict[int, int]:
        """Expose one native model per worker to the shared LoRA readback verifier."""
        return {0: 1}

    def loaded_lora_checksums(self, *, adapter_id: int) -> dict:
        """Read back the effective PEFT adapter, including its B-matrix scaling."""
        del adapter_id
        with self._generate_lock, torch.no_grad():
            layers = {}
            for key, (_, parameter) in self._adapter_parameters().items():
                value = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
                module, suffix = key.rsplit(".lora_", 1)
                if suffix == "B.weight":
                    value = value * (self._lora_cfg.alpha / self._lora_cfg.rank)
                layers.setdefault(module, {})["lora_" + suffix[0].lower()] = fingerprint_tensor(value)
            return {0: [layers]}
