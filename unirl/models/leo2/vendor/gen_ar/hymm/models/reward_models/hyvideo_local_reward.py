import os
import sys
from pathlib import Path
from typing import Dict, List, Union

import loguru

logger = loguru.logger

_DEFAULT_DEP_PATH = (
    Path(__file__).resolve().parents[3] / "deps" / "HYVideoReWard"
)
_DEP_PATH = Path(os.environ.get("HYVIDEO_LOCAL_REWARD_PATH", _DEFAULT_DEP_PATH))

if not _DEP_PATH.exists():
    raise ImportError(
        f"HYVideoReWard repo not found at {_DEP_PATH}. "
        f"Clone/symlink it there or set $HYVIDEO_LOCAL_REWARD_PATH."
    )

_QWENVL_ROOT = _DEP_PATH / "reward_model" / "qwen-vl-finetune"
if not _QWENVL_ROOT.exists():
    raise ImportError(
        f"Expected '{_QWENVL_ROOT}' (HYVideoReWard standard layout). "
        f"Check that $HYVIDEO_LOCAL_REWARD_PATH points to the repo root."
    )

if str(_QWENVL_ROOT) not in sys.path:
    sys.path.insert(0, str(_QWENVL_ROOT))

from qwenvl.evaluation.eval import (  # noqa: E402
    load_eval_config,
    build_inferencer,
)

_DEFAULT_LOGIT2DIM = {0: "TA", 1: "VQ", 2: "MQ", 3: "ID", 4: "AVC"}


class HyVideoRewardLocal:
    """In-process HYVideoScore reward, output-compatible with HyVideoRewardRemote.

    ``reward(videos, prompts)`` returns ``List[Dict[str, float]]`` with keys
    ``{TA, VQ, MQ, AES}`` -- matching :class:`HyVideoRewardRemote` so the GRPO
    reward aggregator can swap between local and remote without changes.
    """

    def __init__(self, model_config: Union[str, Path], model_path: str = None, gpu_id: int = None) -> None:
        """
        Args:
            model_config: path to a HYVideoReWard eval YAML (string or
                ``pathlib.Path``). Parsed by upstream
                :func:`qwenvl.evaluation.eval.load_eval_config` so the
                ``inferencer:`` block, ``model_args`` / ``data_args``
                fallbacks, and ``data`` / ``run`` exclusions all follow
                exactly the same rules as ``eval.py``.
            gpu_id: CUDA device index to load the full model onto. Defaults to
                the process's current device (``torch.cuda.current_device()``),
                which the trainer has already set via ``torch.cuda.set_device``
                -- so each rank loads its own full copy on its own GPU (required
                before :meth:`apply_fsdp` shards it). Using ``current_device``
                (instead of reading ``$LOCAL_RANK``) stays correct under both
                torchrun and deepspeed launchers and respects any per-process
                ``CUDA_VISIBLE_DEVICES`` remapping. This also avoids HF
                ``device_map="auto"`` spreading one copy across all visible
                GPUs, which is wrong under FSDP.
        """
        if not isinstance(model_config, (str, Path)):
            raise TypeError(
                f"model_config must be a YAML path (str/Path), got "
                f"{type(model_config).__name__}"
            )
        if gpu_id is None:
            import torch
            gpu_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
        cfg = load_eval_config(str(model_config))["inferencer"]
        if model_path is not None:
            cfg["model_path"] = model_path
            cfg["ref_model_path"] = model_path
        # Constructor `mode` here selects the matching video_processor variant.
        # Default `with_grad` is friendly for ReFL-style end-to-end training;
        # ``reward(..., mode="server")`` below still returns per-dim vectors.
        cfg.setdefault("mode", "with_grad")

        # Whole inferencer construction (registry resolution, kwarg filtering,
        # video_res bucket lookup, etc.) is upstream's job.
        self.inferencer = build_inferencer(cfg, gpu_id=gpu_id, mode=cfg["mode"])
        self.logit2dim = getattr(
            type(self.inferencer), "Logit2Dim", _DEFAULT_LOGIT2DIM
        )
        logger.info(
            f"HyVideoRewardLocal ready"
            f"model_class={cfg.get('model_class') or cfg.get('model_type')})"
        )

    # ------------------------------------------------------------------
    # nn.Module-ish passthroughs so callers can treat this wrapper like the
    # other reward models (the trainer calls `.eval()` / `.to()` generically).
    # ------------------------------------------------------------------
    @property
    def model(self):
        """The underlying HF reward model (``Qwen3VLForConditionalGeneration*``)."""
        return self.inferencer.model

    @property
    def patch_factors(self):
        """``(temporal_factor, spatial_factor)`` the video processor requires.

        Frames must be divisible by ``temporal_factor`` and each frame's H/W by
        ``spatial_factor``. Mirrors the asserts in the Qwen3-VL video processor
        (``T % temporal_patch_size`` and ``H/W % (patch_size * 2)``). Read off
        the live processor so this stays correct if the backbone/config changes.
        """
        vp = self.inferencer.processor.video_processor
        temporal_factor = int(getattr(vp, "temporal_patch_size", 2))
        spatial_factor = int(getattr(vp, "patch_size", 16)) * 2
        return temporal_factor, spatial_factor

    def train(self):
        self.inferencer.model.train()
        return self

    def eval(self):
        self.inferencer.model.eval()
        return self

    def to(self, *args, **kwargs):
        self.inferencer.model.to(*args, **kwargs)
        return self

    def parameters(self, *args, **kwargs):
        return self.inferencer.model.parameters(*args, **kwargs)

    # ------------------------------------------------------------------
    # FSDP2
    # ------------------------------------------------------------------
    def _iter_decoder_layers(self):
        """Yield the HF decoder layers to shard (Qwen3-VL text + vision blocks).

        Defensive across transformers versions: the text tower lives at
        ``model.model.language_model.layers`` (newer Qwen3-VL) or
        ``model.model.layers`` (older), and the vision tower at
        ``model.model.visual.blocks``.
        """
        hf = self.inferencer.model
        base = getattr(hf, "model", hf)

        lang = getattr(base, "language_model", None)
        text_layers = getattr(lang, "layers", None) if lang is not None else None
        if text_layers is None:
            text_layers = getattr(base, "layers", None)
        if text_layers is not None:
            yield from text_layers

        visual = getattr(base, "visual", None)
        vis_blocks = getattr(visual, "blocks", None) if visual is not None else None
        if vis_blocks is not None:
            yield from vis_blocks

    def apply_fsdp(self, mesh=None, param_dtype=None, reduce_dtype=None):
        """Shard the underlying HF model in-place with FSDP2 (``fully_shard``).

        Unlike the internal PureTorch reward models (LRM / ref-model), this
        model is loaded eagerly by HF ``from_pretrained`` -- full weights are
        already materialized on every rank, so there is NO DCP load step. We
        only need to *shard* the already-loaded module, mirroring how the
        frozen HF text encoder is sharded in ``hymm/models/text_encoder``.

        Args:
            mesh: FSDP device mesh. Defaults to the pure-torch
                ``default_fsdp_mesh`` from the global parallel state.
            param_dtype / reduce_dtype: mixed-precision dtypes; default to the
                model's current parameter dtype (no cast).
        """
        import torch
        from torch.distributed._composable.fsdp import (
            fully_shard,
            MixedPrecisionPolicy,
        )

        if mesh is None:
            from hymm.core.global_vars import get_parallel_state

            p_state = get_parallel_state()
            assert getattr(p_state, "backend", None) == "pure_torch", (
                "HyVideoRewardLocal.apply_fsdp requires the pure_torch backend."
            )
            mesh = p_state.backend_state.default_fsdp_mesh

        if param_dtype is None:
            param_dtype = next(self.inferencer.model.parameters()).dtype
        if reduce_dtype is None:
            reduce_dtype = param_dtype

        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype,
            cast_forward_inputs=True,
        )
        fsdp_config = {"mesh": mesh, "mp_policy": mp_policy}

        num_sharded = 0
        for layer in self._iter_decoder_layers():
            fully_shard(layer, **fsdp_config)
            num_sharded += 1
        if num_sharded == 0:
            raise RuntimeError(
                "HyVideoRewardLocal.apply_fsdp found no decoder layers to shard; "
                "the HF model layout may have changed -- check _iter_decoder_layers."
            )
        # Shard the root last so any unwrapped params (embeddings, reward head,
        # lm_head, norms) are covered too.
        fully_shard(self.inferencer.model, **fsdp_config)

        self.inferencer.model.eval()
        for p in self.inferencer.model.parameters():
            p.requires_grad = False

        logger.info(
            f"HyVideoRewardLocal: applied FSDP2 to {num_sharded} layers "
            f"(param_dtype={param_dtype})"
        )
        return self

    def _parse_score(self, score):
        """Split a ``[B, N]`` score tensor into ``{dim_name: [B, 1]}`` per dim.

        ``score`` is the differentiable per-dim reward (N = number of heads,
        e.g. TA/VQ/MQ/AES). Each column is kept as a ``[B, 1]`` tensor (grad
        preserved) so the ReFL loss can weight + ``.mean()`` each dim.
        """
        return {
            self.logit2dim[i]: score[:, i:i + 1] for i in range(score.shape[1])
        }

    def reward(self, videos: List[str], prompts: List[str], **kwargs):
        """Score videos, returning ``{dim_name: [B, 1]}`` (grad-preserving)."""
        scores = self.inferencer.reward(
            videos=videos, prompts=prompts, **kwargs
        )  # [B, N]
        return self._parse_score(scores)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test HyVideoRewardLocal with a HYVideoReWard eval YAML."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to an HYVideoReWard eval YAML (e.g. qwenvl/configs/hyvideo/*.yaml).",
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    rm = HyVideoRewardLocal(args.config)
    print(rm.reward(videos=[args.video], prompts=[args.prompt]))
