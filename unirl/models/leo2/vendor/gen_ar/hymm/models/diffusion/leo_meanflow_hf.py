# Leo MeanFlow (iMF) inference model + sampling dispatch.
#
# This is a NEW, self-contained module. It does NOT modify any existing file.
#
# ============================ Why this file is needed ============================
# Sampling always goes through the HF model: the sampler entry appends "HF" to
# `--model-structure` (e.g. `LeoModelMeanFlow` -> `LeoModelMeanFlowHF`), and the diffusion
# model is built as `LeoModelHF`. Crucially, the transformer `forward` lives on the shared
# base class `LeoModelBase`, and:
#       LeoModel(LeoModelBase)            <- used for TRAINING
#       LeoModelHF(LeoModelBase, ...)     <- used for SAMPLING / inference
# The MeanFlow forward written for training (`LeoModelMeanFlow`, a subclass of `LeoModel`)
# therefore does NOT apply to the inference model. This module supplies the inference
# counterpart `LeoModelMeanFlowHF` (a subclass of `LeoModelHF`) that REUSES the exact same
# MeanFlow forward + helpers (late 1/4 layers + head do AdaLN with the second time `r`).
# Because those methods only touch members defined on `LeoModelBase`, the train-time and
# sample-time backbones are numerically identical.
#
# It adds NO parameters (r reuses the same `time_embed` / `final_layer`), so existing
# checkpoints load unchanged.
#
# ============================ Selection (no existing file changed) ============================
# Importing this module installs (idempotently) a dispatch extension recognizing
# `model-structure: LeoModelMeanFlowHF`, and overrides `build_diffusion_pipeline` so the
# model uses the MeanFlow sampling pipeline (which feeds `r` at every step). Select it via:
#   ... run_sample_leo_imf.sh torchrun <config> ... --model-structure LeoModelMeanFlow
# (the sampler entry turns `LeoModelMeanFlow` into `LeoModelMeanFlowHF`).

from .leo_hf import LeoModelHF
from .leo_meanflow import LeoModelMeanFlow


class LeoModelMeanFlowHF(LeoModelHF):
    """`LeoModelHF` whose late layers + head do AdaLN with a second time input ``r``."""

    # ---- Reuse the MeanFlow backbone forward + helpers verbatim (bound to this class). ----
    # They only reference members that live on the shared `LeoModelBase`, and their
    # __globals__ still point at the leo_meanflow module (so LeoOutput / get_parallel_state /
    # maybe_gather_seq / torch resolve correctly).
    _mf_num_t_layers = LeoModelMeanFlow._mf_num_t_layers
    _mf_time_states = staticmethod(LeoModelMeanFlow._mf_time_states)
    _mf_stack = staticmethod(LeoModelMeanFlow._mf_stack)
    _mf_run_tail = LeoModelMeanFlow._mf_run_tail
    _mf_head = LeoModelMeanFlow._mf_head
    forward = LeoModelMeanFlow.forward

    def load_generation_config(self, generation_config_path):
        super().load_generation_config(generation_config_path)
        # iMF / MeanFlow MUST run with NO classifier-free guidance: the guidance is already
        # distilled into the model (training target v_tgt = u_t + (1 - 1/w)(v_cond - v_uncond)),
        # so a single cond forward already outputs the guided average velocity. Applying CFG
        # again would double-count it. `diff_guidance_scale` drives BOTH the input-prep
        # cfg_factor (prepare_model_inputs: 2 if > 1 else 1) AND the sampling loop, so we force
        # it to 1.0 here to keep the whole pipeline consistent (one cond forward per step).
        gscale = getattr(self.generation_config, "diff_guidance_scale", 1.0)
        if gscale != 1.0:
            print(
                f"[LeoModelMeanFlowHF] iMF requires guidance_scale == 1.0 (no CFG); "
                f"overriding diff_guidance_scale {gscale} -> 1.0.",
                flush=True,
            )
            self.generation_config.diff_guidance_scale = 1.0

    def build_diffusion_pipeline(self):
        # Build the standard pipeline, then upgrade it in-place to the MeanFlow pipeline,
        # which injects the second time input `r` (next timestep) at every denoising step.
        if self._diffusion_pipeline is None:
            super().build_diffusion_pipeline()
            from ...ar.pipelines.pipeline_leo_meanflow import Leo2PipelineMeanFlow
            self._diffusion_pipeline.__class__ = Leo2PipelineMeanFlow


# --------------------------------------------------------------------------- #
# Non-invasive dispatch extension for `model-structure: LeoModelMeanFlowHF`.
# --------------------------------------------------------------------------- #
def _build_meanflow_hf(args, **kw):
    from .leo_config import LeoConfig, core_model_config_from_args

    device = kw.get("device", None)
    dtype = kw.get("dtype", None)
    factor_kwargs = {"device": device, "dtype": dtype}

    model_name = args.model_name.split(".")[-1]
    model_config = LeoConfig.from_name(model_name, **core_model_config_from_args(args))

    if args.text_branch_model_name is not None:
        text_branch_config = LeoConfig.from_name(
            args.text_branch_model_name.split(".")[-1],
            **core_model_config_from_args(args, prefix="text_branch_"),
        )
    else:
        text_branch_config = None

    if args.audio_branch_model_name is not None:
        audio_branch_config = LeoConfig.from_name(
            args.audio_branch_model_name.split(".")[-1],
            **core_model_config_from_args(args, prefix="audio_branch_"),
        )
    else:
        audio_branch_config = None

    model = LeoModelMeanFlowHF(
        args, model_config, txt_config=text_branch_config, audio_config=audio_branch_config,
        **factor_kwargs,
    )
    return model, dict(
        main_branch=model_config, text_branch=text_branch_config, audio_branch=audio_branch_config,
    )


def install_meanflow_hf_dispatch():
    import hymm.models.diffusion as _pkg

    current = getattr(_pkg, "build_model", None)
    if current is None or getattr(current, "_meanflow_hf_patched", False):
        return

    real_build = current

    def patched_build_model(args, *a, **kw):
        if getattr(args, "model_structure", None) == "LeoModelMeanFlowHF":
            return _build_meanflow_hf(args, *a, **kw)
        return real_build(args, *a, **kw)

    patched_build_model._meanflow_hf_patched = True
    patched_build_model._orig_build_model = real_build
    _pkg.build_model = patched_build_model


# Auto-install on import.
install_meanflow_hf_dispatch()
