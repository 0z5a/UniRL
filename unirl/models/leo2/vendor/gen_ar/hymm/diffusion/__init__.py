# This script contains two functions to create diffusion and transport objects.
#
# For the `create_diffusion` function, it is modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
#
# For the `create_transport` function, it is modified from the SiT repo:
#     SiT:   https://github.com/willisma/SiT/blob/main/transport/__init__.py
#

from .ddpm import gaussian_diffusion as gd
from .flow.transport import *
from .pipelines import load_pipeline
from .ddpm.respace import SpacedDiffusion, space_timesteps
from .schedulers import DDPMScheduler, FlowMatchDiscreteScheduler, EulerDiscreteScheduler
from ..utils.helpers import default


def create_diffusion(
        *,
        steps=1000,                         # DDPM
        learn_sigma=False,                  # Improved DDPM
        sigma_small=False,                  # DDPM
        noise_schedule="linear",            # DDPM(linear), StableDiffusion(scaled_linear)
        enforce_zero_terminal_snr=False,    # whether to enforce terminal SNR to 0
        use_kl=False,
        predict_type='epsilon',             # DDPM(sample or epsilon), Progressive Distillation(v_prediction)
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
        mse_loss_weight_type='constant',    # Min-SNR
        beta_start=0.0001,                  # DDPM
        beta_end=0.02,                      # DDPM
        noise_offset=0.0,                   # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        shift_snr=1.0,                      # Simple Diffusion
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps, beta_start, beta_end)
    if enforce_zero_terminal_snr:
        if predict_type == "v_prediction":
            betas = gd.enforce_zero_terminal_snr(betas)
        else:
            raise ValueError("We only support for enforcing terminal SNR to 0 when predict_type==v_prediction.")
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [steps]
    mean_type = gd.predict_type_dict[predict_type]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=mean_type,
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        mse_loss_weight_type=mse_loss_weight_type,
        noise_offset=noise_offset,
        shift_snr=shift_snr,
    )


def create_transport(
        *,
        path_type,
        prediction,
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
        snr_type="uniform",
        snr_mix_uniform_ratio=0.5,
        reverse=False,
        shift=1.0,
        use_flux_shift=False,
        use_flux2_shift=False,
        flux2_empirical_num_steps=None,
        flux_base_num_tokens=256,
        flux_base_log_shift=0.5,
        flux_max_num_tokens=4096,
        flux_max_log_shift=1.15,
):
    """
    Create a Transport object for the flow matching schedulers.

    Args:
        *: A barrier to prevent positional arguments.
        path_type (str): Type of path to use. Can be "linear", "gvp", or "vp".
        prediction (str): Model prediction type. Can be "velocity", "score", or "noise".
        loss_weight (str, optional): Loss weight type. Can be "velocity", "likelihood", or None.
        train_eps (float, optional): Small epsilon for avoiding instability during training.
        sample_eps (float, optional): Small epsilon for avoiding instability during sampling.
        snr_type (str): Type of SNR to use. Can be "lognorm", "uniform", or "uniform_lognorm_mix".
        snr_mix_uniform_ratio (float): Proportion of uniform sampling in uniform+lognorm mix,
            range [0.0, 1.0]. 0.0 means all lognorm, 1.0 means all uniform. Only used when snr_type is "uniform_lognorm_mix".
        shift (float): SNR shift value. It equals to sqrt(m/n) which is formulated by SD3.
        reverse (bool): Whether to reverse the flow.
        use_flux_shift (bool): Whether to use flux shift.
        use_flux2_shift (bool): Whether to use empirical Flux2 shift (overrides use_flux_shift when True).
        flux2_empirical_num_steps (int, optional): num_steps for empirical mu during training; defaults to training_timesteps.
        flux_base_log_shift (float): Base log shift value for flux.
        flux_max_log_shift (float): Max log shift value for flux.

    Returns:
        state (Transport): A Transport object for the flow matching schedulers.
    """
    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    if snr_type == "lognorm":
        snr_type = SNRType.LOGNORM
    elif snr_type == "uniform":
        snr_type = SNRType.UNIFORM
    elif snr_type == "uniform_lognorm_mix":
        snr_type = SNRType.UNIFORM_LOGNORM_MIX
    else:
        raise ValueError(f"Invalid snr type {snr_type}")

    path_choice = {
        "linear": PathType.LINEAR,
        "gvp": PathType.GVP,
        "vp": PathType.VP,
    }

    path_type = path_choice[path_type.lower()]

    if path_type in [PathType.VP]:
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    elif path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY:
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    else:  # velocity & [GVP, LINEAR] is stable everywhere
        train_eps = 0
        sample_eps = 0

    # create flow state
    state = Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
        snr_type=snr_type,
        snr_mix_uniform_ratio=snr_mix_uniform_ratio,
        shift=shift,
        reverse=reverse,
        use_flux_shift=use_flux_shift,
        use_flux2_shift=use_flux2_shift,
        flux2_empirical_num_steps=flux2_empirical_num_steps,
        flux_base_num_tokens=flux_base_num_tokens,
        flux_base_log_shift=flux_base_log_shift,
        flux_max_num_tokens=flux_max_num_tokens,
        flux_max_log_shift=flux_max_log_shift,
    )

    return state


def load_denoiser(diffusion_config, **kwargs):
    if diffusion_config.denoise_type == "flow":
        # backward compatibility for flux_base_shift and flux_max_shift
        config = dict(
            path_type=diffusion_config.flow_path_type,
            prediction=diffusion_config.flow_predict_type,
            loss_weight=diffusion_config.flow_loss_weight,
            train_eps=diffusion_config.flow_train_eps,
            sample_eps=diffusion_config.flow_sample_eps,
            snr_type=diffusion_config.flow_snr_type,
            snr_mix_uniform_ratio=getattr(diffusion_config, 'flow_snr_mix_uniform_ratio', 0.5),
            reverse=diffusion_config.flow_reverse,
            shift=diffusion_config.flow_shift,
            use_flux_shift=diffusion_config.use_flux_shift,
            use_flux2_shift=getattr(diffusion_config, "use_flux2_shift", False),
            flux2_empirical_num_steps=getattr(diffusion_config, "flux2_empirical_num_steps", None),
            flux_base_num_tokens=diffusion_config.flux_base_num_tokens,
            flux_base_log_shift=default(diffusion_config.flux_base_log_shift, diffusion_config.flux_base_shift),
            flux_max_num_tokens=diffusion_config.flux_max_num_tokens,
            flux_max_log_shift=default(diffusion_config.flux_max_log_shift, diffusion_config.flux_max_shift),
        )
        config.update(kwargs)
        denoiser = create_transport(**config)
    elif diffusion_config.denoise_type == "ddpm":
        denoiser = create_diffusion(noise_schedule=diffusion_config.ddpm_noise_schedule,
                                    predict_type=diffusion_config.ddpm_predict_type,
                                    enforce_zero_terminal_snr=diffusion_config.enforce_zero_terminal_snr,
                                    learn_sigma=diffusion_config.ddpm_learn_sigma,
                                    beta_start=diffusion_config.ddpm_beta_start,
                                    beta_end=diffusion_config.ddpm_beta_end,
                                    noise_offset=diffusion_config.ddpm_noise_offset,
                                    shift_snr=diffusion_config.ddpm_shift_snr,
                                    )
    else:
        raise ValueError(f"Unknown denoise type: {diffusion_config.denoise_type}")
    return denoiser


def load_scheduler(args):
    """ Load the denoising scheduler for inference. """
    if args.denoise_type == "ddpm":
        rescale_betas_zero_snr = False
        if args.enforce_zero_terminal_snr:
            if args.predict_type == "v_prediction":
                rescale_betas_zero_snr = True
            else:
                raise ValueError("We only support for enforcing terminal SNR to 0 when predict_type==v_prediction.")
        scheduler = DDPMScheduler(beta_start=args.ddpm_beta_start,
                                  beta_end=args.ddpm_beta_end,
                                  beta_schedule=args.ddpm_noise_schedule,
                                  variance_type='learned_range' if args.ddpm_learn_sigma else 'fixed_small',
                                  prediction_type=args.ddpm_predict_type,
                                  steps_offset=1,
                                  clip_sample=False,
                                  rescale_betas_zero_snr=rescale_betas_zero_snr,
                                  )
    elif args.denoise_type == "flow":
        # backward compatibility for flux_base_shift and flux_max_shift
        scheduler = FlowMatchDiscreteScheduler(shift=default(args.sample_flow_shift, args.flow_shift),
                                               reverse=args.flow_reverse,
                                               solver=args.flow_solver,
                                               use_flux_shift=default(args.sample_use_flux_shift, args.use_flux_shift),
                                               use_flux2_shift=default(
                                                   getattr(args, "sample_use_flux2_shift", False),
                                                   getattr(args, "use_flux2_shift", False),
                                               ),
                                               flux_base_num_tokens=args.flux_base_num_tokens,
                                               flux_base_log_shift=default(args.flux_base_log_shift, args.flux_base_shift),
                                               flux_max_num_tokens=args.flux_max_num_tokens,
                                               flux_max_log_shift=default(args.flux_max_log_shift, args.flux_max_shift),
                                               start_sigma=getattr(args, "flow_start_sigma", 1.0),
                                               end_sigma=getattr(args, "flow_end_sigma", 0.0),
                                               )
    elif args.denoise_type == "euler":
        scheduler = EulerDiscreteScheduler(
                                            # "_diffusers_version": "0.21.2",
                                            # FIXME move argument below to config file
                                            beta_end =  0.012,
                                            beta_schedule =  "scaled_linear",
                                            beta_start =  0.00085,
                                            # clip_sample =  False,
                                            interpolation_type =  "linear",
                                            num_train_timesteps =  1000,
                                            prediction_type =  "epsilon",
                                            # sample_max_value =  1.0, # Difference of diffusers  "0.21.2",
                                            # set_alpha_to_one =  False,
                                            # skip_prk_steps =  True,
                                            steps_offset =  1,
                                            timestep_spacing =  "leading",
                                            trained_betas =  None,
                                            use_karras_sigmas =  False,
                                            )
    else:
        raise ValueError(f"Invalid denoise type {args.denoise_type}")
    return scheduler


def load_diffusion_pipeline(args, rank, diffusion_model, pipeline_name, device=None, progress_bar_config=None, **extra_model_dict):
    scheduler = load_scheduler(args)
    # Only enable progress bar for rank 0
    progress_bar_config = progress_bar_config or {'leave': True, 'disable': rank != 0}

    pipeline = load_pipeline(pipeline_name)(
        args=args,
        diffusion_model=diffusion_model,
        scheduler=scheduler,
        progress_bar_config=progress_bar_config,
        **extra_model_dict
    )

    pipeline = pipeline.to(device)

    return pipeline
