import enum
import math
from typing import Callable, Optional

import numpy as np
import torch as th

from . import path
from .integrators import ode, sde
from .utils import mean_flat

__all__ = ["ModelType", "PathType", "WeightType", "Transport", "Sampler", "SNRType"]


class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)


class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class SNRType(enum.Enum):
    UNIFORM = enum.auto()
    LOGNORM = enum.auto()
    UNIFORM_LOGNORM_MIX = enum.auto()


# Copied from Flux.1
def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def compute_empirical_mu(image_seq_len: int, num_steps: int = None) -> float:
    """Empirical log-mu for Flux2-style timestep shift (paired with exp(mu) in time_shift)."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300 or num_steps is None:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


# Copied from Flux.1
def time_shift(mu: float, sigma: float, t: th.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


class Transport:
    def __init__(self, *, model_type, path_type, loss_type, train_eps, sample_eps, snr_type,
                 snr_mix_uniform_ratio=0.5,
                 training_timesteps=1000, reverse_time_schedule=False, shift=1.0, reverse=False,
                 use_flux_shift=False, use_flux2_shift=False, flux2_empirical_num_steps: Optional[int] = None,
                 flux_base_num_tokens=256, flux_base_log_shift=0.5,
                 flux_max_num_tokens=4096, flux_max_log_shift=1.15):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type](reverse=reverse)
        self.train_eps = train_eps
        self.sample_eps = sample_eps

        self.snr_type = snr_type
        self.snr_mix_uniform_ratio = snr_mix_uniform_ratio  # proportion of uniform sampling in uniform+lognorm mix, range [0.0, 1.0]
        # timestep shift: http://arxiv.org/abs/2403.03206
        self.shift = shift  # flow matching shift factor, =sqrt(m/n)
        self.reverse = reverse
        assert not (use_flux_shift and use_flux2_shift), (
            "use_flux_shift and use_flux2_shift are mutually exclusive; enable at most one."
        )
        self.use_flux_shift = use_flux_shift
        self.use_flux2_shift = use_flux2_shift
        self.flux2_empirical_num_steps = flux2_empirical_num_steps
        self.flux_base_num_tokens = flux_base_num_tokens
        self.flux_base_log_shift = flux_base_log_shift
        self.flux_max_num_tokens = flux_max_num_tokens
        self.flux_max_log_shift = flux_max_log_shift

        self.training_timesteps = training_timesteps
        self.reverse_time_schedule = reverse_time_schedule  # deprecated

    def prior_logp(self, z):
        """
        Standard multivariate normal prior
        Assume z is batched
        """
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2.0 * np.log(2 * np.pi) - th.sum(x**2) / 2.0
        return th.vmap(_fn)(z)

    def sample_start(self, x1, generator=None):
        """
        Args:
          x1 - data point; [batch, *dim]
          generator: generator for random noise, the initial seed must be args.seed + dp_rank
        """
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        if self.reverse:
            t = th.ones((len(x1),)) * t0
        else:
            t = th.ones((len(x1),)) * t1
        t = t.to(x1[0])

        # torch.empty_like(x1).normal_() is equivalent to torch.randn_like(x1)
        if isinstance(x1, (list, tuple)):
            x0 = [th.empty_like(img_start).normal_(generator=generator) for img_start in x1]
        else:
            x0 = th.empty_like(x1).normal_(generator=generator)

        return t, x0, x1

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if type(self.path_sampler) in [path.VPCPlan]:
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) and (
            self.model_type != ModelType.VELOCITY or sde
        ):  # avoid numerical issue by taking a first semi-implicit step
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def _apply_shift(self, t, n_tokens=None):
        """Apply time shift to sampled timesteps.

        Supports flux2 shift, flux shift (token-count-dependent), and constant shift.
        Returns the shifted timesteps (or the original if no shift is needed).
        """
        if self.use_flux2_shift:
            assert n_tokens is not None, "n_tokens must be provided for flux2 shift"
            num_steps = int(self.flux2_empirical_num_steps) if self.flux2_empirical_num_steps is not None else None
            mu = compute_empirical_mu(int(n_tokens), num_steps)
            if self.reverse:
                t = time_shift(mu, 1.0, t)
            else:
                t = 1 - time_shift(mu, 1.0, 1 - t)
        elif self.use_flux_shift:
            assert n_tokens is not None, "n_tokens must be provided for flux shift"
            mu = get_lin_function(
                x1=self.flux_base_num_tokens, y1=self.flux_base_log_shift,
                x2=self.flux_max_num_tokens, y2=self.flux_max_log_shift)(n_tokens)
            if self.reverse:
                t = time_shift(mu, 1.0, t)
            else:
                t = 1 - time_shift(mu, 1.0, 1 - t)
        elif self.shift != 1.:
            if self.reverse:
                # xt = (1 - t) * x1 + t * x0
                t = (self.shift * t) / (1 + (self.shift - 1) * t)
            else:
                # xt = t * x1 + (1 - t) * x0
                t = t / (self.shift - (self.shift - 1) * t)
        return t

    def sample(self, x1, n_tokens=None, generator=None):
        """Sampling x0 & t based on shape of x1 (if needed)
        Args:
          x1 - data point; [batch, *dim]
          generator: generator for random noise, the initial seed must be args.seed + dp_rank
        """
        # torch.empty_like(x1).normal_() is equivalent to torch.randn_like(x1)
        if isinstance(x1, (list, tuple)):
            x0 = [th.empty_like(img_start).normal_(generator=generator) for img_start in x1]
        else:
            x0 = th.empty_like(x1).normal_(generator=generator)

        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)

        if self.snr_type == SNRType.UNIFORM:
            t = th.rand((len(x1),), generator=generator, device=x1.device) * (t1 - t0) + t0
            t = self._apply_shift(t, n_tokens)
        elif self.snr_type == SNRType.LOGNORM:
            u = th.normal(mean=0.0, std=1.0, size=(len(x1),), generator=generator, device=x1.device)
            t = 1 / (1 + th.exp(-u)) * (t1 - t0) + t0
            t = self._apply_shift(t, n_tokens)
        elif self.snr_type == SNRType.UNIFORM_LOGNORM_MIX:
            n = len(x1)
            # mask=True -> uniform, mask=False -> lognorm
            mask = th.rand((n,), generator=generator, device=x1.device) < self.snr_mix_uniform_ratio
            # uniform part (no shift applied)
            t_uniform = th.rand((n,), generator=generator, device=x1.device) * (t1 - t0) + t0
            # lognorm part (shift applied)
            u = th.normal(mean=0.0, std=1.0, size=(n,), generator=generator, device=x1.device)
            t_lognorm = 1 / (1 + th.exp(-u)) * (t1 - t0) + t0
            t_lognorm = self._apply_shift(t_lognorm, n_tokens)
            t = th.where(mask, t_uniform, t_lognorm)
        else:
            raise ValueError(f"Unknown snr type: {self.snr_type}")

        t = t.to(x1[0])
        return t, x0, x1

    def get_model_t(self, t):
        if self.reverse_time_schedule:
            return (1 - t) * self.training_timesteps
        else:
            return t * self.training_timesteps

    def get_scheduler_t(self, model_t):
        if self.reverse_time_schedule:
            return 1 - model_t / self.training_timesteps
        else:
            return model_t / self.training_timesteps

    def training_losses(self, model, x1, model_kwargs=None, timestep=None, n_tokens=None):
        """Loss for training the score model
        Args:
            model: backbone model; could be score, noise, or velocity
            x1: datapoint
            model_kwargs: additional arguments for the model
            timestep: the timestep at which to evaluate loss.
            n_tokens: number of tokens for flux shift
        """
        if model_kwargs == None:
            model_kwargs = {}
        t, x0, x1 = self.sample(x1, n_tokens)
        if timestep is not None:
            t = th.ones_like(t) * timestep
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        model_output = model(xt, self.get_model_t(t), **model_kwargs)['x']
        assert model_output.size() == xt.size(), f"Output shape from model does not match input shape: " \
                                                 f"{model_output.size()} != {xt.size()}"

        terms = {}
        if self.model_type == ModelType.VELOCITY:
            terms["loss"] = mean_flat(((model_output - ut) ** 2))
        else:
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()

            if self.model_type == ModelType.NOISE:
                terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))

        return terms

    @staticmethod
    def ragged_mse(pred, ref):
        if isinstance(pred, th.Tensor):
            assert pred.size() == ref.size(), f"Pred and ref must have the same shape: {pred.size()} != {ref.size()}"
            return mean_flat((pred - ref) ** 2)
        else:
            batch_mse = []
            for pred_i, ref_i in zip(pred, ref):
                if isinstance(pred_i, th.Tensor):
                    # corresponds to pred as a list of 4-D tensors [B x (N, C, H, W)]
                    assert isinstance(ref_i, th.Tensor)
                    assert pred_i.size() == ref_i.size(), f"Pred and ref must have the same shape: {pred_i.size()} != {ref_i.size()}"
                    batch_mse.append(th.mean((pred_i - ref_i) ** 2)[None])
                else:
                    # corresponds to pred as a list of list of 4-D tensors [B x (N_i x [1, C, H_ij, W_ij])]
                    mse_i = []
                    for pred_ij, ref_ij in zip(pred_i, ref_i):
                        if pred_ij.ndim - ref_ij.ndim == 1:
                            ref_ij = ref_ij[None, ...]
                        assert pred_ij.size() == ref_ij.size(), f"Pred and ref must have the same shape: {pred_ij.size()} != {ref_ij.size()}"
                        mse_i.append(th.mean((pred_ij - ref_ij) ** 2)[None])
                    batch_mse.append(sum(mse_i) / len(mse_i))
            return th.cat(batch_mse)

    def training_losses_fn(self, t, x0, xt, ut, model_output):
        """used by Transfusion to generate a partial loss fn for each micro batch
        """

        terms = {}
        if self.model_type == ModelType.VELOCITY:
            terms["loss"] = self.ragged_mse(model_output, ut)
        else:
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()

            if self.model_type == ModelType.NOISE:
                terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))

        return terms

    def get_drift(self):
        """member function for obtaining the drift of the probability flow ODE"""

        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return -drift_mean + drift_var * model_output  # by change of variable

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return -drift_mean + drift_var * score

        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn

    def get_score(
        self,
    ):
        """member function for obtaining score of
        x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = (
                lambda x, t, model, **kwargs: model(x, t, **kwargs)
                / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
            )
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(
                model(x, t, **kwargs), x, t
            )
        else:
            raise NotImplementedError()

        return score_fn


class Sampler:
    """Sampler class for the transport model"""

    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """

        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):
        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion

        sde_drift = lambda x, t, model, **kwargs: self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(
            x, t, model, **kwargs
        )

        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion

    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""

        if last_step is None:
            last_step_fn = lambda x, t, model, **model_kwargs: x
        elif last_step == "Mean":
            last_step_fn = (
                lambda x, t, model, **model_kwargs: x + sde_drift(x, t, model, **model_kwargs) * last_step_size
            )
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t  # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = lambda x, t, model, **model_kwargs: x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][
                0
            ] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = (
                lambda x, t, model, **model_kwargs: x + self.drift(x, t, model, **model_kwargs) * last_step_size
            )
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample

    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        time_shifting_factor=None,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            time_shifting_factor=time_shifting_factor,
        )
        self.ode = _ode
        return _ode.sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
    ):
        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """

        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(th.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn
