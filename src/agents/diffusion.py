import math
from functools import partial
from typing import Any, Optional, Union

from beartype import beartype
import numpy as np
import torch
from torch import nn
from torch.nn import functional as ff


EPS = 1e-8
LOG_EPS = 1e-20
PROB_CLIP = 10.0
DEFAULT_BETA_SCHEDULE = "linear"
DEFAULT_TIMESTEPS = 10
DEFAULT_CLAMP_MAGNITUDE = 10.0
DIFFUSION_HIDDEN_DIM = 128
TIME_EMB_DIM = 16


@beartype
def extract(a: torch.Tensor,
            t: torch.Tensor,
            x_shape: torch.Size) -> torch.Tensor:
    batch_size, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


@beartype
def cosine_beta_schedule(timesteps: int,
                         *,
                         device: Union[str, torch.device],
                         s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = np.clip(alphas_cumprod, EPS, None)
    alphas_cumprod /= alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=EPS, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=torch.float, device=device)


@beartype
def linear_beta_schedule(timesteps: int,
                         *,
                         device: Union[str, torch.device],
                         beta_start: float = 1e-4,
                         beta_end: float = 2e-2) -> torch.Tensor:
    betas = np.linspace(beta_start, beta_end, timesteps)
    return torch.tensor(betas, dtype=torch.float, device=device)


@beartype
def vp_beta_schedule(timesteps: int,
                     *,
                     device: Union[str, torch.device]) -> torch.Tensor:
    """VP stands for variance preserving."""
    t = np.arange(1, timesteps + 1)
    b_max = 10.0
    b_min = 0.1
    alpha = np.exp(-b_min / timesteps - 0.5 * (b_max - b_min) * (2 * t - 1) / timesteps ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=torch.float, device=device)


@beartype
def l2_to_prob(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(-x, min=-PROB_CLIP, max=PROB_CLIP))


class SinusoidalPosEmb(nn.Module):

    @beartype
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


@beartype
def mean_flat(x: torch.Tensor) -> torch.Tensor:
    """Take the mean over all non-batch dimensions."""
    return x.mean(dim=list(range(1, len(x.shape))))


@beartype
def normal_kl(mean1: Any,
              logvar1: Any,
              mean2: Any,
              logvar2: Any) -> torch.Tensor:
    """Compute KL divergence between two gaussians."""
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


class MLP(nn.Module):

    @beartype
    def __init__(self,
                 x_dim: int,
                 hidden_dims: tuple[int, ...],
                 device: Union[str, torch.device],
                 t_dim: int = TIME_EMB_DIM):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2, device=device),
            # official code uses Mish; avoid it here due to NaNs.
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(t_dim * 2, t_dim, device=device),
        )

        in_dim = x_dim + t_dim
        mid_layers = []
        prev_dim = in_dim
        for hid_dim in hidden_dims:
            mid_layers.extend([
                nn.Linear(prev_dim, hid_dim, device=device),
                nn.LeakyReLU(negative_slope=0.2),
            ])
            prev_dim = hid_dim
        self.mid_layer = nn.Sequential(*mid_layers)

        self.final_layer = nn.Linear(hidden_dims[-1], x_dim, device=device)

    @beartype
    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x = torch.cat([x, t_emb], dim=1)
        x = self.mid_layer(x)
        return self.final_layer(x)


class DiffusionDiscriminator(nn.Module):

    @beartype
    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 max_ac: torch.Tensor,
                 input_mode: str,
                 device: Union[str, torch.device],
                 beta_schedule: str = DEFAULT_BETA_SCHEDULE,
                 n_timesteps: int = DEFAULT_TIMESTEPS,
                 clamp_magnitude: float = DEFAULT_CLAMP_MAGNITUDE,
                 hidden_dims: tuple[int, ...] = (DIFFUSION_HIDDEN_DIM,) * 3,
                 *,
                 clip_denoised: bool = True,
                 predict_epsilon: bool = True):
        super().__init__()
        ob_dim = ob_shape[-1]
        self.ac_dim = ac_shape[-1]
        self.input_mode = input_mode

        in_dim = ob_dim
        match self.input_mode:
            case "ss":
                in_dim += ob_dim
            case "sa":
                in_dim += self.ac_dim
            case _:
                raise ValueError("invalid input mode")

        self.max_ac = max_ac
        self.model = MLP(x_dim=in_dim, hidden_dims=hidden_dims, device=device)
        self.clamp_magnitude = clamp_magnitude

        match beta_schedule:
            case "linear":
                betas = linear_beta_schedule(n_timesteps, device=device)
            case "cosine":
                betas = cosine_beta_schedule(n_timesteps, device=device)
            case "vp":
                betas = vp_beta_schedule(n_timesteps, device=device)
            case _:
                raise NotImplementedError

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([
            torch.ones((1,), dtype=torch.float, device=device),
            alphas_cumprod[:-1],
        ])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(torch.clamp(alphas_cumprod, min=EPS)),
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(torch.clamp(1.0 - alphas_cumprod, min=EPS)),
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod",
            torch.log(1.0 - alphas_cumprod + EPS),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.sqrt(1.0 / (alphas_cumprod + EPS)),
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(torch.clamp(1.0 / alphas_cumprod - 1.0, min=EPS)),
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + EPS)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=LOG_EPS) + EPS),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod + EPS),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod + EPS),
        )

        self.loss_fn = partial(ff.mse_loss, reduction="none")

    @beartype
    def forward(self):
        raise NotImplementedError

    @beartype
    def predict_start_from_noise(self,
                                 x_t: torch.Tensor,
                                 t: torch.Tensor,
                                 noise: torch.Tensor) -> torch.Tensor:
        """Map the model output to x_0 when needed."""
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        return noise

    @beartype
    def q_posterior(self,
                    x_start: torch.Tensor,
                    x_t: torch.Tensor,
                    t: torch.Tensor,
                    *,
                    is_dict: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        if not is_dict:
            posterior_variance = extract(self.posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        else:
            posterior_variance = extract(self.posterior_variance, t, x_t.shape) * torch.ones_like(x_t)
            posterior_log_variance_clipped = (
                extract(self.posterior_log_variance_clipped, t, x_t.shape) * torch.ones_like(x_t)
            )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @beartype
    def p_mean_variance(self,
                        x: torch.Tensor,
                        t: torch.Tensor,
                        *,
                        is_dict: bool = False) -> Union[
                            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                            dict[str, torch.Tensor],
                        ]:
        noise = self.model(x, t)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)

        if self.clip_denoised:
            x_recon[:, :self.ac_dim].clamp_(-self.max_ac, self.max_ac)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon,
            x_t=x,
            t=t,
            is_dict=is_dict,
        )

        if not is_dict:
            return model_mean, posterior_variance, posterior_log_variance
        return {
            "mean": model_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance,
            "pred_xstart": x_recon,
        }

    @beartype
    def p_sample(self,
                 x: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
        batch_size, *_ = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(batch_size, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @beartype
    def p_sample_loop(self,
                      shape: torch.Size,
                      *,
                      verbose: bool = False,
                      return_diffusion: bool = False) -> Union[
                          torch.Tensor,
                          tuple[torch.Tensor, torch.Tensor],
                      ]:
        _ = verbose
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        if return_diffusion:
            diffusion = [x]

        for i in reversed(range(self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps)
            x = torch.clamp(x, min=-self.clamp_magnitude, max=self.clamp_magnitude)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        return x

    @beartype
    def q_mean_variance(self,
                        x_start: torch.Tensor,
                        t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    @beartype
    def q_sample(self,
                 x_start: torch.Tensor,
                 t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @beartype
    def p_losses(self,
                 x_start: torch.Tensor,
                 t: torch.Tensor,
                 *,
                 disc_ddpm: bool = False) -> torch.Tensor:
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, t)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            if not disc_ddpm:
                return self.loss_fn(x_recon, noise)
            l2_ = torch.mean(self.loss_fn(x_recon, noise), dim=1)
            loss = l2_to_prob(l2_)
            return torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)
        return self.loss_fn(x_recon, x_start)

    @beartype
    def q_posterior_mean_variance(self,
                                  x_start: torch.Tensor,
                                  x_t: torch.Tensor,
                                  t: torch.Tensor) -> tuple[torch.Tensor,
                                                            torch.Tensor,
                                                            torch.Tensor]:
        """Compute mean and variance of q(x_{t-1} | x_t, x_0)."""
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @beartype
    def loss(self,
             x: torch.Tensor,
             *,
             disc_ddpm: bool = False) -> torch.Tensor:
        batch_size = x.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, t, disc_ddpm=disc_ddpm)

    @beartype
    def calc_reward(self, x_start: torch.Tensor) -> torch.Tensor:
        batch_size = x_start.shape[0]

        vb = []
        for t in reversed(range(self.n_timesteps)):
            t_batch = torch.empty(batch_size, device=x_start.device, dtype=torch.int64).fill_(t)
            noise = torch.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)

            with torch.no_grad():
                x_recon = self.model(x_t, t_batch)
                l2_ = self.loss_fn(x_recon, noise)
                loss_t = l2_to_prob(l2_)
            vb.append(loss_t)

        vb = torch.stack(vb, dim=1)
        disc_cost = vb.sum(dim=1) / self.n_timesteps
        return disc_cost.sum(dim=1) / x_start.shape[1]
