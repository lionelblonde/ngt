import math
from collections import OrderedDict
from collections.abc import Callable
from typing import Optional, Union

from beartype import beartype
from einops import pack
import torch
from torch import nn
from torch.nn import functional as ff
from torch.distributions.transforms import TanhTransform
from torch.distributions import TransformedDistribution, Normal
from torch.nn.utils.parametrizations import spectral_norm

from helpers import logger


LOG_STD_BOUNDS = [-5.0, 2.0]
LOG_2PI = float(math.log(2.0 * math.pi))  # cache it
HIDDEN_BASE = [256, 256]


@beartype
def _activation_from_name(name: str) -> tuple[type[nn.Module], float]:
    key = name.strip().lower()
    if key == "silu":
        return nn.SiLU, 1.0
    if key == "relu":
        return nn.ReLU, nn.init.calculate_gain("relu")
    raise ValueError(f"unsupported nonlinearity: {name}. choose 'relu' or 'silu'")


@beartype
def clamp_to_within_unit_ball(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    return torch.clamp(x, -1.0 + eps, 1.0 - eps)


@beartype
def log_module_info(model: nn.Module):

    def _fmt(n) -> str:
        if n // 10 ** 6 > 0:
            out = str(round(n / 10 ** 6, 2)) + " M"
        elif n // 10 ** 3:
            out = str(round(n / 10 ** 3, 2)) + " k"
        else:
            out = str(n)
        return out

    logger.info("logging model specs")
    logger.info(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"total trainable params: {_fmt(num_params)}.")


@beartype
def init(*,
         gain: float,
         constant_bias: float = 0.0,
    ) -> Callable[[nn.Module], None]:
    """Perform orthogonal initialization"""

    def _init(m: nn.Module) -> None:

        if (isinstance(m, (nn.Conv2d, nn.Linear))):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, constant_bias)
        elif (isinstance(m, (nn.BatchNorm2d, nn.LayerNorm))):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    return _init


@beartype
def snwrap(*,
           enabled: bool = False
    ) -> Callable[[nn.Module], nn.Module]:
    """Spectral normalization wrapper"""

    def _snwrap(m: nn.Module) -> nn.Module:
        if enabled and isinstance(m, (nn.Linear, nn.Conv2d)):
            return spectral_norm(m)
        return m

    return _snwrap


@beartype
def build_fc_stack(*,
                   in_dim: int,
                   hidden_dims: list[int],
                   make_fc: Callable[[int, int], nn.Module],
                   make_norm: Optional[Callable[[int], nn.Module]] = None,
                   make_act: Callable[[], nn.Module],
    ) -> nn.Sequential:
    layers = []
    prev_dim = in_dim
    for i, hid_dim in enumerate(hidden_dims, start=1):
        block_layers = OrderedDict([
            ("fc", make_fc(prev_dim, hid_dim)),
        ])
        if make_norm is not None:
            block_layers["ln"] = make_norm(hid_dim)
        block_layers["nl"] = make_act()
        layers.append((
            f"fc_block_{i}",
            nn.Sequential(block_layers),
        ))
        prev_dim = hid_dim
    return nn.Sequential(OrderedDict(layers))


class Discriminator(nn.Module):

    @beartype
    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 input_mode: str,
                 *,
                 device: Union[str, torch.device],
                 hidden_dims: Optional[tuple[int, ...]] = None,
        ):
        super().__init__()
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]

        leak = 0.2  # hard-coded

        # define the input dimension
        in_dim = ob_dim
        match input_mode:
            case "ss":
                in_dim += ob_dim
            case "sa":
                in_dim += ac_dim
            case _:
                raise ValueError("invalid input mode")

        hidden_dims_ = list(hidden_dims) if hidden_dims is not None else HIDDEN_BASE.copy()

        # assemble the last layers and output heads
        self.fc_stack = build_fc_stack(
            in_dim=in_dim,
            hidden_dims=hidden_dims_,
            make_fc=lambda din, dout: nn.Linear(din, dout, device=device),
            make_act=lambda: nn.LeakyReLU(leak, inplace=True),
        )
        self.head = nn.Linear(hidden_dims_[-1], 1, device=device)

        # perform initialization
        self.fc_stack.apply(init(gain=nn.init.calculate_gain("leaky_relu", param=leak)))
        self.head.apply(init(gain=1.0))

    @beartype
    def forward(self, input_a: torch.Tensor, input_b: Optional[torch.Tensor]) -> torch.Tensor:
        if input_b is not None:
            x, _ = pack([input_a, input_b], "b *")  # concatenate along last dim
        else:
            x = input_a
        return self.head(self.fc_stack(x))  # no sigmoid here


class RandomPredictor(nn.Module):

    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 input_mode: str,
                 *,
                 out_emb_dim: int,
                 device: Union[str, torch.device],
                 hidden_dims: Optional[tuple[int, ...]] = None,
                 predictor_spectral_norm: bool = False,
                 prior_spectral_norm: bool = False,
                 make_untrainable: bool = False,
        ):
        super().__init__()
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]
        self.input_mode = input_mode

        apply_sn = snwrap(
            enabled=(prior_spectral_norm if make_untrainable else predictor_spectral_norm),
        )
        leak = 0.2  # hard-coded

        # define the input dimension
        in_dim = ob_dim
        match self.input_mode:
            case "ss":
                in_dim += ob_dim
            case "sa":
                in_dim += ac_dim
            case _:
                raise ValueError("invalid input mode")

        hidden_dims_ = list(hidden_dims) if hidden_dims is not None else HIDDEN_BASE.copy()

        # assemble the layers and output heads
        self.fc_stack = build_fc_stack(
            in_dim=in_dim,
            hidden_dims=hidden_dims_,
            make_fc=lambda din, dout: apply_sn(nn.Linear(din, dout, device=device)),
            make_act=lambda: nn.LeakyReLU(leak, inplace=True),
        )
        self.head = apply_sn(nn.Linear(hidden_dims_[-1], out_emb_dim, device=device))

        # perform initialization
        self.fc_stack.apply(init(gain=nn.init.calculate_gain("leaky_relu", param=leak)))
        self.head.apply(init(gain=1.0))

        if make_untrainable:
            # prevent the weights from ever being updated
            for param in self.fc_stack.parameters():
                param.requires_grad = False
            for param in self.head.parameters():
                param.requires_grad = False

    @beartype
    def forward(self, input_a: torch.Tensor, input_b: Optional[torch.Tensor]) -> torch.Tensor:
        if input_b is not None:
            x, _ = pack([input_a, input_b], "b *")  # concatenate along last dim
        else:
            x = input_a
        return self.head(self.fc_stack(x))


class Critic(nn.Module):

    @beartype
    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 *,
                 layer_norm: bool,
                 nonlinearity: str = "silu",
                 hidden_dims: Optional[tuple[int, ...]] = None,
                 device: Union[str, torch.device]):
        super().__init__()
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]
        ln_or_not = nn.LayerNorm if layer_norm else nn.Identity
        act_cls, act_gain = _activation_from_name(nonlinearity)

        hidden_dims_ = list(hidden_dims) if hidden_dims is not None else HIDDEN_BASE.copy()

        # assemble the last layers and output heads
        self.fc_stack = build_fc_stack(
            in_dim=ob_dim + ac_dim,
            hidden_dims=hidden_dims_,
            make_fc=lambda din, dout: nn.Linear(din, dout, device=device),
            make_norm=lambda dim: ln_or_not(dim, device=device),
            make_act=lambda: act_cls(inplace=True),
        )
        self.head = nn.Linear(hidden_dims_[-1], 1, device=device)

        # perform initialization
        self.fc_stack.apply(init(gain=act_gain))
        self.head.apply(init(gain=1e-3))

    @beartype
    def forward(self, ob: torch.Tensor, ac: torch.Tensor) -> torch.Tensor:
        x, _ = pack([ob, ac], "b *")
        return self.head(self.fc_stack(x))


class TanhGaussActor(nn.Module):

    @beartype
    def __init__(self,
                 ob_shape: tuple[int, ...],
                 ac_shape: tuple[int, ...],
                 min_ac: torch.Tensor,
                 max_ac: torch.Tensor,
                 *,
                 layer_norm: bool,
                 nonlinearity: str = "silu",
                 device: Union[str, torch.device],
        ):
        super().__init__()
        ob_dim = ob_shape[-1]
        ac_dim = ac_shape[-1]
        ln_or_not = nn.LayerNorm if layer_norm else nn.Identity
        act_cls, act_gain = _activation_from_name(nonlinearity)

        # register buffers: action rescaling
        self.register_buffer("action_scal", (max_ac - min_ac) / 2.0)
        self.register_buffer("action_bias", (max_ac + min_ac) / 2.0)

        # feature extractor
        self.fc_stack = nn.Sequential(OrderedDict([
            ("fc_block_1", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(ob_dim, HIDDEN_BASE[0], device=device)),
                ("ln", ln_or_not(HIDDEN_BASE[0], device=device)),
                ("nl", act_cls(inplace=True)),
            ]))),
            ("fc_block_2", nn.Sequential(OrderedDict([
                ("fc", nn.Linear(HIDDEN_BASE[0], HIDDEN_BASE[1], device=device)),
                ("ln", ln_or_not(HIDDEN_BASE[1], device=device)),
                ("nl", act_cls(inplace=True)),
            ]))),
        ]))
        self.head = nn.Linear(HIDDEN_BASE[1], 2 * ac_dim, device=device)

        # perform initialization
        self.fc_stack.apply(init(gain=act_gain))
        self.head.apply(init(gain=1e-3))

    @beartype
    def rescale(self, ac: torch.Tensor):
        """Rescale a (-1, 1)-bounded tensor to [min_ac, max_ac]"""
        return (ac * self.action_scal) + self.action_bias

    @staticmethod
    @beartype
    def bound_log_std(log_std: torch.Tensor) -> torch.Tensor:
        """Stability trick from OpenAI SpinUp / Denis Yarats"""
        log_std = clamp_to_within_unit_ball(torch.tanh(log_std))
        lo, hi = LOG_STD_BOUNDS
        return lo + 0.5 * (hi - lo) * (log_std + 1.0)

    @beartype
    def get_mu_ls(self,
                  ob: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the sufficient statistics"""
        return self.head(self.fc_stack(ob)).chunk(2, dim=-1)

    @beartype
    def forward(self, ob: torch.Tensor) -> TransformedDistribution:
        """Produce a distribution over the action space"""
        mu, ls = self.get_mu_ls(ob)
        ls = self.bound_log_std(ls)
        # wrap in a tanh transform
        return TransformedDistribution(Normal(mu, ls.exp(), validate_args=False),
                                       [TanhTransform(cache_size=1)],
                                       validate_args=False)

    @staticmethod
    @beartype
    def analytical_kl(mu1: torch.Tensor,
                      ls1: torch.Tensor,
                      mu2: torch.Tensor,
                      ls2: torch.Tensor,
        ) -> list[torch.Tensor]:
        """Return the non-reduced KL. To reduce: `.sum(-1)`"""
        # pre-clamp the log stds
        ls1 = ls1.clamp(*LOG_STD_BOUNDS)
        ls2 = ls2.clamp(*LOG_STD_BOUNDS)
        # convert log std to variance
        var1 = torch.exp(2 * ls1)
        var2 = torch.exp(2 * ls2)
        # compute separate KLs
        kl_mu = ((mu1 - mu2).pow(2)) / var2
        kl_ls = (ls2 - ls1) + (0.5 * var1 / var2) - 0.5
        return [kl_mu, kl_ls]

    @beartype
    def get_action(self,
                   ob: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
        """Draw a reparameterized sample, compute its log-prob, and compute the mode"""

        dist = self(ob)

        # sample
        samp_raw = dist.rsample()

        # log_prob and mode
        log_prob = dist.log_prob(samp_raw).sum(-1, keepdim=True)
        # extract the base distribution's mean, then apply each transform
        mode_raw = dist.base_dist.mean
        for tranform in dist.transforms:
            mode_raw = tranform(mode_raw)

        # rescale sample and mode
        samp = self.rescale(samp_raw)
        mode = self.rescale(mode_raw)
        log_prob -= torch.log(self.action_scal).sum().view(1, 1)

        # return (with complete key names):
        # - reparametrized sample in action space
        # - log prob summed over action dims, shape [B, 1]
        # - deterministic best action
        return {"sample": samp, "log_prob": log_prob, "mode": mode}

    @beartype
    def get_log_prob(self,
                     ob: torch.Tensor,
                     ac: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
        """Compute the log-prob of a *given* action"""

        dist = self(ob)

        # map env action -> (-1, 1) space
        xx = (ac - self.action_bias) / self.action_scal
        xx = clamp_to_within_unit_ball(xx)

        log_prob = dist.log_prob(xx).sum(-1, keepdim=True)
        log_prob -= torch.log(self.action_scal).sum().view(1, 1)

        # return (with complete key names):
        # - log prob summed over action dims, shape [B, 1]
        return {"log_prob": log_prob}

    @beartype
    def get_action_metal(self,
                         ob: torch.Tensor,
                         eps: torch.Tensor,  # i.i.d. N(0, 1)
        ) -> dict[str, torch.Tensor]:
        """Same as above but cudagraphs-compatible, i.e. takes the noise an input."""
        mu, ls = self.get_mu_ls(ob)

        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        ls = torch.where(torch.isfinite(ls), ls, torch.zeros_like(ls))
        ls = self.bound_log_std(ls)  # sanitize with custom util

        ls = ls.clamp(*LOG_STD_BOUNDS)

        zz = mu + (ls.exp().clamp(min=1e-6) * eps)

        samp = (torch.tanh(zz) * self.action_scal) + self.action_bias
        mode = (torch.tanh(mu) * self.action_scal) + self.action_bias

        gaus = -0.5 * ((eps ** 2) + (2 * ls) + LOG_2PI)
        gaus = gaus.sum(dim=-1, keepdim=True)
        gaus -= torch.log(self.action_scal).sum().view(1, 1)
        jaco = correction(zz).sum(dim=-1, keepdim=True)
        log_prob = gaus - jaco

        return {"sample": samp, "log_prob": log_prob, "mode": mode}


    @beartype
    def get_log_prob_metal(self,
                           ob: torch.Tensor,
                           ac: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
        mu, ls = self.get_mu_ls(ob)

        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        ls = torch.where(torch.isfinite(ls), ls, torch.zeros_like(ls))
        ls = self.bound_log_std(ls)  # sanitize with custom util

        ls = ls.clamp(*LOG_STD_BOUNDS)

        # invert the squashing and scaling to get pre-tanh variable zz
        # because we assume: ac = tanh(zz) * action_scal + action_bias
        xx = (ac - self.action_bias) / self.action_scal
        xx = clamp_to_within_unit_ball(xx)
        zz = torch.atanh(xx)

        eps = (zz - mu) / (ls.exp().clamp(min=1e-6))

        gaus = -0.5 * ((eps ** 2) + (2 * ls) + LOG_2PI)
        gaus = gaus.sum(dim=-1, keepdim=True)
        gaus -= torch.log(self.action_scal).sum().view(1, 1)
        jaco = correction(zz).sum(dim=-1, keepdim=True)
        log_prob = gaus - jaco

        return {"log_prob": log_prob}


@beartype
def correction(x: torch.Tensor) -> torch.Tensor:
    # stable version of: log(1 - tanh(x)^2)
    ax = x.clamp(-80.0, 80.0).abs()
    return 2.0 * (math.log(2.0) - ax - ff.softplus(-2.0 * ax))
