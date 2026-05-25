import tempfile
from pathlib import Path
from functools import partial
from typing import Optional, Union, Any
from collections.abc import MutableMapping

from einops import rearrange, pack, unpack
from beartype import beartype
from omegaconf import OmegaConf, DictConfig
import wandb
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import functional as ff
from torch.nn.utils import clip_grad
from torch import autograd
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, CudaGraphModule
from torchrl.data import ReplayBuffer

from helpers import logger
from helpers.running_moments import RMSConfig, RunningMeanStd
from agents.nets import (
    log_module_info, TanhGaussActor, Critic, Discriminator, RandomPredictor,
)
from agents.losses import HistogramLossGaussian
from agents.pwil import PWILRewarder

from agents.diffusion import DiffusionDiscriminator

from tensordict.nn.functional_modules import _exclude_td_from_pytree  # noqa
_exclude_td_from_pytree().set()  # required; or set env: EXCLUDE_TD_FROM_PYTREE=1


SHAPING_FLOOR = 1e-10


@beartype
def resolve_reward_hidden_dims(reward_archi: str) -> tuple[int, ...]:
    match reward_archi:
        case "128x128x128":
            hidden_dims = (128, 128, 128)
        case "256x256":
            hidden_dims = (256, 256)
        case "512x512":
            hidden_dims = (512, 512)
        case "256x256x256":
            hidden_dims = (256, 256, 256)
        case _:
            raise ValueError("invalid reward archi")
    return hidden_dims


class Agent(object):

    @beartype
    def __init__(self,
                 *,
                 net_shapes: dict[str, tuple[int, ...]],
                 min_ac: np.ndarray,
                 max_ac: np.ndarray,
                 device: torch.device,
                 hps: MutableMapping[Any, Any],
                 replay_dataset: Optional[ReplayBuffer] = None,
                 expert_dataset: Optional[ReplayBuffer] = None,
                 expert_atoms: Optional[list[TensorDict]] = None):

        ob_shape = net_shapes["ob_shape"]
        ac_shape = net_shapes["ac_shape"]
        self.ac_dim = ac_shape[-1]  # used in orchestrator when "metal" is ON + targ ent

        self.device = device

        self.min_ac = torch.tensor(min_ac, dtype=torch.float, device=self.device)
        self.max_ac = torch.tensor(max_ac, dtype=torch.float, device=self.device)

        assert isinstance(hps, DictConfig)
        self.hps = hps
        assert self.hps.input_mode in {"sa", "ss"}

        self.metal = self.hps.cudagraphs

        self.normalize_obs_for_actor_critic = bool(self.hps.get("normalize_obs", False))
        self.normalize_obs_for_reward = bool(
            self.hps.get("normalize_obs_for_reward", False),
        )

        self.obs_rms = None
        if self.normalize_obs_for_actor_critic or self.normalize_obs_for_reward:
            self.obs_rms = RunningMeanStd(
                shape=(ob_shape[-1],),
                cfg=RMSConfig(device=self.device),
            )

        self.qloss_func = ff.mse_loss

        self.timesteps_so_far = 0

        self.actor_updates_so_far = 0
        self.qnets_updates_so_far = 0

        self.reward_updates_so_far = 0

        self.best_eval_ep_ret = -float("inf")  # updated in orchestrator

        assert self.hps.segment_len <= self.hps.batch_size
        if self.hps.clip_norm <= 0:
            logger.info("clip_norm <= 0, hence disabled")

        # replay and expert dataset
        self.replay_dataset = replay_dataset
        self.expert_dataset = expert_dataset

        # define cap for the batch size on expert side
        self.expert_batch_size = self.hps.batch_size
        if self.hps.cap_e_batch_size_to_dataset_len and (self.expert_dataset is not None):
            self.expert_batch_size = min(self.expert_batch_size, len(self.expert_dataset))
        # create constant vector for reward assembly
        self.constant_one = torch.ones((1,), device=self.device)
        # create constant vector for gradient penalty
        self.grad_outputs = rearrange(
            torch.ones(self.expert_batch_size, device=self.device),
            "b -> b 1",
        )

        # expert atoms
        if self.hps.method == "pwil":  # tools for training only
            assert expert_atoms is not None
            # pwil rewarder
            self.pwil_rewarder = [
                PWILRewarder(
                    self.vectorize_expert_atoms(expert_atoms),
                    self.device,
                    self.hps.input_mode,
                    ob_shape,
                    ac_shape,
                    horizon=1000,
                )
                for _ in range(self.hps.num_envs)
            ]

        # POLICY: create online (and target) nets
        actor_net_args = [ob_shape, ac_shape, self.min_ac, self.max_ac]
        actor_net_kwargs = {"layer_norm": self.hps.layer_norm,
                            "nonlinearity": self.hps.nonlinearity}  # leave device out ("meta" stuff)
        actor_class = TanhGaussActor
        self.actor = actor_class(*actor_net_args, **actor_net_kwargs, device=self.device)
        self.actor_params = TensorDict.from_module(self.actor, as_module=True)
        # discard params of net
        self.actor = actor_class(*actor_net_args, **actor_net_kwargs, device="meta")
        self.actor_params.to_module(self.actor)
        self.actor_detach = actor_class(*actor_net_args, **actor_net_kwargs, device=self.device)
        # copy params to actor_detach without grad
        TensorDict.from_module(self.actor).data.to_module(self.actor_detach)
        # create wrapped policy functions for exploration and exploitation
        in_keys = ["observations"]
        get_action = self.actor_detach.eval().get_action
        if self.metal:
            in_keys.append("noise")
            get_action = self.actor_detach.eval().get_action_metal
        self.policy_exploit = TensorDictModule(get_action, in_keys=in_keys, out_keys=[
            "mode",
        ])
        self.policy_explore = TensorDictModule(get_action, in_keys=in_keys, out_keys=[
            "sample",
        ])
        if self.metal:
            # wrap to use cudagraphs
            self.policy_exploit = CudaGraphModule(self.policy_exploit)
            self.policy_explore = CudaGraphModule(self.policy_explore)

        # QNETS: create online and target nets
        qnet_net_args = [ob_shape, ac_shape]
        qnet_net_kwargs = {"layer_norm": self.hps.layer_norm,
                           "nonlinearity": self.hps.nonlinearity}
        self.qnet1 = Critic(*qnet_net_args, **qnet_net_kwargs, device=self.device)
        if self.hps.method == "p2il":
            self.qnet2 = self.qnet1
            self.qnets_params = TensorDict.from_module(self.qnet1, as_module=True)
        else:
            self.qnet2 = Critic(*qnet_net_args, **qnet_net_kwargs, device=self.device)
            self.qnets_params = TensorDict.from_modules(self.qnet1, self.qnet2, as_module=True)
        self.qnets_target = self.qnets_params.data.clone()
        # discard params of net
        self.qnet = Critic(*qnet_net_args, **qnet_net_kwargs, device="meta")
        self.qnets_params.to_module(self.qnet)

        # RENET: create reward net
        renet_args = [ob_shape, ac_shape, self.hps.input_mode]
        reward_hidden_dims = resolve_reward_hidden_dims(self.hps.reward_archi)
        renet_kwargs = {
            "hidden_dims": reward_hidden_dims,
            "device": self.device,
        }
        # treat each method separatly in a switch
        match self.hps.method:
            case "ngt":
                # create the predictor and prior nets
                random_predictor = partial(RandomPredictor, *renet_args, **renet_kwargs)
                out_dim_multiplier = self.hps.hlg_bins if self.hps.criteria == "hl_gauss" else 1
                self.predictor = random_predictor(
                    out_emb_dim=(self.hps.out_emb_dim * out_dim_multiplier),
                    predictor_spectral_norm=self.hps.predictor_spectral_norm,
                )
                self.prior = random_predictor(
                    out_emb_dim=self.hps.out_emb_dim,
                    prior_spectral_norm=self.hps.prior_spectral_norm,
                    make_untrainable=True,
                )
                # self.predictor_params = TensorDict.from_module(self.predictor, as_module=True)
                # self.predictor_lagged = self.predictor_params.data.clone()
                # create the inner criteria
                match self.hps.criteria:
                    case "huber-max":
                        t_func = partial(ff.huber_loss, reduction="none")
                        # pure L_infinity norm
                        # performance is bottlenecked by the worst dimension
                        # only the single biggest error matters

                        def i_func(x, y): return (x - y).abs().max(dim=-1, keepdim=True).values

                    case "huber-softmax_max":
                        t_func = partial(ff.huber_loss, reduction="none")
                        # smooth approximation of L_infinity norm

                        def i_func(x, y): return 0.05 * torch.logsumexp(
                            (x - y).abs() / 0.05, dim=-1, keepdim=True)

                    case "huber-lp8":
                        t_func = partial(ff.huber_loss, reduction="none")
                        # smooth, high-p norm, sitting between L1/L2 and L_infinity
                        # more aggressive than L2
                        # but still norm-like, smooth, all coordinates contribute

                        def i_func(x, y): return (
                            (x - y).abs().pow(8).sum(dim=-1, keepdim=True) + 1e-8).pow(1 / 8)

                    case "huber-topk":
                        t_func = partial(ff.huber_loss, reduction="none")
                        # average of the worst k% components
                        # only the largest k% matter, everyone else is ignored

                        def i_func(x, y): return torch.topk(
                            (x - y).abs(),
                            k=max(1, int(0.2 * x.size(-1))),  # the top k = the worst 20%
                            dim=-1).values.mean(dim=-1, keepdim=True)
                        # picks the worst 20 percent along the last dimension

                    case "huber-huber":
                        t_func = partial(ff.huber_loss, reduction="none")
                        i_func = partial(ff.huber_loss, reduction="none")
                    case "huber-mse":
                        t_func = partial(ff.huber_loss, reduction="none")
                        # L2
                        i_func = partial(ff.mse_loss, reduction="none")
                    case "huber-mse_softmax":
                        t_func = partial(ff.huber_loss, reduction="none")

                        def i_func(x, y): return partial(ff.mse_loss, reduction="none")(
                            ff.softmax(x, dim=1), ff.softmax(y, dim=1))

                    case "hl_gauss":
                        bin_width = (2 * self.hps.hlg_radius) / float(self.hps.hlg_bins)
                        sigma = self.hps.hlg_coeff * bin_width
                        hlg_loss = partial(
                            HistogramLossGaussian,
                            min_value=-self.hps.hlg_radius,
                            max_value=+self.hps.hlg_radius,
                            num_bins=int(self.hps.hlg_bins),
                            device=self.device,
                        )
                        t_func = hlg_loss(sigma=sigma)
                        i_func = hlg_loss(sigma=sigma)
                    case _:
                        raise ValueError("invalid NGT loss")
                match self.hps.reward_shaping:
                    case "linear" | "symexp":
                        pass
                    case _:
                        raise ValueError("invalid NGT reward shaping")
                self.criteria = {"t": t_func, "i": i_func}
            case "dac" | "wdac":
                # create the discriminator net
                self.discriminator = Discriminator(*renet_args, **renet_kwargs)
            case "diffail":
                # create the discriminator net
                self.discriminator = DiffusionDiscriminator(
                    ob_shape,
                    ac_shape,
                    self.max_ac,
                    self.hps.input_mode,
                    device=self.device,
                    beta_schedule=self.hps.diffail_beta_schedule,
                    n_timesteps=int(self.hps.diffail_n_timesteps),
                    clamp_magnitude=float(self.hps.diffail_clamp_magnitude),
                    hidden_dims=reward_hidden_dims,
                )
            case "p2il":
                # create reward net with same architecture as qnet
                self.rnet1 = Critic(
                    *qnet_net_args,
                    **qnet_net_kwargs,
                    hidden_dims=reward_hidden_dims,
                    device=self.device,
                )
                self.rnet2 = self.rnet1
                self.rnets_params = TensorDict.from_module(self.rnet1, as_module=True)
                # discard params of net
                self.rnet = Critic(
                    *qnet_net_args,
                    **qnet_net_kwargs,
                    hidden_dims=reward_hidden_dims,
                    device="meta",
                )
                self.rnets_params.to_module(self.rnet)
            case "pwil" | "bc" | "random" | "iqlearn":
                pass
            case _:
                raise ValueError("invalid method")
        # create wrapped reward functions for fast and easy computation
        self.reward = TensorDictModule(
            self.compute_reward,
            in_keys=["observations", "actions", "next_observations"],
            out_keys=["rewards"],
        )

        # set up the optimizers

        self.q_optimizer = Adam(
            self.qnet.parameters(),
            lr=self.hps.qnets_lr,
            capturable=False,
        )
        self.actor_optimizer = Adam(
            self.actor.parameters(),
            lr=self.hps.actor_lr,
            capturable=self.metal,
        )

        # setup log(alpha) [Lagrange Multiplier] for SAC: temperature parameter
        self.log_alpha = torch.nn.Parameter(
            torch.as_tensor(self.hps.alpha_init, device=self.device).log())
        if self.hps.autotune:
            # create learnable Lagrangian multiplier
            # common trick: learn log(alpha) instead of alpha directly
            self.targ_ent = -self.ac_dim  # set target entropy to -|A|
            if self.hps.ball_targ_ent:
                self.targ_ent /= 2.0  # set it to -|A|/2
            self.alpha_optimizer = Adam(
                [self.log_alpha],
                lr=self.hps.log_alpha_lr,
                capturable=self.metal,
            )

        log_module_info(self.actor)
        log_module_info(self.qnet1)
        if self.hps.method != "p2il":
            log_module_info(self.qnet2)

        if self.hps.method == "p2il":
            log_module_info(self.rnet1)
            self.p2il_reward_optimizer = Adam(
                self.rnet.parameters(),
                lr=self.hps.qnets_lr,
                capturable=False,
            )
        elif self.hps.method in {"pwil", "bc", "random", "iqlearn"}:
            pass  # not a "learned reward" method
        else:
            self.reward_optimizer = Adam(
                (
                    renet := (  # walrus for the module logger below
                        self.predictor if self.hps.method == "ngt" else self.discriminator
                    )
                ).parameters(),
                lr=self.hps.reward_lr,
                capturable=False,
            )

            log_module_info(renet)

    @beartype
    def vectorize_expert_atoms(self, expert_atoms: list[TensorDict]) -> torch.Tensor:
        """Aggregate all the transitions from the demos into one vector"""
        vector_list = []

        for td in expert_atoms:  # TensorDicts

            match self.hps.input_mode:
                case "sa":
                    vec = torch.cat(
                        [
                            td["observations"],
                            td["actions"],
                        ],
                        dim=1,
                    )  # shape: [T, ob_dim + ac_dim]
                case "ss":
                    vec = torch.cat(
                        [
                            td["observations"],
                            td["next_observations"],
                        ],
                        dim=1,
                    )  # shape: [T, 2 * ob_dim]
                case _:
                    raise ValueError("invalid input mode")

            vector_list.append(vec)

        return torch.cat(vector_list, dim=0)

    @beartype
    def batched_qf(self,
                   params: Any,
                   ob: torch.Tensor,
                   action: torch.Tensor,
                   next_q_value: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Use qnet from params"""
        with params.to_module(self.qnet):
            vals = self.qnet(ob, action)  # (B, 1) or (BxK, 1)
            if next_q_value is not None:
                next_q_value = next_q_value.unsqueeze(-1)  # (B, 1)
                return self.qloss_func(vals, next_q_value)
            return vals

    @beartype
    def batched_rf(self,
                   params: Any,
                   ob: torch.Tensor,
                   action: torch.Tensor) -> torch.Tensor:
        """Use reward net from params"""
        with params.to_module(self.rnet):
            return self.rnet(ob, action)

    @property
    @beartype
    def alpha(self) -> torch.Tensor:
        return ff.softplus(self.log_alpha) + 1e-10

    @beartype
    def predict(self,
                in_td: TensorDict,
                *,
                explore: bool,
        ) -> np.ndarray:
        """Predict with policy, with or without perturbation"""
        if self.metal and ("noise" not in in_td.keys()):
            obs = in_td["observations"]
            if explore:
                in_td.update({
                    "noise": torch.empty(
                        obs.size(0),
                        self.ac_dim,
                        device=self.device,
                        dtype=obs.dtype,
                    ).normal_(),
                })
            else:
                in_td.update({
                    "noise": torch.zeros(
                        obs.size(0),
                        self.ac_dim,
                        device=self.device,
                        dtype=obs.dtype,
                    ),
                })

        out_td = self.policy_explore(in_td) if explore else self.policy_exploit(in_td)
        action = out_td["sample" if explore else "mode"]
        action = torch.nan_to_num(action, nan=0.0, posinf=100.0, neginf=-100.0)  # arbitrary -/+

        return action.clamp(self.min_ac, self.max_ac).cpu().numpy()

    @beartype
    def update_qnets(self, batch: TensorDict) -> TensorDict:
        """Update critics with SAC targets"""

        self.q_optimizer.zero_grad()

        with torch.no_grad():

            # compute target action
            outs = self.actor.get_action(batch["next_observations"])
            next_action, next_state_log_pi = outs["sample"], outs["log_prob"]

            qf_next_target = torch.vmap(self.batched_qf, (0, None, None))(
                self.qnets_target, batch["next_observations"], next_action,
            )

            qf_min = qf_next_target.min(0).values
            if self.hps.bcq_style_targ_mix:
                # use BCQ style of target mixing: soft minimum
                qf_max = qf_next_target.max(0).values
                q_prime = ((0.75 * qf_min) + (0.25 * qf_max))
            else:
                # use hard minimum
                q_prime = qf_min

            # add the causal entropy regularization term
            q_prime -= self.alpha * next_state_log_pi

            # assemble the Bellman target
            targ_q = batch["rewards"].flatten() + (
                ~batch["dones"].flatten()
            ).float() * self.hps.gamma * q_prime.view(-1)

        qf_a_values = torch.vmap(self.batched_qf, (0, None, None, None))(
            self.qnets_params, batch["observations"], batch["actions"], targ_q,
        )
        qf_loss = qf_a_values.sum(0)

        qf_loss.backward()
        if self.hps.clip_norm > 0:
            clip_grad.clip_grad_norm_(self.qnet.parameters(), self.hps.clip_norm)
        self.q_optimizer.step()

        return TensorDict(
            {
                "stats/targ_q_abs_mean": targ_q.abs().mean(),
                "stats/next_state_log_pi": next_state_log_pi.mean(),
                "loss/qf_loss": qf_loss.detach(),
                **{
                    f"reward/p{str(k).zfill(2)}": batch["rewards"].quantile(q=(k / 100.0))
                    for k in [1, 5, 10, 50, 90, 95, 99]
                },
            },
        )

    @beartype
    def update_qnets_iq(self, p_batch: TensorDict, e_batch: TensorDict) -> TensorDict:
        """Update critics with the IQ-Learn objective."""

        self.q_optimizer.zero_grad()

        p_batch_size = p_batch["observations"].size(0)
        obs = torch.cat([p_batch["observations"], e_batch["observations"]], dim=0)
        next_obs = torch.cat([p_batch["next_observations"], e_batch["next_observations"]], dim=0)
        actions = torch.cat([p_batch["actions"], e_batch["actions"]], dim=0)
        e_dones = e_batch["dones"] if "dones" in e_batch else torch.zeros(
            e_batch["observations"].size(0),
            1,
            device=self.device,
            dtype=torch.bool,
        )
        dones = torch.cat([p_batch["dones"], e_dones], dim=0)

        outs = self.actor.get_action(obs)
        action_from_actor, state_log_pi = outs["sample"], outs["log_prob"]
        qf_pi = torch.vmap(self.batched_qf, (0, None, None))(
            self.qnets_params, obs, action_from_actor,
        )
        current_v = qf_pi.min(0).values - (self.alpha.detach() * state_log_pi)

        with torch.no_grad():
            outs = self.actor.get_action(next_obs)
            next_action, next_state_log_pi = outs["sample"], outs["log_prob"]
            qf_next_target = torch.vmap(self.batched_qf, (0, None, None))(
                self.qnets_target, next_obs, next_action,
            )
            next_v = qf_next_target.min(0).values - (self.alpha.detach() * next_state_log_pi)

        current_q = torch.vmap(self.batched_qf, (0, None, None))(
            self.qnets_params, obs, actions,
        )

        y = ((~dones).float() * self.hps.gamma * next_v)
        expert_slice = slice(p_batch_size, None)
        v0 = current_v[expert_slice].mean()

        def _compute_q_loss(
            current_q_: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            reward = (current_q_ - y)[expert_slice]
            divergence = self.hps.iq_div
            if divergence is not None:
                divergence = divergence.strip().lower()
            with torch.no_grad():
                match divergence:
                    case "hellinger":
                        phi_grad = 1 / (1 + reward).pow(2)
                    case "kl":
                        phi_grad = torch.exp(-reward - 1.0)
                    case "kl2":
                        phi_grad = ff.softmax(-reward, dim=0) * reward.shape[0]
                    case "kl_fix":
                        phi_grad = torch.exp(-reward)
                    case "js":
                        phi_grad = torch.exp(-reward) / (2.0 - torch.exp(-reward))
                    case _:
                        phi_grad = 1.0

            softq_loss = -(phi_grad * reward).mean()
            match self.hps.iq_loss:
                case "value_expert":
                    value_loss = (current_v - y)[expert_slice].mean()
                case "value":
                    value_loss = (current_v - y).mean()
                case "v0":
                    value_loss = (1.0 - self.hps.gamma) * v0
                case _:
                    raise ValueError("invalid iq loss")
            total_q_loss = softq_loss + value_loss

            alpha = float(self.hps.iq_alpha)

            q_logs = {
                "loss/iq_softq_loss": softq_loss.detach(),
                "loss/iq_value_loss": value_loss.detach(),
            }

            if bool(self.hps.iq_chi) or divergence == "chi":
                chi2_loss = (0.25 / alpha) * reward.pow(2).mean()
                total_q_loss += chi2_loss
                q_logs.update({"loss/iq_chi2_loss": chi2_loss.detach()})

            if bool(self.hps.iq_regularize):
                regularize_loss = (0.25 / alpha) * (current_q_ - y).pow(2).mean()
                total_q_loss += regularize_loss
                q_logs.update({"loss/iq_regularize_loss": regularize_loss.detach()})

            return total_q_loss, q_logs

        q1_loss, q1_logs = _compute_q_loss(current_q[0])
        q2_loss, q2_logs = _compute_q_loss(current_q[1])
        qf_loss = 0.5 * (q1_loss + q2_loss)
        qf_loss.backward()

        if self.hps.clip_norm > 0:
            clip_grad.clip_grad_norm_(self.qnet.parameters(), self.hps.clip_norm)
        self.q_optimizer.step()
        with torch.no_grad():
            self.qnets_target.lerp_(self.qnets_params, self.hps.polyak)

        def _avg(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
            return 0.5 * (v1 + v2)

        logs = {
            "stats/iq_v0": v0.detach(),
            "stats/iq_current_v": current_v.mean().detach(),
            "stats/iq_next_v": next_v.mean().detach(),
            "stats/next_state_log_pi": next_state_log_pi.mean().detach(),
            "loss/qf_loss": qf_loss.detach(),
        }
        for k in q1_logs:
            logs[k] = _avg(q1_logs[k], q2_logs[k])

        return TensorDict(logs)

    @beartype
    def update_qnets_p2il(self, p_batch: TensorDict, e_batch: TensorDict) -> TensorDict:
        """Update critics and reward nets with the P2IL objective."""
        p_batch_size = p_batch["observations"].size(0)
        reward_obs = torch.cat([p_batch["observations"], e_batch["observations"]], dim=0)
        next_obs = torch.cat([p_batch["next_observations"], e_batch["next_observations"]], dim=0)
        actions = torch.cat([p_batch["actions"], e_batch["actions"]], dim=0)
        e_dones = e_batch["dones"] if "dones" in e_batch else torch.zeros(
            e_batch["observations"].size(0),
            1,
            device=self.device,
            dtype=torch.bool,
        )
        dones = torch.cat([p_batch["dones"], e_dones], dim=0)
        obs = reward_obs
        if self.normalize_obs_for_actor_critic and (self.obs_rms is not None):
            obs = self.obs_rms.normalize(obs)
            next_obs = self.obs_rms.normalize(next_obs)
        if self.normalize_obs_for_reward and (self.obs_rms is not None):
            reward_obs = self.obs_rms.normalize(reward_obs)

        inner_steps = int(self.hps.p2il_inner_steps)
        if inner_steps <= 0:
            raise ValueError("p2il_inner_steps must be >= 1")
        targ_update_every = int(self.hps.p2il_target_update_frequency)
        if targ_update_every <= 0:
            raise ValueError("p2il_target_update_frequency must be >= 1")

        for l in range(inner_steps):
            qf_loss, logs = self.compute_p2il_loss(
                obs=obs,
                next_obs=next_obs,
                reward_obs=reward_obs,
                actions=actions,
                dones=dones,
                p_batch_size=p_batch_size,
            )
            if l < (inner_steps - 1):
                self.q_optimizer.zero_grad()
                self.p2il_reward_optimizer.zero_grad()
                qf_loss.backward()
                if self.hps.clip_norm > 0:
                    clip_grad.clip_grad_norm_(self.qnet.parameters(), self.hps.clip_norm)
                    clip_grad.clip_grad_norm_(self.rnet.parameters(), self.hps.clip_norm)
                self.q_optimizer.step()
                self.p2il_reward_optimizer.step()

            if (l % targ_update_every == 0) and (l < (inner_steps - 1)):
                self.update_p2il_qtarget()

        self.q_optimizer.zero_grad()
        self.p2il_reward_optimizer.zero_grad()
        qf_loss.backward()
        if self.hps.clip_norm > 0:
            clip_grad.clip_grad_norm_(self.qnet.parameters(), self.hps.clip_norm)
            clip_grad.clip_grad_norm_(self.rnet.parameters(), self.hps.clip_norm)
        self.q_optimizer.step()
        self.p2il_reward_optimizer.step()

        if ((self.qnets_updates_so_far + 1) % targ_update_every) == 0:
            self.update_p2il_qtarget()

        return TensorDict(logs)

    @beartype
    def compute_p2il_loss(self,
                          *,
                          obs: torch.Tensor,
                          next_obs: torch.Tensor,
                          reward_obs: torch.Tensor,
                          actions: torch.Tensor,
                          dones: torch.Tensor,
                          p_batch_size: int,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the strict official P2IL objective on one batch."""
        with torch.no_grad():
            outs = self.actor.get_action(obs)
            action_from_actor, state_log_pi = outs["sample"], outs["log_prob"]
        qf_pi = self.batched_qf(self.qnets_params, obs, action_from_actor)
        current_v = qf_pi - (self.alpha.detach() * state_log_pi)

        if bool(self.hps.p2il_use_target):
            with torch.no_grad():
                outs = self.actor.get_action(next_obs)
                next_action, next_state_log_pi = outs["sample"], outs["log_prob"]
                qf_next_target = self.batched_qf(self.qnets_target, next_obs, next_action)
                next_v = qf_next_target - (self.alpha.detach() * next_state_log_pi)
        else:
            with torch.no_grad():
                outs = self.actor.get_action(next_obs)
                next_action, next_state_log_pi = outs["sample"], outs["log_prob"]
            qf_next = self.batched_qf(self.qnets_params, next_obs, next_action)
            next_v = qf_next - (self.alpha.detach() * next_state_log_pi)

        current_q = self.batched_qf(self.qnets_params, obs, actions)
        current_r = self.batched_rf(self.rnets_params, reward_obs, actions)

        y = ((~dones).float() * self.hps.gamma * next_v)
        policy_slice = slice(0, p_batch_size)
        expert_slice = slice(p_batch_size, None)
        v0 = current_v[expert_slice].mean()

        tau = float(self.hps.p2il_tau)
        online = bool(self.hps.p2il_online)

        with torch.no_grad():
            delta_loss = -current_q + current_r + y
            if online:
                weights = ff.softmax(tau * delta_loss[policy_slice], dim=0)
            else:
                weights = ff.softmax(tau * delta_loss[expert_slice], dim=0)

        def _w_dot(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return lhs.flatten().dot(rhs.flatten())

        if online:
            reward_loss = -(current_r[expert_slice]).mean() + _w_dot(current_r[policy_slice], weights)
            logistic_loss = _w_dot((-current_q + y)[policy_slice], weights)
        else:
            reward_loss = -(current_r[expert_slice]).mean() + _w_dot(current_r[expert_slice], weights)
            logistic_loss = _w_dot((-current_q + y)[expert_slice], weights)

        match self.hps.iq_loss:
            case "value_expert":
                value_loss = (current_v - y)[expert_slice].mean()
            case "value":
                value_loss = (current_v - y).mean()
            case "v0":
                value_loss = (1.0 - self.hps.gamma) * v0
            case _:
                raise ValueError("invalid iq loss")

        qf_loss = reward_loss + logistic_loss + value_loss
        logs = {
            "loss/p2il_reward_loss": reward_loss.detach(),
            "loss/p2il_logistic_loss": logistic_loss.detach(),
            "loss/p2il_value_loss": value_loss.detach(),
        }

        if self.hps.grad_pen:
            p2il_gp = self.p2il_grad_pen(
                obs,
                actions,
                p_batch_size=p_batch_size,
            )
            qf_loss += p2il_gp
            logs.update({"loss/p2il_grad_pen": p2il_gp.detach()})

        alpha = float(self.hps.iq_alpha)
        divergence = self.hps.iq_div
        if divergence is not None:
            divergence = divergence.strip().lower()

        if bool(self.hps.iq_chi) or divergence == "chi":
            chi2_loss = (0.25 / alpha) * current_r[expert_slice].pow(2).mean()
            qf_loss += chi2_loss
            logs.update({"loss/p2il_chi2_loss": chi2_loss.detach()})

        if bool(self.hps.iq_regularize):
            regularize_loss = (0.25 / alpha) * (
                current_r.pow(2).mean() + current_q.pow(2).mean()
            )
            qf_loss += regularize_loss
            logs.update({"loss/p2il_regularize_loss": regularize_loss.detach()})

        logs.update({
            "stats/p2il_v0": v0.detach(),
            "stats/p2il_current_v": current_v.mean().detach(),
            "stats/p2il_next_v": next_v.mean().detach(),
            "stats/next_state_log_pi": next_state_log_pi.mean().detach(),
            "loss/qf_loss": qf_loss.detach(),
        })

        return qf_loss, logs

    @beartype
    def p2il_grad_pen(self,
                      obs: torch.Tensor,
                      actions: torch.Tensor,
                      p_batch_size: int,
        ) -> torch.Tensor:
        """Compute the official-style gradient penalty for P2IL."""
        policy_obs = obs[:p_batch_size]
        policy_actions = actions[:p_batch_size]
        expert_obs = obs[p_batch_size:]
        expert_actions = actions[p_batch_size:]
        xs = min(expert_obs.size(0), policy_obs.size(0))
        if xs <= 0:
            return torch.zeros((), device=self.device)

        expert_data = torch.cat([expert_obs[:xs], expert_actions[:xs]], dim=1)
        policy_data = torch.cat([policy_obs[:xs], policy_actions[:xs]], dim=1)

        alpha = torch.rand(xs, 1, device=self.device)
        alpha = alpha.expand_as(expert_data)
        interpolated = alpha * expert_data + (1.0 - alpha) * policy_data
        interpolated.requires_grad_(requires_grad=True)

        ob_dim = obs.size(-1)
        int_obs = interpolated[:, :ob_dim]
        int_actions = interpolated[:, ob_dim:]
        q_interp = self.batched_qf(self.qnets_params, int_obs, int_actions)
        ones = torch.ones_like(q_interp)
        grad = autograd.grad(
            outputs=q_interp,
            inputs=interpolated,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        lam = float(self.hps.p2il_lambda_gp)
        return lam * (grad.norm(2, dim=1) - 1.0).pow(2).mean()

    @beartype
    def update_p2il_qtarget(self):
        """Apply the official P2IL target critic update."""
        with torch.no_grad():
            if bool(self.hps.p2il_soft_target_update):
                self.qnets_target.lerp_(self.qnets_params, self.hps.polyak)
            else:
                self.qnets_target.lerp_(self.qnets_params, 1.0)

    @beartype
    def update_actor(self, batch: TensorDict) -> TensorDict:

        self.actor_optimizer.zero_grad()

        inputs = [batch["observations"]]
        funcn = "get_action"
        if self.metal:
            inputs.append(batch["noise"])
            funcn += "_metal"
        outs = getattr(self.actor, funcn)(*inputs)
        action_from_actor, state_log_pi = outs["sample"], outs["log_prob"]

        if self.hps.method == "p2il":
            min_qf_pi = self.batched_qf(
                self.qnets_params.data, batch["observations"], action_from_actor,
            )
        else:
            qf_pi = torch.vmap(self.batched_qf, (0, None, None))(
                self.qnets_params.data, batch["observations"], action_from_actor)
            min_qf_pi = qf_pi.min(0).values
        state_log_pi, min_qf_pi = state_log_pi.squeeze(-1), min_qf_pi.squeeze(-1)
        actor_loss = (self.alpha.detach() * state_log_pi) - min_qf_pi
        actor_loss = actor_loss.mean()

        actor_loss.backward()
        if self.hps.clip_norm > 0:
            clip_grad.clip_grad_norm_(self.actor.parameters(), self.hps.clip_norm)
        self.actor_optimizer.step()

        if self.hps.autotune:
            self.alpha_optimizer.zero_grad()
            alpha_loss = self.alpha * (-state_log_pi - self.targ_ent).detach()
            alpha_loss = alpha_loss.mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            return TensorDict(
                {
                    "loss/actor_loss": actor_loss.detach(),
                    "loss/alpha_loss": alpha_loss.detach(),
                    "stats/state_log_pi": state_log_pi.mean().detach(),
                    "vitals/alpha": self.alpha.detach(),
                },
            )

        return TensorDict(
            {
                "loss/actor_loss": actor_loss.detach(),
                "stats/state_log_pi": state_log_pi.mean().detach(),
                "vitals/alpha": self.alpha.detach(),
            },
        )

    @beartype
    def assemble_diffail_input(self,
                               input_a: torch.Tensor,
                               input_b: Optional[torch.Tensor]) -> torch.Tensor:
        """Assemble the diffusion-discriminator input in official order."""
        match self.hps.input_mode:
            case "sa":
                assert input_b is not None
                # official code uses action first in state-action mode
                return torch.cat([input_b, input_a], dim=1)
            case "ss":
                assert input_b is not None
                return torch.cat([input_a, input_b], dim=1)
            case _:
                raise ValueError("invalid input mode")

    @beartype
    def update_reward(self, p_batch: TensorDict, e_batch: TensorDict) -> TensorDict:

        if self.hps.method not in {"pwil", "iqlearn", "p2il"}:
            self.reward_optimizer.zero_grad()

        p_input_a = p_batch["observations"]
        e_input_a = e_batch["observations"]
        match self.hps.input_mode:
            case "ss":
                p_input_b = p_batch["next_observations"]
                e_input_b = e_batch["next_observations"]
            case "sa":
                p_input_b = p_batch["actions"]
                e_input_b = e_batch["actions"]
            case _:
                raise ValueError("invalid input mode")

        match self.hps.method:

            case "ngt":

                p_inputs = (p_input_a, p_input_b)
                e_inputs = (e_input_a, e_input_b)

                p_loss_ = self.criteria["t"](self.predictor(*p_inputs), self.prior(*p_inputs))
                e_loss_ = self.criteria["t"](self.predictor(*e_inputs), self.prior(*e_inputs))
                # NOTE: the scores above are not reduced

                # use only the desired proportion of experience per update
                p_loss = p_loss_.mean(dim=-1)
                e_loss = e_loss_.mean(dim=-1)
                p_mask = p_loss.clone().detach().uniform_()
                e_mask = e_loss.clone().detach().uniform_()

                p_mask = (p_mask < self.hps.p_proportion_of_exp_per_update).float()
                e_mask = (e_mask < self.hps.e_proportion_of_exp_per_update).float()
                p_loss = (p_mask * p_loss).sum() / torch.max(self.constant_one, p_mask.sum())
                e_loss = (e_mask * e_loss).sum() / torch.max(self.constant_one, e_mask.sum())

                # squeeze to get 0-dim tensors
                p_loss = p_loss.squeeze()
                e_loss = e_loss.squeeze()

                losses = {
                    "loss/p_loss": p_loss.detach(),
                    "loss/e_loss": e_loss.detach(),
                }

                # gradient descent on expert data; gradient ascent on policy data
                reward_loss = e_loss - (self.hps.advers_p_ascent_scale * p_loss)

                if self.hps.grad_pen:

                    # add gradient penalty to loss
                    grad_pen = self.grad_pen(p_input_a, p_input_b, e_input_a, e_input_b)
                    reward_loss += (self.hps.grad_pen_scale * grad_pen)
                    losses.update(
                        {
                            "loss/grad_pen": grad_pen.detach(),
                        },
                    )

            case "dac":
                # compute scores
                p_scores = self.discriminator(p_input_a, p_input_b)
                e_scores = self.discriminator(e_input_a, e_input_b)

                # entropy loss
                scores, _ = pack([p_scores, e_scores], "* d")  # concat along the batch dim, d is 1
                entropy = ff.binary_cross_entropy_with_logits(
                    input=scores,
                    target=torch.sigmoid(scores),
                )
                entropy_loss = -self.hps.ent_reg_scale * entropy

                # create labels
                fake_labels = 0. * torch.ones_like(p_scores)
                real_labels = 1. * torch.ones_like(e_scores)

                # binary classification
                p_loss = ff.binary_cross_entropy_with_logits(
                    input=p_scores,
                    target=fake_labels,
                )
                e_loss = ff.binary_cross_entropy_with_logits(
                    input=e_scores,
                    target=real_labels,
                )
                p_e_loss = p_loss + e_loss

                # sum losses
                reward_loss = p_e_loss + entropy_loss

                losses = {
                    "loss/entropy_loss": entropy_loss.detach(),
                    "loss/p_loss": p_loss.detach(),
                    "loss/e_loss": e_loss.detach(),
                    "loss/p_e_loss": p_e_loss.detach(),
                }

                if self.hps.grad_pen:

                    # add gradient penalty to loss
                    grad_pen = self.grad_pen(p_input_a, p_input_b, e_input_a, e_input_b)
                    reward_loss += (self.hps.grad_pen_scale * grad_pen)
                    losses.update(
                        {
                            "loss/grad_pen": grad_pen.detach(),
                        },
                    )

            case "diffail":
                p_inputs = self.assemble_diffail_input(p_input_a, p_input_b)
                e_inputs = self.assemble_diffail_input(e_input_a, e_input_b)
                p_scores = self.discriminator.loss(p_inputs, disc_ddpm=True)
                e_scores = self.discriminator.loss(e_inputs, disc_ddpm=True)
                p_scores = rearrange(p_scores, "b -> b 1")
                e_scores = rearrange(e_scores, "b -> b 1")

                # create labels
                fake_labels = 0. * torch.ones_like(p_scores)
                real_labels = 1. * torch.ones_like(e_scores)

                # binary classification
                p_loss = ff.binary_cross_entropy(
                    input=p_scores,
                    target=fake_labels,
                )
                e_loss = ff.binary_cross_entropy(
                    input=e_scores,
                    target=real_labels,
                )
                p_e_loss = p_loss + e_loss

                losses = {
                    "loss/p_e_loss": p_e_loss.detach(),
                }

                reward_loss = p_e_loss
                if self.hps.grad_pen:
                    # add gradient penalty to loss
                    grad_pen = self.grad_pen(p_input_a, p_input_b, e_input_a, e_input_b)
                    reward_loss += (self.hps.grad_pen_scale * grad_pen)
                    losses.update(
                        {
                            "loss/grad_pen": grad_pen.detach(),
                        },
                    )

            case "wdac":
                # compute scores
                p_scores = self.discriminator(p_input_a, p_input_b)
                e_scores = self.discriminator(e_input_a, e_input_b)

                # compute the dual EMD distance (== Wasserstein-1)
                dual_emd = e_scores.mean() - p_scores.mean()

                # compute the reward loss as "minus" the W1 loss
                reward_loss = -dual_emd

                # compute gradient penalty
                grad_pen = self.grad_pen(p_input_a, p_input_b, e_input_a, e_input_b)

                reward_loss += (self.hps.grad_pen_scale * grad_pen)

                losses = {
                    "loss/dual_emd": dual_emd.detach(),
                    "loss/grad_pen": grad_pen.detach(),
                }

            case "pwil":
                # the reward is not learned in this method
                return TensorDict({})

            case "iqlearn":
                # IQ-Learn does not fit this reward-updater API
                return TensorDict({})

            case "p2il":
                # P2IL does not fit this reward-updater API
                return TensorDict({})

            case _:
                raise ValueError("invalid method")

        reward_loss.backward()
        self.reward_optimizer.step()

        return TensorDict(
            {
                **losses,
            },
        )

    @beartype
    def grad_pen(self,
                 p_input_a: torch.Tensor,
                 p_input_b: Optional[torch.Tensor],
                 e_input_a: torch.Tensor,
                 e_input_b: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute the gradient penalty"""

        match self.hps.method:
            case "diffail":
                p_input = self.assemble_diffail_input(
                    p_input_a[:self.expert_batch_size],
                    p_input_b[:self.expert_batch_size] if p_input_b is not None else None,
                )
                e_input = self.assemble_diffail_input(
                    e_input_a[:self.expert_batch_size],
                    e_input_b[:self.expert_batch_size] if e_input_b is not None else None,
                )
                eps = torch.rand(p_input.size(0), 1, device=self.device)
                i_input = (eps * p_input) + ((1. - eps) * e_input)
                i_input.requires_grad_(requires_grad=True)

                outputs = self.discriminator.loss(i_input, disc_ddpm=True)
                outputs = rearrange(outputs, "b -> b 1")

                grads = autograd.grad(
                    inputs=i_input,
                    outputs=outputs,
                    grad_outputs=self.grad_outputs[:i_input.size(0)],
                    retain_graph=True,
                    create_graph=True,
                )[0]
                grads_norm = grads.norm(2, dim=-1)

            case _:
                assert p_input_b is not None
                assert e_input_b is not None
                # concat the inputs along the last (2nd) dim
                p_input, ps = pack(
                    [p_input_a[:self.expert_batch_size], p_input_b[:self.expert_batch_size]],
                    "b *",
                )
                e_input, ps = pack(
                    [e_input_a[:self.expert_batch_size], e_input_b[:self.expert_batch_size]],
                    "b *",
                )
                # assemble interpolated inputs (point on segment)
                eps = torch.rand(p_input_a[:self.expert_batch_size].size(0), 1, device=self.device)
                i_input = eps * p_input + ((1. - eps) * e_input)

                # unpack
                u_i_input = unpack(i_input, ps, "b *")
                for e in u_i_input:
                    e.requires_grad_(requires_grad=True)

                match self.hps.method:
                    case "ngt":
                        outputs = self.compute_ngt_raw_score(
                            *u_i_input,
                            criterion_key="i",
                        )
                    case _:
                        outputs = self.discriminator(*u_i_input)

                grads = autograd.grad(
                    inputs=u_i_input,
                    outputs=outputs,
                    grad_outputs=self.grad_outputs,
                    retain_graph=True,
                    create_graph=True,
                )
                packed_grads, _ = pack(list(grads), "b *")
                grads_norm = packed_grads.norm(2, dim=-1)

        if self.hps.one_sided_pen:
            # penalize the gradient for having a norm GREATER than k
            grad_pen = torch.max(
                torch.zeros_like(grads_norm),
                grads_norm - self.hps.grad_pen_targ,
            )
        else:
            # penalize the gradient for having a norm LOWER OR GREATER than k
            grad_pen = grads_norm - self.hps.grad_pen_targ

        return grad_pen.pow(2).mean()

    @beartype
    def compute_ngt_raw_score(self,
                              input_a: torch.Tensor,
                              input_b: Optional[torch.Tensor],
                              *,
                              criterion_key: str) -> torch.Tensor:

        assert self.hps.method == "ngt", "invalid method"

        score = self.criteria[criterion_key](self.predictor(input_a, input_b),
                                             self.prior(input_a, input_b))
        return score.mean(dim=-1, keepdim=True)

    @beartype
    def check_prior_outputs(self, batch_or_demos: TensorDict, *, tag: str) -> TensorDict:

        assert self.hps.method == "ngt", "invalid method"

        input_a = batch_or_demos["observations"]
        match self.hps.input_mode:
            case "ss":
                input_b = batch_or_demos["next_observations"]
            case "sa":
                input_b = batch_or_demos["actions"]
            case _:
                raise ValueError("invalid input mode")

        outputs = self.prior(input_a, input_b)

        def compute_quantile(x: torch.Tensor, k: int) -> torch.Tensor:
            y = x.quantile(q=k / 100.0, dim=1)  # reduce along last
            return y.quantile(0.5)  # scalar

        return TensorDict(
            {
                f"prior_outputs_{tag}/p{str(k).zfill(2)}": compute_quantile(outputs, k)
                for k in [0, 1, 5, 50, 95, 99, 100]
            },
        )

    @beartype
    def compute_reward(self,
                       state: torch.Tensor,
                       action: torch.Tensor,
                       next_state: torch.Tensor,
        ) -> torch.Tensor:

        input_a = state
        match self.hps.input_mode:
            case "ss":
                input_b = next_state
            case "sa":
                input_b = action
            case _:
                raise ValueError("invalid input mode")

        match self.hps.method:

            case "ngt":
                inputs = (input_a, input_b)
                with torch.no_grad():
                    # with self.predictor_lagged.to_module(self.predictor):
                    #     p_loss = self.criteria["i"](self.predictor(*inputs), self.prior(*inputs))
                    u = self.compute_ngt_raw_score(*inputs, criterion_key="i")

                ema_decay_rate = 0.001
                reward_bound = 5.0
                q_lo, q_hi = 0.05, 0.95
                if self.hps.tighter_percentile_range:
                    q_lo, q_hi = 0.10, 0.90
                batch_lo = u.quantile(q=q_lo)
                batch_hi = u.quantile(q=q_hi)
                batch_sh = u.quantile(q=0.50)

                # lazily create EMA state on first call
                if not hasattr(self, "u_lo_ema"):
                    self.u_lo_ema = batch_lo.detach().clone()
                    self.u_hi_ema = batch_hi.detach().clone()
                    self.u_sh_ema = batch_sh.detach().clone()

                # normalize with EMA stats for a steadier reward scale
                scale = (self.u_hi_ema - self.u_lo_ema).clamp_min(SHAPING_FLOOR)
                u = (u - self.u_sh_ema) / scale

                with torch.no_grad():
                    self.u_lo_ema.lerp_(batch_lo.detach(), ema_decay_rate)
                    self.u_hi_ema.lerp_(batch_hi.detach(), ema_decay_rate)
                    self.u_sh_ema.lerp_(batch_sh.detach(), ema_decay_rate)

                match self.hps.reward_shaping:
                    case "linear":
                        u = -u
                    case "symexp":
                        u = -torch.sign(u) * (torch.exp(torch.abs(u)) - 1.0)
                    case _:
                        raise ValueError("invalid NGT reward shaping")
                u = torch.clamp(u, -reward_bound, reward_bound)

                return u

            case "dac":
                with torch.no_grad():
                    score = self.discriminator(input_a, input_b)
                pos_reward = -torch.log(1.0 - torch.sigmoid(score) + SHAPING_FLOOR)
                if self.hps.real_valued_reward:
                    neg_reward = torch.log(torch.sigmoid(score) + SHAPING_FLOOR)
                    return pos_reward + neg_reward
                else:
                    return pos_reward

            case "diffail":
                x = self.assemble_diffail_input(input_a, input_b)
                with torch.no_grad():
                    logits = self.discriminator.calc_reward(x)
                pos_reward = -torch.log(1.0 - logits + SHAPING_FLOOR)
                if self.hps.real_valued_reward:
                    neg_reward = torch.log(logits + SHAPING_FLOOR)
                    return pos_reward + neg_reward
                else:
                    return pos_reward

            case "wdac":

                # compute score
                with torch.no_grad():
                    reward = self.discriminator(input_a, input_b)

                return reward

            case "pwil":
                raise ValueError("should not be here!")

            case "iqlearn":
                raise ValueError("should not be here!")

            case "p2il":
                raise ValueError("should not be here!")

            case _:
                raise ValueError("invalid method")

    @beartype
    def behavioral_cloning(self, batch: TensorDict) -> TensorDict:
        """Train actor with behavioral cloning"""

        self.actor_optimizer.zero_grad()

        inputs = [batch["observations"]]
        funcn = "get_action"
        if self.metal:
            inputs.append(batch["noise"])
            funcn += "_metal"
        # using stochastic actor
        action_from_actor = getattr(self.actor, funcn)(*inputs)["sample"]

        actor_loss = ff.mse_loss(
            action_from_actor, batch["actions"])

        actor_loss.backward()
        self.actor_optimizer.step()

        return TensorDict(
            {
                "loss/bc_loss": actor_loss.detach(),
            },
        )

    @beartype
    def save(self, path: Path, sfx: Optional[str] = None):
        """Save the agent to disk and wandb servers"""
        # prep checkpoint
        fname = (f"ckpt_{sfx}"
                 if sfx is not None
                 else f".ckpt_{self.timesteps_so_far}ts")
        # design choice: hide the ckpt saved without an extra qualifier
        path = (parent := path) / f"{fname}.pth"
        checkpoint = {
            "hps": self.hps,  # handy for archeology
            "timesteps_so_far": self.timesteps_so_far,
            # and now the state_dict objects
            "actor": self.actor.state_dict(),
            "qnet1": self.qnet1.state_dict(),
            "qnet2": self.qnet2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
            # keep keys for old checkpoint schema stability
            "log_eta": None,
            "log_betas": None,
        }
        if self.hps.autotune:
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        if self.obs_rms is not None:
            checkpoint["obs_rms"] = self.obs_rms.state()
        # save checkpoint to filesystem
        torch.save(checkpoint, path)
        logger.info(f"{sfx} model saved to disk")
        if sfx == "best":
            # upload the model to wandb servers
            wandb.save(str(path), base_path=parent)
            logger.warn("model saved to wandb")

    @beartype
    def load_from_disk(self, path: Path):
        """Load another agent into this one"""
        checkpoint = torch.load(path, weights_only=False)
        if "timesteps_so_far" in checkpoint:
            self.timesteps_so_far = checkpoint["timesteps_so_far"]
        # the "strict" argument of `load_state_dict` is True by default
        self.actor.load_state_dict(checkpoint["actor"])
        self.qnet1.load_state_dict(checkpoint["qnet1"])
        if self.hps.method != "p2il":
            self.qnet2.load_state_dict(checkpoint["qnet2"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        if "log_alpha" in checkpoint:
            self.log_alpha = checkpoint["log_alpha"]
        if "log_eta" in checkpoint and checkpoint["log_eta"] is not None:
            self.log_eta = checkpoint["log_eta"]
        if "log_betas" in checkpoint and checkpoint["log_betas"] is not None:
            self.log_betas = checkpoint["log_betas"]
        if self.hps.autotune and "alpha_optimizer" in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        if self.obs_rms is not None and "obs_rms" in checkpoint:
            self.obs_rms.load_state(checkpoint["obs_rms"])
            self.obs_rms.cfg.device = self.device

    @staticmethod
    @beartype
    def compare_dictconfigs(
        dictconfig1: MutableMapping[Any, Any],
        dictconfig2: MutableMapping[Any, Any],
    ) -> dict[str, dict[str, Union[str, int, list[int], dict[str, Union[str, int, list[int]]]]]]:
        """Compare two DictConfig objects of depth=1 and return the differences.
        Returns a dictionary with keys "added", "removed", and "changed".
        """
        assert isinstance(dictconfig1, DictConfig)
        assert isinstance(dictconfig2, DictConfig)

        differences = {"added": {}, "removed": {}, "changed": {}}

        keys1 = set(dictconfig1.keys())
        keys2 = set(dictconfig2.keys())

        # added keys
        for key in keys2 - keys1:
            differences["added"][key] = dictconfig2[key]

        # removed keys
        for key in keys1 - keys2:
            differences["removed"][key] = dictconfig1[key]

        # changed keys
        for key in keys1 & keys2:
            if dictconfig1[key] != dictconfig2[key]:
                differences["changed"][key] = {
                    "from": dictconfig1[key], "to": dictconfig2[key]}

        return differences

    @beartype
    def load(self, wandb_run_path: str, model_name: str = "ckpt_best.pth"):
        """Download a model from wandb and load it"""
        api = wandb.Api()
        run = api.run(wandb_run_path)
        # compare the current cfg with the cfg of the loaded model
        wandb_cfg_dict: dict[str, Any] = run.config
        wandb_cfg: DictConfig = OmegaConf.create(wandb_cfg_dict)
        a, r, c = self.compare_dictconfigs(wandb_cfg, self.hps).values()
        # N.B.: in Python 3.7 and later, dicts preserve the insertion order
        logger.warn(f"added  : {a}")
        logger.warn(f"removed: {r}")
        logger.warn(f"changed: {c}")
        # create a temporary directory to download to
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            file = run.file(model_name)
            # download the model file from wandb servers
            file.download(root=tmp_dir_name, replace=True)
            logger.warn("model downloaded from wandb to disk")
            tmp_file_path = Path(tmp_dir_name) / model_name
            # load the agent stored in this file
            self.load_from_disk(tmp_file_path)
            logger.warn("model loaded")
