import os
import sys
import time
from pathlib import Path
from itertools import cycle
from tqdm import tqdm
from typing import Union
from collections import deque, defaultdict
from collections.abc import Callable, Generator, Iterator

from beartype import beartype
from omegaconf import OmegaConf, DictConfig
from einops import rearrange
import wandb
from wandb.errors import CommError
import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule

from gymnasium.core import Env
from gymnasium.vector.vector_env import VectorEnv

from helpers import logger
from agents.agent import Agent

from tensordict.nn.functional_modules import _exclude_td_from_pytree  # noqa
_exclude_td_from_pytree().set()  # required; or set env: EXCLUDE_TD_FROM_PYTREE=1


@beartype
def segment(env: Union[Env, VectorEnv],
            agent: Agent,
            seed: int,
            segment_len: int,
            learning_starts: int,
            /,
    ) -> Generator[None, None, None]:
    assert agent.replay_dataset is not None

    obs, _ = env.reset(seed=seed)  # for the very first reset, we give a seed (and never again)
    obs = torch.as_tensor(obs, device=agent.device, dtype=torch.float)
    if agent.hps.method == "pwil":
        assert hasattr(agent, "pwil_rewarder")
        for idx in range(env.num_envs):
            agent.pwil_rewarder[idx].reset()

    t = 0

    segment_contents = defaultdict(list)

    while True:

        # predict action
        if (agent.hps.method == "random") or (agent.timesteps_so_far < learning_starts):
            actions = env.action_space.sample()
        else:
            if agent.normalize_obs_for_actor_critic and (rms := agent.obs_rms) is not None:
                inputs = {"observations": rms.normalize(obs)}
            else:
                inputs = {"observations": obs}
            if agent.metal:
                noise = torch.randn(
                    obs.size(0),
                    agent.ac_dim,
                    device=agent.device,
                    dtype=obs.dtype,
                )
                inputs.update({"noise": noise})
            actions = agent.predict(
                TensorDict(inputs, device=agent.device),
                explore=True,
            )

        if t > 0 and t % segment_len == 0:

            stacks = {}
            for k, v in segment_contents.items():
                stacks[k] = torch.stack(v)  # dim=0 by default

            if agent.obs_rms is not None:
                agent.obs_rms.update(
                    rearrange(stacks["observations"], "t n ... -> (t n) ..."),
                )

            # add transitions to replay buffer
            for k, v in stacks.items():
                stacks[k] = rearrange(v, "t n ... -> (t n) ...")
            # adding each of the TxN transitions
            agent.replay_dataset.extend(
                TensorDict(
                    stacks,
                    batch_size=stacks[next(iter(stacks))].size(0),
                    device=agent.device,
                ),
            )

            # clear the segment_contents
            segment_contents.clear()

            yield

        # interact with env (while avoiding reward leakage)
        next_obs, _, terminations, truncations, infos = env.step(actions)

        dones = np.logical_or(np.array(terminations), np.array(truncations))

        next_obs = torch.as_tensor(next_obs, device=agent.device, dtype=torch.float)
        real_next_obs = next_obs.clone()

        for idx, trunc in enumerate(np.array(truncations)):
            if trunc:
                real_next_obs[idx] = torch.as_tensor(
                    infos["final_observation"][idx], device=agent.device, dtype=torch.float)

        terminations = rearrange(
            torch.as_tensor(terminations, device=agent.device, dtype=torch.bool),
            "b -> b 1",
        )

        transition = {
            "observations": obs,
            "next_observations": real_next_obs,
            "actions": torch.as_tensor(actions, device=agent.device, dtype=torch.float),
            "terminations": terminations,
            "dones": terminations,
        }

        if agent.hps.method == "pwil":
            pwil_rewards = []
            actions_ = transition["actions"]
            for idx, done in enumerate(dones):
                pwil_reward = agent.pwil_rewarder[idx].compute_reward(
                    obs[idx, ...], actions_[idx, ...], real_next_obs[idx, ...])
                pwil_rewards.append(pwil_reward)
                if done:
                    agent.pwil_rewarder[idx].reset()

            pwil_rewards = torch.stack(pwil_rewards).unsqueeze(-1)
            transition.update({"pwil_rewards": pwil_rewards})

        for k, v in transition.items():
            segment_contents[k].append(v)

        obs = next_obs

        t += 1


@beartype
def episode(env: Env,
            agent: Agent,
            /,
            *,
            seeds: Iterator[int],
    ) -> Generator[dict[str, np.ndarray], None, None]:
    # generator that spits out a trajectory collected during a single episode

    # `append` operation is significantly faster on lists than numpy arrays,
    # they will be converted to numpy arrays once complete right before the yield

    ob, _ = env.reset(seed=next(seeds))
    ob = torch.as_tensor(ob, device=agent.device, dtype=torch.float)
    ep_len = 0.0
    ep_ret = 0.0

    while True:

        if agent.hps.method == "random":
            action = env.action_space.sample()
        else:
            # predict action
            if agent.normalize_obs_for_actor_critic and (rms := agent.obs_rms) is not None:
                inputs = {"observations": rms.normalize(ob)}
            else:
                inputs = {"observations": ob}
            if agent.metal:
                inputs.update({
                    "noise": torch.randn(
                        ob.size(0),
                        agent.ac_dim,
                        device=agent.device,
                        dtype=ob.dtype,
                    ),
                })
            action = agent.predict(
                TensorDict(inputs, device=agent.device),
                explore=False,
            )

        new_ob, reward, termination, truncation, _ = env.step(action)
        ep_len += 1
        ep_ret += reward

        done = termination or truncation

        new_ob = torch.as_tensor(new_ob, device=agent.device, dtype=torch.float)
        ob = new_ob

        if done:

            yield {
                "length": np.array(ep_len),
                "return": np.array(ep_ret),
            }

            ob, _ = env.reset(seed=next(seeds))
            ob = torch.as_tensor(ob, device=agent.device, dtype=torch.float)
            ep_len = 0.0
            ep_ret = 0.0


@beartype
def train(cfg: DictConfig,
          env: Union[Env, VectorEnv],
          eenv: Env,
          agent_wrapper: Callable[[], Agent],
          name: str,
          progress_files: dict[str, Path]):

    assert isinstance(cfg, DictConfig)
    if "pretrain" in cfg:
        raise ValueError("config key `pretrain` has been removed")
    if "load_ckpt" in cfg:
        raise ValueError(
            "train-side warmstart is disabled; --load_ckpt is only supported in evaluate",
        )

    agent = agent_wrapper()

    assert agent.replay_dataset is not None

    # set up model save directory
    ckpt_dir = Path(cfg.checkpoint_dir) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # set up wandb
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    group = ".".join(name.split(".")[:-1])  # everything in name except seed
    logger.warn(f"{name=}")
    logger.warn(f"{group=}")
    while True:
        try:
            config = OmegaConf.to_object(cfg)
            assert isinstance(config, dict)
            wandb.init(
                project=cfg.wandb_project,
                name=name,
                id=name,
                group=group,
                config=config,
                dir=cfg.root,
            )
            break
        except CommError:
            pause = 10
            logger.info(f"wandb co error. Retrying in {pause} secs.")
            time.sleep(pause)
    logger.info("wandb co established!")

    run_start_time = time.time()
    start_time = None
    measure_burnin = None
    pbar = tqdm(range(cfg.num_timesteps), disable=not sys.stderr.isatty())
    time_spent_eval = 0
    prev_eval_time = run_start_time
    prev_eval_timestep = agent.timesteps_so_far

    tlog = TensorDict({})
    ret_buff = deque(maxlen=cfg.num_eval_passes_to_average)

    update_actor = agent.update_actor
    # only wrap actor update; keep critic/reward updates out of cudagraph capture
    if cfg.cudagraphs:
        # in and out keys are `[]` means it expect 1 TensorDict in and it returns 1 TensorDict out
        update_actor = CudaGraphModule(update_actor, in_keys=[], out_keys=[])

    # create generators for training and evaluating [after compilation setup]
    seg_gen = segment(env, agent, cfg.seed, cfg.segment_len, cfg.learning_starts)
    ep_gen = episode(eenv, agent, seeds=cycle([10_000 + s for s in range(cfg.eval_steps)]))

    while agent.timesteps_so_far <= cfg.num_timesteps:

        if ((agent.timesteps_so_far >= (cfg.measure_burnin + cfg.learning_starts)) and
            (start_time is None)):
            start_time = time.time()
            measure_burnin = agent.timesteps_so_far

        logger.info(("interact").upper())
        next(seg_gen)
        agent.timesteps_so_far += (increment := cfg.segment_len * cfg.num_envs)
        pbar.update(increment)

        if agent.timesteps_so_far <= cfg.learning_starts:
            # start training when enough data
            pbar.set_description("not learning yet")
            continue

        logger.info(("train").upper())

        if cfg.method != "random":
            for _ in range(cfg.num_updates_per_iter):

                # sample from replay and demos
                with torch.no_grad():
                    batch = agent.replay_dataset.sample(cfg.batch_size)
                    demos = agent.expert_dataset.sample(agent.expert_batch_size)

                if cfg.method in {"iqlearn", "p2il"}:
                    if cfg.method == "iqlearn":
                        if agent.normalize_obs_for_actor_critic and (rms := agent.obs_rms) is not None:
                            batch["observations"] = rms.normalize(batch["observations"])
                            batch["next_observations"] = rms.normalize(batch["next_observations"])
                            demos["observations"] = rms.normalize(demos["observations"])
                            demos["next_observations"] = rms.normalize(demos["next_observations"])
                        tlog.update(agent.update_qnets_iq(batch, demos))
                        actor_obs = torch.cat(
                            [batch["observations"], demos["observations"]],
                            dim=0,
                        )
                    else:
                        tlog.update(agent.update_qnets_p2il(batch, demos))
                        actor_obs = torch.cat(
                            [batch["observations"], demos["observations"]],
                            dim=0,
                        )
                        if agent.normalize_obs_for_actor_critic and (rms := agent.obs_rms) is not None:
                            actor_obs = rms.normalize(actor_obs)
                    agent.qnets_updates_so_far += 1

                    actor_batch = TensorDict(
                        {
                            "observations": actor_obs,
                        },
                        device=agent.device,
                    )
                    if cfg.cudagraphs:
                        dtype = (obs := actor_batch["observations"]).dtype
                        noise_shape = [obs.size(0), agent.ac_dim]
                        actor_batch["noise"] = torch.empty(
                            *noise_shape, device=agent.device, dtype=dtype,
                        ).normal_()
                    tlog.update(update_actor(actor_batch))
                    agent.actor_updates_so_far += 1
                    continue

                if cfg.method == "ngt" and cfg.monitor_prior_emb:
                    tlog.update(agent.check_prior_outputs(batch, tag="batch"))
                    tlog.update(agent.check_prior_outputs(demos, tag="demos"))

                if agent.normalize_obs_for_reward and (rms := agent.obs_rms) is not None:
                    p_obs = batch["observations"]
                    p_next_obs = batch["next_observations"]
                    e_obs = demos["observations"]
                    e_next_obs = demos["next_observations"]
                    batch["observations"] = rms.normalize(p_obs)
                    batch["next_observations"] = rms.normalize(p_next_obs)
                    demos["observations"] = rms.normalize(e_obs)
                    demos["next_observations"] = rms.normalize(e_next_obs)

                # update reward
                tlog.update(agent.update_reward(batch, demos))
                agent.reward_updates_so_far += 1
                if agent.normalize_obs_for_reward and (rms := agent.obs_rms) is not None:
                    batch["observations"] = p_obs
                    batch["next_observations"] = p_next_obs
                    demos["observations"] = e_obs
                    demos["next_observations"] = e_next_obs
                # with torch.no_grad():
                #     agent.predictor_lagged.lerp_(agent.predictor_params.data, cfg.polyak)

                # sample new batch of transitions for actor and critic (extra decorrelation trick)
                with torch.no_grad():
                    batch = agent.replay_dataset.sample(cfg.batch_size)

                if cfg.method == "pwil":
                    # override rewards with the PWIL ones
                    batch["rewards"] = batch["pwil_rewards"]
                else:  # noqa
                    # populate batch with rewards
                    if agent.normalize_obs_for_reward and (rms := agent.obs_rms) is not None:
                        obs = batch["observations"]
                        next_obs = batch["next_observations"]
                        batch["observations"] = rms.normalize(obs)
                        batch["next_observations"] = rms.normalize(next_obs)
                        batch = agent.reward(batch)
                        batch["observations"] = obs
                        batch["next_observations"] = next_obs
                    else:
                        batch = agent.reward(batch)

                if agent.normalize_obs_for_actor_critic and (rms := agent.obs_rms) is not None:
                    for k in batch.keys():  # noqa
                        if "observations" in k:
                            batch[k] = rms.normalize(batch[k])

                # update qnets
                tlog.update(agent.update_qnets(batch))
                agent.qnets_updates_so_far += 1
                with torch.no_grad():
                    agent.qnets_target.lerp_(agent.qnets_params, cfg.polyak)
                # update actor (and alpha)
                if cfg.cudagraphs:
                    dtype = (obs := batch["observations"]).dtype
                    noise_shape = [obs.size(0), agent.ac_dim]
                    batch["noise"] = torch.empty(
                        *noise_shape, device=agent.device, dtype=dtype,
                    ).normal_()
                tlog.update(update_actor(batch))
                agent.actor_updates_so_far += 1

        if agent.timesteps_so_far % (cfg.eval_every * cfg.segment_len * cfg.num_envs) == 0:
            logger.info(("eval").upper())
            eval_start = time.time()

            len_list = []
            ret_list = []
            for _ in range(cfg.eval_steps):
                ep = next(ep_gen)
                len_list.append(torch.as_tensor(ep["length"], dtype=torch.float))  # cpu
                ret_list.append(torch.as_tensor(ep["return"], dtype=torch.float))  # cpu

            with torch.no_grad():

                @beartype
                def _wrapper(tensor_list: list[torch.Tensor]) -> torch.Tensor:
                    return torch.stack(tensor_list).mean()

                ret = _wrapper(ret_list)
                ret_buff.append(ret)

                eval_metrics = {
                    "length": _wrapper(len_list),
                    "return": ret,
                    "return_smooth": torch.stack(list(ret_buff)).mean(),
                }

            # log with logger in progress file
            logger.record_tabular("timestep", agent.timesteps_so_far)
            for k, v in eval_metrics.items():
                logger.record_tabular(k, v.numpy())

            # wall time from train loop start (for return-vs-time plots)
            eval_snapshot_time = time.time()
            wall_time_total_s = eval_snapshot_time - run_start_time
            logger.record_tabular("wall_time_total_s", wall_time_total_s)

            # wall time spent outside eval (for throughput plots)
            current_eval_elapsed = eval_snapshot_time - eval_start
            wall_time_train_s = wall_time_total_s - (time_spent_eval + current_eval_elapsed)
            logger.record_tabular("wall_time_train_s", wall_time_train_s)

            # interval speed since previous eval snapshot
            interval_time = eval_snapshot_time - prev_eval_time
            interval_steps = agent.timesteps_so_far - prev_eval_timestep
            sps_interval = interval_steps / max(interval_time, 1e-8)
            logger.record_tabular("sps_interval", sps_interval)
            prev_eval_time = eval_snapshot_time
            prev_eval_timestep = agent.timesteps_so_far

            logger.dump_tabular()

            # log with wandb
            if (new_best := eval_metrics["return"].item()) > agent.best_eval_ep_ret:
                logger.info("new best eval! -- saving model to disk and wandb")
                agent.best_eval_ep_ret = new_best
                agent.save(ckpt_dir, sfx="best")
            for v in progress_files.values():
                wandb.save(v, base_path=str(v.parent))
            wandb.log(
                {
                    **tlog.to_dict(),
                    **{f"eval/{k}": v for k, v in eval_metrics.items()},
                    "vitals/replay_buffer_numel": len(agent.replay_dataset),
                },
                step=agent.timesteps_so_far,
            )

            time_spent_eval += time.time() - eval_start

            if start_time is not None:
                # compute the speed in steps per second
                speed = (
                    (agent.timesteps_so_far - measure_burnin) /
                    (time.time() - start_time - time_spent_eval)
                )
                desc = f"speed={speed: 4.4f} sps"
                pbar.set_description(desc)
                wandb.log(
                    {
                        "vitals/speed": speed,
                    },
                    step=agent.timesteps_so_far,
                )

        tlog.clear()

    # mark a run as finished, and finish uploading all data (from docs)
    wandb.finish()
    logger.warn("bye")


@beartype
def evaluate(cfg: DictConfig,
             eenv: Env,
             agent_wrapper: Callable[[], Agent]):

    assert isinstance(cfg, DictConfig)

    agent = agent_wrapper()

    agent.load(cfg.load_ckpt)

    # create episode generator
    ep_gen = episode(eenv, agent, seeds=cycle([10_000 + s for s in range(cfg.eval_steps)]))

    pbar = tqdm(range(cfg.num_episodes), disable=not sys.stderr.isatty())
    pbar.set_description("evaluating")

    len_list = []
    ret_list = []

    for _ in pbar:

        ep = next(ep_gen)
        len_list.append(torch.as_tensor(ep["length"], dtype=torch.float))  # cpu
        ret_list.append(torch.as_tensor(ep["return"], dtype=torch.float))  # cpu

    with torch.no_grad():

        @beartype
        def _wrapper(tensor_list: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(tensor_list).mean()

        eval_metrics = {
            "length": _wrapper(len_list),
            "return": _wrapper(ret_list),
        }

    # log with logger
    for k, v in eval_metrics.items():
        logger.record_tabular(k, v.numpy())
    logger.dump_tabular()


@beartype
def clone(cfg: DictConfig,
          eenv: Env,
          agent_wrapper: Callable[[], Agent],
          name: str,
          progress_files: dict[str, Path]):

    assert isinstance(cfg, DictConfig)
    if "pretrain" in cfg:
        raise ValueError("config key `pretrain` has been removed")
    if "load_ckpt" in cfg:
        raise ValueError(
            "train-side warmstart is disabled; --load_ckpt is only supported in evaluate",
        )

    agent = agent_wrapper()

    assert agent.replay_dataset is not None

    # set up model save directory
    ckpt_dir = Path(cfg.checkpoint_dir) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # set up wandb
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    group = ".".join(name.split(".")[:-1])  # everything in name except seed
    logger.warn(f"{name=}")
    logger.warn(f"{group=}")
    while True:
        try:
            config = OmegaConf.to_object(cfg)
            assert isinstance(config, dict)
            wandb.init(
                project=cfg.wandb_project,
                name=name,
                id=name,
                group=group,
                config=config,
                dir=cfg.root,
            )
            break
        except CommError:
            pause = 10
            logger.info(f"wandb co error. Retrying in {pause} secs.")
            time.sleep(pause)
    logger.info("wandb co established!")

    # create episode generator for evaluating the agent
    ep_gen = episode(eenv, agent, seeds=cycle([10_000 + s for s in range(cfg.eval_steps)]))

    run_start_time = time.time()
    start_time = None
    measure_burnin = None
    pbar = tqdm(range(cfg.num_bc_iters), disable=not sys.stderr.isatty())
    time_spent_eval = 0
    prev_eval_time = run_start_time
    prev_eval_iter = 0

    tlog = TensorDict({})
    ret_buff = deque(maxlen=cfg.num_eval_passes_to_average)

    update_actor = agent.behavioral_cloning
    if cfg.cudagraphs:
        update_actor = CudaGraphModule(update_actor, in_keys=[], out_keys=[])
        # in and out keys are `[]` means it expect 1 TensorDict in and it returns 1 TensorDict out

    for i in pbar:

        if (agent.actor_updates_so_far >= cfg.measure_burnin) and (start_time is None):
            start_time = time.time()
            measure_burnin = i

        # sample batch of expert data
        demos = agent.expert_dataset.sample(cfg.batch_size)
        if agent.normalize_obs_for_actor_critic and (rms := agent.obs_rms) is not None:
            rms.update(demos["observations"])
            demos["observations"] = rms.normalize(demos["observations"])
        if cfg.cudagraphs:
            dtype = (obs := demos["observations"]).dtype
            noise_shape = [obs.size(0)]
            noise_shape.append(agent.ac_dim)
            demos["noise"] = torch.empty(
                *noise_shape, device=agent.device, dtype=dtype,
            ).normal_()
        # update actor
        tlog.update(update_actor(demos))

        if i % cfg.eval_every == 0:
            logger.info(("eval").upper())
            eval_start = time.time()

            len_list = []
            ret_list = []
            for _ in range(cfg.eval_steps):
                ep = next(ep_gen)
                len_list.append(torch.as_tensor(ep["length"], dtype=torch.float))  # cpu
                ret_list.append(torch.as_tensor(ep["return"], dtype=torch.float))  # cpu

            with torch.no_grad():

                @beartype
                def _wrapper(tensor_list: list[torch.Tensor]) -> torch.Tensor:
                    return torch.stack(tensor_list).mean()

                ret = _wrapper(ret_list)
                ret_buff.append(ret)

                eval_metrics = {
                    "length": _wrapper(len_list),
                    "return": ret,
                    "return_smooth": torch.stack(list(ret_buff)).mean(),
                }

            # log with logger
            logger.record_tabular("iteration", i)
            for k, v in eval_metrics.items():
                logger.record_tabular(k, v.numpy())

            # wall time from train loop start (for return-vs-time plots)
            eval_snapshot_time = time.time()
            wall_time_total_s = eval_snapshot_time - run_start_time
            logger.record_tabular("wall_time_total_s", wall_time_total_s)

            # wall time spent outside eval (for throughput plots)
            current_eval_elapsed = eval_snapshot_time - eval_start
            wall_time_train_s = wall_time_total_s - (time_spent_eval + current_eval_elapsed)
            logger.record_tabular("wall_time_train_s", wall_time_train_s)

            # interval speed since previous eval snapshot
            interval_time = eval_snapshot_time - prev_eval_time
            interval_steps = i - prev_eval_iter
            sps_interval = interval_steps / max(interval_time, 1e-8)
            logger.record_tabular("sps_interval", sps_interval)
            prev_eval_time = eval_snapshot_time
            prev_eval_iter = i

            logger.dump_tabular()

            # log with wandb
            if (new_best := eval_metrics["return"].item()) > agent.best_eval_ep_ret:
                # save the new best model
                logger.info("new best eval! -- saving model to disk and wandb")
                agent.best_eval_ep_ret = new_best
                agent.save(ckpt_dir, sfx="best")
            for v in progress_files.values():
                wandb.save(v, base_path=str(v.parent))
            wandb.log(
                {
                    **tlog.to_dict(),
                    **{f"eval/{k}": v for k, v in eval_metrics.items()},
                    "vitals/replay_buffer_numel": len(agent.replay_dataset),
                },
                step=i,
            )

            time_spent_eval += time.time() - eval_start

            if start_time is not None:
                # compute the speed in steps per second
                speed = (
                    (i - measure_burnin) /
                    (time.time() - start_time - time_spent_eval)
                )
                desc = f"speed={speed: 4.4f} sps"
                pbar.set_description(desc)
                wandb.log(
                    {
                        "vitals/speed": speed,
                    },
                    step=i,
                )

        tlog.clear()

    # mark a run as finished, and finish uploading all data (from docs)
    wandb.finish()
    logger.warn("bye")
