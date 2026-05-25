import os
os.environ["MUJOCO_GL"] = "egl"
from typing import Union, Optional, Callable
from pathlib import Path

from beartype import beartype
import numpy as np

import gymnasium as gym

from gymnasium.core import Env
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnasium.vector.async_vector_env import AsyncVectorEnv
from gymnasium.wrappers import (
    TimeLimit,
    RecordVideo,
    ClipAction,
)


BENCHMARKS = {
    "gym": [
        *[f"{name}-v4" for name in
            [
                # "InvertedDoublePendulum",
                # "InvertedPendulum",
                "Pusher",
                # "Reacher",
                # "Swimmer",
                "Hopper",
                "HalfCheetah",
                "Walker2d",
                "Ant",
                "Humanoid",
                "HumanoidStandup",
            ]
        ],
    ],
}


@beartype
def get_benchmark(env_id: str) -> str:
    # verify that the specified env is amongst the admissible ones
    benchmark = None
    for k, v in BENCHMARKS.items():
        if env_id in v:
            benchmark = k
            continue
    assert benchmark is not None, "unsupported environment"
    return benchmark


@beartype
def make_env(env_id: str,
             seed: int,
             *,
             sync_vec_env: bool,
             num_envs: int,
             video_path: Optional[Path] = None,
             horizon: Optional[int] = None,
    ) -> (tuple[Union[Env, SyncVectorEnv, AsyncVectorEnv],
          dict[str, tuple[int, ...]],
          np.ndarray,
          np.ndarray]):

    get_benchmark(env_id)

    @beartype
    def make_env() -> Callable[[], Env]:
        @beartype
        def thunk() -> Env:
            if video_path is not None:
                assert sync_vec_env and (num_envs == 1)
                env = gym.make(env_id, render_mode="rgb_array")
                env = RecordVideo(env, str(video_path))
            else:
                try:
                    env = gym.make(env_id, terminate_when_unhealthy=False)
                except TypeError:
                    env = gym.make(env_id)
                env = ClipAction(env)
                if horizon is not None:
                    env = TimeLimit(env, max_episode_steps=horizon)
            return env
        return thunk

    # create env
    vec_env = (SyncVectorEnv if sync_vec_env else AsyncVectorEnv)(
        [
            make_env() for _ in range(num_envs)
        ],
    )
    vec_env.action_space.seed(seed)  # to be fully reproducible

    ob_space = vec_env.observation_space
    ac_space = vec_env.action_space

    # due diligence checks
    assert isinstance(ob_space, gym.spaces.Box)
    if isinstance(ac_space, gym.spaces.Discrete):
        raise TypeError("actions must be continuous")
    assert isinstance(ac_space, gym.spaces.Box)
    net_shapes = {"ob_shape": ob_space.shape, "ac_shape": ac_space.shape}

    # assert that all envs have the same action bounds
    assert np.all(ac_space.low == ac_space.low[0])
    assert np.all(ac_space.high == ac_space.high[0])

    return vec_env, net_shapes, ac_space.low[0], ac_space.high[0]
