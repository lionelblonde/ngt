import os
import random
import subprocess
from pathlib import Path
from typing import Optional

import fire
from beartype import beartype
from omegaconf import DictConfig, OmegaConf
import numpy as np

import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

from gymnasium.core import Env

import orchestrator
from helpers import logger
from helpers.env_makers import make_env
from agents.agent import Agent

torch.set_float32_matmul_precision("high")
torch._dynamo.config.inline_inbuilt_nn_modules = True

BASE_CFG_PATH = Path(__file__).resolve().parent.parent / "conf" / "base.yml"


@beartype
def make_uuid(num_syllables: int = 2, num_parts: int = 3) -> str:
    """Randomly create a semi-pronounceable uuid"""
    part1 = ["s", "t", "r", "ch", "b", "c", "w", "z", "h", "k", "p", "ph", "sh", "f", "fr"]
    part2 = ["a", "oo", "ee", "e", "u", "er"]
    seps = ["_"]  # [ "-", "_", "."]
    result = ""
    for i in range(num_parts):
        if i > 0:
            result += seps[random.randrange(len(seps))]
        indices1 = [random.randrange(len(part1)) for _ in range(num_syllables)]
        indices2 = [random.randrange(len(part2)) for _ in range(num_syllables)]
        for i1, i2 in zip(indices1, indices2, strict=True):
            result += part1[i1] + part2[i2]
    return result


@beartype
def get_name(*,
             uuid: str,
             algo: str,
             env_id: str,
             num_demos: int,
             subsampling_rate: int,
             seed: int) -> str:
    """Assemble long experiment name"""
    name = uuid
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        sha = out.strip().decode("ascii")
        name += f".gitSHA_{sha}"
    except OSError:
        pass
    name += f".algo_{algo}"
    name += f".{env_id}"
    name += f".dems{str(num_demos).zfill(2)}"
    name += f".subr{str(subsampling_rate).zfill(2)}"
    name += f".seed{str(seed).zfill(2)}"
    return name


class MagicRunner(object):

    LOGGER_LEVEL: int = logger.WARN

    @beartype
    def __init__(self,
                 *,
                 # value given in arg overrides one in cfg if any
                 seed: int,
                 env_id: str,
                 num_demos: Optional[int] = None,
                 subsampling_rate: Optional[int] = None,
                 expert_path: Optional[str] = None,
                 # can override the value from cfg
                 wandb_project: Optional[str] = None,
                 # never in cfg, but not forced to give in arg either
                 uuid: Optional[str] = None,
                 variant_id: Optional[str] = None,
                 load_ckpt: Optional[str] = None,
                 patch: Optional[str] = None):

        logger.configure_default_logger()
        logger.set_level(self.LOGGER_LEVEL)

        # retrieve config from filesystem
        base_config = OmegaConf.load(BASE_CFG_PATH)
        if patch is not None:
            patch_cfg = OmegaConf.load(Path(patch))
            self._cfg = OmegaConf.merge(base_config, patch_cfg)
        else:
            self._cfg = base_config
        assert isinstance(self._cfg, DictConfig)

        logger.info("the config loaded:")
        logger.info(OmegaConf.to_yaml(self._cfg))

        proj_root = Path(__file__).resolve().parent.parent
        self._cfg.root = str(proj_root)  # in config: used by wandb
        for k in ("checkpoint", "log", "video"):
            new_k = f"{k}_dir"
            self._cfg[new_k] = str(proj_root / k)  # for yml saving

        # set only if nonexistant key in cfg
        self._cfg.seed = seed
        self._cfg.env_id = env_id
        self._cfg.num_demos = num_demos
        self._cfg.subsampling_rate = subsampling_rate
        self._cfg.expert_path = expert_path

        # safety net
        assert self._cfg.method in {
            "ngt", "random", "bc", "dac", "wdac", "pwil", "diffail", "iqlearn", "p2il"}
        assert self._cfg.input_mode in {"sa", "ss"}

        # override value from cfg when given as an argument
        assert "wandb_project" in self._cfg  # if not in cfg file, abort
        if wandb_project is not None:
            # override
            self._cfg.wandb_project = wandb_project

        assert "uuid" not in self._cfg  # uuid should never be in the cfg file
        self._cfg.uuid = uuid if uuid is not None else make_uuid()

        assert "variant_id" not in self._cfg  # variant_id should never be in the cfg file
        if variant_id is not None:
            self._cfg.variant_id = variant_id

        assert "load_ckpt" not in self._cfg  # load_ckpt should never be in the cfg file
        if load_ckpt is not None:
            self._cfg.load_ckpt = load_ckpt  # add in cfg
        else:
            logger.info("no ckpt to load: key will not exist in cfg")

        self.name = get_name(
            uuid=self._cfg.uuid,
            algo=self._cfg.method,
            env_id=self._cfg.env_id,
            num_demos=self._cfg.num_demos,
            subsampling_rate=self._cfg.subsampling_rate,
            seed=self._cfg.seed,
        )

        # set the cfg to read-only for safety
        OmegaConf.set_readonly(self._cfg, value=True)

    @beartype
    def setup_device(self) -> torch.device:
        if self._cfg.cuda:
            # use cuda
            assert torch.cuda.is_available()
            torch.cuda.manual_seed(self._cfg.seed)
            torch.cuda.manual_seed_all(self._cfg.seed)  # if using multiple GPUs
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            device = torch.device("cuda:0")
        else:
            # default case: just use plain old cpu, no cuda or m-chip gpu
            device = torch.device("cpu")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # kill any possibility of usage
        logger.info(f"device in use: {device}")
        return device

    @beartype
    def setup_expert_struct(self,
                            *,
                            device: torch.device,
        ) -> tuple[TensorDictReplayBuffer, Optional[list[TensorDict]]]:

        expert_dataset = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                50_000,  # max imaginable number of demonstrations TRANSITIONS in use
                device=device,
            ),
        )

        expert_atoms = None
        if self._cfg.method == "pwil":
            expert_atoms = []

        expert_path = Path(self._cfg.expert_path) / self._cfg.env_id
        if not expert_path.exists():
            raise ValueError(f"expert path does not exist: {expert_path}")

        keys = ["observations", "actions", "next_observations"]
        for i, fpath in enumerate(sorted(expert_path.glob("*.h5"))):
            if i == self._cfg.num_demos:
                break  # enough demos unpacked
            td = TensorDict.from_h5(fpath)
            logger.warn(f"demo #{i} | return=" + str(td["return"]))
            logger.warn(td["length"])
            start = torch.randint(
                0,
                (n := self._cfg.subsampling_rate),  # exclusive
                (1,),
                device=device,
            ).item()
            start += self._cfg.demo_start_trim
            for j in range(start, int(td["length"].item()), n):
                with torch.no_grad():
                    dt = TensorDict(
                        {
                            k: td[k][j].detach() for k in keys
                        },
                        batch_size=1,
                        device=device,
                    )
                expert_dataset.extend(dt)
                if self._cfg.method == "pwil":
                    expert_atoms.append(dt)

        logger.warn(f"expert dataset contains: {len(expert_dataset)} transitions")

        print(f"expert dataset: {len(expert_dataset)} transitions")

        return expert_dataset, expert_atoms

    @beartype
    def train(self):
        if "pretrain" in self._cfg:
            raise ValueError("config key `pretrain` has been removed")
        if "load_ckpt" in self._cfg:
            raise ValueError(
                "train-side warmstart is disabled; --load_ckpt is only supported in evaluate",
            )

        # logger
        log_path = Path(self._cfg.log_dir) / self.name
        log_path.mkdir(parents=True, exist_ok=True)
        logger.configure(directory=log_path, format_strs=[
            # "stdout",
            "log",
            "json",
            "csv",
        ])
        logger.set_level(self.LOGGER_LEVEL)
        progress_files = {"json": log_path / "progress.json", "csv": log_path / "progress.csv"}

        # save config in log dir
        OmegaConf.save(config=self._cfg, f=(log_path / "cfg.yml"))

        # video capture
        video_path = None
        if self._cfg.capture_video:
            video_path = Path(self._cfg.video_dir) / self.name
            video_path.mkdir(parents=True, exist_ok=True)

        # seed and device
        random.seed(self._cfg.seed)  # after uuid creation, otherwise always same uuid
        np.random.seed(self._cfg.seed)
        torch.manual_seed(self._cfg.seed)
        device = self.setup_device()

        # envs
        env, net_shapes, min_ac, max_ac = make_env(
            self._cfg.env_id,
            self._cfg.seed,
            sync_vec_env=self._cfg.sync_vec_env,
            num_envs=self._cfg.num_envs,
        )
        eenv, _, _, _ = make_env(
            self._cfg.env_id,
            self._cfg.seed,
            sync_vec_env=True,
            num_envs=1,
            video_path=video_path,
        )
        assert isinstance(eenv, Env)

        # expert
        expert_dataset = None
        expert_atoms = None
        if self._cfg.method != "random":
            expert_dataset, expert_atoms = self.setup_expert_struct(
                device=device,
            )

        # agent
        replay_dataset = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                self._cfg.rb_capacity, device=device,
            ),
        )

        @beartype
        def agent_wrapper() -> Agent:
            return Agent(
                net_shapes=net_shapes,
                min_ac=min_ac,
                max_ac=max_ac,
                device=device,
                hps=self._cfg,
                replay_dataset=replay_dataset,
                expert_dataset=expert_dataset,
                expert_atoms=expert_atoms,
            )

        # train
        if self._cfg.method == "bc":
            orchestrator.clone(
                cfg=self._cfg,
                eenv=eenv,
                agent_wrapper=agent_wrapper,
                name=self.name,
                progress_files=progress_files,
            )
        else:
            orchestrator.train(
                cfg=self._cfg,
                env=env,
                eenv=eenv,
                agent_wrapper=agent_wrapper,
                name=self.name,
                progress_files=progress_files,
            )

        # cleanup
        env.close()
        eenv.close()

    @beartype
    def evaluate(self):
        if "load_ckpt" not in self._cfg:
            raise ValueError("evaluate requires --load_ckpt=<wandb_run_path>")

        # logger
        logger.configure(directory=None, format_strs=["stdout"])
        logger.set_level(self.LOGGER_LEVEL)

        # video capture
        video_path = None
        if self._cfg.capture_video:
            video_path = Path(self._cfg.video_dir) / self.name
            video_path.mkdir(parents=True, exist_ok=True)

        # seed and device
        random.seed(self._cfg.seed)  # after uuid creation, otherwise always same uuid
        np.random.seed(self._cfg.seed)
        torch.manual_seed(self._cfg.seed)
        device = self.setup_device()

        # env
        eenv, net_shapes, min_ac, max_ac = make_env(
            self._cfg.env_id,
            self._cfg.seed,
            sync_vec_env=True,
            num_envs=1,
            video_path=video_path,
        )
        assert isinstance(eenv, Env)

        # expert
        expert_atoms = None
        if self._cfg.method == "pwil":
            _, expert_atoms = self.setup_expert_struct(
                device=device,
            )

        # agent
        @beartype
        def agent_wrapper() -> Agent:
            return Agent(
                net_shapes=net_shapes,
                min_ac=min_ac,
                max_ac=max_ac,
                device=device,
                hps=self._cfg,
                expert_atoms=expert_atoms,
            )

        # evaluate
        with torch.inference_mode():
            orchestrator.evaluate(
                cfg=self._cfg,
                eenv=eenv,
                agent_wrapper=agent_wrapper,
            )

        # cleanup
        eenv.close()


if __name__ == "__main__":
    fire.Fire(MagicRunner)
