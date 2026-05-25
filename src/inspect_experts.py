from pathlib import Path
import json

import fire
from beartype import beartype
from tensordict import TensorDict
import numpy as np

from helpers import logger


@beartype
def log_perf(expert_path: str, env_id: str = ""):
    """Print baselines.json-ready expert entries from h5 demos."""
    logger.configure(directory=None, format_strs=["stdout"])
    logger.set_level(logger.WARN)

    expert_path = Path(expert_path)
    for directory in sorted(expert_path.iterdir()):
        if not directory.is_dir():
            continue
        if env_id and directory.name != env_id:
            continue

        returns = []
        for fpath in sorted((expert_path / directory.name).glob("*.h5")):
            td = TensorDict.from_h5(fpath)
            returns.append(float(td["return"]))

        if not returns:
            logger.warn(f"skip {directory.name}: no .h5 files found")
            continue

        out = {}
        for i in range(1, len(returns) + 1):
            subset = np.array(returns[:i], dtype=float)
            out[f"dems{i:02d}"] = {
                "avg": float(subset.mean()),
                "std": float(subset.std(ddof=1)) if i > 1 else 0.0,
            }

        payload = json.dumps(out, indent=2)
        logger.warn(f'"{directory.name}": {payload},')


if __name__ == "__main__":
    fire.Fire(log_perf)
