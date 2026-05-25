from pathlib import Path
from typing import Any, Optional

import fire
from beartype import beartype
import wandb
from wandb.errors import CommError
from wandb.apis.public.runs import Run

from helpers import logger


@beartype
def download_file(file_name: str, run: Run, run_dir: Path):
    try:
        file = run.file(file_name)
        file.download(root=str(run_dir), replace=True)
        logger.warn(f"downloaded {file_name} for run {run.name}")
        logger.warn(f" @@ {run_dir.resolve()}")
    except CommError:
        logger.warn(f"{file_name} not found for run {run.name}")


@beartype
def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return int(float(text))
    return None


@beartype
def parse_run_metadata(run: Run) -> Optional[dict[str, str]]:
    cfg = dict(run.config)
    runset_id = cfg.get("uuid")
    env_id = cfg.get("env_id")
    variant_id = cfg.get("variant_id")
    algo_id = variant_id if isinstance(variant_id, str) and variant_id.strip() else cfg.get("method")
    num_demos = _as_int(cfg.get("num_demos"))
    subsampling_rate = _as_int(cfg.get("subsampling_rate"))
    seed = _as_int(cfg.get("seed"))

    if not isinstance(runset_id, str):
        return None
    if not isinstance(env_id, str):
        return None
    if not isinstance(algo_id, str):
        return None
    if num_demos is None:
        return None
    if subsampling_rate is None:
        return None
    if seed is None:
        return None

    return {
        "runset_id": runset_id,
        "env_id": env_id,
        "dems": f"dems{num_demos:02d}",
        "subr": f"subr{subsampling_rate:02d}",
        "algo_id": algo_id,
        "seed": f"seed{seed:02d}",
    }


@beartype
def retrieve_from_wandb(
    wandb_id: str,
    wandb_project: str,
    download_dir: str = "~/Downloads/results",
    *,
    runset_id: Optional[str] = None,
    group_name: Optional[str] = None,
    only_finished: bool = True,
    include_csv: bool = False,
):
    """Retrieve progress files and materialize campaign-based hierarchy."""
    if runset_id is None and group_name is None:
        raise ValueError("must provide runset_id or group_name")

    base_filters: dict[str, Any] = {}
    if runset_id is not None:
        base_filters["config.uuid"] = runset_id
    if group_name is not None:
        base_filters["group"] = group_name

    filters = dict(base_filters)
    if only_finished:
        filters["state"] = "finished"

    api = wandb.Api()
    runs = api.runs(f"{wandb_id}/{wandb_project}", filters=filters)
    logger.warn(f"filters={filters}")

    if only_finished:
        skipped_runs = api.runs(
            f"{wandb_id}/{wandb_project}",
            filters={**base_filters, "state": {"$ne": "finished"}},
        )
        for run in skipped_runs:
            logger.warn(f"skip {run.name}: run state is {run.state}, not finished")

    root = Path(download_dir).expanduser()
    n_total = 0
    n_synced = 0
    n_skipped = 0

    for run in runs:
        n_total += 1

        meta = parse_run_metadata(run)
        if meta is None:
            n_skipped += 1
            logger.warn(f"skip {run.name}: missing required metadata in config")
            continue

        run_dir = (root /
                   meta["runset_id"] /
                   meta["env_id"] /
                   meta["dems"] /
                   meta["subr"] /
                   meta["algo_id"] /
                   meta["seed"])
        run_dir.mkdir(parents=True, exist_ok=True)

        file_names = ["progress.json"]
        if include_csv:
            file_names.append("progress.csv")
        for file_name in file_names:
            download_file(file_name, run, run_dir)
        n_synced += 1

    logger.warn(f"done | total={n_total}, synced={n_synced}, skipped={n_skipped}")


@beartype
def list_campaigns(
    wandb_id: str,
    wandb_project: str,
    *,
    only_finished: bool = True,
):
    """List campaign IDs (config.uuid) in a wandb project."""
    filters: dict[str, Any] = {}
    if only_finished:
        filters["state"] = "finished"

    api = wandb.Api()
    runs = api.runs(f"{wandb_id}/{wandb_project}", filters=filters)

    buckets: dict[str, dict[str, Any]] = {}
    for run in runs:
        cfg = dict(run.config)
        runset_id = cfg.get("uuid")
        if not isinstance(runset_id, str):
            continue
        if runset_id not in buckets:
            buckets[runset_id] = {
                "runs": 0,
                "updated_at": run.updated_at,
                "groups": set(),
            }
        info = buckets[runset_id]
        info["runs"] += 1
        if run.updated_at is not None:
            if info["updated_at"] is None or run.updated_at > info["updated_at"]:
                info["updated_at"] = run.updated_at
        if isinstance(run.group, str):
            info["groups"].add(run.group)

    rows = sorted(
        buckets.items(),
        key=lambda x: ((x[1]["updated_at"] or ""), x[0]),
        reverse=True,
    )
    for runset_id, info in rows:
        logger.warn(
            f"runset_id={runset_id} | runs={info['runs']} | "
            f"groups={len(info['groups'])} | updated_at={info['updated_at']}",
        )


if __name__ == "__main__":
    logger.configure(directory=None, format_strs=["stdout"])
    logger.set_level(logger.WARN)
    fire.Fire(
        {
            "sync": retrieve_from_wandb,
            "list_campaigns": list_campaigns,
        },
    )
