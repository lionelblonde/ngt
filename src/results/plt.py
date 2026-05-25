from collections import defaultdict
import hashlib
import itertools
import json
import operator
from pathlib import Path
from typing import Any, Optional

from beartype import beartype
import fire
import matplotlib.pyplot as plt
import numpy as np

from helpers import logger

try:
    from rliable import library as rly
    from rliable import metrics as rly_metrics
except ImportError:
    rly = None
    rly_metrics = None


BASELINES_PATH = Path(__file__).resolve().parent.parent / "helpers" / "baselines.json"
HIERARCHY_DEPTH = 5
HIERARCHY_WITH_RUNSET_DEPTH = 6
EPS = 1e-8
X_AXIS_TIMESTEP = "timestep"
X_AXIS_WALL_CLOCK = "wall_time_total_s"
TWO_D = 2
DEFAULT_PLOTS_DIR = "plots"
OUTPUT_HASH_LEN = 10


class MagicPlotter(object):

    @beartype
    def __init__(
        self,
        root: str = "~/Downloads/results",
        runset_id: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        self.root_path = Path(root).expanduser()
        self.runset_id = runset_id

        if output_dir is None:
            self.output_dir = self._make_default_output_dir()
        else:
            self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with BASELINES_PATH.open("r", encoding="utf-8") as f:
            baselines = json.load(f)
        self.experts = baselines["experts"]
        self.randoms = baselines["randoms"]

        self.colors = {
            "blue": "#4285F4",
            "red": "#EA4335",
            "yellow": "#FBBC05",
            "pink": "#EA4C89",
            "purple": "#673AB7",
            "grey": "#5f6368",
            "teal": "#00B7C3",
            "forest": "#228B22",
        }
        self.algo_info = {
            "ngt": {"full_name": "NGT state-action", "color": self.colors["blue"]},
            "ablation-ngt-reward_shaping_symexp": {
                "full_name": "NGT (symexp reward shaping)",
                "color": self.colors["blue"],
            },
            "ablation-ngt-reward_shaping_linear": {
                "full_name": "NGT (linear reward shaping)",
                "color": self.colors["pink"],
            },
            "diffail": {"full_name": "DiffAIL", "color": self.colors["red"]},
            "pwil": {"full_name": "PWIL", "color": self.colors["purple"]},
            "bc": {"full_name": "BC", "color": self.colors["grey"]},
            "random": {"full_name": "Random", "color": self.colors["grey"]},
            "dac": {"full_name": "DAC", "color": self.colors["yellow"]},
            "wdac": {"full_name": "W-DAC", "color": self.colors["yellow"]},
            "iqlearn": {"full_name": "IQ-Learn", "color": self.colors["teal"]},
            "p2il": {"full_name": "P2IL", "color": self.colors["forest"]},
        }

    @beartype
    def _make_default_output_dir(self) -> Path:
        digest_src = f"{self.root_path}|{self.runset_id or ''}"
        digest = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:OUTPUT_HASH_LEN]
        return self.root_path / DEFAULT_PLOTS_DIR / digest

    @beartype
    @staticmethod
    def _parse_progress_file(file_path: Path) -> list[dict[str, Any]]:
        with file_path.open("r", encoding="utf-8") as f:
            out = []
            for line_id, line in enumerate(f, 1):
                text = line.strip()
                if not text:
                    continue
                try:
                    out.append(json.loads(text))
                except json.JSONDecodeError:
                    logger.warn(f"skip bad json line at {file_path}:{line_id}")
            return out

    @beartype
    @staticmethod
    def _decode_hierarchy_parts(parts: tuple[str, ...]) -> Optional[dict[str, str]]:
        if len(parts) == HIERARCHY_DEPTH:
            env_id, dems, subr, algo_id, seed_id = parts
            return {
                "runset_id": "",
                "env_id": env_id,
                "dems": dems,
                "subr": subr,
                "algo_id": algo_id,
                "seed_id": seed_id,
            }
        if len(parts) == HIERARCHY_WITH_RUNSET_DEPTH:
            runset_id, env_id, dems, subr, algo_id, seed_id = parts
            return {
                "runset_id": runset_id,
                "env_id": env_id,
                "dems": dems,
                "subr": subr,
                "algo_id": algo_id,
                "seed_id": seed_id,
            }
        return None

    @beartype
    def traverse_hierarchy(self) -> dict[str, Any]:
        results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

        for file_path in self.root_path.rglob("progress.json"):
            rel = file_path.parent.relative_to(self.root_path)
            decoded = self._decode_hierarchy_parts(rel.parts)
            if decoded is None:
                continue

            if self.runset_id is not None:
                runset = decoded["runset_id"]
                if runset and runset != self.runset_id:
                    continue

            env_id = decoded["env_id"]
            if env_id not in self.experts or env_id not in self.randoms:
                continue
            dems = decoded["dems"]
            subr = decoded["subr"]
            algo_id = decoded["algo_id"]
            seed_id = decoded["seed_id"]

            data = self._parse_progress_file(file_path)
            if data:
                results[env_id][dems][subr][algo_id].append((seed_id, data))

        return results

    @beartype
    @staticmethod
    def _compute_mean_and_std(
        algo_data: list[tuple[str, list[dict[str, Any]]]],
        algo_id: str,
        metric: str = "return",
        x_axis: str = X_AXIS_TIMESTEP,
    ) -> dict[str, list[Any]]:
        if x_axis not in {X_AXIS_TIMESTEP, X_AXIS_WALL_CLOCK}:
            raise ValueError(f"unsupported x_axis: {x_axis}")

        anchor_key = "iteration" if algo_id == "bc" else "timestep"

        all_anchor = sorted({
            int(record[anchor_key])
            for _, data in algo_data
            for record in data
            if anchor_key in record
        })
        metrics: dict[str, list[Any]] = defaultdict(list)

        for anchor in all_anchor:
            values = []
            x_values = []
            for _, data in algo_data:
                record = next(
                    (
                        entry
                        for entry in data
                        if anchor_key in entry and int(entry[anchor_key]) == anchor
                    ),
                    None,
                )
                if record is None:
                    continue
                if metric not in record:
                    continue

                values.append(float(record[metric]))
                if x_axis == X_AXIS_TIMESTEP:
                    x_values.append(float(anchor))
                elif x_axis in record:
                    x_values.append(float(record[x_axis]))
            if values and x_values:
                metrics["anchor"].append(anchor)
                metrics["x"].append(float(np.mean(x_values)))
                metrics["mean"].append(float(np.mean(values)))
                metrics["std"].append(float(np.std(values)))

        return metrics

    @beartype
    @staticmethod
    def _rescale_values(
        values: list[float],
        mean_expert: float,
        mean_random: float,
    ) -> list[float]:
        scale = mean_expert - mean_random
        if abs(scale) < EPS:
            return [0.0 for _ in values]
        return [(value - mean_random) / scale for value in values]

    @beartype
    @staticmethod
    def _get_final_bc_perf(
        results: dict[str, Any],
        env_id: str,
        dems: str,
        subr: str,
        metric: str = "return",
    ) -> dict[str, np.float64]:
        bc_perf: dict[str, np.float64] = {}
        if "bc" in results[env_id][dems][subr]:
            bc_data = results[env_id][dems][subr]["bc"]
            final_values = [
                data[-1][metric]
                for _, data in bc_data
                if data and metric in data[-1]
            ]
            if final_values:
                bc_perf[subr] = np.mean(final_values)
        return bc_perf

    @beartype
    def _resolve_algo_entry(self, algo_id: str) -> dict[str, str]:
        if algo_id in self.algo_info:
            return self.algo_info[algo_id]
        logger.warn(f"missing algo map entry for {algo_id}; using fallback")
        return {
            "full_name": algo_id,
            "color": self.colors["grey"],
        }

    @beartype
    @staticmethod
    def _extract_seed_score(
        data: list[dict[str, Any]],
        algo_id: str,
        *,
        metric: str,
        x_axis: str,
        target_x: Optional[float],
    ) -> Optional[float]:
        if x_axis == X_AXIS_TIMESTEP:
            key = "iteration" if algo_id == "bc" else "timestep"
        else:
            key = x_axis

        valid = [record for record in data if key in record and metric in record]
        if not valid:
            return None

        if target_x is None:
            chosen = max(valid, key=lambda x: float(x[key]))
        else:
            chosen = min(valid, key=lambda x: abs(float(x[key]) - float(target_x)))

        return float(chosen[metric])

    @beartype
    def _build_rliable_score_dict(
        self,
        results: dict[str, Any],
        *,
        algo_ids: list[str],
        subr: str,
        dems_list: list[str],
        metric: str,
        x_axis: str,
        target_x: Optional[float],
    ) -> tuple[dict[str, np.ndarray], list[str], list[tuple[str, str]]]:
        if x_axis not in {X_AXIS_TIMESTEP, X_AXIS_WALL_CLOCK}:
            raise ValueError(f"unsupported x_axis: {x_axis}")

        per_algo_task_scores: dict[str, dict[tuple[str, str], list[tuple[str, float]]]] = {
            algo_id: defaultdict(list) for algo_id in algo_ids
        }
        all_tasks: set[tuple[str, str]] = set()

        for env_id, env_data in results.items():
            for dems in dems_list:
                if dems not in env_data:
                    continue
                if env_id not in self.experts or dems not in self.experts[env_id]:
                    continue
                if env_id not in self.randoms:
                    continue
                if subr not in env_data[dems]:
                    continue

                task = (env_id, dems)
                mean_expert = float(self.experts[env_id][dems]["avg"])
                mean_random = float(self.randoms[env_id])
                all_tasks.add(task)

                for algo_id in algo_ids:
                    algo_id_ = algo_id
                    if algo_id_ not in env_data[dems][subr]:
                        continue

                    algo_data = env_data[dems][subr][algo_id_]
                    for seed_id, data in algo_data:
                        score = self._extract_seed_score(
                            data,
                            algo_id_,
                            metric=metric,
                            x_axis=x_axis,
                            target_x=target_x,
                        )
                        if score is None:
                            continue
                        score = self._rescale_values([score], mean_expert, mean_random)[0]
                        per_algo_task_scores[algo_id][task].append((seed_id, score))

        if not all_tasks:
            raise ValueError("no tasks found for rliable computation")

        common_tasks = sorted(all_tasks)
        for algo_id in algo_ids:
            algo_tasks = {
                task for task, vals in per_algo_task_scores[algo_id].items() if len(vals) > 0
            }
            common_tasks = [task for task in common_tasks if task in algo_tasks]
        if not common_tasks:
            raise ValueError("no common tasks across selected algorithms")

        score_dict: dict[str, np.ndarray] = {}
        kept_algo_ids: list[str] = []

        for algo_id in algo_ids:
            task_series = []
            min_seeds = None
            for task in common_tasks:
                vals = sorted(per_algo_task_scores[algo_id][task], key=operator.itemgetter(0))
                scores = [s for _, s in vals]
                if not scores:
                    min_seeds = 0
                    break
                min_seeds = len(scores) if min_seeds is None else min(min_seeds, len(scores))
                task_series.append(scores)

            if min_seeds is None or min_seeds == 0:
                logger.warn(f"skip {algo_id}: not enough seed coverage for common tasks")
                continue

            matrix = np.array([scores[:min_seeds] for scores in task_series], dtype=float).T
            score_dict[algo_id] = matrix
            kept_algo_ids.append(algo_id)
            logger.warn(
                f"rliable dataset | algo={algo_id} | runs={matrix.shape[0]} | "
                f"tasks={matrix.shape[1]}",
            )

        if not score_dict:
            raise ValueError("no algorithms left after task/seed alignment")

        return score_dict, kept_algo_ids, common_tasks

    @beartype
    def plot_main(
        self,
        metric: str = "return",
        figsize: tuple[int, int] = (20, 10),
        truncate_at: int = 3_000_000,
        out_name: str = "main.png",
        x_axis: str = X_AXIS_TIMESTEP,
        *,
        thicken_ngt: bool = True,
    ):
        results = self.traverse_hierarchy()
        env_ids = sorted(results.keys())
        dems_values = sorted({dems for env_data in results.values() for dems in env_data})

        if not env_ids or not dems_values:
            raise ValueError("no progress files found under root")

        fig, axes = plt.subplots(
            len(dems_values),
            len(env_ids),
            figsize=figsize,
            squeeze=False,
        )

        for j, env_id in enumerate(env_ids):
            for i, dems in enumerate(dems_values):
                ax = axes[i, j]

                if env_id not in self.experts or dems not in self.experts[env_id]:
                    logger.warn(f"skip panel {env_id}/{dems}: baseline stats missing")
                    ax.axis("off")
                    continue
                if env_id not in self.randoms:
                    logger.warn(f"skip panel {env_id}/{dems}: random baseline missing")
                    ax.axis("off")
                    continue

                mean_expert = float(self.experts[env_id][dems]["avg"])
                mean_random = float(self.randoms[env_id])

                plot_subr = "subr20"
                subr_data = results[env_id][dems].get(plot_subr, {})
                for algo_id, algo_data in subr_data.items():
                    if algo_id == "bc":
                        continue

                    metrics = self._compute_mean_and_std(
                        algo_data,
                        algo_id,
                        metric=metric,
                        x_axis=x_axis,
                    )
                    if "x" not in metrics:
                        continue

                    x_values = np.array(metrics["x"])
                    mean_values = np.array(
                        self._rescale_values(metrics["mean"], mean_expert, mean_random),
                    )
                    scale = mean_expert - mean_random
                    std_values = np.array([
                        std / scale if abs(scale) > EPS else 0.0
                        for std in metrics["std"]
                    ])

                    if x_values.size == 0:
                        continue

                    entry = self._resolve_algo_entry(algo_id)
                    kwargs = {"label": entry["full_name"], "color": entry["color"]}

                    ax.plot(x_values, mean_values, **kwargs)
                    ax.fill_between(
                        x_values,
                        mean_values - 0.5 * std_values,
                        mean_values + 0.5 * std_values,
                        color=entry["color"],
                        alpha=0.2,
                    )

                bc_perf = self._get_final_bc_perf(
                    results,
                    env_id,
                    dems,
                    plot_subr,
                    metric=metric,
                )
                linestyles = itertools.cycle(["-", ":"])
                for subr, final_value in bc_perf.items():
                    scale = mean_expert - mean_random
                    if abs(scale) > EPS:
                        rescaled_value = (final_value - mean_random) / scale
                    else:
                        rescaled_value = 0.0
                    line_label = f"BC ({subr[-2:]})"
                    ax.axhline(
                        y=rescaled_value,
                        color="gray",
                        linestyle=next(linestyles),
                        label=line_label,
                    )

                if x_axis == X_AXIS_TIMESTEP:
                    ax.set_xlim(0, 10_000_000)
                else:
                    ax.set_xlim(left=0)
                ax.set_yticks([0.0, 0.5, 1.0])
                ax.xaxis.set_ticks_position("bottom")
                ax.yaxis.set_ticks_position("left")
                ax.set_title(
                    f"{env_id} [{int(dems[4:])} demo{'' if int(dems[4:]) == 1 else 's'}]",
                    fontsize=15,
                )
                if i == (len(dems_values) - 1):
                    if x_axis == X_AXIS_TIMESTEP:
                        ax.set_xlabel("Timestep")
                    else:
                        ax.set_xlabel("Wall-clock time (s)")
                if j == 0:
                    ax.set_ylabel(f"Normalized {metric}", fontsize=14)
                ax.grid(visible=False)

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            sorted_handles_labels = sorted(
                zip(labels, handles, strict=True),
                key=operator.itemgetter(0),
            )
            sorted_labels, sorted_handles = zip(*sorted_handles_labels, strict=True)
            fig.legend(
                sorted_handles,
                sorted_labels,
                loc="lower center",
                ncol=6,
                fontsize=14,
            )

        plt.tight_layout(rect=[0, 0.07, 1, 1])
        file_name = self.output_dir / out_name
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        logger.warn(f"saved {file_name}")

    @beartype
    def plot_rliable(
        self,
        algo_ids: Optional[list[str]] = None,
        subr: str = "subr20",
        dems_list: Optional[list[str]] = None,
        metric: str = "return",
        x_axis: str = X_AXIS_TIMESTEP,
        target_x: Optional[float] = None,
        reps: int = 5_000,
        *,
        out_prefix: str = "rliable",
        profile_xmin: float = 0.0,
        profile_xmax: float = 1.5,
        profile_num_thresholds: int = 101,
    ):
        if rly is None or rly_metrics is None:
            raise ImportError(
                "rliable is required for plot_rliable; add it to dependencies first",
            )

        results = self.traverse_hierarchy()

        if algo_ids is None:
            algo_ids = sorted({
                algo_id
                for env_data in results.values()
                for dems_data in env_data.values()
                for subr_data in dems_data.values()
                for algo_id in subr_data
            })
        if dems_list is None:
            dems_list = sorted({
                dems
                for env_data in results.values()
                for dems in env_data
            })

        score_dict, kept_algo_ids, common_tasks = self._build_rliable_score_dict(
            results,
            algo_ids=algo_ids,
            subr=subr,
            dems_list=dems_list,
            metric=metric,
            x_axis=x_axis,
            target_x=target_x,
        )
        logger.warn(f"rliable common tasks: {len(common_tasks)}")

        @beartype
        def aggregate_func(x: np.ndarray) -> np.ndarray:
            return np.array([
                rly_metrics.aggregate_median(x),
                rly_metrics.aggregate_iqm(x),
                rly_metrics.aggregate_mean(x),
            ])

        aggregate_scores, aggregate_cis = rly.get_interval_estimates(
            score_dict,
            aggregate_func,
            reps=reps,
        )

        # aggregate interval estimates
        metric_names = ["Median", "IQM", "Mean"]
        _fig, axes = plt.subplots(1, len(metric_names), figsize=(13, 4), squeeze=False)
        for m_id, metric_name in enumerate(metric_names):
            ax = axes[0, m_id]
            for a_id, algo_id in enumerate(kept_algo_ids):
                point = float(np.asarray(aggregate_scores[algo_id])[m_id])
                ci = np.asarray(aggregate_cis[algo_id])
                if ci.ndim != TWO_D:
                    low = point
                    high = point
                elif ci.shape[0] == TWO_D:
                    low = float(ci[0, m_id])
                    high = float(ci[1, m_id])
                else:
                    low = float(ci[m_id, 0])
                    high = float(ci[m_id, 1])

                yerr = np.array([[max(0.0, point - low)], [max(0.0, high - point)]])
                entry = self._resolve_algo_entry(algo_id)
                ax.errorbar(
                    a_id,
                    point,
                    yerr=yerr,
                    fmt="o",
                    capsize=4,
                    color=entry["color"],
                )

            ax.set_title(metric_name)
            ax.set_xticks(range(len(kept_algo_ids)))
            ax.set_xticklabels(
                [self._resolve_algo_entry(x)["full_name"] for x in kept_algo_ids],
                rotation=25,
                ha="right",
            )
            ax.grid(alpha=0.2)

        plt.tight_layout()
        aggregate_path = self.output_dir / f"{out_prefix}_aggregate.png"
        plt.savefig(aggregate_path, dpi=300, bbox_inches="tight")
        logger.warn(f"saved {aggregate_path}")

        # performance profile
        thresholds = np.linspace(profile_xmin, profile_xmax, profile_num_thresholds)
        profiles, profiles_cis = rly.create_performance_profile(score_dict, thresholds)

        _fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for algo_id in kept_algo_ids:
            entry = self._resolve_algo_entry(algo_id)
            y = np.asarray(profiles[algo_id], dtype=float)
            ax.plot(thresholds, y, label=entry["full_name"], color=entry["color"])

            ci = np.asarray(profiles_cis[algo_id], dtype=float)
            if ci.ndim == TWO_D:
                if ci.shape[0] == TWO_D:
                    low, high = ci[0], ci[1]
                elif ci.shape[1] == TWO_D:
                    low, high = ci[:, 0], ci[:, 1]
                else:
                    continue
                ax.fill_between(thresholds, low, high, color=entry["color"], alpha=0.15)

        ax.set_xlim(profile_xmin, profile_xmax)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel(f"Normalized {metric} threshold")
        ax.set_ylabel("Fraction of tasks above threshold")
        ax.grid(alpha=0.2)
        ax.legend()

        plt.tight_layout()
        profile_path = self.output_dir / f"{out_prefix}_profile.png"
        plt.savefig(profile_path, dpi=300, bbox_inches="tight")
        logger.warn(f"saved {profile_path}")


if __name__ == "__main__":
    logger.configure(directory=None, format_strs=["stdout"])
    logger.set_level(logger.WARN)
    fire.Fire(MagicPlotter)
