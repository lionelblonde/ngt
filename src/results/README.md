# Results retrieval and plotting

Run the commands below from the repository root.

## progress-file retrieval workflow

Show available downloader commands:

```bash
uv run -m src.results.get -- --help
```

Show downloader options for campaign/group sync:

```bash
uv run -m src.results.get sync -- --help
```

Retrieve full campaigns by `uuid` (recommended):

```bash
uv run -m src.results.get sync \
  "your_entity" \
  "your_project" \
  --runset_id="your_uuid" \
  --download_dir="/abs/path/to/results"
```

Retrieve one group only:

```bash
uv run -m src.results.get sync \
  "your_entity" \
  "your_project" \
  --group_name="your.group.name" \
  --download_dir="/abs/path/to/results"
```

List campaign IDs available in a project:

```bash
uv run -m src.results.get list_campaigns "your_entity" "your_project"
```

Optional flags:

- `--only_finished=true|false` (default: `true`)
- `--include_csv=true|false` (default: `false`, only `progress.json` by default)

The downloader materializes this hierarchy:

`<download_dir>/<runset_id>/<env_id>/demsXX/subrYY/<algo_id>/seedZZ/progress.json`

## plotting from local progress files

The plotter reads expert/random baselines from `src/helpers/baselines.json`.
By default, plots are written under `<root>/plots/<hash>/`. Pass
`--output_dir="/abs/path/to/dir"` to override that.

If your results live under one campaign root:

```bash
uv run -m src.results.plt plot_main --root="/abs/path/to/results/<runset_id>"
```

Return vs wall-clock time:

```bash
uv run -m src.results.plt plot_main \
  --root="/abs/path/to/results/<runset_id>" \
  --x_axis="wall_time_total_s" \
  --out_name="main_vs_walltime.png"
```

If your root contains multiple campaigns, pass one explicitly:

```bash
uv run -m src.results.plt plot_main \
  --root="/abs/path/to/results" \
  --runset_id="your_uuid"
```

RLiable plots (interval estimates and performance profile):

```bash
uv run -m src.results.plt plot_rliable \
  --root="/abs/path/to/results/<runset_id>" \
  --subr="subr20" \
  --x_axis="timestep" \
  --target_x=10000000 \
  --out_prefix="rliable_10M"
```

## update expert baselines for a new environment

Generate `baselines.json`-ready expert entries from local expert demos:

```bash
uv run -m src.inspect_experts \
  --expert_path="/abs/path/to/experts" \
  --env_id="YourEnv-v0"
```

The output is copy-paste ready for the `"experts"` section in
`src/helpers/baselines.json`.
