# NGT (PyTorch)

Official PyTorch implementation of Noise-Guided Transport (NGT),
along with imitation learning baselines.

_The NGT paper, [Noise-Guided Transport: Imitation Learning from Random Priors](https://proceedings.mlr.press/v306/blonde26a.html), was published at ICML 2026. This repository also includes functionality developed after the experiments reported in the paper._

## setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). The commands below use
`uv run` to create and use the project environment automatically.

Training uses Weights & Biases logging. If needed, set your API key:

```bash
export WANDB_API_KEY="your_key_here"
```

The released configurations target CUDA execution. Expert demonstrations for imitation-learning
methods are available [here](https://drive.google.com/drive/folders/17CyhQeiUYZBykiEgdzzix9nA2vOWLHFX?usp=sharing).

## available baselines

The code currently supports these methods:

- `ngt` (main method)
- `random`
- `bc`
- `dac`
- `wdac`
- `pwil`
- `diffail`
- `iqlearn`
- `p2il`

Use these patch files:

| method | default patch | extra per-env patches |
| --- | --- | --- |
| `ngt` | `conf/patch/ngt.yml` | none |
| `random` | `conf/patch/random.yml` | none |
| `bc` | `conf/patch/bc.yml` | none |
| `dac` | `conf/patch/dac.yml` | none |
| `wdac` | `conf/patch/wdac.yml` | none |
| `pwil` | `conf/patch/pwil.yml` | none |
| `diffail` | `conf/patch/diffail.yml` | `diffail_ant.yml`, `diffail_halfcheetah.yml`, `diffail_hopper.yml`, `diffail_walker2d.yml` |
| `iqlearn` | `conf/patch/iqlearn.yml` | `iqlearn_ant.yml`, `iqlearn_halfcheetah.yml`, `iqlearn_hopper.yml`, `iqlearn_walker2d.yml` |
| `p2il` | `conf/patch/p2il.yml` | `p2il_ant.yml`, `p2il_halfcheetah.yml` |

## train one run

Example (NGT):

```bash
uv run -m src.main train \
  --env_id="Hopper-v4" \
  --seed=1 \
  --num_demos=4 \
  --subsampling_rate=20 \
  --expert_path="/abs/path/to/experts" \
  --patch="conf/patch/ngt.yml"
```

Example (baseline, here IQ-Learn):

```bash
uv run -m src.main train \
  --env_id="Hopper-v4" \
  --seed=1 \
  --num_demos=4 \
  --subsampling_rate=20 \
  --expert_path="/abs/path/to/experts" \
  --patch="conf/patch/iqlearn_hopper.yml"
```

Notes:
- `conf/base.yml` is always loaded first.
- `--patch` overlays base config and selects the method.
- NGT ablation patches live under `conf/patch/ablations/`, including: `ngt_reward_shaping_linear.yml`.
- Reward-shaping ablation compares default NGT (`symexp`) against the ablation override (`linear`).
- Expert demos are expected under `--expert_path/<env_id>/*.h5`.

## results and plotting

For instructions on downloading logged results and generating plots, see
[`src/results/README.md`](src/results/README.md).

## citation

If you use this work, please cite the [official paper](https://proceedings.mlr.press/v306/blonde26a.html):

```bibtex
@InProceedings{pmlr-v306-blonde26a,
  title = {Noise-Guided Transport: Imitation Learning from Random Priors},
  author = {Blond\'{e}, Lionel and Candido Ramos, Joao and Kalousis, Alexandros},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  pages = {8618--8643},
  year = {2026},
  editor = {Zhang, Tong and Dudik, Miroslav and Jaggi, Martin and Agarwal, Alekh and Li, Sharon and Schuurmans, Dale and Zhu, Jerry and Berkenkamp, Felix and Dong, Hanze and Bietti, Alberto},
  volume = {306},
  series = {Proceedings of Machine Learning Research},
  month = {06--11 Jul},
  publisher = {PMLR},
  pdf = {https://raw.githubusercontent.com/mlresearch/v306/main/assets/blonde26a/blonde26a.pdf},
  url = {https://proceedings.mlr.press/v306/blonde26a.html}
}
```
