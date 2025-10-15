# Hierarchical Simulation-Based Inference Benchmark

This repository contains a simulation-based inference benchmark framework, `sbibm`, focused on **hierarchical** tasks for multi-level inference benchmarking. It builds on the original benchmark described in ["Benchmarking Simulation-based Inference"](http://proceedings.mlr.press/v130/lueckmann21a.html); the standard (non-hierarchical) tasks have been removed in this fork — every task is now a hierarchical variant.

The framework includes hierarchical tasks, reference-free metrics, plotting, and integrations with SBI toolboxes. It is designed to be highly extensible and easily used in new research projects as we show below.

If you have questions or comments, please [open an issue](https://github.com/giovannic/sbibm/issues). We [invite contributions](CONTRIBUTING.md), e.g., of new tasks, novel metrics, or wrappers for other SBI toolboxes.


## Installation

Assuming you have a working Python environment, install `sbibm` via `pip`:
```commandline
$ pip install sbibm
```

## Quickstart

```python
import sbibm

# Every task is hierarchical; n_l sets the number of local contexts
task = sbibm.get_task("hierarchical_two_moons", n_l=5)
prior = task.get_prior()
simulator = task.get_simulator()
observation = task.get_observation(num_observation=1)  # 10 per task

# These objects can be used for custom inference algorithms, e.g.
# we might want to generate simulations by sampling from prior:
thetas = prior(num_samples=10_000)
xs = simulator(thetas)

# Alternatively, import an existing algorithm:
from sbibm.algorithms import snpe
posterior_samples, _, _ = snpe(
    task=task, num_samples=10_000, num_observation=1, num_simulations=100_000
)

# Reference-free evaluation (no MCMC reference posterior needed):
from sbibm.metrics import reverse_kl, lc2st
rkl = reverse_kl(posterior_samples, task, num_observation=1)
```

## Tasks

All tasks are hierarchical and share the `hierarchical_` prefix. Hierarchical tasks feature parameters naturally organized into global (shared) and local (group-specific) levels, enabling benchmarking of algorithms on multi-site studies, pooling problems, and hierarchical inference scenarios.

Available tasks:
- `hierarchical_two_moons`
- `hierarchical_gaussian_mixture`
- `hierarchical_gaussian_linear`
- `hierarchical_gaussian_linear_uniform`
- `hierarchical_slcp`
- `hierarchical_sir`

Hierarchical tasks are parameterized by `n_l` (number of local contexts). For example:

```python
import sbibm

# Load hierarchical task with n_l=5 local contexts
task = sbibm.get_task("hierarchical_two_moons", n_l=5)

# Prior and simulator
prior = task.get_prior()
simulator = task.get_simulator()
observation = task.get_observation(num_observation=1)

# All SBI algorithms work seamlessly with hierarchical tasks
from sbibm.algorithms import snpe
posterior_samples, _, _ = snpe(
    task=task,
    num_samples=10_000,
    num_simulations=100_000,
    num_observation=1
)

## Running Benchmarks

Hierarchical benchmark experiments are configured with [Hydra](https://hydra.cc/). The single entry point is `scripts/run_benchmark.py`.

### Single run

```bash
python scripts/run_benchmark.py algorithm=snpe task=hierarchical_two_moons \
    num_simulations=1000 num_observation=1 device=cuda
```

### Experiment sweeps

Three pre-defined experiment configs run full sweeps using Hydra multirun (`-m`):

```bash
# Full benchmark: 6 tasks x 7 algorithms x 3 budgets x 10 observations
python scripts/run_benchmark.py +experiment=benchmark -m

# Ablation study for (Tokenised Flow Matching for Posterior Estimation): 6 tasks x 8 ablation variants x 3 budgets x 10 observations
python scripts/run_benchmark.py +experiment=ablation -m

# n_l scaling: 6 tasks x 3 algorithms x 5 n_l values x 10 observations
python scripts/run_benchmark.py +experiment=n_l_scaling -m
```

### HPC with SLURM

Each multirun job can be dispatched to SLURM via the [submitit launcher](https://hydra.cc/docs/plugins/submitit_launcher/):

```bash
python scripts/run_benchmark.py +experiment=benchmark hydra/launcher=submitit_slurm -m
```

### Configuration

Configs live under `conf/` with the following structure:

```
conf/
  config.yaml                  # top-level defaults
  algorithm/                   # one file per algorithm (snpe, fmpe, bottom_up, ...)
    ablation/                  # ablation variants (only used by bottom_up)
  task/                        # one file per hierarchical task
  experiment/                  # sweep definitions (benchmark, ablation, n_l_scaling)
```

Override any parameter from the CLI:

```bash
python scripts/run_benchmark.py algorithm=bottom_up algorithm/ablation=mlp \
    task=hierarchical_sir num_simulations=5000 seed=123
```

### Outputs

Results are saved in two locations:
- **Hydra output directory** (`outputs/<date>/<time>/`) — includes the full resolved config, logs, and CSV results
- **Aggregate directory** (`results_aggregated/`) — flat CSV collection compatible with the analysis scripts (`scripts/generate_ablation_table.py`, `scripts/plot_ablation.py`, etc.)

## Algorithms

`sbibm` wraps a number of third-party packages to run various algorithms. Each algorithm specifies a `run` function that gets `task` and hyperparameters as arguments, and eventually returns the required `num_posterior_samples`. Wrappers for external toolboxes live in `sbibm/algorithms` and currently cover `sbi`, `tfmpe`, `simformer`, `deepset`, and `pyro`/`pytorch`-based baselines.

## Metrics

Hierarchical tasks are evaluated primarily with **reference-free** metrics, which do not require expensive MCMC reference posteriors. The available metrics in `sbibm/metrics` are:

- `lc2st` — Likelihood Classifier Two-Sample Test (reference-free)
- `reverse_kl` — Reverse KL divergence (reference-free)

For example:
```python
from sbibm.metrics import reverse_kl
rkl = reverse_kl(posterior_samples, task, num_observation=1)
```

## Figures

`sbibm` includes code for plotting results. Analysis and plotting scripts under `scripts/` (e.g., `generate_ablation_table.py`, `plot_ablation.py`) consume the aggregate CSVs in `results_aggregated/`.

## License

MIT
