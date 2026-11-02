# Repository Guidelines

## Project Structure & Module Organization

- `main.py` launches self-supervised training and Optuna searches.
- `src/` contains the experiment trainer, student/teacher architecture, and optimizer helpers; `models/tau_scheduler.py` supplies adaptive threshold control.
- `dinov2/` contains the DINOv2 backbone, layers, losses, datasets, distributed utilities, and upstream evaluation code.
- `evals/` implements experiment-specific linear probes; `utils/` handles configuration, reproducibility, attributes, and visualization.
- `configs/train/` and `configs/eval/` hold dataset-specific YAML settings for ColoredMNIST, CelebA, dSprites, and Shapes3D. Shared defaults live in `dinov2/configs/ssl_default_config.yaml`.
- `slurm_scripts/dinov2/` contains cluster launchers. Datasets are external; `outputs/`, `wandb/`, and `slurm_outputs/` contain generated artifacts.

## Build, Test, and Development Commands

Run commands from the repository root in a CUDA-capable Python environment with the project's dependencies installed.

```bash
WANDB_MODE=offline torchrun --standalone --nproc_per_node=1 main.py \
  --config_file configs/train/cmnist/vits16.yaml \
  --eval_fname configs/eval/cmnist/eval_dinov2.yaml \
  --run_name cmnist_baseline --dataset cmnist \
  --dataset_path /path/to/colored_mnist --output_dir outputs/cmnist \
  --seed 0 lcal.loss_weight=0.0
```

This launches single-GPU training. Put `key=value` configuration overrides after all named arguments. Set the evaluation YAML's `meta.cache_dir` to a writable local path.

- `sbatch slurm_scripts/dinov2/mnist/aisurrey/train_dinov2.sub` submits a cluster experiment; adapt resource requests, environment activation, and absolute paths first.
- `python -m compileall -q main.py src models evals utils dinov2` checks Python syntax.

There is no package build configuration. `apptainer.def` requires an `environment.yml` that is currently absent; restore it before building the container.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, and `PascalCase` classes. Follow surrounding code and preserve upstream license headers. Use two-space YAML indentation and existing configuration keys. No formatter or linter is configured.

## Testing Guidelines

Tests use Python's built-in `unittest`: run `python -m unittest discover -s tests -v`. Name tests `tests/test_*.py`; no coverage threshold is configured. For training changes, run a reduced GPU experiment and check finite losses and linear-probe metrics. Record the seed, configuration, command, and hardware.

## Commit & Pull Request Guidelines

History uses brief, informal subjects without a fixed prefix convention. Prefer specific, imperative subjects such as `Fix Optuna worker seeding`. PRs should describe the change, affected datasets/configs, validation commands, and relevant metrics; link issues when applicable.

## Configuration & Artifacts

Supply credentials through environment variables. Keep datasets, checkpoints, caches, and run logs out of commits; `outputs/` is not currently ignored. Replace user-specific paths when sharing launch scripts.
