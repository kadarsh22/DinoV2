import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"

import argparse
import json
import pprint
import hydra
import torch
import optuna
import wandb
from omegaconf import OmegaConf
from torch.distributed import destroy_process_group
from torchinfo import summary
import torch.distributed as dist

from dinov2.data.datasets.data_loader import build_dataloaders
from src.helper import build_optimizer, build_schedulers
from src.ssl_meta_arch import SSLMetaArch
from src.trainer import Trainer
from utils.config import setup, write_config
from utils.distributed import ddp_setup, is_main_process, set_seed_custom
from utils.utils import load_yaml
from utils.visualisation import plot_crops_from_dataset


@hydra.main(config_path="configs/", config_name="defaults", version_base=None)
def main(args):
    ddp_setup()
    try:
        tmp_params = setup(args)
        max_iter = tmp_params.optim.epochs * tmp_params.train.OFFICIAL_EPOCH_LENGTH

        def run_one(trial=None):
            set_seed_custom(seed=args.seed)
            train_params = setup(args)
            train_params.train.seed = int(args.seed)
            train_params.train.dataset_path = args.dataset_path
            eval_params = load_yaml(args.eval_fname)
            if args.eval_cache_dir is not None:
                eval_params["meta"]["cache_dir"] = args.eval_cache_dir
            if trial is not None:
                train_params.lcal.tau_target_hard_frac = trial.suggest_float("tau_target_hard_frac", 0.1, 0.7)
                train_params.lcal.loss_weight = trial.suggest_float("loss_weight", 1, 6,
                                                                    log=True)  # change key if different

            # Persist the effective configuration, including actual data paths.
            write_config(train_params, args.output_dir)
            if is_main_process():
                with open(os.path.join(args.output_dir, "eval_config.yaml"), "w") as handle:
                    OmegaConf.save(OmegaConf.create(eval_params), handle)

            if is_main_process():
                pprint.pprint(eval_params)

            if is_main_process():
                wandb_cfg = OmegaConf.to_container(train_params, resolve=True)
                run_name = args.run_name if trial is None else f"{args.run_name}-t{trial.number}"
                wandb.init(project=train_params.wandb.project,
                           entity=train_params.wandb.get("entity"),
                           name=run_name, config=wandb_cfg, dir=args.output_dir)
                wandb.run.log_code('.')

            (unsupervised_loader, unsupervised_sampler,
             probe_train_loader, probe_eval_loader, probe_train_sampler, probe_test_sampler, attributes, normalisation_params) = build_dataloaders(
                args.dataset, train_params, eval_params
            )

            if is_main_process():
                fig = plot_crops_from_dataset(
                    unsupervised_loader.dataset, n=10, upsample_locals_to=224,
                    denorm_mean=normalisation_params['mean'], denorm_std=normalisation_params['std']
                )
                wandb.log({"Training samples": wandb.Image(fig)})

            model = SSLMetaArch(train_params).to(torch.device("cuda"))
            if is_main_process():
                summary(model.teacher.backbone,
                        input_size=(1, 3, train_params.crops.global_crops_size, train_params.crops.global_crops_size),
                        device="cuda")

            model.prepare_for_distributed_training()

            optimizer = build_optimizer(train_params, model.get_params_groups())
            lr_schedule, wd_schedule, momentum_schedule, teacher_temp_schedule, last_layer_lr_schedule = build_schedulers(
                train_params)

            trainer = Trainer(
                model, unsupervised_loader, unsupervised_sampler, optimizer,
                lr_schedule, wd_schedule, momentum_schedule, teacher_temp_schedule, last_layer_lr_schedule,
                model.fp16_scaler, probe_train_loader, probe_eval_loader, probe_train_sampler, probe_test_sampler, attributes, train_params, eval_params
            )

            OFFICIAL_EPOCH_LENGTH = train_params.train.OFFICIAL_EPOCH_LENGTH
            max_iter = train_params.optim.epochs * OFFICIAL_EPOCH_LENGTH

            try:
                if trial is None:
                    trainer.train(max_iter)
                else:
                    trainer.train(max_iter, trial=trial, prune_after=args.prune_after)
            finally:
                if is_main_process():
                    wandb.finish()

            best = getattr(trainer, "best_eval_metric", None)
            if best is None:
                raise RuntimeError("Expose trainer.best_eval_metric (e.g. best eval/eval_attr_avg_top1) for Optuna.")
            if is_main_process():
                result = {
                    "run_name": run_name,
                    "seed": int(args.seed),
                    "dataset_path": args.dataset_path,
                    "metric": "eval/eval_attr_avg_top1",
                    "best_score": float(best),
                    "best_iteration": trainer.best_eval_iteration,
                    "best_metrics": trainer.best_eval_metrics,
                    "lcal": OmegaConf.to_container(train_params.lcal, resolve=True),
                    "training_iterations": max_iter,
                }
                temporary = os.path.join(args.output_dir, "result.json.tmp")
                with open(temporary, "w") as handle:
                    json.dump(result, handle, indent=2, allow_nan=False)
                os.replace(temporary, os.path.join(args.output_dir, "result.json"))
            return float(best)

        if not getattr(args, "hpo", False):
            run_one(trial=None)
            return

        if args.storage is None:
            raise ValueError("--storage required for HPO (postgres URL or shared journal file)")

        if args.storage.endswith(".log"):
            storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(args.storage))
        else:
            storage = args.storage

        worker_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        sampler_seed = int(args.seed) + 10_000 * worker_id  # big spacing avoids overlap

        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=sampler_seed, constant_liar=True),
            pruner=optuna.pruners.HyperbandPruner(min_resource=args.prune_after, max_resource=max_iter),
        )

        study.optimize(lambda trial: run_one(trial=trial), n_trials=1)

    finally:

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ---------------------------
    # Config overrides
    # ---------------------------
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        default=None,
        help=(
            "Modify config options at the end of the command.\n"
            '  • Yacs configs:   "PATH.KEY VALUE" pairs (space-separated)\n'
            '  • LazyConfig:     "path.key=value"\n'
        ),
    )

    parser.add_argument("--hpo", action="store_true", help="Run Optuna HPO mode")
    parser.add_argument("--storage", type=str, default=None,
                        help="postgresql+psycopg2://... OR /shared/optuna_journal.log")
    parser.add_argument("--study_name", type=str, default="ssl_hpo")
    parser.add_argument("--prune_after", type=int, default=70000,
                        help="Don’t allow pruning before this iteration")

    parser.add_argument(
        "--config_file",
        default="configs/train/cmnist/vits16.yaml",
        metavar="FILE",
        help="Path to training config file",
    )
    parser.add_argument(
        "--eval_fname",
        type=str,
        default="configs/eval/cmnist/eval_dinov2.yaml",
        help="Path to evaluation config file",
    )
    parser.add_argument("--eval_cache_dir", default=None,
                        help="Override probe cache directory; use a distinct path for each run")

    # ---------------------------
    # Run / logging
    # ---------------------------
    parser.add_argument("--run_name", type=str, required=True, help="W&B run name")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/fast/nobackup/scratch4weeks/ak03476/cmnist/dinov2_outputs",
        help="Output directory for logs/checkpoints",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # ---------------------------
    # Dataset
    # ---------------------------
    parser.add_argument("--dataset", type=str, default="cmnist", help="Dataset name")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt/fast/nobackup/users/ak03476/data/verfied_cmnist/uncorrelated/ColoredMNIST-Skewed0.9-Severity1",
        help="Path to the dataset root",
    )

    arguments = parser.parse_args()
    main(arguments)
