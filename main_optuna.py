import argparse
import pprint
import os
import hydra
import torch
import wandb
from omegaconf import OmegaConf
from torch.distributed import destroy_process_group
from torchinfo import summary

from data.data_loader import build_dataloaders
from src.helper import build_optimizer, build_schedulers
from src.ssl_meta_arch import SSLMetaArch
from src.trainer import Trainer
from utils.config import setup
from utils.distributed import ddp_setup, is_main_process, set_seed_custom
from utils.utils import load_yaml
from utils.visualisation import plot_crops_from_dataset


def run_train(args, trial=None):
    ddp_setup()
    set_seed_custom(seed=args.seed)

    train_params = setup(args)
    train_params.train.dataset_path = args.dataset_path
    eval_params = load_yaml(args.eval_fname)

    # ---- Apply Optuna-sampled hyperparams into train_params (if trial provided)
    if trial is not None:
        # sample from your grids
        tau_start = trial.suggest_categorical("tau_start", [9.5, 8.5, 7.5, 6.0])
        tau_end = trial.suggest_categorical("tau_end", [1.5, 2.0, 2.5, 3.5])
        warmup_iters = trial.suggest_categorical("warmup_iters", [100, 500])
        loss_weight = trial.suggest_categorical("loss_weight", [2.0, 4.0])

        # optional continuous params
        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)

        # ---- IMPORTANT: map these to YOUR actual config fields
        # These are placeholders; adjust to match your OmegaConf structure.
        train_params.optim.lr = lr
        train_params.optim.weight_decay = wd

        # Example placeholders (edit these paths):
        # train_params.teacher.temp.tau_start = tau_start
        # train_params.teacher.temp.tau_end = tau_end
        # train_params.optim.warmup_iters = warmup_iters
        # train_params.loss.weight = loss_weight

        # If you don't know the exact paths, print the config once and grep:
        # print(OmegaConf.to_yaml(train_params))

    if is_main_process():
        pprint.pprint(eval_params)

    if is_main_process():
        wandb_cfg = OmegaConf.to_container(train_params, resolve=True)
        run_name = args.run_name if trial is None else f"{args.run_name}-t{trial.number}"
        wandb.init(
            project=train_params.wandb.project,
            name=run_name,
            config=wandb_cfg,
        )
        wandb.run.log_code('.')

    (unsupervised_loader, unsupervised_sampler,
    probe_train_loader, probe_eval_loader, attributes, normalisation_params) = build_dataloaders(args.dataset, train_params, eval_params)

    if is_main_process():
        fig = plot_crops_from_dataset(unsupervised_loader.dataset, n=10, upsample_locals_to=224,
                                      denorm_mean=normalisation_params['mean'], denorm_std=normalisation_params['std'])
        wandb.log({"Training samples": wandb.Image(fig)})

    model = SSLMetaArch(train_params).to(torch.device("cuda"))

    if is_main_process():
        summary(model.teacher.backbone,
        input_size=(1, 3, train_params.crops.global_crops_size, train_params.crops.global_crops_size), device="cuda")
        summary(model.teacher.dino_head, input_size=(1, model.embed_dim), device="cuda")
        summary(model.teacher.ibot_head, input_size=(1, model.embed_dim), device="cuda")

    model.prepare_for_distributed_training()

    optimizer = build_optimizer(train_params, model.get_params_groups())
    lr_schedule, wd_schedule, momentum_schedule,teacher_temp_schedule,last_layer_lr_schedule = build_schedulers(train_params)

    trainer = Trainer(model, unsupervised_loader, unsupervised_sampler, optimizer,
                      lr_schedule, wd_schedule, momentum_schedule,teacher_temp_schedule,last_layer_lr_schedule,
                      model.fp16_scaler, probe_train_loader, probe_eval_loader, attributes, train_params, eval_params)

    OFFICIAL_EPOCH_LENGTH = train_params.train.OFFICIAL_EPOCH_LENGTH
    max_iter = train_params.optim.epochs * OFFICIAL_EPOCH_LENGTH
    trainer.train(max_iter)

    destroy_process_group()
    if is_main_process():
        wandb.finish()

    if metric is None:
        raise RuntimeError("Trainer must expose a metric (e.g. best_probe_acc) to optimize.")
    return float(metric)


def optuna_objective(trial, args):
    return run_train(args, trial=trial)


@hydra.main(config_path="configs/", config_name="defaults", version_base=None)
def main(args):
    # Normal single run (no HPO)
    if not getattr(args, "hpo", False):
        run_train(args, trial=None)
        return

    # HPO mode
    if args.storage is None:
        raise ValueError("--storage is required for HPO (DB URL or shared journal file).")

    # Storage: DB URL or journal file path
    if args.storage.endswith(".log"):
        storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(args.storage))
    else:
        storage = args.storage

    study = optuna.create_study(
        study_name=getattr(args, "study_name", "ssl_hpo"),
        storage=storage,
        load_if_exists=True,
        direction=getattr(args, "direction", "maximize"),
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    # One trial per Slurm task
    study.optimize(lambda t: optuna_objective(t, args), n_trials=1)


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

    parser.add_argument("--hpo", action="store_true")
    parser.add_argument("--storage", type=str, default=None,
                        help="postgresql+psycopg2://... or /shared/optuna_journal.log")
    parser.add_argument("--study_name", type=str, default="ssl_hpo")
    parser.add_argument("--direction", type=str, default="maximize", choices=["maximize", "minimize"])
    parser.add_argument("--hpo_max_iter", type=int, default=None,
                        help="Optional: cap iterations per trial during HPO")

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

