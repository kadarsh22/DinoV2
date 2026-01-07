import argparse
import pprint

import torch
import torch.distributed as dist
import wandb
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



def main(args):
    ddp_setup()
    set_seed_custom(seed=args.seed)

    train_params = setup(args)
    train_params.train.dataset_path = args.dataset_path
    eval_params = load_yaml(args.eval_fname)
    combined_config = {"train": train_params, "eval": eval_params,}

    if is_main_process():
        pprint.pprint(train_params)
        pprint.pprint(eval_params)

    if is_main_process():
        wandb.init(
            project="ijepa-cmnist",
            name=args.run_name ,
                 # +'_tau_start_'+str(args.tau_start)+'_tau_end_'+str(args.tau_end)+'_warmup_' +str(args.warmup_epochs),
            config=combined_config,
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
        summary(model.teacher.backbone, input_size=(1, 3, 224, 224), device="cuda")
        summary(model.teacher.dino_head, input_size=(1, model.embed_dim), device="cuda")
        # summary(model.teacher.ibot_head, input_size=(1, model.embed_dim), device="cuda")

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

    parser.add_argument(
        "--config-file",
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
    parser.add_argument("--run_name", type=str, default="test", help="W&B run name")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/fast/nobackup/scratch4weeks/ak03476/cache/dinv2_outputs",
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

