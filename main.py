import argparse
import pprint
import torch
import wandb
import torch.distributed as dist
from data.data_loader import build_dataloaders
from src.trainer import Trainer
from src.helper import build_optimizer, build_schedulers
from src.ssl_meta_arch import SSLMetaArch
from utils.distributed import ddp_setup, set_seed_custom, is_main_process, get_world_size

from torch.distributed import destroy_process_group
from utils.utils import setup_logging, load_yaml
from utils.config import setup
from utils.visualisation import plot_crops_from_dataset


def main(args):
    ddp_setup()
    set_seed_custom(seed=args.seed)
    rank = dist.get_rank()
    train_params = setup(args)
    eval_params = load_yaml(args.eval_fname)
    combined_config = {"train": train_params, "eval": eval_params,}

    if rank == 0:
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
    probe_train_loader, probe_eval_loader, attributes) = build_dataloaders(args.dataset, args.dataset_path,  train_params, eval_params)

    fig = plot_crops_from_dataset(unsupervised_loader.dataset, n=10, upsample_locals_to=224)
    if is_main_process():
        wandb.log({"Training samples": wandb.Image(fig)})

    model = SSLMetaArch(train_params).to(torch.device("cuda"))
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='name of config file to load',
                        default='/mnt/fast/nobackup/scratch4weeks/ak03476/dino_outputs')
    parser.add_argument("opts",
                        help=""" Modify config options at the end of the command. 
                        For Yacs configs, use space-separated "PATH.KEY VALUE" pairs.
                        For python-based LazyConfig, use "path.key=value". """.strip(),
                        default=None, nargs=argparse.REMAINDER,)
    parser.add_argument("--config-file", default="configs/train/vits16.yaml", metavar="FILE",
                        help="path to config file")

    parser.add_argument('--run_name', type=str, help='wandb run name',
                        default='test')

    parser.add_argument('--dataset', type=str, help='wandb run name', default='cmnist')
    parser.add_argument('--dataset_path', type=str, help='path to the dataset',
                        default='/mnt/fast/nobackup/users/ak03476/data/ColoredMNIST-Skewed0.9-Severity1')


    parser.add_argument('--eval_fname', type=str, help='name of config file to load', default='configs/dinov2/mnist/eval_dinov2.yaml')
    parser.add_argument('--seed', type=int, help='random seed', default=123)

    parser.add_argument('--temp_start', type=float, help='starting temperature', default=0.9)
    parser.add_argument('--temp_end', type=float, help='ending temperature', default=0.01)
    parser.add_argument('--temp_warmup_epochs', type=int, help='number of warmup epochs for temperature schedule', default=1)

    parser.add_argument('--global_crop_scale', type=float, nargs=2, metavar=('MIN', 'MAX'),
                        help='min/max scale for global crops', default=[0.4, 1.0])
    parser.add_argument('--local_crop_scale', type=float, nargs=2, metavar=('MIN', 'MAX'),
                        help='min/max scale for local crops', default=[0.05, 0.4])

    parser.add_argument('--lr', type=float, help='learning rate', default=0.0005)
    parser.add_argument('--warmup_epochs', type=int, help='number of warmup epochs for learning rate schedule', default=10)


    arguments = parser.parse_args()
    main(arguments)
