import argparse
import pprint
import copy
import os
import sys
import wandb
import torch
import torch.distributed as dist
from models.dinov2 import DINOv2
from data.data_loader import build_dataloaders
from src.trainer import Trainer
from evals.evaluator import Evaluator
from helper import init_optimizers
from utils.distributed import ddp_setup, set_seed_custom, is_main_process, get_world_size

from torch.distributed import destroy_process_group
from utils.utils import setup_logging, load_yaml


def main(args):
    ddp_setup()
    set_seed_custom(seed=args.seed)
    rank = dist.get_rank()
    logger = setup_logging(rank)

    params = load_yaml(args.train_fname)
    eval_params = load_yaml(args.eval_fname)

    params['data']['global_crop_scale'] = args.global_crop_scale
    params['data']['local_crop_scale'] = args.local_crop_scale
    params['optimization']['lr'] = args.lr
    params['optimization']['temp_start'] = args.temp_start
    params['optimization']['temp_end'] = args.temp_end
    params['optimization']['temp_warmup_epochs'] = args.temp_warmup_epochs
    params['optimization']['warmup_epochs'] = args.warmup_epochs

    params['meta']['snapshot_path'] = os.path.join(params['meta']['snapshot_path'], params['data']['dataset'], params['meta']['method_name'])

    if os.path.exists(params['meta']['snapshot_path']):
        raise FileExistsError(f"Folder already exists: {params['meta']['snapshot_path']}")
    combined_config = {"train": params, "eval": eval_params,}

    if rank == 0:
        pprint.pprint(params)
        pprint.pprint(eval_params)

    if is_main_process():
        wandb.init(
            project="ijepa-cmnist",
            name=params['tag'] ,
                 # +'_tau_start_'+str(args.tau_start)+'_tau_end_'+str(args.tau_end)+'_warmup_' +str(args.warmup_epochs),
            config=combined_config,
        )
        wandb.run.log_code('.')

    model = DINOv2(params['model'], params['data']).to(int(os.environ["LOCAL_RANK"]))

    (unsupervised_loader, unsupervised_sampler,
    probe_train_loader, probe_eval_loader, attributes) = build_dataloaders(params['meta']['method_name'], params['data'], eval_params)

    num_gpus = get_world_size()
    optimizer, lr_scheduler, wd_scheduler = init_optimizers(model, params['optimization'], 
                                                    params['data']['batch_size_per_gpu']*num_gpus,
                                                     unsupervised_loader)

    trainer = Trainer(model, unsupervised_loader, unsupervised_sampler, optimizer, lr_scheduler, wd_scheduler, params)

    for epoch in range(trainer.epochs_run, params['optimization']['epochs']):
        trainer.train(epoch)

        if (epoch + 1) % params['meta']['eval_every'] == 0:

            dist.barrier()

            if dist.get_rank() == 0:

                target_encoder = copy.deepcopy(trainer.student.module.backbone)

                evaluator = Evaluator(target_encoder, probe_train_loader, probe_eval_loader, attributes, eval_params, logger)

                eval_metrics = evaluator.train_and_eval_probe(epoch)
                eval_metrics['epoch'] = epoch
                del target_encoder, evaluator
                wandb.log(eval_metrics)

            dist.barrier()
        
        if epoch > 11:
            sys.exit()

        # if (epoch + 1) % params['meta']['save_every'] == 0:
        #     trainer._save_snapshot(epoch)

    destroy_process_group()
    if is_main_process():
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fname', type=str, help='name of config file to load', default='configs/dinov2/mnist/train_dinov2.yaml')
    parser.add_argument('--eval_fname', type=str, help='name of config file to load', default='configs/dinov2/mnist/eval_dinov2.yaml')
    parser.add_argument('--seed', type=int, help='random seed', default=123)

    parser.add_argument('--temp_start', type=float, help='starting temperature', default=0.9)
    parser.add_argument('--temp_end', type=float, help='ending temperature', default=0.01)
    parser.add_argument('--temp_warmup_epochs', type=int, help='number of warmup epochs for temperature schedule', default=1)

    parser.add_argument('--global_crop_scale', type=float, nargs=2, metavar=('MIN', 'MAX'),
                        help='min/max scale for global crops', default=[0.4, 1.0])
    parser.add_argument('--local_crop_scale', type=float, nargs=2, metavar=('MIN', 'MAX'),
                        help='min/max scale for local crops', default=[0.05, 0.4])

    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('--warmup_epochs', type=int, help='number of warmup epochs for learning rate schedule', default=10)


    arguments = parser.parse_args()
    main(arguments)
