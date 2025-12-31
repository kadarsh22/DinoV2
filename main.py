import argparse
import pprint
import copy
import os
import wandb
import torch.distributed as dist
from models.dinov2 import DINOv2
from data.data_loader import build_dataloaders
from src.trainer import Trainer
from evals.evaluator import Evaluator
from helper import init_optimizers
from utils.distributed import ddp_setup, set_seed_custom, is_main_process

from torch.distributed import destroy_process_group
from utils.utils import setup_logging, load_yaml


def main(args):
    ddp_setup()
    set_seed_custom(seed=args.seed)
    rank = dist.get_rank()
    logger = setup_logging(rank)

    params = load_yaml(args.train_fname)
    eval_params = load_yaml(args.eval_fname)
    params['meta']['snapshot_path'] = os.path.join(params['meta']['snapshot_path'], params['data']['dataset'], params['meta']['method_name'])

    if os.path.exists(params['meta']['snapshot_path']):
        raise FileExistsError(f"Folder already exists: {params['meta']['snapshot_path']}")
    combined_config = {"train": params, "eval": eval_params,}

    if rank == 0:
        pprint.pprint(params)
        pprint.pprint(eval_params)

    # if is_main_process():
    #     wandb.init(
    #         project="ijepa-cmnist",
    #         name=params['logging']['write_tag'] ,
    #              # +'_tau_start_'+str(args.tau_start)+'_tau_end_'+str(args.tau_end)+'_warmup_' +str(args.warmup_epochs),
    #         config=combined_config,
    #     )
    #     wandb.run.log_code('.')

    model = DINOv2(params['model'], params['data'])
    (unsupervised_loader, unsupervised_sampler,
    probe_train_loader, probe_eval_loader, attributes) = build_dataloaders(params['meta']['method_name'], params['data'])

    optimizer, scheduler, wd_scheduler = init_optimizers(model, params['optimization'], params['data']['batch_size'], unsupervised_loader)

    trainer = Trainer(model, unsupervised_loader, unsupervised_sampler, optimizer, scheduler, wd_scheduler, params)

    for epoch in range(trainer.epochs_run, params['optimization']['epochs']):
        trainer.train(epoch)

        if (epoch + 1) % params['meta']['eval_every'] == 0:

            dist.barrier()

            target_encoder = copy.deepcopy(trainer.target_encoder)

            opt_eval, scaler_eval, scheduler_eval, wd_scheduler_eval = init_opt_eval(classifier, eval_params['optimization']['lr'],
                                                                use_float16=eval_params['optimization']['use_float16'],
                                                                wd=eval_params['optimization']['weight_decay'], )
            evaluator = Evaluator(target_encoder, classifier, opt_eval, scaler_eval, scheduler_eval,
                                wd_scheduler_eval, val_loader, train_loader, lidar_val_loader, params, logger)

            evaluator.evaluate(epoch, eval_params['optimization']['num_epochs'], eval_linear_probe=True)
            del target_encoder, evaluator

            dist.barrier()

        if (epoch + 1) % params['meta']['save_every'] == 0:
            trainer._save_snapshot(epoch)

    destroy_process_group()
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fname', type=str, help='name of config file to load', default='configs/dinov2/mnist/train_dinov2.yaml')
    parser.add_argument('--eval_fname', type=str, help='name of config file to load', default='configs/dinov2/mnist/eval_dinov2.yaml')
    parser.add_argument('--seed', type=int, help='random seed', default=123)

    arguments = parser.parse_args()
    main(arguments)
