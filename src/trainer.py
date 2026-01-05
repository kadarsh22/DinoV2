from __future__ import annotations
import os
import torch
import torch.distributed as dist
import dinov2.distributed as distributed
import wandb
import math
import copy
from dinov2.logging import MetricLogger
from utils.distributed import is_main_process
from src.utils.logging import AverageMeter, ddp_mean
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from evals.evaluator import Evaluator


class Trainer:
    def __init__(self, model, unsupervised_loader, unsupervised_sampler, optimizer,
                      lr_scheduler, wd_scheduler, momentum_schedule,teacher_temp_schedule,last_layer_lr_schedule,
                        fp16_scaler,
                        probe_train_loader, probe_eval_loader, attributes, train_params, eval_params) -> None:
        self.train_params = train_params
        self.eval_params = eval_params
        self.epochs_run = 0
        self.model = model
        self.fp16_scaler = fp16_scaler

        self.unsupervised_loader = unsupervised_loader
        self.unsupervised_sampler = unsupervised_sampler

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler
        self.momentum_schedule = momentum_schedule
        self.teacher_temp_schedule = teacher_temp_schedule
        self.last_layer_lr_schedule = last_layer_lr_schedule

        self.probe_train_loader = probe_train_loader
        self.probe_eval_loader = probe_eval_loader
        self.attributes = attributes

        self.total_loss_meter = AverageMeter()
        self.dino_loss_meter = AverageMeter()
        self.ibot_loss_meter = AverageMeter()
        self.koleo_loss_meter = AverageMeter()

        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.global_rank = int(os.environ.get("RANK", "0"))

        metrics_file = os.path.join(train_params.train.output_dir, "training_metrics.json")
        self.metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
        self.start_iter = 0
        self.iteration = self.start_iter


    def train(self, max_iter: int):
        header = "Training"
        for data in self.metric_logger.log_every( self.unsupervised_loader, 10, header, max_iter, self.start_iter,):
            current_batch_size = data["collated_global_crops"].shape[0] / 2
            if self.iteration > max_iter:
                return

            # apply schedules

            lr = self.lr_scheduler[self.iteration]
            wd = self.wd_scheduler[self.iteration]
            mom = self.momentum_schedule[self.iteration]
            teacher_temp = self.teacher_temp_schedule[self.iteration]
            last_layer_lr = self.last_layer_lr_schedule[self.iteration]
            self.apply_optim_scheduler(lr, wd, last_layer_lr)

            # compute losses

            self.optimizer.zero_grad(set_to_none=True)
            loss_dict = self.model.forward_backward(data, teacher_temp=teacher_temp)

            # clip gradients

            if self.fp16_scaler is not None:
                if self.train_params.optim.clip_grad:
                    self.fp16_scaler.unscale_(self.optimizer)
                    for v in self.model.student.values():
                        v.clip_grad_norm_(self.train_params.optim.clip_grad)
                self.fp16_scaler.step(self.optimizer)
                self.fp16_scaler.update()
            else:
                if self.train_params.optim.clip_grad:
                    for v in self.model.student.values():
                        v.clip_grad_norm_(self.train_params.optim.clip_grad)
                self.optimizer.step()

            # perform teacher EMA update

            self.model.update_teacher(mom)

            # logging

            if distributed.get_global_size() > 1:
                for v in loss_dict.values():
                    torch.distributed.all_reduce(v)
            loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

            if math.isnan(sum(loss_dict_reduced.values())):
                print("NaN detected")
                raise AssertionError
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            self.metric_logger.update(lr=lr)
            self.metric_logger.update(wd=wd)
            self.metric_logger.update(mom=mom)
            self.metric_logger.update(last_layer_lr=last_layer_lr)
            self.metric_logger.update(current_batch_size=current_batch_size)
            self.metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

            self.iteration = self.iteration + 1
            # --------- Logging ----------
            if self.global_rank == 0 and self.iteration % self.eval_params["meta"]["log_every"] == 0:
                wandb.log(
                    {
                        "train/total_loss": self.metric_logger.meters['total_loss'].avg,
                        "train/dino_local_crop_loss": self.metric_logger.meters['dino_local_crops_loss'].avg,
                        "train/ibot_loss": self.metric_logger.meters['ibot_loss'].avg,
                        "train/koleo_loss": self.metric_logger.meters['koleo_loss'].avg,
                        "train/lr": self.metric_logger.meters['lr'].value,
                        "train/wd": self.metric_logger.meters['wd'].value,
                        "iteration": self.iteration,
                    }
                )

            if self.iteration % self.eval_params['meta']['eval_every'] == 0:

                dist.barrier()

                backbone_fsdp = self.model.student.backbone  # or self.model.teacher.backbone
                backbone_fsdp.eval()

                evaluator = Evaluator(
                    backbone_fsdp,
                    self.probe_train_loader,
                    self.probe_eval_loader,
                    self.attributes,
                    self.eval_params,
                )
                eval_metrics = evaluator.train_and_eval_probe(self.iteration)
                eval_metrics["iteration"] = self.iteration
                if is_main_process():
                    wandb.log(eval_metrics)

                dist.barrier()

                backbone_fsdp.train()
                dist.barrier()

        # if (epoch + 1) % params['meta']['save_every'] == 0:
        #     trainer._save_snapshot(epoch)

    def apply_optim_scheduler(self, lr, wd, last_layer_lr):
        for param_group in self.optimizer.param_groups:
            is_last_layer = param_group["is_last_layer"]
            lr_multiplier = param_group["lr_multiplier"]
            wd_multiplier = param_group["wd_multiplier"]
            param_group["weight_decay"] = wd * wd_multiplier
            param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier
