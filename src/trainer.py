from __future__ import annotations
import os
import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils.logging import AverageMeter, ddp_mean
from lightly.loss import DINOLoss, IBOTPatchLoss, KoLeoLoss
from lightly.models.utils import random_block_mask, update_momentum
from lightly.utils.scheduler import cosine_schedule


@torch.no_grad()
def checksum(module):
    s = torch.zeros((), device=torch.cuda.current_device())
    for p in module.parameters():
        if p.requires_grad:
            s += p.detach().float().sum()
    return s


class Trainer:
    def __init__(self, student_model, teacher_model, unsupervised_loader, unsupervised_sampler,
                 optimizer, lr_scheduler, wd_scheduler, params) -> None:
        self.params = params
        self.epochs_run = 0

        self.unsupervised_loader = unsupervised_loader
        self.unsupervised_sampler = unsupervised_sampler

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler

        self.total_loss_meter = AverageMeter()
        self.dino_loss_meter = AverageMeter()
        self.ibot_loss_meter = AverageMeter()
        self.koleo_loss_meter = AverageMeter()

        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.global_rank = int(os.environ.get("RANK", "0"))

        self.student = student_model.to(self.local_rank)
        self.student = DDP(self.student, device_ids=[self.local_rank])

        self.teacher = teacher_model.to(self.local_rank)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.dino_criterion = DINOLoss(
            output_dim=self.params["model"]["dino"]["output_dim"],
            warmup_teacher_temp=self.params["optimization"]["temp_start"],
            teacher_temp=self.params["optimization"]["temp_end"],
            warmup_teacher_temp_epochs=self.params["optimization"]["temp_warmup_epochs"],
        ).to(self.local_rank)

        self.ibot_criterion = IBOTPatchLoss(
            output_dim=self.params["model"]["dino"]["output_dim"],
            teacher_temp=self.params["optimization"]["temp_end"],
        ).to(self.local_rank)

        self.koleo_criterion = KoLeoLoss().to(self.local_rank)

    def train(self, epoch: int):
        for data in metric_logger.log_every(
                data_loader,
                10,
                header,
                max_iter,
                start_iter,
        ):
            current_batch_size = data["collated_global_crops"].shape[0] / 2
            if iteration > max_iter:
                return

            # apply schedules

            lr = lr_schedule[iteration]
            wd = wd_schedule[iteration]
            mom = momentum_schedule[iteration]
            teacher_temp = teacher_temp_schedule[iteration]
            last_layer_lr = last_layer_lr_schedule[iteration]
            apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

            # compute losses

            optimizer.zero_grad(set_to_none=True)
            loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

            # clip gradients

            if fp16_scaler is not None:
                if cfg.optim.clip_grad:
                    fp16_scaler.unscale_(optimizer)
                    for v in model.student.values():
                        v.clip_grad_norm_(cfg.optim.clip_grad)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            else:
                if cfg.optim.clip_grad:
                    for v in model.student.values():
                        v.clip_grad_norm_(cfg.optim.clip_grad)
                optimizer.step()

            # perform teacher EMA update

            model.update_teacher(mom)

            # logging

            if distributed.get_global_size() > 1:
                for v in loss_dict.values():
                    torch.distributed.all_reduce(v)
            loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

            if math.isnan(sum(loss_dict_reduced.values())):
                logger.info("NaN detected")
                raise AssertionError
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            metric_logger.update(lr=lr)
            metric_logger.update(wd=wd)
            metric_logger.update(mom=mom)
            metric_logger.update(last_layer_lr=last_layer_lr)
            metric_logger.update(current_batch_size=current_batch_size)
            metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)


            # --------- Logging ----------
            if self.global_rank == 0 and global_step % self.params["meta"]["log_every"] == 0:
                wandb.log(
                    {
                        "train/total_loss": self.total_loss_meter.avg,
                        "train/dino_loss": self.dino_loss_meter.avg,
                        "train/ibot_loss": self.ibot_loss_meter.avg,
                        "train/koleo_loss": self.koleo_loss_meter.avg,
                        "train/lr": lr,
                        "train/wd": wd,
                        "epoch": epoch,
                        "global_step": global_step,
                    }
                )

        avg_loss = total_loss / num_batches
        if self.global_rank == 0:
            print(f"epoch: {epoch:02d}, loss: {avg_loss.item():.5f}")

