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
    def __init__(self, model, unsupervised_loader, unsupervised_sampler,
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

        # ---- student DDP ----
        model = model.to(self.local_rank)
        self.student = model.student.to(self.local_rank)
        self.student = DDP(self.student, device_ids=[self.local_rank],
                           broadcast_buffers=False, find_unused_parameters=True)  # keep True while iBOT exists but is disabled)

        # ---- teacher frozen, not DDP ----
        self.teacher = model.teacher.to(self.local_rank)
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

        # Toggle flags (so it’s obvious)
        self.use_ibot = False
        self.use_koleo = False

    def train(self, epoch: int):
        num_batches = len(self.unsupervised_loader)
        total_steps = self.params["optimization"]["epochs"] * num_batches

        total_loss = torch.zeros((), device=self.local_rank)

        self.total_loss_meter.reset()
        self.dino_loss_meter.reset()
        self.ibot_loss_meter.reset()
        self.koleo_loss_meter.reset()

        if self.unsupervised_sampler is not None:
            self.unsupervised_sampler.set_epoch(epoch)

        self.student.train()
        self.teacher.eval()

        for batch_idx, batch in enumerate(self.unsupervised_loader):
            global_step = epoch * num_batches + batch_idx

            views = batch["views"]
            global_views = torch.cat(views[:2]).to(self.local_rank, non_blocking=True)
            local_views = torch.cat(views[2:]).to(self.local_rank, non_blocking=True)

            # ---- Masking ----
            B = global_views.shape[0]
            sequence_length = self.teacher.backbone.sequence_length
            mask = global_views.new_zeros((B, sequence_length), dtype=torch.bool)

            H, W = self.teacher.backbone.vit.patch_embed.grid_size
            assert H * W == sequence_length - 1, f"Unexpected grid size {H}x{W}, seq_len {sequence_length}"

            block_mask = random_block_mask(size=(B, H, W), device=mask.device)
            mask[:, 1:] = block_mask.flatten(start_dim=1)

            # ---- Teacher forward (ONCE) ----
            with torch.no_grad():
                t_cls, t_feats = self.teacher.forward_backbone(global_views)
                t_out = self.teacher.head.dino_head(t_cls)  # [2B, D]

                if self.use_ibot:
                    t_masked_out = self.teacher.head.ibot_head(t_feats[mask])

            all_views = torch.cat([global_views, local_views], dim=0)  # [(2+n_local)*B, C, H, W]

            seq_len = self.teacher.backbone.sequence_length
            mask_global = mask  # [2B, seq_len] (already built)
            mask_local = torch.zeros((local_views.shape[0], seq_len),
                                     dtype=torch.bool,
                                     device=mask.device)
            all_mask = torch.cat([mask_global, mask_local], dim=0)  # [(2+n_local)*B, seq_len]

            s_out = self.student(all_views, mask=all_mask)  # [(2+n_local)*B, D]

            # ---- Loss (DINO only for now) ----
            dino_loss = self.dino_criterion(
                teacher_out=t_out.chunk(2),
                student_out=s_out.chunk(len(views)),
            )
            loss = dino_loss

            # ---- Schedules: apply BEFORE optimizer.step() ----
            lr = float(self.lr_scheduler[global_step])
            wd = float(self.wd_scheduler[global_step])
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
                if pg.get("weight_decay", 0.0) != 0.0:
                    pg["weight_decay"] = wd

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # ---- Logging stats (global mean) ----
            loss_avg = ddp_mean(loss.detach())
            dino_avg = ddp_mean(dino_loss.detach())

            total_loss += loss_avg
            self.total_loss_meter.update(loss_avg.item())
            self.dino_loss_meter.update(dino_avg.item())

            # ---- EMA teacher update (match modules) ----
            momentum = cosine_schedule(
                step=global_step,
                max_steps=total_steps,
                start_value=self.params["optimization"]["mom_start"],
                end_value=self.params["optimization"]["mom_end"],
            )
            with torch.no_grad():
                s_mod = self.student.module
                update_momentum(s_mod.backbone, self.teacher.backbone, m=momentum)
                update_momentum(s_mod.dino_head, self.teacher.head.dino_head, m=momentum)
                if self.use_ibot:
                    update_momentum(s_mod.ibot_head, self.teacher.head.ibot_head, m=momentum)

            if self.global_rank == 0 and global_step % self.params["meta"]["log_every"] == 0:
                wandb.log(
                    {
                        "train/total_loss": self.total_loss_meter.avg,
                        "train/dino_loss": self.dino_loss_meter.avg,
                        "train/lr": lr,
                        "train/wd": wd,
                        "epoch": epoch,
                        "global_step": global_step,
                    }
                )

        avg_loss = total_loss / num_batches
        if self.global_rank == 0:
            print(f"epoch: {epoch:02d}, loss: {avg_loss.item():.5f}")
