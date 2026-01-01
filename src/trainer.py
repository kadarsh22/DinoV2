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

        model = model.to(self.local_rank)
        self.student = model.student.to(self.local_rank)
        self.student = DDP(self.student, device_ids=[self.local_rank])

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


            # --------- Pack all views into ONE DDP forward ----------
            global_views = torch.cat(views[:2]).to(self.local_rank, non_blocking=True)  # [2B, C, H, W]
            local_views = torch.cat(views[2:]).to(self.local_rank, non_blocking=True)
            x_all = torch.cat([global_views, local_views], dim=0)

            B_global = global_views.shape[0]  # 2B
            seq_len = self.teacher.backbone.sequence_length

            # Build global mask (for iBOT) and extend to all views (locals get all-false)
            mask_global = torch.zeros((B_global, seq_len), dtype=torch.bool, device=self.local_rank)

            H, W = self.teacher.backbone.vit.patch_embed.grid_size
            assert H * W == seq_len - 1, f"Unexpected grid size {H}x{W}, seq_len {seq_len}"

            block_mask = random_block_mask(size=(B_global, H, W), device=mask_global.device)  # [2B,H,W]
            mask_global[:, 1:] = block_mask.flatten(1)

            mask_all = torch.zeros((x_all.shape[0], seq_len), dtype=torch.bool, device=self.local_rank)
            mask_all[:B_global] = mask_global  # locals remain all-false

            # --------- Teacher forward (globals only) ----------
            with torch.no_grad():
                t_cls, t_feats = self.teacher.forward_backbone(global_views)  # [2B,C], [2B,T,C]
                t_dino = self.teacher.head.dino_head(t_cls)  # [2B,D]
                t_ibot = self.teacher.head.ibot_head(t_feats[mask_global])  # [N_masked,D]

            # --------- Student forward (ONE CALL through DDP wrapper) ----------
            s_dino_all, s_ibot_masked, s_cls_all = self.student(x_all, mask=mask_all)

            # --------- Losses ----------
            dino_loss = self.dino_criterion(
                teacher_out=t_dino.chunk(2),
                student_out=s_dino_all.chunk(len(views)),
            )

            ibot_loss = self.ibot_criterion(teacher_out=t_ibot, student_out=s_ibot_masked, mask=block_mask,)

            s_cls_global = s_cls_all[:B_global]  # [2B,C]
            koleo_loss = self.params["optimization"]["koleo_weight"] * sum(self.koleo_criterion(t) for t in s_cls_global.chunk(2))

            loss = dino_loss + 0*ibot_loss + 0*koleo_loss

            # --------- Schedules BEFORE step ----------
            lr = float(self.lr_scheduler[global_step])
            wd = float(self.wd_scheduler[global_step])
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
                if pg.get("weight_decay", 0.0) != 0.0:
                    pg["weight_decay"] = wd

            # Optional last layer freeze (if you use that param_group convention)
            if epoch < 1:
                for pg in self.optimizer.param_groups:
                    if "last_layer" in pg:
                        pg["lr"] = 0.0

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # --------- Global-avg meters (DDP mean) ----------
            loss_avg = ddp_mean(loss.detach())
            dino_avg = ddp_mean(dino_loss.detach())
            ibot_avg = ddp_mean(ibot_loss.detach())
            koleo_avg = ddp_mean(koleo_loss.detach())

            total_loss += loss_avg
            self.total_loss_meter.update(loss_avg.item())
            self.dino_loss_meter.update(dino_avg.item())
            self.ibot_loss_meter.update(ibot_avg.item())
            self.koleo_loss_meter.update(koleo_avg.item())

            # --------- EMA teacher update (backbone + BOTH heads) ----------
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
                update_momentum(s_mod.ibot_head, self.teacher.head.ibot_head, m=momentum)

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

