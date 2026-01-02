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
            B = views[0].shape[0]
            V = len(views)

            # ---- pack views ----
            global_views = torch.cat(views[:2], dim=0).to(self.local_rank, non_blocking=True)  # [2B,C,224,224]
            local_views = torch.cat(views[2:], dim=0).to(self.local_rank, non_blocking=True)
            x_all = torch.cat([global_views, local_views], dim=0)

            B_global = global_views.shape[0]  # 2B
            seq_len = self.teacher.backbone.sequence_length

            H, W = self.teacher.backbone.grid_size
            assert H * W == seq_len - 1

            # block_mask: [2B,H,W] with variable masked count per image (ok)
            block_mask = random_block_mask(size=(B_global, H, W), device=self.local_rank)


            # token mask for backbone: [2B,1+N] (CLS unmasked)
            mask_global_tokens = torch.zeros((B_global, seq_len), dtype=torch.bool, device=self.local_rank)
            mask_global_tokens[:, 1:] = block_mask.flatten(1)

            # extend to all views: locals have no masking
            mask_all_tokens = torch.zeros((x_all.shape[0], seq_len), dtype=torch.bool, device=self.local_rank)
            mask_all_tokens[:B_global] = mask_global_tokens

            # ---- teacher forward (globals only, NO mask token replacement) ----
            with torch.no_grad():
                # teacher uses mask=None (no replacement), but selects same masked positions via block_mask
                t_dino, t_ibot_masked, _ = self.teacher(global_views, mask=None, ibot_mask_hw=block_mask)

            # ---- student forward (ONE DDP call) ----
            # student uses mask_all_tokens for replacement, and uses block_mask only on globals for iBOT selection
            s_dino_all, s_ibot_masked, s_cls_all = self.student(
                x_all,
                mask=mask_all_tokens,
                ibot_mask_hw=torch.cat(
                    [block_mask, block_mask.new_zeros((x_all.shape[0] - B_global, H, W), dtype=torch.bool)], dim=0)
            )

            # But simplest: only compute iBOT for globals inside the model by passing globals separately:
            # (If you keep as above, just ensure ibot_mask_hw matches batch size.)

            # ---- DINO loss ----
            dino_loss = self.dino_criterion(
                teacher_out=t_dino.chunk(2),
                student_out=s_dino_all.chunk(V),
            )

            # ---- iBOT loss ----
            ibot_loss = self.ibot_criterion(
                teacher_out=t_ibot_masked,  # [2B*N_masked, D_out]
                student_out=s_ibot_masked,  # same
                mask=block_mask,  # [2B,H,W]
            )

            # ---- KoLeo loss (student global CLS feats only) ----
            s_cls_global = s_cls_all[:B_global]  # [2B,C]
            koleo_loss = 0.1 * sum(self.koleo_criterion(t) for t in s_cls_global.chunk(2))

            loss = dino_loss + 0 * ibot_loss + 0*koleo_loss

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
                for ps, pt in zip(self.student.module.parameters(), self.teacher.parameters()):
                    pt.data.mul_(momentum).add_(ps.data, alpha=(1.0 - momentum))

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

