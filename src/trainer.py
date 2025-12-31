from __future__ import annotations
import torch
import os
import random
import wandb
import numpy as np
from src.utils.logging import AverageMeter
from lightly.loss import DINOLoss, IBOTPatchLoss, KoLeoLoss
from lightly.models.utils import (random_block_mask, update_momentum,)
from lightly.utils.scheduler import cosine_schedule


class Trainer:
    def __init__(self, model, unsupervised_loader, unsupervised_sampler, optimizer, lr_scheduler, wd_scheduler, params) -> None:
        self.params = params
        self.epochs_run = 0
        self.unsupervised_loader = unsupervised_loader
        self.unsupervised_sampler = unsupervised_sampler
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler
        self.total_loss_meter = AverageMeter()
        self.dino_loss_meter = AverageMeter()
        self.ibot_loss_meter = AverageMeter()
        self.koleo_loss_meter = AverageMeter()
        self.time_meter = AverageMeter()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.dino_criterion = DINOLoss(output_dim=self.params['model']['dino']['output_dim'],
                                warmup_teacher_temp = self.params['optimization']['temp_start'],
                                teacher_temp = self.params['optimization']['temp_end'],
                                warmup_teacher_temp_epochs = self.params['optimization']['temp_warmup_epochs']).to(self.local_rank)
        self.ibot_criterion = IBOTPatchLoss(output_dim=self.params['model']['dino']['output_dim'],
                                            teacher_temp = self.params['optimization']['temp_end']).to(self.local_rank)
        self.koleo_criterion = KoLeoLoss().to(self.local_rank)
        self.optimizer = optimizer

    def train(self, epoch):
        num_batches = len(self.unsupervised_loader)
        total_steps = self.params['optimization']['epochs'] * num_batches
        total_loss = 0
        self.total_loss_meter.reset()
        self.dino_loss_meter.reset()
        self.ibot_loss_meter.reset()
        self.koleo_loss_meter.reset()
        self.unsupervised_sampler.set_epoch(epoch)
        self.model.train()
        for batch_idx, batch in enumerate(self.unsupervised_loader):
            global_step = epoch * num_batches + batch_idx
            views, _ = batch['views'], batch['label']
            global_views = torch.cat(views[:2]).to(self.local_rank)
            local_views = torch.cat(views[2:]).to(self.local_rank)

            # Masking
            B = len(global_views)
            sequence_length = self.model.teacher_backbone.sequence_length
            mask = global_views.new_zeros((B, sequence_length), dtype=torch.bool)

            # Mask patches except class token.
            H, W = self.model.teacher_backbone.vit.patch_embed.grid_size
            assert (
                H * W == sequence_length - 1
            ), f"Unexpected grid size: {H}x{W}, sequence_length {sequence_length}"
            block_mask = random_block_mask(size=(B, H, W), device=mask.device)
            mask[:, 1:] = block_mask.flatten(start_dim=1)

            # Teacher forward
            with torch.no_grad():
                teacher_cls_token, teacher_features = self.model.forward_teacher(global_views)
                teacher_cls_out = self.model.teacher_head.dino_head.forward(teacher_cls_token)
                teacher_masked_out = self.model.teacher_head.ibot_head.forward(
                    teacher_features[mask]
                )

            # Student forward
            (
                student_global_cls_token,
                student_global_masked_features,
            ) = self.model.forward_student(global_views, mask=mask)
            student_global_cls_out = self.model.student_head.dino_head.forward(
                student_global_cls_token
            )
            student_global_masked_out = self.model.student_head.ibot_head.forward(
                student_global_masked_features
            )
            student_local_cls_token, _ = self.model.forward_student(local_views, mask=None)
            student_local_cls_out = self.model.student_head.dino_head.forward(
                student_local_cls_token
            )
            student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out])


            dino_loss = self.dino_criterion(teacher_out=teacher_cls_out.chunk(2),
                                            student_out=student_cls_out.chunk(len(views)))


            ibot_loss = self.ibot_criterion(teacher_out=teacher_masked_out, student_out=student_global_masked_out,
                                            mask=block_mask)
            koleo_loss = self.params['optimization']['koleo_weight']* sum(
                self.koleo_criterion(t) for t in student_global_cls_token.chunk(2)
            )
            loss = dino_loss + 0*ibot_loss + 0*koleo_loss

            total_loss += loss.detach()
            loss.backward()

            # Optionally zero out the learning rate of the last layer.
            if epoch < 1:
                for param_group in self.optimizer.param_groups:
                    if "last_layer" in param_group:
                        param_group["lr"] = 0.0

            # Update weight decay directly for all parameter groups.
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr_scheduler[global_step]
                if param_group["weight_decay"] != 0.0:
                    param_group["weight_decay"] = self.lr_scheduler[global_step]

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.total_loss_meter.update(loss)
            self.dino_loss_meter.update(dino_loss)
            self.ibot_loss_meter.update(ibot_loss)
            self.koleo_loss_meter.update(koleo_loss)

            # Momentum update teacher.
            momentum = cosine_schedule(step=global_step, max_steps=total_steps,
                                       start_value=self.params['optimization']['mom_start'],
                                       end_value=self.params['optimization']['mom_end'])
            update_momentum(self.model.student_backbone, self.model.teacher_backbone, m=momentum)
            update_momentum(self.model.student_head, self.model.teacher_head, m=momentum)

            if self.global_rank == 0 and global_step % self.params['meta']['log_every'] == 0:
                self._log_stats(self.total_loss_meter.avg,
                                self.dino_loss_meter.avg,
                                self.ibot_loss_meter.avg,
                                self.koleo_loss_meter.avg,
                                self.lr_scheduler[global_step],
                                self.wd_scheduler[global_step],
                                self.dino_criterion.teacher_temp_schedule[epoch],
                                epoch,
                                global_step
                                )

        avg_loss = total_loss / len(self.unsupervised_loader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

    @staticmethod
    def _log_stats(total_loss, dino_loss, ibot_loss, koleo_loss, lr, wd, teacher_temp, epoch, global_step,):
        wandb.log({"train/total_loss": total_loss,
                   "train/dino_loss": dino_loss,
                   "train/ibot_loss": ibot_loss,
                   "train/koleo_loss": koleo_loss,
                   "train/lr": lr,
                   "train/wd":wd,
                   "train/teacher_temp": teacher_temp,
                   "epoch": epoch,
                   "global_step": global_step})


    def _save_snapshot(self, epoch):
        if self.global_rank == 0:
            snapshot = {
                'encoder': self.encoder.module.state_dict(),
                'predictor': self.predictor.module.state_dict(),
                'target_encoder': self.target_encoder.state_dict(),
                'opt': self.optimizer.state_dict(),
                'scaler': None if self.scaler is None else self.scaler.state_dict(),
                'epoch': epoch,
                'global_step': self.global_step,
                'scheduler_step': self.scheduler._step,
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "python_random_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "wandb_run_id" : wandb.run.id
            }
            os.makedirs(self.snapshot_path, exist_ok=True)
            save_path = f"{self.snapshot_path}/train_epoch{epoch}.pt"
            torch.save(snapshot, save_path)
            print(f"Epoch {epoch} | Training snapshot saved at {save_path}")

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc, weights_only=False)
        self.encoder.load_state_dict(snapshot["encoder"])
        self.predictor.load_state_dict(snapshot["predictor"])
        state_dict = snapshot["target_encoder"]

        # Handle DDP / DataParallel checkpoints
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        msg = self.target_encoder.load_state_dict(state_dict)
        self.logger.info(f'loaded pretrained model with msg: {msg}')
        self.logger.info(f'loaded pretrained encoder from epoch: {snapshot["epoch"]}\n path: {snapshot_path}')
        self.optimizer.load_state_dict(snapshot["opt"])
        if snapshot["scaler"] is not None:
            self.scaler.load_state_dict(snapshot["scaler"])
        self.epochs_run = snapshot["epoch"] + 1
        self.scheduler.optimizer = self.optimizer
        self.wd_scheduler.optimizer = self.optimizer
        self.scheduler._step = snapshot["scheduler_step"]
        self.wd_scheduler._step = snapshot["scheduler_step"]
        self.global_step = snapshot["global_step"] + 1
        rng_state = snapshot.get("torch_rng_state", None)

        if rng_state is not None:
            if not isinstance(rng_state, torch.ByteTensor):
                rng_state = torch.tensor(rng_state, dtype=torch.uint8)

            rng_state = rng_state.cpu().type(torch.ByteTensor)
            torch.set_rng_state(rng_state)

        cuda_rng_state = snapshot.get("cuda_rng_state", None)
        if cuda_rng_state is not None and torch.cuda.is_available():
            fixed_states = []
            for s in cuda_rng_state:
                state_tensor = s.cpu()
                state_tensor = state_tensor.type(torch.ByteTensor)
                fixed_states.append(state_tensor)
            torch.cuda.set_rng_state_all(fixed_states)

        random.setstate(snapshot["python_random_state"])
        np.random.set_state(snapshot["numpy_rng_state"])
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")