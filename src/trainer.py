from __future__ import annotations
import copy
from functools import partial
import torch
import torchvision
from timm.models.vision_transformer import vit_small_patch16_224
from torch import Tensor
from torch.nn import Module
from models.dinov2 import DINOv2
from torch.optim import AdamW
import os
from lightly.loss import DINOLoss, IBOTPatchLoss, KoLeoLoss
from lightly.models.modules import DINOv2ProjectionHead, MaskedVisionTransformerTIMM
from lightly.models.utils import (
    random_block_mask,
    update_drop_path_rate,
    update_momentum,
)
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule, linear_warmup_schedule


class Trainer:
    def __init__(self, model, unsupervised_loader, unsupervised_sampler, optimizer, scheduler, wd_scheduler, params) -> None:
        self.params = params
        self.epochs_run = 0
        self.unsupervised_loader = unsupervised_loader
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.dino_criterion = DINOLoss().to(self.local_rank)
        self.ibot_criterion = IBOTPatchLoss().to(self.local_rank)
        self.koleo_criterion = KoLeoLoss().to(self.local_rank)
        self.optimizer = optimizer

    def train(self, max_epochs):
        num_batches = len(self.unsupervised_loader)
        total_loss = 0
        for batch_idx, batch in enumerate(self.unsupervised_loader):
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

            # Calculate current global step based on epoch and batch index.
            global_step = epoch * num_batches + batch_idx

            # Calculate the loss.
            teacher_temp = linear_warmup_schedule(
                step=global_step,
                warmup_steps=int(30 / max_epochs * total_steps),
                start_value=0.04,
                end_value=0.07,
            )
            dino_loss = self.dino_criterion(teacher_out=teacher_cls_out.chunk(2),
                                            student_out=student_cls_out.chunk(len(views)), teacher_temp=teacher_temp,)
            ibot_loss = self.ibot_criterion(teacher_out=teacher_masked_out, student_out=student_global_masked_out,
                                            mask=block_mask, teacher_temp=teacher_temp,)
            koleo_loss = 0.1 * sum(
                self.koleo_criterion(t) for t in student_global_cls_token.chunk(2)
            )
            loss = dino_loss + ibot_loss + koleo_loss

            total_loss += loss.detach()
            loss.backward()

            # Optionally zero out the learning rate of the last layer.
            if epoch < 1:
                for param_group in self.optimizer.param_groups:
                    if "last_layer" in param_group:
                        param_group["lr"] = 0.0

            # Apply weight decay schedule.
            weight_decay = cosine_schedule(
                step=global_step,
                max_steps=total_steps,
                start_value=0.04,
                end_value=0.4,
            )

            # Update weight decay directly for all parameter groups.
            for group in self.optimizer.param_groups:
                if group["weight_decay"] != 0.0:
                    group["weight_decay"] = weight_decay

            self.optimizer.step()
            self.optimizer.zero_grad()

            # Momentum update teacher.
            momentum = cosine_schedule(
                step=global_step,
                max_steps=total_steps,
                start_value=0.996,
                end_value=1.0,
            )
            update_momentum(self.model.student_backbone, self.model.teacher_backbone, m=momentum)
            update_momentum(self.model.student_head, self.model.teacher_head, m=momentum)

        avg_loss = total_loss / len(self.unsupervised_loader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")