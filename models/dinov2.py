from __future__ import annotations

import copy
from functools import partial
from timm.models.vision_transformer import vit_tiny_patch4_28
from torch import Tensor
from torch.nn import Module

from lightly.models.modules import DINOv2ProjectionHead, MaskedVisionTransformerTIMM
from lightly.models.utils import update_drop_path_rate

def freeze_eval_module(module: Module) -> None:
    """Freeze the parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


class DINOv2Head(Module):
    def __init__(
            self, dino_head: DINOv2ProjectionHead, ibot_head: DINOv2ProjectionHead
    ) -> None:
        super().__init__()
        self.dino_head = dino_head
        self.ibot_head = ibot_head


class DINOv2(Module):
    def __init__(
            self,
            model_config,
            data_config,
            ibot_separate_head: bool = False,
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.data_config = data_config
        # Backbones
        vit_teacher = vit_tiny_patch4_28(
            pos_embed="learn",
            dynamic_img_size=False,
            init_values=1e-5,
        )
        self.teacher_backbone = MaskedVisionTransformerTIMM(
            vit=vit_teacher,
            antialias=False,
            pos_embed_initialization="skip",
        )
        self.student_backbone = copy.deepcopy(self.teacher_backbone)
        update_drop_path_rate(
            self.student_backbone.vit,
            drop_path_rate=self.model_config['drop_path_rate'],  # we recommend using smaller rates like 0.1 for vit-s-14
            mode="uniform",
        )

        freeze_eval_module(self.teacher_backbone)

        # Heads
        dino_head = partial(
            DINOv2ProjectionHead,
            input_dim=self.model_config['dino']['input_dim'],
            hidden_dim=self.model_config['dino']['hidden_dim'],
            output_dim=self.model_config['dino']['output_dim']

        )

        teacher_dino_head = dino_head()
        student_dino_head = dino_head()

        ibot_head = partial(
            DINOv2ProjectionHead,
            input_dim=self.model_config['dino']['input_dim'],
            hidden_dim=self.model_config['dino']['hidden_dim'],
            output_dim=self.model_config['dino']['output_dim']
        )

        if ibot_separate_head:
            teacher_ibot_head = ibot_head()
            student_ibot_head = ibot_head()
        else:
            teacher_ibot_head = teacher_dino_head
            student_ibot_head = student_dino_head

        self.teacher_head = DINOv2Head(
            dino_head=teacher_dino_head,
            ibot_head=teacher_ibot_head,
        )
        self.student_head = DINOv2Head(
            dino_head=student_dino_head,
            ibot_head=student_ibot_head,
        )

        freeze_eval_module(self.teacher_head)

    def forward(self, x: Tensor) -> Tensor:
        return self.teacher_backbone(x)

    def forward_teacher(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.teacher_backbone.encode(x)
        cls_tokens = features[:, 0]
        return cls_tokens, features

    def forward_student(
            self, x: Tensor, mask: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        features = self.student_backbone.encode(x, mask=mask)
        cls_tokens = features[:, 0]
        masked_features = None if mask is None else features[mask]
        return cls_tokens, masked_features