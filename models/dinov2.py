from __future__ import annotations
import copy
from functools import partial
import torch
from torch import Tensor
from torch.nn import Module

from timm.models.vision_transformer import vit_tiny_patch4_28
from lightly.models.modules import DINOv2ProjectionHead, MaskedVisionTransformerTIMM
from lightly.models.utils import update_drop_path_rate


def freeze_eval_module(module: Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


class DINOv2(Module):
    def __init__(self, model_config, ibot_separate_head: bool = True) -> None:
        super().__init__()
        self.model_config = model_config

        vit = vit_tiny_patch4_28(pos_embed="learn", dynamic_img_size=False, init_values=1e-5)
        teacher_backbone = MaskedVisionTransformerTIMM(vit=vit, antialias=False, pos_embed_initialization="skip")
        student_backbone = copy.deepcopy(teacher_backbone)
        teacher_backbone.vit.reset_classifier(0)
        student_backbone.vit.reset_classifier(0)

        update_drop_path_rate(
            student_backbone.vit,
            drop_path_rate=self.model_config["drop_path_rate"],
            mode="uniform",
        )

        dino_head = partial(
            DINOv2ProjectionHead,
            input_dim=self.model_config["dino"]["input_dim"],
            hidden_dim=self.model_config["dino"]["hidden_dim"],
            output_dim=self.model_config["dino"]["output_dim"],
        )
        ibot_head = partial(
            DINOv2ProjectionHead,
            input_dim=self.model_config["dino"]["input_dim"],
            hidden_dim=self.model_config["dino"]["hidden_dim"],
            output_dim=self.model_config["dino"]["output_dim"],
        )

        teacher_dino = dino_head()
        student_dino = dino_head()

        teacher_ibot = ibot_head() if ibot_separate_head else teacher_dino
        student_ibot = ibot_head() if ibot_separate_head else student_dino

        teacher_head = DINOv2Head(teacher_dino, teacher_ibot)

        self.student = DINOv2Student(student_backbone, student_dino, student_ibot)
        self.teacher = DINOv2Teacher(teacher_backbone, teacher_head)


class DINOv2Head(Module):
    def __init__(self, dino_head: DINOv2ProjectionHead, ibot_head: DINOv2ProjectionHead) -> None:
        super().__init__()
        self.dino_head = dino_head
        self.ibot_head = ibot_head


class DINOv2Student(torch.nn.Module):
    def __init__(self, backbone, dino_head, ibot_head) -> None:
        super().__init__()
        self.backbone = backbone
        self.dino_head = dino_head
        self.ibot_head = ibot_head

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        feats = self.backbone.encode(x, mask=mask)
        cls = feats[:, 0]                      # [B, C]
        dino_out = self.dino_head(cls)         # [B, D]
        masked_tokens = feats[mask]
        ibot_out = self.ibot_head(masked_tokens)   # [N_masked, D] (or [0, D])
        return dino_out, ibot_out, cls


class DINOv2Teacher(Module):
    """Frozen, eval-only, EMA-updated."""
    def __init__(self, backbone: MaskedVisionTransformerTIMM, head: DINOv2Head) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        freeze_eval_module(self.backbone)
        freeze_eval_module(self.head)

    @torch.no_grad()
    def forward_backbone(self, x: Tensor):
        feats = self.backbone.encode(x)
        cls = feats[:, 0]
        return cls, feats





