# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)


logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                # transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                # transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        # color_jittering = transforms.Compose(
        #     [
        #         transforms.RandomApply(
        #             [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
        #             p=0.8,
        #         ),
        #         transforms.RandomGrayscale(p=0.2),
        #     ]
        # )
        #
        # global_transfo1_extra = GaussianBlur(p=1.0)
        #
        # global_transfo2_extra = transforms.Compose(
        #     [
        #         GaussianBlur(p=0.1),
        #         transforms.RandomSolarize(threshold=128, p=0.2),
        #     ]
        # )
        #
        # local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        # self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        # self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        # self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

        self.global_transfo1 = transforms.Compose([self.normalize])
        self.global_transfo2 = transforms.Compose([self.normalize])
        self.local_transfo = transforms.Compose([self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output


def repeat_3_channels(x):
    # x: (1, H, W) -> (3, H, W)
    return x.repeat(3, 1, 1)


import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

# ---- helpers ----

def repeat_3_channels(x: torch.Tensor) -> torch.Tensor:
    # x: (1,H,W) -> (3,H,W)
    if x.ndim == 3 and x.shape[0] == 1:
        return x.repeat(3, 1, 1)
    if x.ndim == 2:
        return x.unsqueeze(0).repeat(3, 1, 1)
    return x

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, sigma=0.02, p=0.5):
        super().__init__()
        self.sigma = float(sigma)
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0 or random.random() > self.p:
            return x
        noise = torch.randn_like(x) * self.sigma
        return (x + noise).clamp(0.0, 1.0)

class SafeRandomTranslation(torch.nn.Module):
    """
    Small translation that won't change image size.
    max_frac: fraction of width/height to translate.
    """
    def __init__(self, max_frac=0.08, p=0.8):
        super().__init__()
        self.max_frac = float(max_frac)
        self.p = float(p)

    def forward(self, img: Image.Image) -> Image.Image:
        if self.p <= 0 or random.random() > self.p:
            return img
        w, h = img.size
        max_dx = int(round(w * self.max_frac))
        max_dy = int(round(h * self.max_frac))
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)
        # PIL affine expects (a,b,c,d,e,f) for:
        # x' = a*x + b*y + c
        # y' = d*x + e*y + f
        return img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, dx, 0, 1, dy),
            resample=Image.BILINEAR,
            fillcolor=0,
        )

def make_normalize_transform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    return transforms.Normalize(mean=mean, std=std)

# ---- main augmentation ----

class DataAugmentationDINODsprites:
    """
    DINO-style multi-crop augmentations tuned for dSprites:
    - Keep appearance simple (no color jitter)
    - Use *constrained* random crops/translation to create multi-view signal
    - Optional small rotation (off by default)
    """

    def __init__(
        self,
        global_crops_size=64,
        local_crops_size=32,
        local_crops_number=6,
        global_scale=(0.8, 1.0),
        local_scale=(0.5, 0.8),
        translation_p=0.8,
        translation_max_frac=0.08,
        noise_p=0.5,
        noise_sigma=0.02,
        rotation_p=0.0,
        rotation_degrees=15,
    ):
        self.global_crops_size = int(global_crops_size)
        self.local_crops_size = int(local_crops_size)
        self.local_crops_number = int(local_crops_number)

        # --- geometry (stochastic, but constrained) ---
        self.pre_geo = transforms.Compose(
            [
                SafeRandomTranslation(max_frac=translation_max_frac, p=translation_p),
                transforms.RandomApply(
                    [transforms.RandomRotation(degrees=rotation_degrees, fill=0)],
                    p=rotation_p,
                ),
            ]
        )

        # RandomResizedCrop creates the "two views of the same instance"
        self.global_geo = transforms.RandomResizedCrop(
            self.global_crops_size, scale=global_scale, ratio=(1.0, 1.0), interpolation=transforms.InterpolationMode.BILINEAR
        )
        self.local_geo = transforms.RandomResizedCrop(
            self.local_crops_size, scale=local_scale, ratio=(1.0, 1.0), interpolation=transforms.InterpolationMode.BILINEAR
        )

        # --- tensor / channel / normalize ---
        self.to_tensor_norm = transforms.Compose(
            [
                transforms.ToTensor(),               # (1,H,W) for grayscale PIL
                transforms.Lambda(repeat_3_channels), # -> (3,H,W)
                AddGaussianNoise(sigma=noise_sigma, p=noise_p),
                make_normalize_transform(),
            ]
        )

        # Student/teacher can be same for dsprites; keep separate hooks anyway
        self.global_transfo1 = transforms.Compose([self.to_tensor_norm])
        self.global_transfo2 = transforms.Compose([self.to_tensor_norm])
        self.local_transfo = transforms.Compose([self.to_tensor_norm])

    def __call__(self, image: Image.Image):
        """
        Returns dict matching common DINOv2-style pipelines:
          - global_crops: [g1, g2]
          - global_crops_teacher: [g1, g2]
          - local_crops: [l1...lK]
        """
        # 1) mild pre-geometry (translation / optional rotation)
        base = self.pre_geo(image)

        # 2) global crops (two different stochastic views)
        g1_pil = self.global_geo(base)
        g2_pil = self.global_geo(base)

        g1 = self.global_transfo1(g1_pil)
        g2 = self.global_transfo2(g2_pil)

        # 3) local crops (multiple stochastic smaller views)
        locals_ = []
        for _ in range(self.local_crops_number):
            l_pil = self.local_geo(base)
            locals_.append(self.local_transfo(l_pil))

        return {
            "global_crops": [g1, g2],
            "global_crops_teacher": [g1, g2],
            "local_crops": locals_,
            "offsets": (),  # keep for compatibility if your pipeline expects it
        }
