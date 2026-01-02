from typing import Any, Dict, Tuple
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from functools import partial
from torchvision.transforms import (
    Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda
)
import torch.distributed as dist
from torch.utils.data import random_split
from data.attributes import AttributeSpec
from lightly.data.collate import IJEPAMaskCollator
from lightly.transforms.ijepa_transform import IJEPATransform
from data.loaders import make_data_loader, SamplerType
from data.augmentations import DataAugmentationDINO
from data.masking import MaskingGenerator
from data.collate import collate_data_and_cast
from utils.attribute_dataset import AttributeDataset



class DictDataset(Dataset):
    def __init__(self, base: Dataset, x_key: str, y_key: str, transform=None, target_transform=None):
        self.base = base
        self.x_key = x_key
        self.y_key = y_key
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x, y = self.base[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return {self.x_key: x, self.y_key: y}


def _imagenet_norm():
    return Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def _infer_dataset(data_cfg: Dict[str, Any]) -> Tuple[Dataset, Dataset, int, Any]:
    """
    Returns (train_dataset, val_dataset, num_classes, attributes)
    """
    name = data_cfg['dataset'].lower()

    if name in {"cifar10", "cifar-10"}:
        train = torchvision.datasets.CIFAR10(data_cfg['root'], train=True, download=True)
        val = torchvision.datasets.CIFAR10(data_cfg['root'], train=False, download=True)
        return train, val, 10, False

    if name in {"cmnist"}:
        transform_probe = Compose([Resize(data_cfg['img_size']), CenterCrop(data_cfg['img_size']), ToTensor(),
                                   Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                                   Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        train = AttributeDataset(root=data_cfg['root'], split='train', transform=transform_probe)
        val = AttributeDataset(root=data_cfg['root'], split='valid', transform=transform_probe)
        attributes = [AttributeSpec("digit", num_classes=10, index=0), AttributeSpec("color", num_classes=10,
                                                                                     index=1)]
        return train, val, 10, attributes

    if name in {"mnist"}:
        transform_probe = Compose([Resize(data_cfg['img_size']), CenterCrop(data_cfg['img_size']), ToTensor(),
                                   Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                                   Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        train = torchvision.datasets.MNIST(data_cfg['root'], train=True, transform=transform_probe)
        val = torchvision.datasets.MNIST(data_cfg['root'], train=False, transform=transform_probe)
        attributes = [AttributeSpec("digit", num_classes=10, index=0)]
        return train, val, 10, attributes

    raise ValueError(
        f"Unknown dataset '{data_cfg['dataset']}'. Supported: cifar10, cifar100, mnist/cmnist"
    )


def build_dataloaders(method_name: str, data_config, eval_params):
    train_dataset, val_dataset, num_classes, attributes = _infer_dataset(data_config)

    train_size = int(0.8 * len(val_dataset))
    test_size = int(0.2 * len(val_dataset))

    g = torch.Generator().manual_seed(123)
    probe_train_dataset, probe_test_dataset = random_split(val_dataset, [train_size, test_size],  generator=g)

    probe_train_loader = DataLoader(probe_train_dataset, batch_size=eval_params['probe']['batch_size'], shuffle=True, drop_last=False,
                                    num_workers=data_config['num_workers'], pin_memory=True)
    probe_test_loader = DataLoader(probe_test_dataset, batch_size=eval_params['probe']['batch_size'], shuffle=True, drop_last=False,
                                   num_workers=data_config['num_workers'], pin_memory=True, )
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if method_name == "dinov2":
        img_size = cfg.crops.global_crops_size
        patch_size = cfg.student.patch_size
        n_tokens = (img_size // patch_size) ** 2
        mask_generator = MaskingGenerator(
            input_size=(img_size // patch_size, img_size // patch_size),
            max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
        )

        data_transform = DataAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
        )

        collate_fn = partial(
            collate_data_and_cast,
            mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
            mask_probability=cfg.ibot.mask_sample_probability,
            n_tokens=n_tokens,
            mask_generator=mask_generator,
            dtype=inputs_dtype,
        )

        train_ds = DictDataset(train_dataset, x_key="views", y_key="label", transform=data_transform)
        sampler_type = SamplerType.DISTRIBUTED
        train_loader, train_sampler = make_data_loader(
            dataset=train_ds,
            batch_size=cfg.train.batch_size_per_gpu,
            num_workers=cfg.train.num_workers,
            shuffle=True,
            seed=start_iter,  # TODO: Fix this -- cfg.train.seed
            sampler_type=sampler_type,
            sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
            drop_last=True,
            collate_fn=collate_fn,
        )


    elif method_name == "ijepa":
        train_transform = IJEPATransform(normalize={'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]})
        collator = IJEPAMaskCollator(input_size=(224, 224), patch_size=cfg.patch_size,
                                     enc_mask_scale=cfg.enc_mask_scale, pred_mask_scale=cfg.pred_mask_scale,
                                     aspect_ratio=cfg.aspect_ratio,
                                     nenc=cfg.num_enc_masks,
                                     npred=cfg.num_pred_masks,
                                     min_keep=cfg.min_keep,
                                     allow_overlap=False,
                                     )
        train_ds = DictDataset(train_dataset, x_key="views", y_key="label", transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_ds, collate_fn=collator,
                                                   batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                                                   persistent_workers=False)
    else:
        raise ValueError(f"Unknown method name '{method_name}'. Supported: dino, ijepa")

    return train_loader, train_sampler, probe_train_loader, probe_test_loader, attributes
