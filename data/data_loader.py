from typing import Any, Dict, Tuple
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda
)
import torch.distributed as dist
from torch.utils.data import random_split
from data.attributes import AttributeSpec
from lightly.data.collate import IJEPAMaskCollator
from lightly.transforms.ijepa_transform import IJEPATransform
from lightly.transforms.dino_transform import DINOTransform
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


def build_dataloaders(method_name: str, data_config):
    train_dataset, val_dataset, num_classes, attributes = _infer_dataset(data_config)

    train_size = int(0.8 * len(val_dataset))
    test_size = int(0.2 * len(val_dataset))

    probe_train_dataset, probe_test_dataset = random_split(val_dataset, [train_size, test_size])

    probe_train_loader = DataLoader(probe_train_dataset, batch_size=data_config['batch_size'], shuffle=True, drop_last=False,
                                    num_workers=data_config['num_workers'], pin_memory=True)
    probe_test_loader = DataLoader(probe_test_dataset, batch_size=data_config['batch_size'], shuffle=True, drop_last=False,
                                   num_workers=data_config['num_workers'], pin_memory=True, )
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if method_name == "dinov2":
        dino_tf = DINOTransform(
            global_crop_size=data_config['global_crops_size'],
            local_crop_size=data_config['local_crops_size'],
            global_crop_scale=data_config['global_crop_scale'],
            local_crop_scale=data_config['local_crop_scale'],
            n_local_views=data_config['n_local_views'],
            normalize={"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
        )

        def train_transform(img):
            return dino_tf(img)

        train_ds = DictDataset(train_dataset, x_key="views", y_key="label", transform=train_transform)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_ds, num_replicas=world_size, shuffle=True, rank=rank)

        train_loader = DataLoader(train_ds, batch_size=data_config['batch_size'], drop_last=True, sampler=train_sampler,
                                  num_workers=data_config['num_workers'], pin_memory=True, )

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
