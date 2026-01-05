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


def _infer_dataset(dataset, dataset_path, cfg: Dict[str, Any]) -> Tuple[Dataset, Dataset, int, Any]:
    """
    Returns (train_dataset, val_dataset, num_classes, attributes)
    """

    if dataset in {"cifar10", "cifar-10"}:
        train = torchvision.datasets.CIFAR10(dataset_path, train=True, download=True)
        val = torchvision.datasets.CIFAR10(dataset_path, train=False, download=True)
        return train, val, 10, False

    if dataset in {"cmnist"}:
        img_size = 224
        transform_probe = Compose([Resize(img_size), CenterCrop(img_size), ToTensor(),
                                   Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                                   Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        train = AttributeDataset(root=dataset_path, split='train')
        val = AttributeDataset(root=dataset_path, split='valid', transform=transform_probe)
        attributes = [AttributeSpec("digit", num_classes=10, index=0), AttributeSpec("color", num_classes=10,
                                                                                     index=1)]
        return train, val, 10, attributes

    if dataset in {"mnist"}:

        img_size = 224
        transform_probe = Compose([Resize(img_size), CenterCrop(img_size), ToTensor(),
                                   Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                                   Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        train = torchvision.datasets.MNIST(dataset_path, train=True, transform=transform_probe)
        val = torchvision.datasets.MNIST(dataset_path, train=False, transform=transform_probe)
        attributes = [AttributeSpec("digit", num_classes=10, index=0)]
        return train, val, 10, attributes


def build_dataloaders(dataset, dataset_path, train_params, eval_params):
    train_dataset, val_dataset, num_classes, attributes = _infer_dataset(dataset, dataset_path, train_params)

    train_size = int(0.8 * len(val_dataset))
    test_size = int(0.2 * len(val_dataset))

    g = torch.Generator().manual_seed(123)
    probe_train_dataset, probe_test_dataset = random_split(val_dataset, [train_size, test_size],  generator=g)

    probe_train_loader = DataLoader(probe_train_dataset, batch_size=eval_params['probe']['batch_size'], shuffle=True, drop_last=False,
                                    num_workers=eval_params['trainer']['num_workers'], pin_memory=True)
    probe_test_loader = DataLoader(probe_test_dataset, batch_size=eval_params['probe']['batch_size'], shuffle=True, drop_last=False,
                                   num_workers=eval_params['trainer']['num_workers'], pin_memory=True, )
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    img_size = train_params.crops.global_crops_size
    patch_size = train_params.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    data_transform = DataAugmentationDINO(
        train_params.crops.global_crops_scale,
        train_params.crops.local_crops_scale,
        train_params.crops.local_crops_number,
        global_crops_size=train_params.crops.global_crops_size,
        local_crops_size=train_params.crops.local_crops_size,
    )

    setattr(train_dataset, "transform", data_transform)
    setattr(train_dataset, "target_transform", lambda _: ())

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=train_params.ibot.mask_ratio_min_max,
        mask_probability=train_params.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=torch.half,
    )


    sampler_type = SamplerType.DISTRIBUTED
    train_loader, train_sampler = make_data_loader(
        dataset=train_dataset,
        batch_size=train_params.train.batch_size_per_gpu,
        num_workers=train_params.train.num_workers,
        shuffle=True,
        seed=0,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return train_loader, train_sampler, probe_train_loader, probe_test_loader, attributes
