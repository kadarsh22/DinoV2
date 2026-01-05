from typing import Any, Dict, Tuple
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from functools import partial
from torchvision.transforms import (
    Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda
)
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.data import random_split
from data.attributes import AttributeSpec
from data.loaders import make_data_loader, SamplerType
from data.augmentations import DataAugmentationDINO
from data.masking import MaskingGenerator
from data.collate import collate_data_and_cast
from utils.attribute_dataset import AttributeDataset


def _infer_dataset(dataset, cfg: Dict[str, Any]) -> Tuple[Dataset, Dataset, int, Any]:
    """
    Returns (train_dataset, val_dataset, num_classes, attributes)
    """

    if dataset in {"cifar10", "cifar-10"}:
        train = torchvision.datasets.CIFAR10(dataset_path, train=True, download=True)
        val = torchvision.datasets.CIFAR10(dataset_path, train=False, download=True)
        return train, val, 10, False

    if dataset in {"cmnist"}:
        normalisation_params = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
        transform_probe = Compose([Resize(cfg.crops.global_crops_size), CenterCrop(cfg.crops.global_crops_size), ToTensor(),
                                   Normalize(mean=normalisation_params['mean'], std=normalisation_params['std'])])

        train = AttributeDataset(root=cfg.train.dataset_path, split='train')
        val = AttributeDataset(root=cfg.train.dataset_path, split='valid', transform=transform_probe)
        attributes = [AttributeSpec("digit", num_classes=10, index=0), AttributeSpec("color", num_classes=10,
                                                                                     index=1)]
        return train, val, 10, attributes, normalisation_params

    if dataset in {"mnist"}:

        img_size = 224
        transform_probe = Compose([Resize(img_size), CenterCrop(img_size), ToTensor(),
                                   Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                                   Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        train = torchvision.datasets.MNIST(dataset_path, train=True, transform=transform_probe)
        val = torchvision.datasets.MNIST(dataset_path, train=False, transform=transform_probe)
        attributes = [AttributeSpec("digit", num_classes=10, index=0)]
        return train, val, 10, attributes


def build_dataloaders(dataset, train_params, eval_params):
    train_dataset, val_dataset, num_classes, attributes, normalisation_params = _infer_dataset(dataset, train_params)

    train_size = int(0.8 * len(val_dataset))
    test_size = int(0.2 * len(val_dataset))

    g = torch.Generator().manual_seed(0)
    probe_train_dataset, probe_test_dataset = random_split(val_dataset, [train_size, test_size],  generator=g)

    probe_train_sampler = DistributedSampler(probe_train_dataset, shuffle=True)
    probe_test_sampler = DistributedSampler(probe_test_dataset, shuffle=False)
    probe_train_loader = DataLoader(probe_train_dataset, batch_size=eval_params['probe']['batch_size'], drop_last=False,
                        sampler=probe_train_sampler, num_workers=eval_params['trainer']['num_workers'], pin_memory=True)
    probe_test_loader = DataLoader(probe_test_dataset, batch_size=eval_params['probe']['batch_size'], shuffle=False, drop_last=False,
                                   sampler=probe_test_sampler, num_workers=eval_params['trainer']['num_workers'], pin_memory=True, )

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


    sampler_type = SamplerType.SHARDED_INFINITE
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

    return train_loader, train_sampler, probe_train_loader, probe_test_loader, attributes, normalisation_params
