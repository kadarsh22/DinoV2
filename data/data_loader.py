from typing import Any, Dict, Tuple, Union
import torch
import torchvision
import torchvision.transforms as T
from functools import partial
from torch.utils.data import Subset

from torchvision.datasets import CIFAR10, MNIST, CelebA
from torchvision.transforms import (
    Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda
)
from data.datasets.shapes3d import Shapes3d, make_long_tailed_subset , latent_summary
from data.datasets.dsprites import DSprites
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.data import random_split
from data.attributes import AttributeSpec
from data.loaders import make_data_loader, SamplerType
from data.augmentations import DataAugmentationDINO
from data.masking import MaskingGenerator
from data.collate import collate_data_and_cast
from utils.attribute_dataset import AttributeDataset


def _infer_dataset(dataset, cfg: Dict[str, Any], long_tailed=False) -> Union[
    None, tuple[Subset[Any], Subset[Any], int, list[AttributeSpec], dict[str, list[float]]], tuple[
        AttributeDataset, AttributeDataset, int, list[AttributeSpec], dict[str, list[float]]], tuple[
        CIFAR10, CIFAR10, int, bool], tuple[CelebA, CelebA, int, list[AttributeSpec], dict[str, list[float]]], tuple[
        MNIST, MNIST, int, list[AttributeSpec], dict[str, list[float]]]]:
    """
    Returns (train_dataset, val_dataset, num_classes, attributes)
    """

    if dataset in {"cifar10", "cifar-10"}:
        train = torchvision.datasets.CIFAR10(cfg.train.dataset_path, train=True, download=True)
        val = torchvision.datasets.CIFAR10(cfg.train.dataset_path, train=False, download=True)
        return train, val, 10, False

    elif dataset in {"cmnist", "c_mnist"}:
        normalisation_params = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
        transform_probe = Compose(
            [Resize(cfg.crops.global_crops_size), CenterCrop(cfg.crops.global_crops_size), ToTensor(),
             Normalize(mean=normalisation_params['mean'], std=normalisation_params['std'])])

        train = AttributeDataset(root=cfg.train.dataset_path, split='train')
        val = AttributeDataset(root=cfg.train.dataset_path, split='valid', transform=transform_probe)
        attributes = [AttributeSpec("digit", num_classes=10, index=0), AttributeSpec("color", num_classes=10,
                                                                                     index=1)]
        return train, val, 10, attributes, normalisation_params

    elif dataset in {'3dshapes'}:
        normalisation_params = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

        transform_probe = Compose([
            Resize(cfg.crops.global_crops_size, antialias=True),
            CenterCrop(cfg.crops.global_crops_size),
            ToTensor(),
            Normalize(mean=normalisation_params["mean"], std=normalisation_params["std"]),
        ])

        transform_train = None

        dataset_train = Shapes3d(cfg.train.dataset_path, transform=transform_train)
        if long_tailed:
            labels_idx_np = dataset_train.labels_idx.cpu().numpy()
            subset_idx = make_long_tailed_subset(
                labels_idx_np=labels_idx_np,
                factor_id=4,
                total=100_000,
                imbalance_ratio=50,
                head_at_low_index=True, seed=42)

            dataset_train = Shapes3d(cfg.train.dataset_path, transform=transform_train, indices=subset_idx)

            factor_names = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]  # typical order
            summ = latent_summary(dataset_train, factor_names=factor_names)



        dataset_val = Shapes3d(cfg.train.dataset_path, transform=transform_probe)

        N = len(dataset_train)
        train_size = int(0.8 * N)

        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(N, generator=g)

        train_idx = perm[:train_size].tolist()
        val_idx = perm[train_size:].tolist()

        train = Subset(dataset_train, train_idx)
        val = Subset(dataset_val, val_idx)

        attributes = [
            AttributeSpec("floor_hue", num_classes=10, index=0),
            AttributeSpec("wall_hue", num_classes=10, index=1),
            AttributeSpec("object_hue", num_classes=10, index=2),
            AttributeSpec("scale", num_classes=8, index=3),
            AttributeSpec("shape", num_classes=4, index=4),
            AttributeSpec("orientation", num_classes=15, index=5),
        ]

        return train, val, 10, attributes, normalisation_params

    elif dataset in {'dsprites'}:
        normalisation_params = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

        transform_probe = Compose([
            Resize(cfg.crops.global_crops_size, antialias=True),
            CenterCrop(cfg.crops.global_crops_size),
            ToTensor(),
            Normalize(mean=normalisation_params["mean"], std=normalisation_params["std"]),
        ])

        transform_train = None

        dataset_train = DSprites(cfg.train.dataset_path, transform=transform_train)
        if long_tailed:
            labels_idx_np = dataset_train.labels_idx.cpu().numpy()
            subset_idx = make_long_tailed_subset(
                labels_idx_np=labels_idx_np,
                factor_id=4,
                total=100_000,
                imbalance_ratio=50,
                head_at_low_index=True, seed=42)

            dataset_train = Shapes3d(cfg.train.dataset_path, transform=transform_train, indices=subset_idx)

            factor_names = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]  # typical order
            summ = latent_summary(dataset_train, factor_names=factor_names)


        dataset_val = DSprites(cfg.train.dataset_path, transform=transform_probe)

        N = len(dataset_train)
        train_size = int(0.8 * N)

        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(N, generator=g)

        train_idx = perm[:train_size].tolist()
        val_idx = perm[train_size:].tolist()

        train = Subset(dataset_train, train_idx)
        val = Subset(dataset_val, val_idx)

        attributes = [
            AttributeSpec("shape", num_classes=3, index=0),
            AttributeSpec("scale", num_classes=6, index=1),
            AttributeSpec("orientation", num_classes=40, index=2),
            AttributeSpec("posX", num_classes=32, index=3),
            AttributeSpec("posY", num_classes=32, index=4),
        ]

        return train, val, 10, attributes, normalisation_params

    elif dataset in {'CelebA'}:
        normalisation_params = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
        transform_probe = T.Compose(
            [T.Resize(cfg.crops.global_crops_size), T.CenterCrop(cfg.crops.global_crops_size), T.ToTensor(),
             T.Normalize(mean=normalisation_params['mean'], std=normalisation_params['std']), ])
        target_transform = lambda attr: ((attr + 1) // 2).to(torch.uint8)  # 0/1

        train = CelebA(root=cfg.train.dataset_path, split="train", target_type="attr", download=False)
        val = CelebA(root=cfg.train.dataset_path, split="test", target_type="attr", download=False,
                     transform=transform_probe,
                     target_transform=target_transform)
        train.attr_names = train.attr_names[:40]
        val.attr_names = train.attr_names[:40]
        attributes = [AttributeSpec(name=str(a), num_classes=2, index=i)
                      for i, a in enumerate(val.attr_names)]
        return train, val, 2, attributes, normalisation_params

    elif dataset in {"mnist"}:
        normalisation_params = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
        transform_probe = Compose([Resize(cfg.crops.global_crops_size), CenterCrop(cfg.crops.global_crops_size), ToTensor(),
                                   Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                                   Normalize(mean=normalisation_params['mean'], std=normalisation_params['std'])])

        train = torchvision.datasets.MNIST(cfg.train.dataset_path, train=True, transform=transform_probe)
        val = torchvision.datasets.MNIST(cfg.train.dataset_path, train=False, transform=transform_probe)
        attributes = [AttributeSpec("digit", num_classes=10, index=0)]
        return train, val, 10, attributes, normalisation_params
    else:
        raise NotImplementedError


def build_dataloaders(dataset, train_params, eval_params):
    train_dataset, val_dataset, num_classes, attributes, normalisation_params = _infer_dataset(dataset, train_params)

    train_size = int(0.8 * len(val_dataset))
    test_size = len(val_dataset) - train_size

    g = torch.Generator().manual_seed(train_params.train.seed)
    probe_train_dataset, probe_test_dataset = random_split(val_dataset, [train_size, test_size], generator=g)

    probe_train_sampler = DistributedSampler(probe_train_dataset, shuffle=True)
    probe_test_sampler = DistributedSampler(probe_test_dataset, shuffle=False)
    probe_train_loader = DataLoader(probe_train_dataset, batch_size=eval_params['probe']['batch_size'], drop_last=False,
                                    sampler=probe_train_sampler, num_workers=eval_params['trainer']['num_workers'],
                                    pin_memory=True)
    probe_test_loader = DataLoader(probe_test_dataset, batch_size=eval_params['probe']['batch_size'], shuffle=False,
                                   drop_last=False,
                                   sampler=probe_test_sampler, num_workers=eval_params['trainer']['num_workers'],
                                   pin_memory=True, )

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

    if isinstance(train_dataset, Subset):
        train_dataset.dataset.transform = data_transform
        train_dataset.dataset.target_transform = lambda _: ()
    else:
        train_dataset.transform = data_transform
        train_dataset.target_transform = lambda _: ()

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
        seed=train_params.train.seed,
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return train_loader, train_sampler, probe_train_loader, probe_test_loader, attributes, normalisation_params
