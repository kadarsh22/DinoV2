# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import torch
import torch.distributed as dist

from logging import getLogger
from torch.distributed import init_process_group
import numpy as np
import torch
import random

logger = getLogger()


def init_distributed(port=40112, rank_and_world_size=(None, None)):

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ['MASTER_ADDR'] = 'localhost'

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
        except Exception:
            logger.info('SLURM vars not set (distributed training not available)')
            world_size, rank = 1, 0
            return world_size, rank

    try:
        os.environ['MASTER_PORT'] = str(port)
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank)
    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f'distributed training not available {e}')

    return world_size, rank


def set_seed_custom(seed: int = 32) -> None:
    if dist.is_available() and dist.is_initialized():
        seed_tensor = torch.tensor(seed, device="cuda")
        dist.broadcast(seed_tensor, src=0)
        seed = seed_tensor.item()

    os.environ["PYTHONHASHSEED"] = str(seed)     # Set a fixed value for the hash seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if dist.get_rank() == 0:
        print(f"Random seed set as {seed}")

def ddp_setup():
    init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

def is_main_process() -> bool:
    """Check if current process is the main (rank 0) process."""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0