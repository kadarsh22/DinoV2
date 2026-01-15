import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DSprites(Dataset):
    """
    dSprites dataset wrapper with factor-index labels.

    Factors (default order in dSprites):
        0: color (always 1, ignored)
        1: shape
        2: scale
        3: orientation
        4: posX
        5: posY
    """

    def __init__(self, npz_path, transform=None, indices=None, drop_color=True):
        data = np.load(npz_path, allow_pickle=True)

        # Images: (N, 64, 64), values {0,1}
        images = data["imgs"]
        images = (images * 255).astype(np.uint8)
        self.images = images

        # Factor values: (N, 6)
        labels_val = data["latents_values"]
        labels_idx = data["latents_classes"]

        if drop_color:
            labels_val = labels_val[:, 1:]
            labels_idx = labels_idx[:, 1:]

        # Re-index factors so they are contiguous [0..K-1]
        factor_values = [np.unique(labels_val[:, i]) for i in range(labels_val.shape[1])]
        labels_idx_norm = np.empty_like(labels_idx, dtype=np.int64)

        for i, u in enumerate(factor_values):
            labels_idx_norm[:, i] = np.searchsorted(u, labels_val[:, i])

        self.labels_idx = torch.from_numpy(labels_idx_norm).long()
        self.transform = transform

        if indices is None:
            self.indices = None
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return self.images.shape[0] if self.indices is None else self.indices.shape[0]

    def __getitem__(self, index):
        base_idx = index if self.indices is None else int(self.indices[index])

        img = Image.fromarray(self.images[base_idx], mode="L")
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels_idx[base_idx]
