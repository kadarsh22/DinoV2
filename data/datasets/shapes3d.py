import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class Shapes3d(Dataset):
    """
    Fast (slow-disk friendly): loads the full .npz into RAM at init.

    Expects .npz keys:
      images: (N,64,64,3) uint8
      labels: (N,6) latent values

    Returns:
      img: torch.FloatTensor [3,64,64] in [0,1] (then your transform can Normalize)
      factors: torch.LongTensor [6] (latent factor indices)
    """
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)  # load once
        self.images = np.array(data["images"], copy=True)      # (N,64,64,3) uint8 in RAM
        labels_val  = np.array(data["labels"], copy=True)      # (N,6)

        self.transform = transform

        # values -> indices
        factor_values = [np.unique(labels_val[:, i]) for i in range(labels_val.shape[1])]
        labels_idx = np.empty((labels_val.shape[0], labels_val.shape[1]), dtype=np.int64)
        for i, u in enumerate(factor_values):
            labels_idx[:, i] = np.searchsorted(u, labels_val[:, i])
        self.labels_idx = torch.from_numpy(labels_idx).long()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels_idx[index]
