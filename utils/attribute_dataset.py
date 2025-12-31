import os
import torchvision.transforms as T
from torch.utils.data.dataset import Dataset
import numpy as np
import pickle
import torch


class AttributeDataset(Dataset):
    def __init__(self, root, split, transform=None):
        super(AttributeDataset, self).__init__()
        data_path = os.path.join(root, split, "images.npy")
        self.transform = transform
        self.data = np.load(data_path)
        self.data = [T.ToPILImage()(self.data[i]) for i in range(self.data.shape[0])]
        attr_path = os.path.join(root, split, "attrs.npy")
        self.attr = torch.LongTensor(np.load(attr_path))

        attr_names_path = os.path.join(root, "attr_names.pkl")
        with open(attr_names_path, "rb") as f:
            self.attr_names = pickle.load(f)

        self.num_attrs = self.attr.size(1)

    def __len__(self):
        return self.attr.size(0)

    def __getitem__(self, index):
        image, attr = self.data[index], self.attr[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, attr