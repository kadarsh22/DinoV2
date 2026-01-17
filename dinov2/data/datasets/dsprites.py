import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DSprites(Dataset):
    """
    dSprites dataset wrapper with factor-index labels.

    If drop_color=True, factor order becomes:
        0: shape
        1: scale
        2: orientation
        3: posX
        4: posY
    """

    def __init__(
        self,
        npz_path,
        transform=None,
        indices=None,
        drop_color=True,
        # --- NEW: orientation filtering ---
        nonrepetitive_orientation=False,
        same_orientations_across_shapes=True,
        # shape -> symmetry period in ORIENTATION BINS (for 40-bin dSprites)
        symmetry_bins_by_shape=None,
        # optional: balance dataset (cap examples per (shape, orientation))
        max_per_shape_orientation=None,
        seed=0,
    ):
        data = np.load(npz_path, allow_pickle=True)

        images = data["imgs"]  # (N, 64, 64) values {0,1}
        images = (images * 255).astype(np.uint8)
        self.images = images

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

        # Base indices (either provided or full dataset)
        if indices is None:
            base_indices = np.arange(len(self.images), dtype=np.int64)
        else:
            base_indices = np.asarray(indices, dtype=np.int64)

        # --- NEW: build filtered indices for non-repetitive orientations ---
        if nonrepetitive_orientation:
            # Default symmetry periods for dSprites (40 orientation bins):
            # (Assumes shape ids are [0,1,2] in the dataset order.)
            # If your shape-id mapping differs, pass symmetry_bins_by_shape explicitly.
            if symmetry_bins_by_shape is None:
                symmetry_bins_by_shape = {0: 10, 1: 20, 2: 40}  # square, ellipse, heart

            # factor positions after drop_color: shape=0, orientation=2
            shape_all = labels_idx_norm[base_indices, 0]
            ori_all = labels_idx_norm[base_indices, 2]

            if same_orientations_across_shapes:
                # Keep only orientations in the shared canonical set
                k = min(symmetry_bins_by_shape.values())  # -> 10
                allowed_oris = set(range(k))              # bins 0..9
                mask = np.isin(ori_all, list(allowed_oris))
                filtered = base_indices[mask]
            else:
                # Keep per-shape canonical ranges (square: 0..9, ellipse: 0..19, heart: 0..39)
                masks = []
                for s, period in symmetry_bins_by_shape.items():
                    m = (shape_all == s) & (ori_all < period)
                    masks.append(m)
                mask = np.logical_or.reduce(masks)
                filtered = base_indices[mask]

            # Optional balancing: cap #examples per (shape, orientation)
            if max_per_shape_orientation is not None:
                rng = np.random.default_rng(seed)
                # recompute factors on filtered
                shape_f = labels_idx_norm[filtered, 0]
                ori_f = labels_idx_norm[filtered, 2]

                keep = []
                combos = {}
                for idx, s, o in zip(filtered, shape_f, ori_f):
                    combos.setdefault((int(s), int(o)), []).append(int(idx))

                for key, idxs in combos.items():
                    if len(idxs) > max_per_shape_orientation:
                        idxs = rng.choice(idxs, size=max_per_shape_orientation, replace=False).tolist()
                    keep.extend(idxs)

                filtered = np.array(keep, dtype=np.int64)

            self.indices = filtered
        else:
            self.indices = base_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        img = Image.fromarray(self.images[idx], mode="L")
        y = self.labels_idx[idx]  # factor indices

        if self.transform is not None:
            img = self.transform(img)

        return img, y
