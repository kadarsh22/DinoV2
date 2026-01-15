import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class Shapes3d(Dataset):
    def __init__(self, npz_path, transform=None, indices=None):
        data = np.load(npz_path)
        self.images = np.array(data["images"], copy=True)
        labels_val  = np.array(data["labels"], copy=True)
        self.transform = transform

        factor_values = [np.unique(labels_val[:, i]) for i in range(labels_val.shape[1])]
        labels_idx = np.empty((labels_val.shape[0], labels_val.shape[1]), dtype=np.int64)
        for i, u in enumerate(factor_values):
            labels_idx[:, i] = np.searchsorted(u, labels_val[:, i])

        self.labels_idx = torch.from_numpy(labels_idx).long()

        if indices is None:
            self.indices = None
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return self.images.shape[0] if self.indices is None else self.indices.shape[0]

    def __getitem__(self, index):
        base_idx = index if self.indices is None else int(self.indices[index])

        img = Image.fromarray(self.images[base_idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels_idx[base_idx]


def exponential_counts(num_values: int, total: int, alpha: float = None, imbalance_ratio: float = None):
    """
    Returns integer counts per value (length=num_values) summing to total.
    Either specify alpha, or specify imbalance_ratio = head_count / tail_count.
    """
    if alpha is None:
        if imbalance_ratio is None:
            raise ValueError("Provide either alpha or imbalance_ratio.")
        # head/tail = exp(alpha*(K-1))  => alpha = ln(r)/(K-1)
        alpha = np.log(float(imbalance_ratio)) / max(1, (num_values - 1))

    ranks = np.arange(num_values, dtype=np.float64)
    w = np.exp(-alpha * ranks)
    w = w / w.sum()

    raw = w * total
    counts = np.floor(raw).astype(int)
    # fix rounding so sum == total
    remainder = total - counts.sum()
    if remainder > 0:
        frac = raw - np.floor(raw)
        add_idx = np.argsort(-frac)[:remainder]
        counts[add_idx] += 1
    return counts


def make_long_tailed_subset(labels_idx_np: np.ndarray,
                            factor_id: int,
                            total: int,
                            alpha: float = None,
                            imbalance_ratio: float = None,
                            head_at_low_index: bool = True,
                            seed: int = 0):
    """
    labels_idx_np: (N,6) int64 numpy array (factor indices)
    Returns: subset_indices (total,) int64 numpy array
    """
    rng = np.random.default_rng(seed)
    factor_vals = labels_idx_np[:, factor_id]
    num_values = int(factor_vals.max()) + 1

    counts = exponential_counts(num_values=num_values, total=total, alpha=alpha, imbalance_ratio=imbalance_ratio)

    # Optionally put the "head" on the *highest* index instead of lowest
    if not head_at_low_index:
        counts = counts[::-1]

    subset = []
    for v in range(num_values):
        idxs = np.where(factor_vals == v)[0]
        if len(idxs) == 0:
            continue
        take = min(counts[v], len(idxs))
        chosen = rng.choice(idxs, size=take, replace=False)
        subset.append(chosen)

    subset = np.concatenate(subset) if len(subset) else np.array([], dtype=np.int64)
    rng.shuffle(subset)

    # If some classes capped (shouldn't happen for Shapes3D unless total is huge), pad randomly from remaining
    if subset.shape[0] < total:
        remaining = np.setdiff1d(np.arange(labels_idx_np.shape[0]), subset, assume_unique=False)
        pad = rng.choice(remaining, size=(total - subset.shape[0]), replace=False)
        subset = np.concatenate([subset, pad])
        rng.shuffle(subset)

    return subset.astype(np.int64)



from fractions import Fraction
import numpy as np
import torch

def latent_summary(dataset, factor_names=None, topk=None, simplify=True, max_denominator=None):
    """
    Summarize marginal distributions of each latent factor in a Shapes3d dataset,
    respecting dataset.indices if present.

    Prints fractions (count/total) instead of absolute counts.
    """
    # get labels as numpy (N,d) for the FULL dataset
    labels = dataset.labels_idx
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    # restrict to subset if indices exist
    if getattr(dataset, "indices", None) is not None:
        idx = np.asarray(dataset.indices, dtype=np.int64)
        labels = labels[idx]

    n, d = labels.shape
    if factor_names is None:
        factor_names = [f"factor_{i}" for i in range(d)]

    summaries = []
    print(f"Subset size: {n}\n")

    for j in range(d):
        vals = labels[:, j]
        K = int(vals.max()) + 1
        counts = np.bincount(vals, minlength=K)
        total = int(counts.sum())

        probs = counts / total if total > 0 else np.zeros_like(counts, dtype=float)

        # Fractions count/total (optionally simplified)
        fracs = []
        for c in counts.tolist():
            f = Fraction(int(c), total) if total > 0 else Fraction(0, 1)
            if simplify:
                if max_denominator is not None:
                    f = f.limit_denominator(max_denominator)
                else:
                    # already exact; keep as-is
                    pass
            fracs.append(f)

        head = counts.max() if total > 0 else 0
        tail = counts.min() if total > 0 else 0
        head_tail_ratio = (head / tail) if tail > 0 else np.inf

        print(f"{factor_names[j]} (K={K})")
        print("  fractions:", [f"{f.numerator}/{f.denominator}" for f in fracs])
        print("  probs:    ", [round(float(p), 6) for p in probs.tolist()])
        print(f"  head/tail ratio: {head_tail_ratio:.3f}\n")

        summaries.append({
            "name": factor_names[j],
            "counts": counts,
            "total": total,
            "fractions": fracs,
            "probs": probs
        })

        if topk is not None:
            order = np.argsort(-counts)
            top = []
            for v in order[:topk]:
                v = int(v)
                c = int(counts[v])
                f = Fraction(c, total) if total > 0 else Fraction(0, 1)
                top.append((v, f"{f.numerator}/{f.denominator}", float(probs[v])))
            print(f"  top-{topk} values (value, frac, prob): {top}\n")

    return summaries

