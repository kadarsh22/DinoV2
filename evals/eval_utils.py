import os
import torch


def _load_cached(cache_dir, split, map_location="cpu"):
    X = torch.load(os.path.join(cache_dir, f"{split}_X.pt"), map_location=map_location)
    A = torch.load(os.path.join(cache_dir, f"{split}_A.pt"), map_location=map_location)
    return X, A


def _cleanup_cache_dir(cache_dir: str):
    for name in ("train_X.pt", "train_A.pt", "test_X.pt", "test_A.pt"):
        p = os.path.join(cache_dir, name)
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[cache cleanup] could not remove {p}: {e}")
    try:
        os.rmdir(cache_dir)
    except OSError:
        pass

def _need_cache(cache_dir: str, split: str) -> bool:
    x_p, a_p = _cache_paths(cache_dir, split)
    return not (os.path.exists(x_p) and os.path.exists(a_p))

def _cache_paths(cache_dir: str, split: str):
    return (
        os.path.join(cache_dir, f"{split}_X.pt"),
        os.path.join(cache_dir, f"{split}_A.pt"),
    )