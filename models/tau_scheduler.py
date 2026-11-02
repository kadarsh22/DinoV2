import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class TauAdaptiveConfig:
    """Unbounded controller for DINO cross-entropy errors."""

    target_hard_frac: float = 0.3
    lr_tau: float = 0.01


class TauAdaptiveScheduler:
    """Adjust tau directly from the current batch's hard-patch fraction.

    Adapted from source-ijepa-cmnist-main.py:v749 logged by W&B run 8dsyakwz.
    https://forge.coreweave.com/wandb/kadarsh22/ijepa-cmnist/runs/8dsyakwz
    The controller uses each observation immediately, without EMA smoothing
    or threshold bounds.
    Its initial threshold is the first batch's (1 - target_hard_frac) error
    quantile, pooled across ranks. Subsequent batches use proportional updates.
    Labels remain based on DINO's cross-entropy error.
    """

    def __init__(self, cfg: TauAdaptiveConfig):
        if not all(math.isfinite(v) for v in vars(cfg).values()):
            raise ValueError("Adaptive tau settings must be finite")
        if not 0 < cfg.target_hard_frac < 1:
            raise ValueError("Invalid adaptive tau target")
        if cfg.lr_tau < 0:
            raise ValueError("Adaptive tau learning rate must be nonnegative")
        self.cfg = cfg
        self._tau = None

    def tau(self) -> Optional[float]:
        return self._tau

    def threshold(self, prediction_error) -> float:
        """Initialize once from detached errors, then return the current tau."""
        if self._tau is None:
            import torch
            import torch.distributed as dist

            errors = prediction_error.detach().float().reshape(-1)
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                size = torch.tensor([errors.numel()], device=errors.device, dtype=torch.long)
                sizes = [torch.empty_like(size) for _ in range(dist.get_world_size())]
                dist.all_gather(sizes, size)
                counts = [int(n.item()) for n in sizes]
                if not sum(counts):
                    raise ValueError("Initial tau requires nonempty prediction errors")
                padded = torch.nn.functional.pad(errors, (0, max(counts) - errors.numel()))
                gathered = [torch.empty_like(padded) for _ in counts]
                dist.all_gather(gathered, padded)
                errors = torch.cat([values[:n] for values, n in zip(gathered, counts)])
            if not errors.numel() or not torch.isfinite(errors).all():
                raise ValueError("Initial tau requires nonempty, finite prediction errors")
            self._tau = torch.quantile(errors, 1 - self.cfg.target_hard_frac).item()
        return self._tau

    def update(self, observed_hard_frac: float) -> float:
        if not math.isfinite(observed_hard_frac) or not 0 <= observed_hard_frac <= 1:
            raise ValueError("Observed hard fraction must be finite and between 0 and 1")
        if self._tau is None:
            raise RuntimeError("Initialize tau from prediction errors before updating")
        c = self.cfg
        self._tau += c.lr_tau * (observed_hard_frac - c.target_hard_frac)
        return self._tau
