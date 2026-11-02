import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class TauFixedIterConfig:
    mode: str = "cosine"          # "constant" | "linear" | "cosine" | "exponential" | "step"
    tau_start: float = 0.8
    tau_end: float = 0.2
    total_iters: int = 100_000
    warmup_iters: int = 0         # linear warmup from 0 -> tau_start

    # exponential only (per-iter decay)
    exp_gamma: float = 0.99997    # choose based on total_iters (see note below)

    # step only
    step_size: int = 10_000       # iterations between drops
    step_gamma: float = 0.5       # multiply tau by this at each step


class TauFixedIterScheduler:
    def __init__(self, cfg: TauFixedIterConfig):
        self.cfg = cfg

    def _progress(self, it: int) -> float:
        # normalized progress ∈ [0,1] after warmup
        i = max(it - self.cfg.warmup_iters, 0)
        T = max(self.cfg.total_iters - self.cfg.warmup_iters, 1)
        return min(max(i / T, 0.0), 1.0)

    def tau(self, it: int) -> float:
        c = self.cfg

        # warmup: linearly ramp 0 -> tau_start
        if it < c.warmup_iters:
            if c.warmup_iters <= 0:
                return c.tau_start
            return (it / c.warmup_iters) * c.tau_start

        if c.mode == "constant":
            return c.tau_start

        t = self._progress(it)

        if c.mode == "linear":
            return c.tau_start + (c.tau_end - c.tau_start) * t

        if c.mode == "cosine":
            return c.tau_end + 0.5 * (c.tau_start - c.tau_end) * (1 + math.cos(math.pi * t))

        if c.mode == "exponential":
            i = max(it - c.warmup_iters, 0)
            return c.tau_end + (c.tau_start - c.tau_end) * (c.exp_gamma ** i)

        if c.mode == "step":
            i = max(it - c.warmup_iters, 0)
            k = i // max(c.step_size, 1)
            return max(c.tau_end, c.tau_start * (c.step_gamma ** k))

        raise ValueError(f"Unknown mode {c.mode}")


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
    Labels remain based on DINO's cross-entropy error; tau_end and warmup
    apply only to fixed schedules.
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

    def tau(self, iteration: int = 0) -> Optional[float]:
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
