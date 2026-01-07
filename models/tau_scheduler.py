import math
from dataclasses import dataclass

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
