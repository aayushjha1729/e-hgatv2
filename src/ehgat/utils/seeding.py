"""Centralized determinism control.

Every stochastic component (instance generation, BRKGA, GNN init/training, NSGA-II)
draws from seeds derived here so that an entire experiment is reproducible from a
single integer. Torch is configured lazily and only if installed, keeping the
environment/oracle/BRKGA layers free of a hard Torch dependency.
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["make_rng", "seed_everything"]


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and (if available) Torch for reproducible runs.

    Parameters
    seed:
        Master seed.
    deterministic_torch:
        If True and Torch is installed, force deterministic algorithms and disable
        nondeterministic CuDNN/MPS kernels. Execution targets the CPU, where
        deterministic scatter and max aggregation are guaranteed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is an optional extra
        return

    torch.manual_seed(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def make_rng(seed: int) -> np.random.Generator:
    """Return an isolated NumPy Generator for a sub-component.

    Prefer this over the global NumPy state so that components (e.g. instance
    generation vs. BRKGA evolution) do not interfere with one another.
    """
    return np.random.default_rng(seed)
