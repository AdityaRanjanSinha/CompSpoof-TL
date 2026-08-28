"""Deterministic seeding. The only allowed headline seeds are 13, 42, 2027."""
import os
import random

ALLOWED_SEEDS = (13, 42, 2027)


def set_seed(seed: int, strict: bool = True) -> int:
    """Seed python, numpy and torch (if present). Returns the seed used."""
    if strict and seed not in ALLOWED_SEEDS:
        raise ValueError(f"{seed} is not an allowed headline seed {ALLOWED_SEEDS}. "
                         f"Pass strict=False only for quick throwaway experiments.")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed
