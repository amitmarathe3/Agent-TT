"""Reproducibility helper for setting random seeds across Python, NumPy, and PyTorch."""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed for Python random, NumPy, PyTorch, and environment for reproducibility.

    Args:
        seed: Integer random seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        # Seed CUDA if present; never call is_available() (it can segfault on
        # some Windows/Python wheels). The manual calls are safe on CPU-only.
        try:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except ImportError:
        pass
