"""Seeding helpers so that every run is exactly repeatable on the same machine.

Sources of randomness in this project and how each one is controlled:

* Python ``random``   -> ``random.seed``          (not used directly, seeded defensively)
* NumPy               -> a ``np.random.Generator`` created from the seed and passed
                         explicitly to agents (preferred over the global NumPy state)
* PyTorch             -> ``torch.manual_seed`` + deterministic algorithms
* Gymnasium           -> ``env.reset(seed=...)`` on the first reset and
                         ``env.action_space.seed(...)``; done inside the training loop.
"""

from __future__ import annotations

import random

import numpy as np


def set_global_seeds(seed: int, deterministic_torch: bool = True) -> np.random.Generator:
    """Seed every library-level RNG and return a fresh NumPy Generator.

    Args:
        seed: Non-negative integer seed.
        deterministic_torch: If True, force PyTorch to use deterministic kernels and a
            single CPU thread. One thread also makes wall-clock comparisons fair,
            because the tabular agent is single-threaded NumPy code.

    Returns:
        A ``np.random.Generator`` that the caller should pass to the agent.
    """
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    random.seed(seed)
    np.random.seed(seed)  # legacy global state; only for third-party code that uses it

    try:
        import torch
    except ImportError:  # the tabular agent does not need PyTorch
        torch = None

    if torch is not None:
        torch.manual_seed(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True)
            torch.set_num_threads(1)

    return np.random.default_rng(seed)
