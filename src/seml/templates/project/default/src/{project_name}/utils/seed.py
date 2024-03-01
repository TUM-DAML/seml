"""Module for seeding."""

import random

import numpy as np


def set_seed(seed: int | None) -> tuple[int, np.random.Generator]:
    """Set seed for random number generators, numpy and python.random.

    The seed is a big int that wandb converts into a float, destroying the seed, so we
    store it as a string instead.
    """
    big_seed = int(seed) if seed is not None else None
    return manual_seed(big_seed)


def manual_seed(seed: int | None) -> tuple[int, np.random.Generator]:
    """Seed all RNGs manually without reusing the same seed."""
    root_ss = np.random.SeedSequence(seed)

    std_ss, np_ss = root_ss.spawn(2)

    # Python uses a Mersenne twister with 624 words of state, so we provide enough seed to
    # initialize it fully
    random.seed(std_ss.generate_state(624).tobytes())

    # It is best practice not to use numpy's global RNG, so we instantiate one
    rng = np.random.default_rng(np_ss)

    # We seed the global RNG anyway in case some library uses it internally
    np.random.seed(int(root_ss.generate_state(1, np.uint32)))

    # Always initialize the global default (CPU) generator

    seed = root_ss.entropy
    return seed, rng
