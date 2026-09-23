from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedPair:
    factual_seed: int
    counterfactual_seed: int


def paired_seeds(base_seed: int, k: int, use_crn: bool = True) -> list[SeedPair]:
    rng = random.Random(base_seed)
    if use_crn:
        seeds = [rng.randint(0, 2**31 - 1) for _ in range(k)]
        return [SeedPair(seed, seed) for seed in seeds]
    return [SeedPair(rng.randint(0, 2**31 - 1), rng.randint(0, 2**31 - 1)) for _ in range(k)]
