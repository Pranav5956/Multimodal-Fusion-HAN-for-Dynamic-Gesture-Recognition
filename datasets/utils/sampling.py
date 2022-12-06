import numpy as np
from typing import List


def sample_from_frames(num_frames: int, num_samples: int, sampling: str = "uniform") -> List[int]:
    assert sampling in (
        "uniform", "k-random"), f"Valid sampling = 'k-random', 'uniform': received {sampling}!"

    if sampling == "uniform":
        return np.linspace(0, num_frames - 1, num_samples, dtype=np.uint8).tolist()

    if sampling == "k-random":
        return [
            np.random.choice(segment) if len(segment) > 0 else (num_frames - 1) for segment in np.array_split(np.arange(0, num_frames, dtype=np.uint8), num_samples)
        ]

    return np.arange(0, num_frames, 1, dtype=np.uint8).tolist()
