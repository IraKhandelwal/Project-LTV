
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Make the backend directory available for importing the training script.
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from scripts.train_mrcnet import _density_target


def test_case(name, points, size=256, sigma=2.0):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    density = _density_target(points, size, sigma)

    high_count = float(density.sum())

    high_tensor = torch.from_numpy(density).unsqueeze(0).unsqueeze(0)
    low_density = F.avg_pool2d(high_tensor, 4, 4) * 16.0
    low_count = float(low_density.sum())

    expected_count = len(points)

    print(f"\nTest: {name}")
    print(f"Expected people: {expected_count}")
    print(f"High-resolution density sum: {high_count:.6f}")
    print(f"Low-resolution density sum:  {low_count:.6f}")
    print(f"High-resolution error: {high_count - expected_count:.6f}")
    print(f"Low-resolution error:  {low_count - expected_count:.6f}")


def main():
    test_case("No people", [])

    test_case(
        "One person in the center",
        [(128, 128)],
    )

    test_case(
        "Three people",
        [(50, 50), (128, 128), (200, 200)],
    )

    test_case(
        "People near image boundaries",
        [(0, 0), (255, 255), (0, 255), (255, 0)],
    )

    test_case(
        "Several nearby people",
        [(100, 100), (101, 100), (100, 101), (101, 101)],
    )


if __name__ == "__main__":
    main()