#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.mrcnet import MRCNet

MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
PATCH_SIZE = 256


def starts(length):
    return list(range(0, max(length, 1), PATCH_SIZE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--images",
        nargs="+",
        default=["I_10.jpg", "I_25.jpg", "I_14.jpg"],
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = MRCNet().to(device)

    checkpoint = torch.load(
        args.weights, map_location=device, weights_only=True
    )
    state = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    image_dir = args.test_dir / "Images"

    with torch.inference_mode():
        for name in args.images:
            image_path = image_dir / name

            if not image_path.is_file():
                print(f"\nMissing image: {image_path}")
                continue

            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            low_maps = []
            high_maps = []

            for top in starts(height):
                for left in starts(width):
                    patch = image.crop(
                        (left, top, left + PATCH_SIZE, top + PATCH_SIZE)
                    )

                    if patch.size != (PATCH_SIZE, PATCH_SIZE):
                        padded = Image.new(
                            "RGB", (PATCH_SIZE, PATCH_SIZE)
                        )
                        padded.paste(patch, (0, 0))
                        patch = padded

                    pixels = np.asarray(
                        patch, dtype=np.float32
                    ) / 255.0
                    pixels = (pixels - MEAN) / STD

                    tensor = torch.from_numpy(
                        pixels.transpose(2, 0, 1).copy()
                    ).unsqueeze(0).to(device)

                    low, high = model(tensor)

                    low_maps.append(low.squeeze().cpu().numpy())
                    high_maps.append(high.squeeze().cpu().numpy())

            low_values = np.concatenate(
                [x.reshape(-1) for x in low_maps]
            )
            high_values = np.concatenate(
                [x.reshape(-1) for x in high_maps]
            )

            print("\n" + "=" * 65)
            print(f"Image: {name} | Size: {width}x{height}")
            print(f"Tiles evaluated: {len(low_maps)}")

            for label, values, maps in (
                ("LOW-RESOLUTION", low_values, low_maps),
                ("HIGH-RESOLUTION", high_values, high_maps),
            ):
                print(f"\n{label}")
                print(f"  Total predicted count: {values.sum():.3f}")
                print(f"  Minimum value:         {values.min():.8f}")
                print(f"  Maximum value:         {values.max():.8f}")
                print(f"  Mean value:            {values.mean():.8f}")
                print(f"  Standard deviation:    {values.std():.8f}")
                print(f"  Exact zero values:     {(values == 0).mean() * 100:.2f}%")
                print(
                    "  Value percentiles:     "
                    + str(np.percentile(values, [25, 50, 75, 95, 99]).round(6))
                )
                print("  Count per tile:        " +
                      str([round(float(x.sum()), 2) for x in maps]))


if __name__ == "__main__":
    main()
