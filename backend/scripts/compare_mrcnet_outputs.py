#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.crowd_counting import count_people_in_annotation
from app.mrcnet import MRCNet

MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
PATCH_SIZE = 256


def starts(length):
    return list(range(0, max(length, 1), PATCH_SIZE))


@torch.inference_mode()
def predict_both(model, image_path, device):
    low_count = 0.0
    high_count = 0.0

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    width, height = image.size
    model.eval()

    for top in starts(height):
        for left in starts(width):
            patch = image.crop(
                (left, top, left + PATCH_SIZE, top + PATCH_SIZE)
            )

            if patch.size != (PATCH_SIZE, PATCH_SIZE):
                padded = Image.new("RGB", (PATCH_SIZE, PATCH_SIZE))
                padded.paste(patch, (0, 0))
                patch = padded

            pixels = np.asarray(patch, dtype=np.float32) / 255.0
            pixels = (pixels - MEAN) / STD

            tensor = torch.from_numpy(
                pixels.transpose(2, 0, 1).copy()
            ).unsqueeze(0).to(device)

            low_density, high_density = model(tensor)

            low_count += float(low_density.sum().cpu())
            high_count += float(high_density.sum().cpu())

    return low_count, high_count


def main():
    parser = argparse.ArgumentParser(
        description="Compare MRCNet low- and high-resolution counting outputs."
    )

    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Dataset split containing Images/ and Annotation/"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/mrcnet-output-comparison")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()

    image_dir = args.test_dir / "Images"
    annotation_dir = args.test_dir / "Annotation"

    if not image_dir.is_dir() or not annotation_dir.is_dir():
        parser.error("Dataset directory must contain Images/ and Annotation/.")

    if not args.weights.is_file():
        parser.error(f"Checkpoint not found: {args.weights}")

    device = torch.device(args.device)

    model = MRCNet().to(device)

    checkpoint = torch.load(
        args.weights,
        map_location=device,
        weights_only=True
    )

    state = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )

    model.load_state_dict(state, strict=True)
    model.eval()

    images = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if args.limit > 0:
        images = images[:args.limit]

    if not images:
        parser.error("No images found.")

    rows = []

    for index, image_path in enumerate(images, start=1):
        annotation_path = annotation_dir / f"{image_path.stem}.png"

        if not annotation_path.is_file():
            parser.error(f"Missing annotation: {annotation_path}")

        truth = count_people_in_annotation(annotation_path)

        low_count, high_count = predict_both(
            model, image_path, device
        )

        row = {
            "image": image_path.name,
            "ground_truth": truth,
            "low_prediction": round(low_count, 3),
            "low_absolute_error": round(abs(low_count - truth), 3),
            "high_prediction": round(high_count, 3),
            "high_absolute_error": round(abs(high_count - truth), 3),
        }

        rows.append(row)

        print(
            f"[{index}/{len(images)}] {image_path.name} | "
            f"Actual={truth} | "
            f"Low={low_count:.1f} "
            f"(error {abs(low_count-truth):.1f}) | "
            f"High={high_count:.1f} "
            f"(error {abs(high_count-truth):.1f})",
            flush=True
        )

    low_errors = [r["low_absolute_error"] for r in rows]
    high_errors = [r["high_absolute_error"] for r in rows]

    low_signed = [
        r["low_prediction"] - r["ground_truth"] for r in rows
    ]
    high_signed = [
        r["high_prediction"] - r["ground_truth"] for r in rows
    ]

    summary = {
        "checkpoint": str(args.weights),
        "dataset_split": args.test_dir.name,
        "images_evaluated": len(rows),
        "low_resolution": {
            "MAE": sum(low_errors) / len(rows),
            "mean_signed_error": sum(low_signed) / len(rows),
        },
        "high_resolution": {
            "MAE": sum(high_errors) / len(rows),
            "mean_signed_error": sum(high_signed) / len(rows),
        },
        "note": (
            "Diagnostic comparison only. The high-resolution output "
            "may not be calibrated for counting. The dataset split has "
            "previously been used for model comparisons."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8"
    )

    print("\n===== FINAL COMPARISON =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
