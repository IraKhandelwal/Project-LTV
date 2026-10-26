#!/usr/bin/env python3
"""Evaluate a trained MRCNet reproduction on a held-out image split."""

from __future__ import annotations

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


def _starts(length: int) -> list[int]:
    # Non-overlapping tiles avoid counting people twice; PIL pads the final
    # right/bottom tile when it extends beyond the image boundary.
    return list(range(0, max(length, 1), PATCH_SIZE))


@torch.inference_mode()
def predict_image(model: MRCNet, image_path: Path, device: torch.device) -> float:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    count = 0.0
    model.eval()
    for top in _starts(height):
        for left in _starts(width):
            patch = image.crop((left, top, left + PATCH_SIZE, top + PATCH_SIZE))
            if patch.size != (PATCH_SIZE, PATCH_SIZE):
                padded = Image.new("RGB", (PATCH_SIZE, PATCH_SIZE))
                padded.paste(patch, (0, 0))
                patch = padded
            pixels = np.asarray(patch, dtype=np.float32) / 255.0
            pixels = (pixels - MEAN) / STD
            tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
            low_density, _ = model(tensor)
            count += float(low_density.sum().cpu())
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True,
                        help="Held-out split with Images/ and Annotation/ folders")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/mrcnet-dlr-reproduction"))
    parser.add_argument("--image-names", nargs="+", default=None,
                        help="Evaluate only named images, for example a validation event group")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    image_dir = args.test_dir / "Images"
    annotation_dir = args.test_dir / "Annotation"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        parser.error("test-dir must contain Images/ and Annotation/ folders")
    if not args.weights.is_file():
        parser.error(f"Checkpoint does not exist: {args.weights}")
    device = torch.device(args.device)
    model = MRCNet().to(device)
    checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    images = sorted(path for path in image_dir.iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.image_names:
        requested = set(args.image_names)
        images = [path for path in images if path.name in requested]
        missing = requested - {path.name for path in images}
        if missing:
            parser.error(f"Requested image(s) are missing from {image_dir}: {sorted(missing)}")
    if args.limit:
        images = images[:args.limit]
    if not images:
        parser.error("No test images found")

    rows: list[dict[str, str | int | float]] = []
    for index, image_path in enumerate(images, start=1):
        truth = count_people_in_annotation(annotation_dir / f"{image_path.stem}.png")
        prediction = predict_image(model, image_path, device)
        rows.append({"image": image_path.name, "ground_truth_people": truth,
                     "predicted_people": round(prediction, 3),
                     "absolute_error": round(abs(prediction - truth), 3),
                     "signed_error": round(prediction - truth, 3),
                     "normalized_absolute_error": round(abs(prediction - truth) / truth, 6)})
        print(f"[{index}/{len(images)}] {image_path.name}: GT={truth}, prediction={prediction:.1f}",
              flush=True)

    errors = [float(row["absolute_error"]) for row in rows]
    signed = [float(row["signed_error"]) for row in rows]
    normalized = [float(row["normalized_absolute_error"]) for row in rows]
    summary = {
        "model": "Independent paper-inspired MRCNet reproduction",
        "dataset_split": args.test_dir.name,
        "images_evaluated": len(rows),
        "mae_people_per_image": sum(errors) / len(errors),
        "rmse_people_per_image": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "mean_signed_error_people_per_image": sum(signed) / len(signed),
        "mnae": sum(normalized) / len(normalized),
        "scope": "Offline image counts; tiled inference; not persons/m², live-camera, or field validation.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                  encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
