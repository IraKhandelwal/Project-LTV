#!/usr/bin/env python3
"""Compare MRCNet and label counts in 256px tiles for diagnostic images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.mrcnet import MRCNet

MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
TILE = 256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", nargs="+", required=True,
                        help="Image filenames selected for error inspection")
    parser.add_argument("--output", type=Path,
                        default=Path("reports/mrcnet-dlr-reproduction/tile-diagnostics.csv"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = MRCNet().to(device)
    checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()

    rows: list[dict[str, int | float | str]] = []
    for filename in args.images:
        image_path = args.test_dir / "Images" / filename
        annotation_path = args.test_dir / "Annotation" / f"{Path(filename).stem}.png"
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(annotation_path) as source:
            annotation = source.convert("L")
        width, height = image.size
        for top in range(0, height, TILE):
            for left in range(0, width, TILE):
                crop = image.crop((left, top, left + TILE, top + TILE))
                pixels = np.asarray(crop, dtype=np.float32) / 255.0
                pixels = (pixels - MEAN) / STD
                tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
                with torch.inference_mode():
                    low_density, _ = model(tensor)
                ground_truth = int(np.count_nonzero(
                    np.asarray(annotation.crop((left, top, left + TILE, top + TILE))) == 255
                ))
                prediction = float(low_density.sum().cpu())
                rows.append({"image": filename, "tile_x": left, "tile_y": top,
                             "ground_truth_people": ground_truth,
                             "predicted_people": round(prediction, 3),
                             "signed_error": round(prediction - ground_truth, 3)})
        image_rows = [row for row in rows if row["image"] == filename]
        truth_total = sum(int(row["ground_truth_people"]) for row in image_rows)
        prediction_total = sum(float(row["predicted_people"]) for row in image_rows)
        print(f"{filename}: truth={truth_total}, prediction={prediction_total:.1f}, "
              f"tiles={len(image_rows)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved tile details to {args.output}")


if __name__ == "__main__":
    main()
