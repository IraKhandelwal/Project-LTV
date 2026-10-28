#!/usr/bin/env python3
"""Evaluate compatible MRCNet checkpoints on the DLR-ACD test set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

from app.mrcnet import MRCNet


MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

PATCH_SIZE = 256

CHECKPOINTS = [
    "mrcnet-dlr-countloss-v1.pth",
    "mrcnet-dlr-highloss-0001.pth",
    "mrcnet-dlr-highloss-001.pth",
    "mrcnet-dlr-reproduction-v2.pth",
    "mrcnet-dlr-resume-v4.pth",
    "mrcnet-dlr-validation-v3.pth",
]


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def ground_truth_count(path: Path) -> int:
    with Image.open(path) as image:
        annotation = np.asarray(image.convert("L"))
    return int(np.count_nonzero(annotation == 255))


def image_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    pixels = rgb.astype(np.float32) / 255.0
    pixels = (pixels - MEAN) / STD
    return torch.from_numpy(
        np.ascontiguousarray(pixels.transpose(2, 0, 1))
    ).unsqueeze(0).to(device)


def tile_starts(length: int, patch_size: int) -> list[int]:
    """Cover the image, including its far edge, with no overlapping tiles."""
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, patch_size))
    if starts[-1] + patch_size < length:
        starts.append(length - patch_size)
    return starts


def predict_image(
    model: MRCNet,
    rgb: np.ndarray,
    device: torch.device,
) -> tuple[float, float]:
    """Return low-map and high-map estimates summed over image patches."""
    height, width = rgb.shape[:2]
    low_total = 0.0
    high_total = 0.0
    low_covered = 0
    high_covered = 0

    y_starts = tile_starts(height, PATCH_SIZE)
    x_starts = tile_starts(width, PATCH_SIZE)

    for top in y_starts:
        for left in x_starts:
            patch = rgb[
                top:min(top + PATCH_SIZE, height),
                left:min(left + PATCH_SIZE, width),
            ]

            patch_height, patch_width = patch.shape[:2]

            # Pad only at the bottom/right for model compatibility.
            if patch_height != PATCH_SIZE or patch_width != PATCH_SIZE:
                padded = np.zeros(
                    (PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8
                )
                padded[:patch_height, :patch_width] = patch
                patch = padded

            tensor = image_tensor(patch, device)

            with torch.inference_mode():
                low_density, high_density = model(tensor)

            # Ignore padded areas when summing the high-resolution map.
            high_map = high_density[0, 0]
            high_map = high_map[:patch_height, :patch_width]
            high_total += float(high_map.sum().item())

            # The low-resolution map is 1/4 of the input width/height.
            # Scale the valid region to the corresponding low-map dimensions.
            low_map = low_density[0, 0]
            valid_h = max(1, round(patch_height / 4))
            valid_w = max(1, round(patch_width / 4))
            low_total += float(low_map[:valid_h, :valid_w].sum().item())

            low_covered += patch_height * patch_width
            high_covered += patch_height * patch_width

    return low_total, high_total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/public/dlr-acd/DLR_AerialCrowdDataset/Test"),
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/mrcnet-evaluation")
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--checkpoints", nargs="*", default=CHECKPOINTS,
        help="Checkpoint filenames located in --models-dir",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    images_dir = args.data_dir / "Images"
    annotations_dir = args.data_dir / "Annotation"
    images = {p.stem: p for p in images_dir.glob("*.jpg")}
    annotations = {p.stem: p for p in annotations_dir.glob("*.png")}

    names = sorted(set(images) & set(annotations))
    if not names:
        parser.error("No matching test images and annotation PNGs were found.")

    print(f"Device: {device}")
    print(f"Matched test images: {len(names)}")
    print("Ground truth: number of pixels equal to 255 in each annotation")
    print("Note: this initial evaluator uses patch tiling; inspect results before "
          "treating them as final benchmark metrics.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for checkpoint_name in args.checkpoints:
        checkpoint_path = args.models_dir / checkpoint_name
        if not checkpoint_path.is_file():
            print(f"\nSKIP missing checkpoint: {checkpoint_path}")
            continue

        print(f"\nEvaluating: {checkpoint_name}", flush=True)

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        state = checkpoint.get("model_state_dict", checkpoint)

        model = MRCNet().to(device)
        try:
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            print(f"SKIP incompatible checkpoint: {exc}")
            del model
            continue

        model.eval()
        rows = []

        for index, name in enumerate(names, start=1):
            rgb = load_image(images[name])
            truth = ground_truth_count(annotations[name])
            low_prediction, high_prediction = predict_image(model, rgb, device)

            rows.append({
                "image": name,
                "ground_truth": truth,
                "low_prediction": low_prediction,
                "high_prediction": high_prediction,
                "low_absolute_error": abs(low_prediction - truth),
                "high_absolute_error": abs(high_prediction - truth),
                "width": rgb.shape[1],
                "height": rgb.shape[0],
            })

            print(
                f"  [{index}/{len(names)}] {name}: GT={truth}, "
                f"low={low_prediction:.1f}, high={high_prediction:.1f}",
                flush=True,
            )

        for output_key in ("low_prediction", "high_prediction"):
            errors = np.asarray(
                [row[output_key] - row["ground_truth"] for row in rows],
                dtype=np.float64,
            )
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            bias = float(np.mean(errors))

            summary_rows.append({
                "checkpoint": checkpoint_name,
                "output": output_key,
                "images": len(rows),
                "MAE": mae,
                "RMSE": rmse,
                "bias_prediction_minus_GT": bias,
            })

            print(
                f"  {output_key}: MAE={mae:.2f}, RMSE={rmse:.2f}, "
                f"bias={bias:.2f}",
                flush=True,
            )

        per_image_path = args.output_dir / f"{checkpoint_path.stem}_per_image.csv"
        with per_image_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if summary_rows:
        summary_path = args.output_dir / "summary.csv"
        with summary_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"\nSummary saved to: {summary_path}")
        print(f"Per-image reports saved in: {args.output_dir}")
    else:
        print("No compatible checkpoints were evaluated.")


if __name__ == "__main__":
    main()
