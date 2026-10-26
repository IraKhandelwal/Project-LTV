#!/usr/bin/env python3
"""Evaluate CSRNet or DM-Count on an annotated crowd-count split."""

import argparse
import csv
import json
import math
from pathlib import Path

from app.crowd_counting import (
    CrowdCountModel,
    DMCountCrowdModel,
    count_people_in_annotation,
    find_annotation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("csrnet", "dmcount"), default="csrnet",
                        help="candidate architecture to evaluate")
    parser.add_argument("--test-dir", required=True, type=Path,
                        help="Test split folder; may contain images directly or Images/ and Annotation/ subfolders")
    parser.add_argument("--annotation-dir", type=Path,
                        help="Optional separate folder containing matching .mat or binary .png labels")
    parser.add_argument("--weights", required=True, type=Path, help="Local model checkpoint")
    parser.add_argument("--output-dir", type=Path,
                        help="Output folder (defaults to reports/<model>-public-benchmark)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional small smoke run; 0 evaluates every image")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (if available)")
    args = parser.parse_args()

    if not args.test_dir.is_dir():
        parser.error(f"Test directory does not exist: {args.test_dir}")
    if not args.weights.is_file():
        parser.error(f"Checkpoint does not exist: {args.weights}")
    image_dir = args.test_dir / "Images" if (args.test_dir / "Images").is_dir() else args.test_dir
    annotation_dir = args.annotation_dir or (
        args.test_dir / "Annotation" if (args.test_dir / "Annotation").is_dir() else args.test_dir
    )
    images = sorted(path for path in image_dir.iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        parser.error(f"No image files found in {args.test_dir}")

    model_class = CrowdCountModel if args.model == "csrnet" else DMCountCrowdModel
    model = model_class(args.weights, args.device)
    results: list[dict[str, float | int | str]] = []
    for index, image_path in enumerate(images, start=1):
        annotation_path = find_annotation(image_path, annotation_dir)
        ground_truth = count_people_in_annotation(annotation_path)
        prediction, _ = model.predict(image_path)
        results.append({
            "image": image_path.name,
            "ground_truth_people": ground_truth,
            "predicted_people": round(prediction, 3),
            "absolute_error": round(abs(prediction - ground_truth), 3),
            "signed_error": round(prediction - ground_truth, 3),
        })
        print(f"[{index}/{len(images)}] {image_path.name}: GT={ground_truth}, prediction={prediction:.1f}")

    errors = [float(row["absolute_error"]) for row in results]
    signed_errors = [float(row["signed_error"]) for row in results]
    summary = {
        "model": ("CSRNet; ShanghaiTech Part B checkpoint" if args.model == "csrnet"
                  else "DM-Count; NWPU pretrained checkpoint; evaluate-only candidate"),
        "dataset": args.test_dir.parent.name,
        "images_evaluated": len(results),
        "mae_people_per_image": round(sum(errors) / len(errors), 4),
        "rmse_people_per_image": round(math.sqrt(sum(error * error for error in errors) / len(errors)), 4),
        "mean_signed_error_people_per_image": round(sum(signed_errors) / len(signed_errors), 4),
        "scope": "Image-level count only; not persons/m², real-time, or operational validation.",
    }
    output_dir = args.output_dir or Path(f"reports/{args.model}-public-benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
