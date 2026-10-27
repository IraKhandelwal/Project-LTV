# backend/scripts/inspect_dlr_labels.py

from pathlib import Path
import argparse

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def inspect_split(split_dir: Path) -> None:
    image_dir = split_dir / "Images"
    annotation_dir = split_dir / "Annotation"

    print(f"\n{'=' * 75}")
    print(f"Checking split: {split_dir.name}")
    print(f"{'=' * 75}")

    if not image_dir.is_dir():
        print(f"ERROR: Image directory not found: {image_dir}")
        return

    if not annotation_dir.is_dir():
        print(f"ERROR: Annotation directory not found: {annotation_dir}")
        return

    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print("ERROR: No images found.")
        return

    total_people = 0
    total_images = 0
    zero_count_images = []
    missing_annotations = []
    dimension_mismatches = []
    suspicious_annotations = []

    print(
        f"{'Image':<20} {'Image size':<18} "
        f"{'Label size':<18} {'People':>8} {'Nonzero':>8}"
    )
    print("-" * 75)

    for image_path in images:
        annotation_path = annotation_dir / f"{image_path.stem}.png"

        if not annotation_path.is_file():
            missing_annotations.append(image_path.name)
            print(f"{image_path.name:<20} MISSING ANNOTATION")
            continue

        try:
            with Image.open(image_path) as image:
                image_size = image.size

            with Image.open(annotation_path) as annotation:
                label = np.asarray(annotation.convert("L"))

        except Exception as error:
            print(f"ERROR reading {image_path.name}: {error}")
            continue

        label_size = (label.shape[1], label.shape[0])

        # The project documentation says one pixel with value 255
        # represents one annotated person.
        white_pixel_count = int(np.count_nonzero(label == 255))

        # Count all nonzero pixels as a cross-check.
        nonzero_pixel_count = int(np.count_nonzero(label))

        total_images += 1
        total_people += white_pixel_count

        print(
            f"{image_path.name:<20} "
            f"{str(image_size):<18} "
            f"{str(label_size):<18} "
            f"{white_pixel_count:>8} "
            f"{nonzero_pixel_count:>8}"
        )

        if white_pixel_count == 0:
            zero_count_images.append(image_path.name)

        if image_size != label_size:
            dimension_mismatches.append(
                (image_path.name, image_size, label_size)
            )

        # Different counts may indicate unexpected annotation values.
        if white_pixel_count != nonzero_pixel_count:
            suspicious_annotations.append(
                (
                    image_path.name,
                    white_pixel_count,
                    nonzero_pixel_count,
                )
            )

    print("\nSUMMARY")
    print(f"Images checked: {total_images}")
    print(f"Total people counted from white pixels: {total_people}")
    print(f"Images with zero white pixels: {len(zero_count_images)}")
    print(f"Missing annotation files: {len(missing_annotations)}")
    print(f"Image/annotation dimension mismatches: {len(dimension_mismatches)}")
    print(f"Annotations with nonzero pixels other than 255: "
          f"{len(suspicious_annotations)}")

    if zero_count_images:
        print("\nImages with zero annotated people:")
        for name in zero_count_images:
            print(f"  - {name}")

    if missing_annotations:
        print("\nImages with missing annotations:")
        for name in missing_annotations:
            print(f"  - {name}")

    if dimension_mismatches:
        print("\nImage/annotation dimension mismatches:")
        for name, image_size, label_size in dimension_mismatches:
            print(
                f"  - {name}: image={image_size}, annotation={label_size}"
            )

    if suspicious_annotations:
        print("\nAnnotations containing nonzero values other than 255:")
        for name, white_count, nonzero_count in suspicious_annotations:
            print(
                f"  - {name}: white pixels={white_count}, "
                f"nonzero pixels={nonzero_count}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect DLR-ACD image and annotation integrity."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Folder containing Train/ and Test/ directories.",
    )

    args = parser.parse_args()

    if not args.dataset_root.is_dir():
        parser.error(f"Dataset directory does not exist: {args.dataset_root}")

    for split_name in ("Train", "Test"):
        inspect_split(args.dataset_root / split_name)


if __name__ == "__main__":
    main()