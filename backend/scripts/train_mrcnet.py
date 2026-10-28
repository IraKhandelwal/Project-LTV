#!/usr/bin/env python3
"""Train a paper-inspired MRCNet reproduction on DLR-ACD training images only."""

from __future__ import annotations

import argparse
from functools import lru_cache
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from app.mrcnet import MRCNet


# Ground sampling distance in metres per pixel, from Table 1 of the MRCNet paper.
GSD_METRES_PER_PIXEL = {
    2: 0.14, 3: 0.075, 5: 0.09, 6: 0.10, 7: 0.13, 8: 0.14, 15: 0.12,
    17: 0.13, 18: 0.13, 19: 0.09, 20: 0.09, 22: 0.045, 23: 0.045,
    24: 0.045, 28: 0.055, 29: 0.055, 30: 0.055, 32: 0.11, 33: 0.11,
}
MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


@lru_cache(maxsize=2)
def _load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB")).copy()


@lru_cache(maxsize=2)
def _load_points(path: str) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as source:
        annotation = np.asarray(source.convert("L"))
    return np.where(annotation == 255)


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def _density_target(points_xy: np.ndarray, size: int, sigma: float) -> np.ndarray:
    """Place one normalized Gaussian per annotated person; total mass is count."""
    target = np.zeros((size, size), dtype=np.float32)
    if len(points_xy) == 0:
        return target
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(offsets, offsets)
    kernel = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma)).astype(np.float32)
    kernel /= kernel.sum()
    for x, y in points_xy:
        cx, cy = int(x), int(y)
        x0, x1 = max(0, cx - radius), min(size, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(size, cy + radius + 1)
        kx0, ky0 = x0 - (cx - radius), y0 - (cy - radius)
        local = kernel[ky0:ky0 + (y1 - y0), kx0:kx0 + (x1 - x0)]
        local_sum = float(local.sum())
        if local_sum:
            target[y0:y1, x0:x1] += local / local_sum
    return target


class DLRPatchDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Samples random 256px crops from 320px overlapping training tiles."""

    def __init__(self, train_dir: Path, samples_per_tile: int = 2,
                 exclude_image_ids: set[int] | None = None) -> None:
        self.image_dir = train_dir / "Images"
        self.annotation_dir = train_dir / "Annotation"
        self.entries: list[tuple[Path, Path, int, int, float, str]] = []
        self.exclude_image_ids = exclude_image_ids or set()
        for image_path in sorted(self.image_dir.glob("*.jpg")):
            image_id = int(image_path.stem.split("_")[-1])
            if image_id in self.exclude_image_ids:
                continue
            annotation_path = self.annotation_dir / f"{image_path.stem}.png"
            if not annotation_path.is_file():
                raise FileNotFoundError(f"Missing annotation: {annotation_path}")
            if image_id not in GSD_METRES_PER_PIXEL:
                raise ValueError(f"No paper-reported GSD for {image_path.name}")
            with Image.open(image_path) as image:
                width, height = image.size
            all_ys, all_xs = _load_points(str(annotation_path))
            for top in _tile_starts(height, 320, 160):
                for left in _tile_starts(width, 320, 160):
                    within = ((all_xs >= left) & (all_xs < left + 320) &
                              (all_ys >= top) & (all_ys < top + 320))
                    count = int(np.count_nonzero(within))
                    band = ("empty" if count == 0 else "sparse" if count < 10
                            else "medium" if count < 50 else "dense")
                    for _ in range(samples_per_tile):
                        self.entries.append((image_path, annotation_path, left, top,
                                             GSD_METRES_PER_PIXEL[image_id], band))
        if not self.entries:
            raise ValueError("Training split is empty after excluding validation images")
        self.bands = [entry[5] for entry in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        image_path, annotation_path, left, top, gsd, _ = self.entries[index]
        tile = _load_rgb(str(image_path))[top:top + 320, left:left + 320]
        image = Image.fromarray(tile.copy())
        all_ys, all_xs = _load_points(str(annotation_path))
        inside_tile = ((all_xs >= left) & (all_xs < left + 320) &
                       (all_ys >= top) & (all_ys < top + 320))
        xs = all_xs[inside_tile] - left
        ys = all_ys[inside_tile] - top
        crop_left = random.randint(0, 64)
        crop_top = random.randint(0, 64)
        image = image.crop((crop_left, crop_top, crop_left + 256, crop_top + 256))
        inside = ((xs >= crop_left) & (xs < crop_left + 256) &
                  (ys >= crop_top) & (ys < crop_top + 256))
        points = np.column_stack((xs[inside] - crop_left, ys[inside] - crop_top))

        # Random quarter-turns and flips follow the paper's aerial-image augmentation.
        turns = random.randrange(4)
        if turns:
            image = image.rotate(90 * turns)
            points = points.copy()
            for _ in range(turns):
                points = np.column_stack((points[:, 1], 255 - points[:, 0]))
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            points[:, 0] = 255 - points[:, 0]
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            points[:, 1] = 255 - points[:, 1]

        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = (pixels - MEAN) / STD
        density = _density_target(points, 256, float(max(1, math.ceil(0.5 / (3 * gsd)))))
        image_tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy())
        high_target = torch.from_numpy(density).unsqueeze(0)
        # Average pooling is multiplied by its area so count mass is preserved.
        low_target = F.avg_pool2d(high_target.unsqueeze(0), 4, 4).squeeze(0) * 16.0
        return image_tensor, low_target, high_target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--vgg16-weights", type=Path, required=True,
                        help="Torchvision VGG16 state dict used to initialize the encoder")
    parser.add_argument("--output", type=Path, default=Path("models/mrcnet-dlr-reproduction.pth"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=100,
                        help="Bound CPU training time; samples are drawn from the training set with replacement")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Small default for CPU-only machines; paper used 60")
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--high-loss-weight", type=float, default=1e-4,
                        help="Weight for high-resolution density loss")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume model weights from a checkpoint")
    parser.add_argument("--samples-per-tile", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--exclude-image-ids", type=int, nargs="*", default=[],
                        help="Hold out complete event images for validation, e.g. 28 29 30")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.train_dir.is_dir():
        parser.error(f"Training directory does not exist: {args.train_dir}")
    if not args.vgg16_weights.is_file():
        parser.error(f"VGG16 weights do not exist: {args.vgg16_weights}")
    if args.steps_per_epoch < 1:
        parser.error("steps-per-epoch must be at least 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    excluded_ids = set(args.exclude_image_ids)
    dataset = DLRPatchDataset(args.train_dir, args.samples_per_tile, excluded_ids)
    band_counts = {band: dataset.bands.count(band)
                   for band in ("empty", "sparse", "medium", "dense")}
    print(f"Training patches available: {len(dataset)}; device: {device}; "
          f"held-out image IDs: {sorted(excluded_ids)}; tile bands: {band_counts}", flush=True)
    # Preserve empty-background examples while giving dense tiles more training
    # weight, since the first runs missed dense areas and fired on empty tiles.
    target_share = {"empty": 0.35, "sparse": 0.15, "medium": 0.15, "dense": 0.35}
    counts = {band: dataset.bands.count(band) for band in target_share}
    weights = [target_share[band] / counts[band] if counts[band] else 0.0
               for band in dataset.bands]
    sampler_generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(weights, num_samples=args.steps_per_epoch * args.batch_size,
                                    replacement=True, generator=sampler_generator)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=0, pin_memory=device.type == "cuda")
    checkpoint = {}
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        model = MRCNet().to(device)
        model.load_state_dict(state, strict=True)
        start_epoch = int(checkpoint.get("epoch", 0))
        print(f"Loaded checkpoint: {args.resume}", flush=True)
        print(f"Checkpoint epoch: {start_epoch}", flush=True)
    else:
        model = MRCNet(args.vgg16_weights).to(device)
        start_epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Restored optimizer state.", flush=True)
    elif args.resume is not None:
        print("No saved optimizer state; starting with a fresh optimizer.", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume is not None and args.output.resolve() == args.resume.resolve():
        raise ValueError("Use a different --output path to protect the source checkpoint.")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        total_loss = 0.0
        for step, (images, low_target, high_target) in enumerate(loader, start=1):
            images = images.to(device)
            low_target = low_target.to(device)
            high_target = high_target.to(device)
            low_prediction, high_prediction = model(images)
            low_loss = F.mse_loss(low_prediction, low_target)
            high_loss = F.mse_loss(high_prediction, high_target)
            loss = low_loss + args.high_loss_weight * high_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            if step % 10 == 0 or step == len(loader):
                print(f"epoch {epoch + 1}/{args.epochs}, step {step}/{len(loader)}",
                      flush=True)
        average = total_loss / max(1, len(loader))
        print(f"epoch {epoch + 1}/{args.epochs}: loss={average:.6f}", flush=True)
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "training_kind": "independent paper-inspired reproduction",
                    "excluded_validation_image_ids": sorted(excluded_ids),
                    "balanced_tile_target_shares": target_share,
                    "seed": args.seed}, args.output)
    print(f"Saved candidate weights to {args.output}")


if __name__ == "__main__":
    main()
