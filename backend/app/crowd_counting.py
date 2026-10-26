"""Evaluation-only CSRNet implementation for public crowd-count benchmarks."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn


def _make_layers(channels: list[int | str], in_channels: int = 3,
                 dilation: bool = False) -> nn.Sequential:
    rate = 2 if dilation else 1
    layers: list[nn.Module] = []
    for value in channels:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            out_channels = int(value)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          padding=rate, dilation=rate),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    """CSRNet produces a spatial density map; its sum is the image count."""

    def __init__(self) -> None:
        super().__init__()
        self.frontend = _make_layers(
            [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
        )
        self.backend = _make_layers([512, 512, 512, 256, 128, 64], 512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.backend(self.frontend(image)))


class CrowdCountModel:
    """Loads a local checkpoint and exposes count plus density-map inference."""

    def __init__(self, weights_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = CSRNet().to(self.device)
        checkpoint = torch.load(Path(weights_path), map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        # Older CSRNet checkpoints sometimes wrap parameters in DataParallel names.
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image_path: str | Path) -> tuple[float, np.ndarray]:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        padded_width = (width + 7) // 8 * 8
        padded_height = (height + 7) // 8 * 8
        if (padded_width, padded_height) != (width, height):
            image = image.resize((padded_width, padded_height), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32))
        pixels = pixels / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)
        density_map = self.model(tensor)[0, 0].cpu().numpy()
        return float(density_map.sum()), density_map


def _dmcount_features() -> nn.Sequential:
    """Build the VGG-19 feature stack used by the authors' DM-Count model."""
    config: list[int | str] = [
        64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
        512, 512, 512, 512, "M", 512, 512, 512, 512,
    ]
    layers: list[nn.Module] = []
    in_channels = 3
    for value in config:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        channels = int(value)
        layers.extend([
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ])
        in_channels = channels
    return nn.Sequential(*layers)


class DMCount(nn.Module):
    """DM-Count's VGG-19 encoder and density-map decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.features = _dmcount_features()
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1), nn.ReLU())

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(image)
        features = torch.nn.functional.interpolate(
            features, scale_factor=2, mode="bilinear", align_corners=True
        )
        density = self.density_layer(self.reg_layer(features))
        count = density.sum(dim=(1, 2, 3), keepdim=True)
        return density, density / (count + 1e-6)


class DMCountCrowdModel:
    """Load official DM-Count weights and predict count plus density map."""

    def __init__(self, weights_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = DMCount().to(self.device)
        checkpoint = torch.load(Path(weights_path), map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict):
            checkpoint = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image_path: str | Path) -> tuple[float, np.ndarray]:
        # Match the authors' NWPU preprocessing: scale the longer side to at most
        # 1920px, keep the shorter side at least 384px, and make both dimensions
        # divisible by 16. Their dataset loader then applies ImageNet normalization.
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        scale = 1920 / height
        if width * scale > 1920:
            scale = 1920 / width
        resized_height = int(height * scale / 16) * 16
        resized_width = int(width * scale / 16) * 16
        if resized_height < 384:
            scale = 384 / height
        if resized_width < 384:
            scale = 384 / width
        if resized_height < 384 or resized_width < 384:
            resized_height = min(max(int(height * scale / 16) * 16, 384), 1920)
            resized_width = min(max(int(width * scale / 16) * 16, 384), 1920)
        if (resized_width, resized_height) != image.size:
            image = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32))
        pixels = pixels / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)
        density_map, _ = self.model(tensor)
        density = density_map[0, 0].cpu().numpy()
        return float(density.sum()), density


def count_points_in_mat(annotation_path: str | Path) -> int:
    """Read point annotations from common QNRF and ShanghaiTech MAT layouts."""
    from scipy.io import loadmat

    annotations = loadmat(annotation_path, simplify_cells=True)
    for key in ("annPoints", "points", "ann_points"):
        value = annotations.get(key)
        if value is not None:
            points = np.asarray(value)
            if points.size == 0:
                return 0
            return int(points.reshape(-1, points.shape[-1]).shape[0])
    image_info = annotations.get("image_info")
    if image_info is not None:
        candidates: list[np.ndarray] = []
        def collect(value: object) -> None:
            if isinstance(value, np.ndarray):
                if value.ndim == 2 and value.shape[-1] == 2 and value.dtype != object:
                    candidates.append(value)
                elif value.dtype == object:
                    for child in value.flat:
                        collect(child)
        collect(image_info)
        if candidates:
            return int(candidates[0].shape[0])
    raise ValueError(f"Could not find point annotations in {annotation_path}")


def find_annotation(image_path: Path, annotation_root: Path) -> Path:
    candidates = (
        annotation_root / f"{image_path.stem}_anno.mat",
        annotation_root / f"{image_path.stem}.mat",
        annotation_root / f"GT_{image_path.stem}.mat",
        annotation_root / f"{image_path.stem}.png",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No .mat annotation found for {image_path.name} under {annotation_root}")


def count_people_in_annotation(annotation_path: str | Path) -> int:
    """Count annotated people from supported MAT points or binary PNG dots."""
    annotation_path = Path(annotation_path)
    if annotation_path.suffix.lower() == ".png":
        with Image.open(annotation_path) as annotation:
            pixels = np.asarray(annotation.convert("L"))
        return int(np.count_nonzero(pixels == 255))
    return count_points_in_mat(annotation_path)
