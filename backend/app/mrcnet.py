"""Paper-inspired MRCNet for offline DLR-ACD crowd-count research.

This is an independent implementation from the published architecture, not code
released by the MRCNet authors. It is intentionally separate from the dashboard.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _vgg_block(in_channels: int, out_channels: int, convolutions: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(convolutions):
        layers.extend((nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True)))
        in_channels = out_channels
    return nn.Sequential(*layers)


class MRCNet(nn.Module):
    """VGG-16 encoder + multi-resolution decoder.

    Returns (low_resolution_density, full_resolution_density). The low map is
    one quarter of the input width and height and is the map used for counting.
    """

    def __init__(self, vgg16_weights: str | Path | None = None) -> None:
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                _vgg_block(3, 64, 2),
                _vgg_block(64, 128, 2),
                _vgg_block(128, 256, 3),
                _vgg_block(256, 512, 3),
                _vgg_block(512, 512, 3),
            ]
        )
        self.pool = nn.MaxPool2d(2, 2)

        # Top-down pathway. Each lateral 1x1 projection is added to an
        # upsampled decoder feature, then refined by two 3x3 convolutions.
        decoder_channels = (512, 256, 128, 64, 64)
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, o, 1) for c, o in zip((512, 512, 256, 128, 64),
                                                decoder_channels)]
        )
        self.up_project = nn.ModuleList(
            [nn.Conv2d(c, o, 1) for c, o in zip((512, 512, 256, 128, 64),
                                                decoder_channels)]
        )
        self.refine = nn.ModuleList(
            [
                _vgg_block(512, 512, 1),
                _vgg_block(256, 256, 2),
                _vgg_block(128, 128, 2),
                _vgg_block(64, 64, 2),
                _vgg_block(64, 64, 2),
            ]
        )
        self.count_head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.density_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(inplace=True),
        )
        self._initialize_decoder()
        if vgg16_weights is not None:
            self.load_vgg16_weights(vgg16_weights)

    def _initialize_decoder(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def load_vgg16_weights(self, weights_path: str | Path) -> None:
        """Load the feature layers from a torchvision VGG16 state dictionary."""
        checkpoint = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict):
            checkpoint = checkpoint.get("state_dict", checkpoint)
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
        # Feature convolution layer indices in torchvision VGG16 (no batch norm).
        indices = ((0, 2), (5, 7), (10, 12, 14), (17, 19, 21), (24, 26, 28))
        own = self.state_dict()
        copied = 0
        for block_index, block_indices in enumerate(indices):
            convs = [layer for layer in self.encoder[block_index] if isinstance(layer, nn.Conv2d)]
            for conv, feature_index in zip(convs, block_indices):
                for parameter in ("weight", "bias"):
                    source_key = f"features.{feature_index}.{parameter}"
                    dest_key = f"encoder.{block_index}.{2 * block_indices.index(feature_index)}.{parameter}"
                    if source_key not in checkpoint or own[dest_key].shape != checkpoint[source_key].shape:
                        raise ValueError(f"VGG16 checkpoint is missing compatible {source_key}")
                    own[dest_key] = checkpoint[source_key]
                    copied += 1
        self.load_state_dict(own)
        if copied != 26:
            raise ValueError(f"Expected 26 VGG16 parameter tensors; loaded {copied}")

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        features: list[Tensor] = []
        x = image
        for block in self.encoder:
            x = block(x)
            features.append(x)
            x = self.pool(x)

        # Start at 1/32 resolution, then fuse VGG features at 1/16, 1/8,
        # 1/4, 1/2, and full resolution. Counting uses the 1/4 feature.
        x = self.pool(features[-1])
        decoded: list[Tensor] = []
        for stage, feature in enumerate(reversed(features)):
            x = F.interpolate(x, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            x = self.lateral[stage](feature) + self.up_project[stage](x)
            x = self.refine[stage](x)
            decoded.append(x)
        low_density = self.count_head(decoded[2])
        high_density = self.density_head(decoded[-1])
        return low_density, high_density


class MRCNetCrowdModel:
    """Image-level inference helper for a locally trained reproduction."""

    def __init__(self, weights_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        checkpoint = torch.load(Path(weights_path), map_location=self.device, weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model = MRCNet().to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict_tensor(self, image: Tensor) -> tuple[float, Tensor]:
        low_density, high_density = self.model(image.to(self.device))
        return float(low_density.sum().item()), high_density[0, 0].cpu()
