"""Flexible Black and White Rims (FBWR) scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class FBWRConfig:
    center_window_fraction: float = 0.10
    rim_margin: int = 5
    num_angles: int = 12
    threshold_uint8: float = 2000.0
    threshold_normalized: float = 0.03


def _validate_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError("box must contain four xyxy coordinates")
    x1, y1, x2, y2 = (float(value) for value in box)
    if not np.all(np.isfinite((x1, y1, x2, y2))) or x2 <= x1 or y2 <= y1:
        raise ValueError("box must be finite and have positive width and height")
    return x1, y1, x2, y2


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Return central-difference gradient magnitude without extra dependencies."""
    image = image.astype(np.float32, copy=False)
    grad_y, grad_x = np.gradient(image)
    return np.hypot(grad_x, grad_y)


def fbwr_score(
    image: np.ndarray,
    box: Sequence[float],
    *,
    config: FBWRConfig = FBWRConfig(),
) -> float:
    """Compute the flexible radial FBWR score for one grayscale image and xyxy box.

    Image values are not rescaled. Consequently, an 8-bit image produces scores
    on the raw intensity scale, while a [0, 1] image produces normalized scores.
    Sample pairs are ignored when either endpoint is outside the image.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be a two-dimensional grayscale array")
    if image.size == 0:
        return 0.0
    if config.num_angles <= 0 or config.rim_margin < 0:
        raise ValueError("num_angles must be positive and rim_margin non-negative")

    x1, y1, x2, y2 = _validate_box(box)
    height, width = image.shape
    gradients = _gradient_magnitude(image)

    box_width = x2 - x1
    box_height = y2 - y1
    half_size = min(box_width, box_height) / 2.0
    center_radius = config.center_window_fraction * half_size
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    radius_min = max(1.0, half_size - config.rim_margin)
    radius_max = half_size + config.rim_margin
    radii = range(int(np.floor(radius_min)), int(np.ceil(radius_max)) + 1)
    angles = np.arange(config.num_angles, dtype=np.float32) * np.pi / config.num_angles

    scores: list[float] = []
    for offset_y in range(int(np.floor(-center_radius)), int(np.ceil(center_radius)) + 1):
        for offset_x in range(int(np.floor(-center_radius)), int(np.ceil(center_radius)) + 1):
            sample_center_x = center_x + offset_x
            sample_center_y = center_y + offset_y
            for radius in radii:
                for angle in angles:
                    dx = radius * float(np.cos(angle))
                    dy = radius * float(np.sin(angle))
                    point_a = (int(round(sample_center_y + dy)), int(round(sample_center_x + dx)))
                    point_b = (int(round(sample_center_y - dy)), int(round(sample_center_x - dx)))
                    ay, ax = point_a
                    by, bx = point_b
                    if not (0 <= ay < height and 0 <= ax < width):
                        continue
                    if not (0 <= by < height and 0 <= bx < width):
                        continue
                    contrast = abs(float(image[ay, ax]) - float(image[by, bx]))
                    mean_gradient = (float(gradients[ay, ax]) + float(gradients[by, bx])) / 2.0
                    scores.append(contrast * mean_gradient)

    return float(np.mean(scores)) if scores else 0.0


def passes_fbwr(score: float, threshold: float) -> bool:
    """Return whether a score passes the supplied FBWR threshold."""
    return bool(score > threshold)


def fbwr_scores(
    image: np.ndarray,
    boxes: Iterable[Sequence[float]],
    *,
    config: FBWRConfig = FBWRConfig(),
) -> np.ndarray:
    """Score multiple xyxy boxes in one image and return a float array."""
    return np.asarray([fbwr_score(image, box, config=config) for box in boxes], dtype=np.float32)


def fbwr_filter_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    *,
    threshold: float,
    config: FBWRConfig = FBWRConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return kept boxes, their scores, and the Boolean keep mask."""
    boxes = np.asarray(boxes)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [N, 4] in xyxy format")
    scores = fbwr_scores(image, boxes, config=config)
    keep = scores > threshold
    return boxes[keep], scores, keep


def fbwr_score_torch(
    image,
    box,
    *,
    config: FBWRConfig = FBWRConfig(),
):
    """Compute the FBWR score for one grayscale tensor and one xyxy box.

    The returned scalar remains on the image device, including when the image
    is stored on a CUDA device.
    """
    import torch

    if image.ndim != 2:
        raise ValueError("image must be a two-dimensional grayscale tensor")
    if hasattr(box, "detach"):
        box = box.detach().cpu().tolist()
    x1, y1, x2, y2 = _validate_box(box)
    height, width = image.shape
    image = image.float()
    grad_y, grad_x = torch.gradient(image)
    gradients = torch.hypot(grad_x, grad_y)
    values = []
    box_width, box_height = x2 - x1, y2 - y1
    half_size = min(box_width, box_height) / 2.0
    center_radius = config.center_window_fraction * half_size
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    radius_min = max(1.0, half_size - config.rim_margin)
    radius_max = half_size + config.rim_margin
    radii = range(int(np.floor(radius_min)), int(np.ceil(radius_max)) + 1)
    angles = np.arange(config.num_angles, dtype=np.float32) * np.pi / config.num_angles

    for offset_y in range(int(np.floor(-center_radius)), int(np.ceil(center_radius)) + 1):
        for offset_x in range(int(np.floor(-center_radius)), int(np.ceil(center_radius)) + 1):
            for radius in radii:
                for angle in angles:
                    dx = radius * float(np.cos(angle))
                    dy = radius * float(np.sin(angle))
                    ay, ax = int(round(center_y + offset_y + dy)), int(round(center_x + offset_x + dx))
                    by, bx = int(round(center_y + offset_y - dy)), int(round(center_x + offset_x - dx))
                    if 0 <= ay < height and 0 <= ax < width and 0 <= by < height and 0 <= bx < width:
                        values.append(torch.abs(image[ay, ax] - image[by, bx]) * (gradients[ay, ax] + gradients[by, bx]) / 2.0)

    return torch.stack(values).mean() if values else image.new_zeros(())
