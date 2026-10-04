"""Render canonical corrected labels on selected corrected images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def _overlay(image: Image.Image, labels: list[list[float]]) -> Image.Image:
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    for center_x, center_y, radius_x, radius_y in labels:
        draw.ellipse(
            (center_x - radius_x, center_y - radius_y, center_x + radius_x, center_y + radius_y),
            outline=(255, 40, 40),
            width=2,
        )
        draw.ellipse((center_x - 2, center_y - 2, center_x + 2, center_y + 2), fill=(255, 40, 40))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("targets", nargs="+")
    args = parser.parse_args()

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    for tile in args.targets:
        image_path = args.images / f"{tile}_corrected.png"
        with Image.open(image_path) as image:
            _overlay(image, labels[tile]).save(args.output / f"{tile}_canonical_overlay.png")
        print(f"{tile}: labels={len(labels[tile])}")


if __name__ == "__main__":
    main()