"""Render supplied/new label differences on selected source tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


COLORS = {
    "shared": (40, 110, 255),
    "supplied_only": (40, 210, 80),
    "new_only": (245, 50, 45),
}


def _key(label: list[float]) -> tuple[float, ...]:
    return tuple(round(value, 6) for value in label)


def _draw_label(draw: ImageDraw.ImageDraw, label: list[float], color: tuple[int, int, int]) -> None:
    center_x, center_y, radius_x, radius_y = label
    draw.ellipse(
        (center_x - radius_x, center_y - radius_y, center_x + radius_x, center_y + radius_y),
        outline=color,
        width=2,
    )
    draw.ellipse((center_x - 2, center_y - 2, center_x + 2, center_y + 2), fill=color)


def _render(image: Image.Image, supplied: list[list[float]], new: list[list[float]]) -> Image.Image:
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    supplied_by_key = {_key(label): label for label in supplied}
    new_by_key = {_key(label): label for label in new}

    for key in supplied_by_key.keys() - new_by_key.keys():
        _draw_label(draw, supplied_by_key[key], COLORS["supplied_only"])
    for key in new_by_key.keys() - supplied_by_key.keys():
        _draw_label(draw, new_by_key[key], COLORS["new_only"])
    for key in supplied_by_key.keys() & new_by_key.keys():
        _draw_label(draw, supplied_by_key[key], COLORS["shared"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("supplied_labels", type=Path)
    parser.add_argument("new_labels", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("targets", nargs="+")
    args = parser.parse_args()

    supplied = json.loads(args.supplied_labels.read_text(encoding="utf-8"))
    new = json.loads(args.new_labels.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    for tile in args.targets:
        image_path = args.images / f"{tile}_original.png"
        with Image.open(image_path) as image:
            rendered = _render(image, supplied.get(tile, []), new.get(tile, []))
        rendered.save(args.output / f"{tile}_comparison_overlay.png")
        shared = len({_key(label) for label in supplied.get(tile, [])} & {_key(label) for label in new.get(tile, [])})
        print(
            f"{tile}: shared={shared} supplied_only={len(supplied.get(tile, [])) - shared} "
            f"new_only={len(new.get(tile, [])) - shared}"
        )


if __name__ == "__main__":
    main()
