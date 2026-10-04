"""Audit source-tile coverage required by the reference correction transform."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


FILENAME_PATTERN = re.compile(
    r"^lat_(?P<lat>-?\d+(?:\.\d+)?)_long_(?P<lon>-?\d+(?:\.\d+)?)_original\.png$"
)
CANVAS_SIZE = 512.0
TILE_DEGREES = 2.0


def _parse_name(name: str) -> tuple[float, float]:
    match = FILENAME_PATTERN.match(name)
    if match is None:
        raise ValueError(f"Unsupported tile filename: {name}")
    return float(match.group("lat")), float(match.group("lon"))


def _longitude_offset(source_longitude: float, target_longitude: float) -> int:
    delta = (source_longitude - target_longitude + 180.0) % 360.0 - 180.0
    return round(delta / TILE_DEGREES)


def _required_offsets(latitude: float) -> list[int]:
    compression = math.cos(math.radians(abs(latitude)))
    source_width = CANVAS_SIZE / compression
    crop_start = (CANVAS_SIZE - source_width) / 2.0
    crop_end = crop_start + source_width
    first_offset = math.floor(crop_start / CANVAS_SIZE)
    last_offset = math.ceil(crop_end / CANVAS_SIZE) - 1
    return list(range(first_offset, last_offset + 1))


def audit(image_dir: Path) -> dict:
    tiles = {
        path.name: _parse_name(path.name)
        for path in image_dir.glob("lat_*_long_*_original.png")
    }
    rows: dict[float, list[tuple[str, float]]] = {}
    for name, (latitude, longitude) in tiles.items():
        rows.setdefault(latitude, []).append((name, longitude))
    for row in rows.values():
        row.sort(key=lambda item: item[1])

    tile_details = []
    complete_tiles = 0
    gapped_tiles = 0
    for target_name, (latitude, target_longitude) in sorted(tiles.items()):
        row = rows[latitude]
        required_offsets = _required_offsets(latitude)
        available_by_offset = {
            _longitude_offset(longitude, target_longitude): name
            for name, longitude in row
        }
        missing_offsets = [offset for offset in required_offsets if offset not in available_by_offset]
        required_names = [available_by_offset[offset] for offset in required_offsets if offset in available_by_offset]
        status = "complete" if not missing_offsets else "gapped"
        if status == "complete":
            complete_tiles += 1
        else:
            gapped_tiles += 1
        tile_details.append(
            {
                "tile": target_name.removesuffix("_original.png"),
                "latitude": latitude,
                "longitude": target_longitude,
                "required_offsets": required_offsets,
                "required_source_tiles": required_names,
                "missing_offsets": missing_offsets,
                "status": status,
            }
        )

    return {
        "transform": {
            "horizontal_model": "cos(abs(latitude)) compression",
            "tile_extent_degrees": TILE_DEGREES,
            "canvas_size": int(CANVAS_SIZE),
        },
        "summary": {
            "source_images": len(tiles),
            "complete_tiles": complete_tiles,
            "gapped_tiles": gapped_tiles,
            "complete_fraction": complete_tiles / len(tiles) if tiles else 0.0,
        },
        "tile_details": tile_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("output_report", type=Path)
    args = parser.parse_args()

    report = audit(args.images)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()