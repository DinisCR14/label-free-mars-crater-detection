"""Create a dry-run corrected image and label subset from reference labels."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw


FILENAME_PATTERN = re.compile(
    r"^lat_(?P<lat>-?\d+(?:\.\d+)?)_long_(?P<lon>-?\d+(?:\.\d+)?)_original\.png$"
)
CANVAS_SIZE = 512
TILE_DEGREES = 2.0


def _parse_name(name: str) -> tuple[float, float]:
    match = FILENAME_PATTERN.match(name)
    if match is None:
        raise ValueError(f"Unsupported tile filename: {name}")
    return float(match.group("lat")), float(match.group("lon"))


def _longitude_offset(source_longitude: float, target_longitude: float) -> int:
    delta = (source_longitude - target_longitude + 180.0) % 360.0 - 180.0
    return round(delta / TILE_DEGREES)


def _correct_tile(
    image_dir: Path,
    row_tiles: list[tuple[str, float]],
    target_name: str,
) -> tuple[Image.Image, float, float, list[tuple[str, int]]]:
    target_latitude, target_longitude = _parse_name(target_name)
    compression = math.cos(math.radians(abs(target_latitude)))
    source_width = CANVAS_SIZE / compression
    crop_start = (CANVAS_SIZE - source_width) / 2.0
    crop_end = crop_start + source_width
    strip_start = math.floor(crop_start / CANVAS_SIZE) * CANVAS_SIZE
    strip_end = math.ceil(crop_end / CANVAS_SIZE) * CANVAS_SIZE
    strip = Image.new("RGB", (strip_end - strip_start, CANVAS_SIZE), (0, 0, 0))
    included_tiles = []

    for source_name, source_longitude in row_tiles:
        tile_offset = _longitude_offset(source_longitude, target_longitude)
        source_start = tile_offset * CANVAS_SIZE
        x_position = source_start - strip_start
        if x_position >= strip.width or x_position + CANVAS_SIZE <= 0:
            continue
        with Image.open(image_dir / source_name) as source:
            strip.paste(source.convert("RGB"), (x_position, 0))
        included_tiles.append((source_name, source_start))

    crop = strip.crop((crop_start - strip_start, 0, crop_end - strip_start, CANVAS_SIZE))
    corrected = crop.resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.BILINEAR)
    return corrected, compression, crop_start, included_tiles


def _transform_labels(
    labels_by_tile: dict[str, list[list[float]]],
    included_tiles: list[tuple[str, int]],
    crop_start: float,
    compression: float,
) -> tuple[list[list[float]], int]:
    transformed = []
    source_label_count = 0
    for source_name, source_start in included_tiles:
        for center_x, center_y, radius_x, radius_y in labels_by_tile.get(source_name, []):
            source_label_count += 1
            transformed.append(
                [
                    (source_start + center_x - crop_start) * compression,
                    center_y,
                    radius_x * compression,
                    radius_y,
                ]
            )
    return transformed, source_label_count


def _overlay(image: Image.Image, labels: list[list[float]]) -> Image.Image:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    for center_x, center_y, radius_x, radius_y in labels:
        draw.ellipse(
            (center_x - radius_x, center_y - radius_y, center_x + radius_x, center_y + radius_y),
            outline=(255, 40, 40),
            width=2,
        )
        draw.ellipse((center_x - 2, center_y - 2, center_x + 2, center_y + 2), fill=(255, 40, 40))
    return output


def correct_subset(
    labels_by_stem: dict[str, list[list[float]]],
    image_dir: Path,
    output_dir: Path,
    targets: list[str],
    coverage_by_tile: dict[str, dict] | None = None,
) -> dict:
    available_tiles = {
        path.name: _parse_name(path.name)
        for path in image_dir.glob("lat_*_long_*_original.png")
    }
    labels_by_tile = {
        f"{stem}_original.png": labels for stem, labels in labels_by_stem.items()
    }
    output_labels = {}
    tile_reports = []

    for target_stem in targets:
        target_name = f"{target_stem}_original.png"
        if target_name not in available_tiles:
            raise FileNotFoundError(image_dir / target_name)
        target_latitude, _ = available_tiles[target_name]
        row_tiles = sorted(
            (name, longitude)
            for name, (latitude, longitude) in available_tiles.items()
            if latitude == target_latitude
        )
        corrected, compression, crop_start, included_tiles = _correct_tile(
            image_dir, row_tiles, target_name
        )
        transformed, source_label_count = _transform_labels(
            labels_by_tile, included_tiles, crop_start, compression
        )
        output_labels[target_stem] = transformed
        corrected.save(output_dir / f"{target_stem}_corrected.png")
        _overlay(corrected, transformed).save(output_dir / f"{target_stem}_corrected_overlay.png")
        tile_reports.append(
            {
                "tile": target_stem,
                "latitude": target_latitude,
                "compression": compression,
                "included_source_tiles": len(included_tiles),
                "source_labels_included": source_label_count,
                "output_labels": len(transformed),
                "discarded_labels": source_label_count - len(transformed),
                "outside_center_labels": sum(
                    not (0 <= label[0] <= CANVAS_SIZE and 0 <= label[1] <= CANVAS_SIZE)
                    for label in transformed
                ),
                "coverage": (
                    coverage_by_tile.get(target_stem, {"status": "unknown"})
                    if coverage_by_tile is not None
                    else {"status": "not_provided"}
                ),
            }
        )

    (output_dir / "labels_unfiltered.json").write_text(
        json.dumps(output_labels, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "policy": {
            "preserve_all_labels": True,
            "discarded_labels": False,
            "image_transform": "horizontal compression and longitude-row recentering",
        },
        "tile_reports": tile_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("targets", nargs="*")
    parser.add_argument("--coverage-manifest", type=Path)
    parser.add_argument(
        "--all",
        action="store_true",
        dest="process_all",
        help="process every source image with a matching label entry",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if args.process_all:
        targets = sorted(labels)
    elif args.targets:
        targets = args.targets
    else:
        parser.error("provide target stems or use --all")
    coverage_by_tile = None
    if args.coverage_manifest is not None:
        coverage_report = json.loads(args.coverage_manifest.read_text(encoding="utf-8"))
        coverage_by_tile = {
            tile["tile"]: {
                "status": tile["status"],
                "missing_offsets": tile["missing_offsets"],
                "required_source_tiles": tile["required_source_tiles"],
            }
            for tile in coverage_report["tile_details"]
        }
    report = correct_subset(labels, args.images, args.output, targets, coverage_by_tile)
    (args.output / "correction_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()