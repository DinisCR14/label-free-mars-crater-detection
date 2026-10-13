"""Generate uncorrected labels from the Robbins 2012 crater catalogue."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import re
from pathlib import Path

from PIL import Image

FILENAME_PATTERN = re.compile(
    r"^lat_(?P<lat>-?\d+(?:\.\d+)?)_long_(?P<lon>-?\d+(?:\.\d+)?)_original\.png$"
)
CANVAS_SIZE = 512.0
PIXELS_PER_DEGREE = 256.0
TILE_DEGREES = 2.0
MARS_RADIUS_KM = 3390.0


def _parse_name(name: str) -> tuple[float, float]:
    match = FILENAME_PATTERN.match(name)
    if match is None:
        raise ValueError(f"Unsupported tile filename: {name}")
    return float(match.group("lat")), float(match.group("lon"))


def _normalize_longitude(longitude: float) -> float:
    return (longitude + 180.0) % 360.0 - 180.0


def _longitude_delta(longitude: float, tile_longitude: float) -> float:
    return (longitude - tile_longitude + 180.0) % 360.0 - 180.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ellipse(row: dict[str, str]) -> tuple[float, float, float, float]:
    latitude = float(row["LATITUDE_CIRCLE_IMAGE"])
    longitude = _normalize_longitude(float(row["LONGITUDE_CIRCLE_IMAGE"]))
    diameter_km = float(row["DIAM_CIRCLE_IMAGE"])
    if not all(math.isfinite(value) for value in (latitude, longitude, diameter_km)):
        raise ValueError("Catalogue row contains a non-finite coordinate or diameter")
    if diameter_km <= 0:
        raise ValueError("Catalogue row contains a non-positive diameter")
    kilometres_per_degree = 2.0 * math.pi * MARS_RADIUS_KM / 360.0
    radius_km = diameter_km / 2.0
    radius_x = (
        radius_km
        / (kilometres_per_degree * math.cos(math.radians(latitude)))
        * PIXELS_PER_DEGREE
    )
    radius_y = radius_km / kilometres_per_degree * PIXELS_PER_DEGREE
    return latitude, longitude, radius_x, radius_y


def _ellipse_is_fully_black(image: Image.Image, label: list[float]) -> bool:
    grayscale = image if image.mode == "L" else image.convert("L")
    center_x, center_y, radius_x, radius_y = label
    if radius_x <= 0 or radius_y <= 0:
        return False
    left = max(0, math.floor(center_x - radius_x))
    top = max(0, math.floor(center_y - radius_y))
    right = min(grayscale.width - 1, math.ceil(center_x + radius_x))
    bottom = min(grayscale.height - 1, math.ceil(center_y + radius_y))
    if left > right or top > bottom:
        return False
    covered_pixels = 0
    for pixel_y in range(top, bottom + 1):
        for pixel_x in range(left, right + 1):
            normalized_x = (pixel_x + 0.5 - center_x) / radius_x
            normalized_y = (pixel_y + 0.5 - center_y) / radius_y
            if normalized_x * normalized_x + normalized_y * normalized_y <= 1.0:
                covered_pixels += 1
                if grayscale.getpixel((pixel_x, pixel_y)) != 0:
                    return False
    return covered_pixels > 0


def _label_center_is_black(image: Image.Image, label: list[float]) -> bool:
    grayscale = image if image.mode == "L" else image.convert("L")
    center_x, center_y, _radius_x, _radius_y = label
    pixel_x = math.floor(center_x)
    pixel_y = math.floor(center_y)
    if not (0 <= pixel_x < grayscale.width and 0 <= pixel_y < grayscale.height):
        return False
    return grayscale.getpixel((pixel_x, pixel_y)) == 0


def _historical_non_black_percentage(
    image: Image.Image, label: list[float], channel: Image.Image | None = None
) -> float:
    _center_x, _center_y, radius_x, radius_y = label
    left = max(_center_x - radius_x, 0)
    top = max(_center_y - radius_y, 0)
    right = min(_center_x + radius_x, image.width)
    bottom = min(_center_y + radius_y, image.height)
    left, top, right, bottom = (
        round(value) for value in (left, top, right, bottom)
    )
    if left >= right or top >= bottom:
        return 0.0
    if channel is None:
        channel = image if image.mode == "L" else image.convert("RGB").getchannel("B")
    content = channel.crop((left, top, right, bottom))
    histogram = content.histogram()
    return 1.0 - histogram[0] / (content.width * content.height)


def _historical_interception_percentage(label: list[float]) -> float:
    center_x, center_y, radius_x, radius_y = label
    left = center_x - radius_x
    top = center_y - radius_y
    right = center_x + radius_x
    bottom = center_y + radius_y
    if left >= CANVAS_SIZE or right <= 0 or top >= CANVAS_SIZE or bottom <= 0:
        return 0.0
    clipped_width = min(right, CANVAS_SIZE) - max(left, 0)
    clipped_height = min(bottom, CANVAS_SIZE) - max(top, 0)
    bbox_area = (right - left) * (bottom - top)
    if clipped_width <= 0 or clipped_height <= 0 or bbox_area <= 0:
        return 0.0
    return clipped_width * clipped_height / bbox_area


def _apply_historical_filter(
    labels: dict[str, list[list[float]]], image_dir: Path
) -> tuple[dict[str, list[list[float]]], dict[str, int]]:
    filtered = {}
    tile_count_removed = 0
    label_count_removed = 0
    for stem, tile_labels in labels.items():
        with Image.open(image_dir / f"{stem}_original.png") as image:
            channel = image if image.mode == "L" else image.convert("RGB").getchannel("B")
            tile_content = _historical_non_black_percentage(
                image,
                [CANVAS_SIZE / 2, CANVAS_SIZE / 2, CANVAS_SIZE / 2, CANVAS_SIZE / 2],
                channel,
            )
            if tile_content < 0.70:
                filtered[stem] = []
                tile_count_removed += 1
                label_count_removed += len(tile_labels)
                continue
            kept = [
                label
                for label in tile_labels
                if _historical_interception_percentage(label) >= 0.08
                and _historical_non_black_percentage(image, label, channel) >= 0.08
            ]
            label_count_removed += len(tile_labels) - len(kept)
            filtered[stem] = kept
    return filtered, {
        "historical_tiles_removed": tile_count_removed,
        "historical_labels_removed": label_count_removed,
    }


def _remove_fully_black_labels(
    labels: dict[str, list[list[float]]], image_dir: Path
) -> tuple[dict[str, list[list[float]]], int]:
    filtered = {}
    removed = 0
    for stem, tile_labels in labels.items():
        with Image.open(image_dir / f"{stem}_original.png") as image:
            kept = []
            for label in tile_labels:
                if _ellipse_is_fully_black(image, label):
                    removed += 1
                else:
                    kept.append(label)
        filtered[stem] = kept
    return filtered, removed


def _remove_black_center_labels(
    labels: dict[str, list[list[float]]], image_dir: Path
) -> tuple[dict[str, list[list[float]]], int]:
    filtered = {}
    removed = 0
    for stem, tile_labels in labels.items():
        with Image.open(image_dir / f"{stem}_original.png") as image:
            kept = []
            for label in tile_labels:
                if _label_center_is_black(image, label):
                    removed += 1
                else:
                    kept.append(label)
        filtered[stem] = kept
    return filtered, removed


def _tile_labels(
    image_dir: Path,
    catalogue: Path,
    *,
    inclusion: str = "intersects",
    discard_fully_black: bool = False,
    discard_black_center: bool = False,
    historical_filter: bool = False,
) -> tuple[dict[str, list[list[float]]], dict]:
    tiles = {
        path.stem.removesuffix("_original"): _parse_name(path.name)
        for path in image_dir.glob("lat_*_long_*_original.png")
    }
    if not tiles:
        raise FileNotFoundError(f"No *_original.png tiles found in {image_dir}")
    latitudes = sorted({latitude for latitude, _ in tiles.values()})
    longitudes = sorted({longitude for _, longitude in tiles.values()})
    tile_names = {(latitude, longitude): stem for stem, (latitude, longitude) in tiles.items()}
    labels = {stem: [] for stem in sorted(tiles)}
    kilometres_per_degree = 2.0 * math.pi * MARS_RADIUS_KM / 360.0
    rows_read = usable_rows = assigned_rows = assigned_occurrences = 0
    diameter_filtered_rows = 0

    with catalogue.open("r", encoding="latin-1", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "LATITUDE_CIRCLE_IMAGE",
            "LONGITUDE_CIRCLE_IMAGE",
            "DIAM_CIRCLE_IMAGE",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Catalogue is missing fields: {sorted(missing)}")
        for row_number, row in enumerate(reader, 2):
            rows_read += 1
            try:
                if historical_filter and float(row["DIAM_CIRCLE_IMAGE"]) < 0.25:
                    diameter_filtered_rows += 1
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            try:
                latitude, longitude, radius_x, radius_y = _ellipse(row)
            except (KeyError, TypeError, ValueError):
                continue
            usable_rows += 1
            latitude_min = latitude - radius_y / PIXELS_PER_DEGREE
            latitude_max = latitude + radius_y / PIXELS_PER_DEGREE
            first_latitude_index = max(
                0,
                bisect.bisect_left(
                    latitudes, latitude_min - TILE_DEGREES
                ),
            )
            last_latitude_index = min(
                len(latitudes), bisect.bisect_right(latitudes, latitude_max)
            )
            longitude_min = longitude - radius_x / PIXELS_PER_DEGREE
            longitude_max = longitude + radius_x / PIXELS_PER_DEGREE
            first_longitude_index = math.floor((longitude_min + 180.0) / TILE_DEGREES)
            last_longitude_index = math.floor((longitude_max + 180.0) / TILE_DEGREES)
            row_assigned = False
            for latitude_index in range(first_latitude_index, last_latitude_index):
                tile_latitude = latitudes[latitude_index]
                for longitude_index in range(
                    first_longitude_index, last_longitude_index + 1
                ):
                    tile_longitude = _normalize_longitude(
                        -180.0 + longitude_index * TILE_DEGREES
                    )
                    stem = tile_names.get((tile_latitude, tile_longitude))
                    if stem is None:
                        continue
                    center_x = _longitude_delta(longitude, tile_longitude) * PIXELS_PER_DEGREE
                    center_y = (latitude - tile_latitude) * PIXELS_PER_DEGREE
                    intersects = not (
                        center_x + radius_x < 0
                        or center_x - radius_x > CANVAS_SIZE
                        or center_y + radius_y < 0
                        or center_y - radius_y > CANVAS_SIZE
                    )
                    inside = (
                        0 <= center_x <= CANVAS_SIZE
                        and 0 <= center_y <= CANVAS_SIZE
                    )
                    if (inclusion == "center" and not inside) or (
                        inclusion == "intersects" and not intersects
                    ):
                        continue
                    labels[stem].append([center_x, center_y, radius_x, radius_y])
                    assigned_occurrences += 1
                    row_assigned = True
            if row_assigned:
                assigned_rows += 1

    historical_tiles_removed = historical_labels_removed = 0
    if historical_filter:
        labels, historical_report = _apply_historical_filter(labels, image_dir)
        historical_tiles_removed = historical_report["historical_tiles_removed"]
        historical_labels_removed = historical_report["historical_labels_removed"]

    black_labels_removed = 0
    if discard_fully_black:
        labels, black_labels_removed = _remove_fully_black_labels(labels, image_dir)
    black_center_labels_removed = 0
    if discard_black_center:
        labels, black_center_labels_removed = _remove_black_center_labels(labels, image_dir)

    report = {
        "policy": {
            "tile_origin": "lower-left corner",
            "tile_extent_degrees": TILE_DEGREES,
            "pixels_per_degree": PIXELS_PER_DEGREE,
            "longitude_normalization": "[-180, 180)",
            "diameter_field": "DIAM_CIRCLE_IMAGE",
            "inclusion": inclusion,
            "discard_fully_black": discard_fully_black,
            "discard_black_center": discard_black_center,
            "historical_filter": historical_filter,
            "historical_geometric_overlap_threshold": 0.08,
            "historical_non_black_threshold": 0.08,
            "historical_tile_non_black_threshold": 0.70,
            "historical_minimum_diameter_km": 0.25,
        },
        "source": {
            "catalogue": str(catalogue),
            "catalogue_sha256": _sha256(catalogue),
            "image_directory": str(image_dir),
        },
        "catalogue_rows_read": rows_read,
        "catalogue_rows_usable": usable_rows,
        "catalogue_rows_filtered_by_minimum_diameter": diameter_filtered_rows,
        "catalogue_rows_assigned": assigned_rows,
        "assigned_label_occurrences": assigned_occurrences,
        "fully_black_labels_removed": black_labels_removed,
        "black_center_labels_removed": black_center_labels_removed,
        "historical_tiles_removed": historical_tiles_removed,
        "historical_labels_removed": historical_labels_removed,
        "output_tiles": len(labels),
        "output_labels": sum(len(tile_labels) for tile_labels in labels.values()),
    }
    return labels, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalogue", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("output_labels", type=Path)
    parser.add_argument("output_report", type=Path)
    parser.add_argument(
        "--inclusion",
        choices=("intersects", "center"),
        default="intersects",
        help="include ellipses intersecting a tile, or only centers inside it",
    )
    parser.add_argument(
        "--discard-fully-black",
        action="store_true",
        help="discard labels whose covered source-image pixels are all zero",
    )
    parser.add_argument(
        "--discard-black-center",
        action="store_true",
        help="discard labels whose center pixel in the source image is zero",
    )
    parser.add_argument(
        "--historical-filter",
        action="store_true",
        help="apply the historical 0.25 km, 8%%, and 70%% image-content rules",
    )
    args = parser.parse_args()
    labels, report = _tile_labels(
        args.images,
        args.catalogue,
        inclusion=args.inclusion,
        discard_fully_black=args.discard_fully_black,
        discard_black_center=args.discard_black_center,
        historical_filter=args.historical_filter,
    )
    args.output_labels.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_labels.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()