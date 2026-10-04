"""Generate catalog labels for images that were already distortion-corrected."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
from pathlib import Path


FILENAME_PATTERN = re.compile(
    r"^lat_(?P<lat>-?\d+(?:\.\d+)?)_long_(?P<lon>-?\d+(?:\.\d+)?)_original\.png$"
)
TILE_DEGREES = 2.0
PIXELS_PER_DEGREE = 256.0
MARS_RADIUS_KM = 3390.0
CANVAS_SIZE = 512


def _parse_name(name: str) -> tuple[float, float]:
    match = FILENAME_PATTERN.match(name)
    if match is None:
        raise ValueError(f"Unsupported tile filename: {name}")
    return float(match.group("lat")), float(match.group("lon"))


def _normalize_longitude(longitude: float) -> float:
    return (longitude + 180.0) % 360.0 - 180.0


def _longitude_offset(source_lon: float, target_lon: float) -> int:
    delta = (source_lon - target_lon + 180.0) % 360.0 - 180.0
    return round(delta / TILE_DEGREES)


def _find_source_tile(
    latitude: float,
    longitude: float,
    row_tiles: dict[float, list[tuple[str, float]]],
    row_latitudes: list[float],
) -> tuple[str, float] | None:
    row_index = bisect.bisect_right(row_latitudes, latitude) - 1
    if row_index < 0:
        return None
    tile_latitude = row_latitudes[row_index]
    if not tile_latitude <= latitude < tile_latitude + TILE_DEGREES:
        return None

    tiles = row_tiles[tile_latitude]
    longitudes = [tile_longitude for _, tile_longitude in tiles]
    longitude_index = bisect.bisect_right(longitudes, longitude) - 1
    candidate_indices = {longitude_index, longitude_index + 1, longitude_index - 1}
    for index in candidate_indices:
        if not 0 <= index < len(tiles):
            continue
        name, tile_longitude = tiles[index]
        relative_longitude = (longitude - tile_longitude) % 360.0
        if relative_longitude < TILE_DEGREES:
            return name, relative_longitude
    return None


def _catalog_labels_by_tile(
    catalog: Path,
    row_tiles: dict[float, list[tuple[str, float]]],
) -> tuple[dict[str, list[list[float]]], dict[str, int]]:
    row_latitudes = sorted(row_tiles)
    labels_by_tile = {
        name: [] for tiles in row_tiles.values() for name, _ in tiles
    }
    catalog_rows = 0
    accepted_rows = 0

    with catalog.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"LAT_CIRC_IMG", "LON_CIRC_IMG", "DIAM_CIRC_IMG"}
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(f"Catalog is missing fields: {sorted(missing_fields)}")

        for row in reader:
            catalog_rows += 1
            try:
                latitude = float(row["LAT_CIRC_IMG"])
                longitude = _normalize_longitude(float(row["LON_CIRC_IMG"]))
                diameter_km = float(row["DIAM_CIRC_IMG"])
            except (TypeError, ValueError):
                continue
            if not (
                math.isfinite(latitude)
                and math.isfinite(longitude)
                and math.isfinite(diameter_km)
                and diameter_km > 0
            ):
                continue

            source = _find_source_tile(latitude, longitude, row_tiles, row_latitudes)
            if source is None:
                continue
            tile_name, relative_longitude = source
            tile_latitude, _ = _parse_name(tile_name)
            kilometers_per_degree = 2.0 * math.pi * MARS_RADIUS_KM / 360.0
            kilometers_per_longitude_degree = kilometers_per_degree * math.cos(
                math.radians(tile_latitude)
            )
            if kilometers_per_longitude_degree <= 0:
                continue
            radius_km = diameter_km / 2.0
            labels_by_tile[tile_name].append(
                [
                    relative_longitude * PIXELS_PER_DEGREE,
                    (latitude - tile_latitude) * PIXELS_PER_DEGREE,
                    radius_km
                    / kilometers_per_longitude_degree
                    * PIXELS_PER_DEGREE,
                    radius_km / kilometers_per_degree * PIXELS_PER_DEGREE,
                ]
            )
            accepted_rows += 1

    return labels_by_tile, {
        "catalog_rows_read": catalog_rows,
        "catalog_rows_assigned": accepted_rows,
    }


def _transform_labels(
    labels_by_tile: dict[str, list[list[float]]],
    row_tiles: list[tuple[str, float]],
    target_name: str,
) -> list[list[float]]:
    target_latitude, target_longitude = _parse_name(target_name)
    compression = math.cos(math.radians(abs(target_latitude)))
    corrected_tile_width = CANVAS_SIZE * compression
    corrected_tile_start = (CANVAS_SIZE - corrected_tile_width) / 2.0
    transformed: list[list[float]] = []

    for source_name, source_longitude in row_tiles:
        tile_offset = _longitude_offset(source_longitude, target_longitude)
        x_shift = corrected_tile_start + tile_offset * corrected_tile_width
        if x_shift >= CANVAS_SIZE or x_shift + corrected_tile_width <= 0:
            continue
        for center_x, center_y, radius_x, radius_y in labels_by_tile[source_name]:
            corrected_x = x_shift + center_x * compression
            corrected_radius_x = radius_x * compression
            if (
                corrected_x + corrected_radius_x < 0
                or corrected_x - corrected_radius_x > CANVAS_SIZE
                or center_y + radius_y < 0
                or center_y - radius_y > CANVAS_SIZE
            ):
                continue
            transformed.append(
                [corrected_x, center_y, corrected_radius_x, radius_y]
            )
    return transformed


def generate_corrected_labels(
    catalog: Path,
    image_dir: Path,
) -> tuple[dict[str, list[list[float]]], dict[str, object]]:
    tiles = {
        path.name: _parse_name(path.name)
        for path in image_dir.glob("lat_*_long_*_original.png")
    }
    row_tiles: dict[float, list[tuple[str, float]]] = {}
    for name, (latitude, longitude) in tiles.items():
        row_tiles.setdefault(latitude, []).append((name, longitude))
    for tile_row in row_tiles.values():
        tile_row.sort(key=lambda item: item[1])

    labels_by_tile, catalog_report = _catalog_labels_by_tile(catalog, row_tiles)
    labels = {}
    for tile_row in row_tiles.values():
        for target_name, _ in tile_row:
            stem = target_name.removesuffix("_original.png")
            labels[stem] = _transform_labels(labels_by_tile, tile_row, target_name)

    report = {
        **catalog_report,
        "image_count": len(tiles),
        "latitude_count": len(row_tiles),
        "labels_total": sum(len(tile_labels) for tile_labels in labels.values()),
        "labels_per_image": {
            "minimum": min(map(len, labels.values()), default=0),
            "maximum": max(map(len, labels.values()), default=0),
            "mean": (
                sum(map(len, labels.values())) / len(labels)
                if labels
                else 0.0
            ),
        },
        "assumptions": {
            "images_modified": False,
            "tile_extent_degrees": TILE_DEGREES,
            "pixels_per_degree": PIXELS_PER_DEGREE,
            "neighbor_order": "2-degree longitude offsets; missing tiles remain gaps",
            "horizontal_correction": "cos(abs(latitude)) compression",
            "vertical_correction": "none",
            "catalog_longitudes": "normalized from 0..360 to -180..180",
        },
    }
    return labels, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("output_labels", type=Path)
    parser.add_argument("output_report", type=Path)
    args = parser.parse_args()

    labels, report = generate_corrected_labels(args.catalog, args.images)
    args.output_labels.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_labels.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()