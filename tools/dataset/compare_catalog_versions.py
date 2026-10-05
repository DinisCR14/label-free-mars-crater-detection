"""Compare supplied image labels with Robbins and Hynek catalogue versions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


MARS_RADIUS_KM = 3390.0
PIXELS_PER_DEGREE = 256.0
TILE_DEGREES = 2.0
BIN_DEGREES = 0.01
MATCH_TOLERANCE_DEGREES = 0.003
EXACT_TOLERANCE_DEGREES = 1e-8


CATALOG_FIELDS = {
    "2012": {
        "delimiter": "\t",
        "latitude": "LATITUDE_CIRCLE_IMAGE",
        "longitude": "LONGITUDE_CIRCLE_IMAGE",
        "diameter": "DIAM_CIRCLE_IMAGE",
    },
    "2014": {
        "delimiter": "\t",
        "latitude": "LATITUDE_CIRCLE_IMAGE",
        "longitude": "LONGITUDE_CIRCLE_IMAGE",
        "diameter": "DIAM_CIRCLE_IMAGE",
    },
    "2020": {
        "delimiter": ",",
        "latitude": "LAT_CIRC_IMG",
        "longitude": "LON_CIRC_IMG",
        "diameter": "DIAM_CIRC_IMG",
    },
}


def _read_labels(path: Path) -> tuple[list[tuple[float, float, float]], dict]:
    labels_by_tile = json.loads(path.read_text(encoding="utf-8"))
    labels = []
    tile_latitudes = []
    for tile, tile_labels in labels_by_tile.items():
        _, latitude_text, _, longitude_text = tile.split("_")
        tile_latitude = float(latitude_text)
        tile_longitude = float(longitude_text)
        tile_latitudes.append(tile_latitude)
        for center_x, center_y, radius_x, _radius_y in tile_labels:
            latitude = tile_latitude + center_y / PIXELS_PER_DEGREE
            longitude = tile_longitude + center_x / PIXELS_PER_DEGREE
            kilometers_per_degree = (
                2.0 * math.pi * MARS_RADIUS_KM / 360.0
            )
            diameter = (
                2.0
                * radius_x
                / PIXELS_PER_DEGREE
                * kilometers_per_degree
                * math.cos(math.radians(latitude))
            )
            labels.append((latitude, longitude, diameter))
    return labels, {
        "tiles": len(labels_by_tile),
        "labels": len(labels),
        "tile_latitude_min": min(tile_latitudes),
        "tile_latitude_max": max(tile_latitudes),
    }


def _read_catalog(path: Path, version: str, latitude_min: float, latitude_max: float):
    fields = CATALOG_FIELDS[version]
    rows = []
    bins: dict[tuple[int, int], list[int]] = {}
    with path.open("r", newline="", encoding="latin-1") as handle:
        reader = csv.DictReader(handle, delimiter=fields["delimiter"])
        for row in reader:
            try:
                latitude = float(row[fields["latitude"]])
                longitude = float(row[fields["longitude"]])
                diameter = float(row[fields["diameter"]])
            except (KeyError, TypeError, ValueError):
                continue
            if not latitude_min <= latitude <= latitude_max:
                continue
            if version == "2020" and longitude > 180.0:
                longitude -= 360.0
            index = len(rows)
            rows.append((latitude, longitude, diameter))
            cell = (
                math.floor(latitude / BIN_DEGREES),
                math.floor(longitude / BIN_DEGREES),
            )
            bins.setdefault(cell, []).append(index)
    return rows, bins


def _longitude_difference(first: float, second: float) -> float:
    difference = abs(first - second)
    return min(difference, 360.0 - difference)


def _match_labels(labels, catalog, bins):
    matches = []
    distances = []
    diameter_residuals = []
    for latitude, longitude, diameter in labels:
        cell_latitude = math.floor(latitude / BIN_DEGREES)
        cell_longitude = math.floor(longitude / BIN_DEGREES)
        best = None
        for latitude_cell in range(cell_latitude - 1, cell_latitude + 2):
            for longitude_cell in range(cell_longitude - 1, cell_longitude + 2):
                for index in bins.get((latitude_cell, longitude_cell), []):
                    candidate_latitude, candidate_longitude, candidate_diameter = catalog[index]
                    longitude_delta = _longitude_difference(longitude, candidate_longitude)
                    distance = math.hypot(
                        latitude - candidate_latitude,
                        longitude_delta * math.cos(math.radians(latitude)),
                    )
                    if best is None or distance < best[0]:
                        best = (distance, index, candidate_diameter)
        if best is None:
            continue
        distance, index, candidate_diameter = best
        matches.append(index)
        distances.append(distance)
        diameter_residuals.append(diameter - candidate_diameter)
    return matches, distances, diameter_residuals


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return values[min(len(values) - 1, int(len(values) * fraction))]


def _summarize(labels, catalog, bins):
    matches, distances, residuals = _match_labels(labels, catalog, bins)
    distances.sort()
    absolute_residuals = sorted(abs(value) for value in residuals)
    return {
        "catalog_rows_in_image_latitude_range": len(catalog),
        "labels_with_catalog_candidate": len(matches),
        "unique_catalog_rows_matched": len(set(matches)),
        "repeated_label_occurrences": len(matches) - len(set(matches)),
        "coordinate_matches_below_tolerance": sum(
            distance < MATCH_TOLERANCE_DEGREES for distance in distances
        ),
        "coordinate_matches_at_floating_point_precision": sum(
            distance < EXACT_TOLERANCE_DEGREES for distance in distances
        ),
        "coordinate_distance_degrees": {
            "median": _percentile(distances, 0.5),
            "p95": _percentile(distances, 0.95),
            "maximum": _percentile(distances, 1.0),
        },
        "absolute_diameter_residual_km": {
            "median": _percentile(absolute_residuals, 0.5),
            "p95": _percentile(absolute_residuals, 0.95),
            "maximum": _percentile(absolute_residuals, 1.0),
        },
    }


def compare_catalogues(labels_path: Path, catalogues: dict[str, Path]) -> dict:
    labels, label_summary = _read_labels(labels_path)
    latitude_min = label_summary["tile_latitude_min"]
    latitude_max = label_summary["tile_latitude_max"] + TILE_DEGREES
    comparisons = {}
    for version, path in catalogues.items():
        catalog, bins = _read_catalog(path, version, latitude_min, latitude_max)
        comparisons[version] = {
            "source": str(path),
            "fields": CATALOG_FIELDS[version],
            "summary": _summarize(labels, catalog, bins),
        }
    return {
        "labels_source": str(labels_path),
        "assumptions": {
            "pixel_to_degree_scale": PIXELS_PER_DEGREE,
            "tile_size_degrees": TILE_DEGREES,
            "mars_radius_km": MARS_RADIUS_KM,
            "longitude_normalized_to": "[-180, 180]",
            "catalogue_match_tolerance_degrees": MATCH_TOLERANCE_DEGREES,
            "label_diameter_recovered_from": "horizontal ellipse radius",
        },
        "labels": label_summary,
        "catalogues": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("catalog_2012", type=Path)
    parser.add_argument("catalog_2014", type=Path)
    parser.add_argument("catalog_2020", type=Path)
    parser.add_argument("output_report", type=Path)
    args = parser.parse_args()
    report = compare_catalogues(
        args.labels,
        {
            "2012": args.catalog_2012,
            "2014": args.catalog_2014,
            "2020": args.catalog_2020,
        },
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["catalogues"], indent=2))


if __name__ == "__main__":
    main()