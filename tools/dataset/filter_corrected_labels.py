"""Filter invisible and duplicate labels from corrected-tile annotations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CANVAS_SIZE = 512.0
CENTER_TOLERANCE = 1.0
RADIUS_RELATIVE_TOLERANCE = 0.25


def _intersects_canvas(label: list[float]) -> bool:
    center_x, center_y, radius_x, radius_y = label
    return not (
        center_x + radius_x < 0
        or center_x - radius_x > CANVAS_SIZE
        or center_y + radius_y < 0
        or center_y - radius_y > CANVAS_SIZE
    )


def _same_crater(first: list[float], second: list[float]) -> bool:
    if math.hypot(first[0] - second[0], first[1] - second[1]) > CENTER_TOLERANCE:
        return False
    for first_radius, second_radius in zip(first[2:], second[2:]):
        if abs(first_radius - second_radius) / max(first_radius, second_radius, 1e-6) > RADIUS_RELATIVE_TOLERANCE:
            return False
    return True


def _deduplicate(labels: list[list[float]]) -> tuple[list[list[float]], int]:
    kept: list[list[float]] = []
    grid: dict[tuple[int, int], list[int]] = {}
    duplicate_count = 0
    for label in labels:
        cell = (math.floor(label[0] / CENTER_TOLERANCE), math.floor(label[1] / CENTER_TOLERANCE))
        duplicate = False
        for cell_x in range(cell[0] - 1, cell[0] + 2):
            for cell_y in range(cell[1] - 1, cell[1] + 2):
                for kept_index in grid.get((cell_x, cell_y), []):
                    if _same_crater(label, kept[kept_index]):
                        duplicate = True
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if duplicate:
            duplicate_count += 1
            continue
        kept_index = len(kept)
        kept.append(label)
        grid.setdefault(cell, []).append(kept_index)
    return kept, duplicate_count


def filter_labels(labels_by_tile: dict[str, list[list[float]]]) -> tuple[dict, dict]:
    filtered = {}
    tile_reports = []
    total_input = total_visible = total_invisible = total_duplicates = 0
    for tile, labels in labels_by_tile.items():
        visible = [label for label in labels if _intersects_canvas(label)]
        deduplicated, duplicate_count = _deduplicate(visible)
        filtered[tile] = deduplicated
        total_input += len(labels)
        total_visible += len(visible)
        total_invisible += len(labels) - len(visible)
        total_duplicates += duplicate_count
        tile_reports.append(
            {
                "tile": tile,
                "input_labels": len(labels),
                "visible_labels": len(visible),
                "fully_outside_labels": len(labels) - len(visible),
                "duplicate_labels_removed": duplicate_count,
                "output_labels": len(deduplicated),
                "outside_center_labels_preserved": sum(
                    not (0 <= label[0] <= CANVAS_SIZE and 0 <= label[1] <= CANVAS_SIZE)
                    for label in deduplicated
                ),
            }
        )
    report = {
        "criteria": {
            "fully_outside_ellipses_removed": True,
            "center_tolerance_pixels": CENTER_TOLERANCE,
            "radius_relative_tolerance": RADIUS_RELATIVE_TOLERANCE,
        },
        "summary": {
            "tiles": len(labels_by_tile),
            "input_labels": total_input,
            "visible_labels": total_visible,
            "fully_outside_labels_removed": total_invisible,
            "duplicate_labels_removed": total_duplicates,
            "output_labels": total_visible - total_duplicates,
        },
        "tile_reports": tile_reports,
    }
    return filtered, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_labels", type=Path)
    parser.add_argument("output_labels", type=Path)
    parser.add_argument("output_report", type=Path)
    args = parser.parse_args()

    labels = json.loads(args.input_labels.read_text(encoding="utf-8"))
    filtered, report = filter_labels(labels)
    args.output_labels.write_text(json.dumps(filtered, indent=2) + "\n", encoding="utf-8")
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()