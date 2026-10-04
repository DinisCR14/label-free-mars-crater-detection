"""Audit reference labels without filtering edge-overlap craters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CANVAS_SIZE = 512.0


def _intersects_canvas(label: list[float]) -> bool:
    center_x, center_y, radius_x, radius_y = label
    return not (
        center_x + radius_x < 0
        or center_x - radius_x > CANVAS_SIZE
        or center_y + radius_y < 0
        or center_y - radius_y > CANVAS_SIZE
    )


def audit(labels_by_tile: dict[str, list[list[float]]]) -> dict:
    tile_details = []
    total_labels = 0
    outside_center_labels = 0
    visible_ellipse_labels = 0
    fully_outside_labels = 0

    for tile, labels in labels_by_tile.items():
        outside_centers = [
            label
            for label in labels
            if not (0 <= label[0] <= CANVAS_SIZE and 0 <= label[1] <= CANVAS_SIZE)
        ]
        visible_ellipses = [label for label in labels if _intersects_canvas(label)]
        fully_outside = len(labels) - len(visible_ellipses)
        total_labels += len(labels)
        outside_center_labels += len(outside_centers)
        visible_ellipse_labels += len(visible_ellipses)
        fully_outside_labels += fully_outside
        tile_details.append(
            {
                "tile": tile,
                "total_labels": len(labels),
                "outside_center_labels": len(outside_centers),
                "visible_ellipse_labels": len(visible_ellipses),
                "fully_outside_labels": fully_outside,
            }
        )

    return {
        "policy": {
            "preserve_all_labels": True,
            "center_outside_canvas_is_not_discarded": True,
            "fully_outside_ellipse_is_not_discarded": True,
        },
        "summary": {
            "images": len(labels_by_tile),
            "total_labels": total_labels,
            "outside_center_labels": outside_center_labels,
            "visible_ellipse_labels": visible_ellipse_labels,
            "fully_outside_labels": fully_outside_labels,
        },
        "tile_details": tile_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("output_report", type=Path)
    args = parser.parse_args()

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    report = audit(labels)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()