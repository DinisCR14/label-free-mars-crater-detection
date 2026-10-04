"""Build a per-tile manifest for a corrected reference dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _index_by_tile(
    report: dict[str, Any],
    detail_key: str = "tile_reports",
) -> dict[str, dict[str, Any]]:
    return {item["tile"]: item for item in report.get(detail_key, [])}


def build_manifest(
    correction_report_path: Path,
    filter_report_path: Path,
    coverage_manifest_path: Path,
) -> dict[str, Any]:
    correction_report = json.loads(correction_report_path.read_text(encoding="utf-8"))
    filter_report = json.loads(filter_report_path.read_text(encoding="utf-8"))
    coverage_manifest = json.loads(coverage_manifest_path.read_text(encoding="utf-8"))

    correction_by_tile = _index_by_tile(correction_report)
    filter_by_tile = _index_by_tile(filter_report)
    coverage_by_tile = _index_by_tile(coverage_manifest, "tile_details")
    tiles = sorted(correction_by_tile)

    missing_filter_tiles = sorted(set(tiles) - set(filter_by_tile))
    missing_coverage_tiles = sorted(set(tiles) - set(coverage_by_tile))
    if missing_filter_tiles or missing_coverage_tiles:
        raise ValueError(
            "reports do not describe the same tiles: "
            f"missing filter={missing_filter_tiles}, "
            f"missing coverage={missing_coverage_tiles}"
        )

    tile_records = []
    for tile in tiles:
        correction = correction_by_tile[tile]
        filtered = filter_by_tile[tile]
        coverage = coverage_by_tile[tile]
        tile_records.append(
            {
                "tile": tile,
                "corrected_image": f"{tile}_corrected.png",
                "canonical_labels": "labels.json",
                "unfiltered_labels": "labels_unfiltered.json",
                "canonical_label_count": filtered["output_labels"],
                "unfiltered_label_count": correction["output_labels"],
                "filtering": {
                    "fully_outside_labels_removed": filtered["fully_outside_labels"],
                    "duplicate_labels_removed": filtered["duplicate_labels_removed"],
                    "outside_center_labels_preserved": filtered[
                        "outside_center_labels_preserved"
                    ],
                },
                "coverage": {
                    "status": coverage["status"],
                    "missing_offsets": coverage["missing_offsets"],
                    "required_source_tiles": coverage["required_source_tiles"],
                },
            }
        )

    return {
        "source": {
            "correction_report": correction_report_path.name,
            "filter_report": filter_report_path.name,
            "coverage_manifest": coverage_manifest_path.name,
        },
        "summary": {
            "tiles": len(tile_records),
            "canonical_labels": sum(item["canonical_label_count"] for item in tile_records),
            "unfiltered_labels": sum(item["unfiltered_label_count"] for item in tile_records),
            "complete_tiles": sum(
                item["coverage"]["status"] == "complete" for item in tile_records
            ),
            "gapped_tiles": sum(
                item["coverage"]["status"] == "gapped" for item in tile_records
            ),
        },
        "tiles": tile_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("correction_report", type=Path)
    parser.add_argument("filter_report", type=Path)
    parser.add_argument("coverage_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    args = parser.parse_args()

    manifest = build_manifest(
        args.correction_report,
        args.filter_report,
        args.coverage_manifest,
    )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
