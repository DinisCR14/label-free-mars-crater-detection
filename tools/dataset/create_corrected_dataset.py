"""Create a corrected reference dataset in one reproducible command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_correction_coverage import audit
from build_corrected_manifest import build_manifest
from correct_reference_dataset import correct_subset
from filter_corrected_labels import filter_labels


def create_dataset(image_dir: Path, output_dir: Path, labels_path: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage_path = output_dir / "correction_coverage_manifest.json"
    correction_report_path = output_dir / "correction_report.json"
    unfiltered_labels_path = output_dir / "labels_unfiltered.json"
    filter_report_path = output_dir / "label_filter_report.json"
    labels_output_path = output_dir / "labels.json"
    manifest_path = output_dir / "dataset_manifest.json"

    coverage_report = audit(image_dir)
    coverage_path.write_text(
        json.dumps(coverage_report, indent=2) + "\n", encoding="utf-8"
    )
    coverage_by_tile = {
        tile["tile"]: {
            "status": tile["status"],
            "missing_offsets": tile["missing_offsets"],
            "required_source_tiles": tile["required_source_tiles"],
        }
        for tile in coverage_report["tile_details"]
    }

    labels_by_stem = json.loads(labels_path.read_text(encoding="utf-8"))
    correction_report = correct_subset(
        labels_by_stem,
        image_dir,
        output_dir,
        sorted(labels_by_stem),
        coverage_by_tile,
    )
    correction_report_path.write_text(
        json.dumps(correction_report, indent=2) + "\n", encoding="utf-8"
    )

    unfiltered_labels = json.loads(unfiltered_labels_path.read_text(encoding="utf-8"))
    filtered_labels, filter_report = filter_labels(unfiltered_labels)
    labels_output_path.write_text(
        json.dumps(filtered_labels, indent=2) + "\n", encoding="utf-8"
    )
    filter_report_path.write_text(
        json.dumps(filter_report, indent=2) + "\n", encoding="utf-8"
    )

    manifest = build_manifest(
        correction_report_path,
        filter_report_path,
        coverage_path,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--labels",
        type=Path,
        help="source labels JSON; defaults to <images>/labels.json",
    )
    args = parser.parse_args()
    labels_path = args.labels or args.images / "labels.json"
    create_dataset(args.images, args.output, labels_path)


if __name__ == "__main__":
    main()