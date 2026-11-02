"""Audit geographic overlap among corrected manifest images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crater_detection.image_manifest import ImageManifest
from tools.dataset.prepare_training_dataset import _overlap_groups


def audit_spatial_overlap(manifest_path: str | Path) -> dict:
    manifest = ImageManifest.load(manifest_path)
    groups, footprints = _overlap_groups(manifest.images)
    rows: dict[float, int] = {}
    for footprint in footprints.values():
        rows[footprint["latitude"]] = rows.get(footprint["latitude"], 0) + 1
    return {
        "manifest": str(manifest_path),
        "source_manifest": manifest.metadata(),
        "footprint_model": (
            "corrected horizontal extent 2 degrees / "
            "cos(abs(center latitude))"
        ),
        "summary": {
            "image_count": len(manifest.images),
            "overlap_group_count": len(groups),
            "latitude_row_count": len(rows),
            "largest_group_size": max((len(group) for group in groups), default=0),
            "groups_with_multiple_images": sum(len(group) > 1 for group in groups),
        },
        "groups": [
            {
                "id": index,
                "images": [image.file_name for image in group],
                "footprints": {
                    image.file_name: footprints[image.file_name] for image in group
                },
            }
            for index, group in enumerate(groups)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = audit_spatial_overlap(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
