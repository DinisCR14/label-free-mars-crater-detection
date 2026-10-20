"""Validate prediction or ground-truth COCO files against an image manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from crater_detection.image_manifest import ImageManifest, validate_coco_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("coco", type=Path, nargs="+")
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="validate image records while allowing missing mapping metadata",
    )
    args = parser.parse_args()
    manifest = ImageManifest.load(args.manifest)
    for path in args.coco:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_coco_mapping(
            payload, manifest, require_fingerprint=not args.allow_legacy
        )
        print(f"OK {path}")


if __name__ == "__main__":
    main()
