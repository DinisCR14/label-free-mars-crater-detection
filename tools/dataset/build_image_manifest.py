"""Build the canonical inference image manifest from COCO image metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from crater_detection.image_manifest import ManifestImage, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path, help="COCO JSON containing the images list")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.metadata.read_text(encoding="utf-8"))
    images = payload.get("images")
    if not isinstance(images, list):
        raise ValueError("Metadata must contain an images list")
    write_manifest((ManifestImage.from_json(image) for image in images), args.output)


if __name__ == "__main__":
    main()
