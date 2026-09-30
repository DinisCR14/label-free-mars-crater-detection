"""Validate image files and stage a clean dataset directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


def _validate_image(path: Path, expected_size: tuple[int, int]) -> tuple[bool, str | None]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.format != "PNG":
                return False, f"format is {image.format!r}, expected 'PNG'"
            if image.size != expected_size:
                return False, f"size is {image.size!r}, expected {expected_size!r}"
            if image.mode != "RGB":
                return False, f"mode is {image.mode!r}, expected 'RGB'"
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    return True, None


def validate_and_stage(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = "*_original.png",
    width: int = 512,
    height: int = 512,
    hardlink: bool = False,
) -> dict[str, int]:
    """Validate matching source images and stage only valid files."""
    source_dir = Path(source_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {source_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")

    paths = sorted(path for path in source_dir.rglob(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {source_dir}")
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("Source contains duplicate basenames; flat staging is ambiguous")

    valid: list[Path] = []
    invalid: list[tuple[Path, str]] = []
    for path in paths:
        is_valid, reason = _validate_image(path, (width, height))
        if is_valid:
            valid.append(path)
        else:
            invalid.append((path, reason or "validation failed"))

    output_dir.mkdir(parents=True)
    for path in valid:
        destination = output_dir / path.name
        if hardlink:
            destination.hardlink_to(path)
        else:
            shutil.copy2(path, destination)

    (output_dir / "valid-image-list.txt").write_text(
        "".join(f"{path.name}\n" for path in valid), encoding="utf-8"
    )
    (output_dir / "invalid-image-list.txt").write_text(
        "".join(f"{path.name}\t{reason}\n" for path, reason in invalid),
        encoding="utf-8",
    )
    manifest = {
        "source_dir": str(source_dir),
        "pattern": pattern,
        "validation": {
            "format": "PNG",
            "size": [width, height],
            "mode": "RGB",
            "pillow_verify": True,
        },
        "counts": {
            "scanned": len(paths),
            "valid": len(valid),
            "invalid": len(invalid),
        },
        "valid_images": [path.name for path in valid],
        "invalid_images": [
            {"name": path.name, "reason": reason} for path, reason in invalid
        ],
    }
    (output_dir / "validation-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest["counts"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pattern", default="*_original.png")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="stage hard links instead of independent file copies",
    )
    args = parser.parse_args()
    counts = validate_and_stage(
        args.source_dir,
        args.output_dir,
        pattern=args.pattern,
        width=args.width,
        height=args.height,
        hardlink=args.hardlink,
    )
    print(
        "Validated "
        + ", ".join(f"{name}={value}" for name, value in counts.items())
    )


if __name__ == "__main__":
    main()