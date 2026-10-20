"""Canonical image-to-ID mapping for inference and evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


@dataclass(frozen=True)
class ManifestImage:
    """One image record from the canonical inference manifest."""

    id: int
    file_name: str
    width: int
    height: int

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ManifestImage":
        try:
            image = cls(
                id=int(value["id"]),
                file_name=str(value["file_name"]),
                width=int(value["width"]),
                height=int(value["height"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid image record: {value!r}") from error
        if image.id < 0 or not image.file_name or image.width < 1 or image.height < 1:
            raise ValueError(f"Invalid image record: {value!r}")
        return image

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "file_name": self.file_name,
            "width": self.width,
            "height": self.height,
        }


def _fingerprint_records(images: Iterable[ManifestImage]) -> str:
    payload = [image.as_json() for image in images]
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ImageManifest:
    """Validated canonical image records and their stable fingerprint."""

    path: Path
    images: tuple[ManifestImage, ...]
    fingerprint: str

    @classmethod
    def load(cls, path: str | Path) -> "ImageManifest":
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read image manifest: {path}") from error
        raw_images = payload.get("images") if isinstance(payload, dict) else None
        if not isinstance(raw_images, list) or not raw_images:
            raise ValueError(f"Manifest must contain a non-empty images list: {path}")
        images = tuple(ManifestImage.from_json(value) for value in raw_images)
        ids = [image.id for image in images]
        names = [image.file_name for image in images]
        if ids != list(range(len(images))):
            raise ValueError("Manifest image IDs must be contiguous and start at zero")
        if len(set(names)) != len(names):
            raise ValueError("Manifest image filenames must be unique")
        fingerprint = _fingerprint_records(images)
        declared_fingerprint = payload.get("fingerprint")
        if declared_fingerprint is not None and declared_fingerprint != fingerprint:
            raise ValueError(f"Manifest fingerprint does not match its image records: {path}")
        return cls(path=path, images=images, fingerprint=fingerprint)

    def resolve(self, image_root: str | Path, *, limit: int | None = None) -> list[Path]:
        """Resolve manifest records and reject missing or unexpected corrected images."""
        image_root = Path(image_root)
        if not image_root.is_dir():
            raise FileNotFoundError(f"Image root does not exist: {image_root}")
        records = self.images if limit is None else self.images[:limit]
        expected = {image.file_name for image in self.images}
        actual = {path.name for path in image_root.glob("*_corrected.png")}
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={missing[:3]}")
            if unexpected:
                details.append(f"unexpected={unexpected[:3]}")
            raise ValueError(f"Image root does not match manifest ({'; '.join(details)})")
        paths = []
        for image in records:
            path = image_root / image.file_name
            if not path.is_file():
                raise FileNotFoundError(f"Manifest image is not a file: {path}")
            with Image.open(path) as source:
                if source.size != (image.width, image.height):
                    raise ValueError(
                        f"Image dimensions do not match manifest: {path} "
                        f"(expected {(image.width, image.height)}, got {source.size})"
                    )
            paths.append(path)
        return paths

    def metadata(self) -> dict[str, Any]:
        return {
            "manifest": self.path.name,
            "fingerprint": self.fingerprint,
            "image_count": len(self.images),
        }


def write_manifest(images: Iterable[ManifestImage], output: str | Path) -> None:
    """Write a canonical manifest with a fingerprint."""
    image_list = tuple(images)
    fingerprint = _fingerprint_records(image_list)
    payload = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "image_count": len(image_list),
        "images": [image.as_json() for image in image_list],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def validate_coco_mapping(
    payload: dict[str, Any],
    manifest: ImageManifest,
    *,
    require_fingerprint: bool = True,
) -> None:
    """Validate a COCO document against the complete canonical mapping."""
    raw_images = payload.get("images")
    if not isinstance(raw_images, list):
        raise ValueError("COCO document must contain an images list")
    images = tuple(ManifestImage.from_json(image) for image in raw_images)
    if images != manifest.images:
        raise ValueError("COCO image records do not match the canonical manifest")
    mapping = payload.get("mapping")
    if require_fingerprint and mapping != manifest.metadata():
        raise ValueError("COCO document does not contain the canonical manifest metadata")
