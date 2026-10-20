import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from crater_detection.image_manifest import ImageManifest, ManifestImage, write_manifest


class ImageManifestTests(unittest.TestCase):
    def test_manifest_resolves_in_explicit_id_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_root = root / "images"
            image_root.mkdir()
            for name in ("b_corrected.png", "a_corrected.png"):
                Image.new("RGB", (4, 3)).save(image_root / name)
            manifest_path = root / "manifest.json"
            write_manifest(
                (
                    ManifestImage(0, "b_corrected.png", 4, 3),
                    ManifestImage(1, "a_corrected.png", 4, 3),
                ),
                manifest_path,
            )

            manifest = ImageManifest.load(manifest_path)
            self.assertEqual(
                [path.name for path in manifest.resolve(image_root)],
                ["b_corrected.png", "a_corrected.png"],
            )

    def test_manifest_rejects_extra_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_root = root / "images"
            image_root.mkdir()
            Image.new("RGB", (4, 3)).save(image_root / "a_corrected.png")
            Image.new("RGB", (4, 3)).save(image_root / "extra_corrected.png")
            manifest_path = root / "manifest.json"
            write_manifest((ManifestImage(0, "a_corrected.png", 4, 3),), manifest_path)

            with self.assertRaisesRegex(ValueError, "unexpected"):
                ImageManifest.load(manifest_path).resolve(image_root)

    def test_manifest_rejects_tampered_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "manifest.json"
            write_manifest((ManifestImage(0, "a_corrected.png", 4, 3),), path)
            payload = json.loads(path.read_text())
            payload["images"][0]["file_name"] = "renamed_corrected.png"
            path.write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "fingerprint"):
                ImageManifest.load(path)
