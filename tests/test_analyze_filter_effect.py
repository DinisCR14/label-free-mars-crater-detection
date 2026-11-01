import json
import tempfile
import unittest
from pathlib import Path

from crater_detection.image_manifest import ManifestImage, write_manifest
from tools.evaluation.analyze_filter_effect import analyze_filter_effect


class FilterEffectTests(unittest.TestCase):
    def test_reports_false_positive_removal_and_true_positive_retention(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "manifest.json"
            write_manifest((ManifestImage(0, "image.png", 64, 64),), manifest_path)
            mapping = json.loads(manifest_path.read_text())["fingerprint"]
            document = {
                "images": [{"id": 0, "file_name": "image.png", "width": 64, "height": 64}],
                "categories": [{"id": 1, "name": "crater"}],
                "mapping": {
                    "manifest": manifest_path.name,
                    "fingerprint": mapping,
                    "image_count": 1,
                },
                "annotations": [
                    {"id": 1, "image_id": 0, "category_id": 1, "bbox": [10, 10, 10, 10]}
                ],
            }
            before = {
                **document,
                "annotations": [
                    {"id": 10, "image_id": 0, "category_id": 1, "bbox": [10, 10, 10, 10], "score": 0.9},
                    {"id": 11, "image_id": 0, "category_id": 1, "bbox": [40, 40, 5, 5], "score": 0.8},
                ],
            }
            after = {**before, "annotations": [before["annotations"][0]]}
            gt_path = root / "ground-truth.json"
            before_path = root / "before.json"
            after_path = root / "after.json"
            output_path = root / "report.json"
            for path, payload in (
                (gt_path, document),
                (before_path, before),
                (after_path, after),
            ):
                path.write_text(json.dumps(payload))

            report = analyze_filter_effect(
                gt_path, before_path, after_path, manifest_path, output_path
            )

            self.assertEqual(report["metrics_before"]["true_positives"], 1)
            self.assertEqual(report["metrics_before"]["false_positives"], 1)
            self.assertEqual(report["filter_effect"]["removed_false_positives"], 1)
            self.assertEqual(report["filter_effect"]["removed_true_positives"], 0)
            self.assertEqual(report["filter_effect"]["false_positive_removal_rate"], 1.0)
            self.assertEqual(report["filter_effect"]["true_positive_retention_rate"], 1.0)
            self.assertEqual(
                report["size_metrics"]["medium"]["true_positive_retention_rate"], 1.0
            )
            self.assertEqual(
                report["size_metrics"]["small"]["false_positive_removal_rate"], 1.0
            )
            self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
