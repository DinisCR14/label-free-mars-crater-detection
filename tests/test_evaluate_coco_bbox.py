import json
import tempfile
import unittest
from pathlib import Path

from crater_detection.image_manifest import ManifestImage, write_manifest
from tools.evaluation.evaluate_coco_bbox import evaluate_coco_bbox
from tools.evaluation.plot_bbox_diagnostics import _match_counts


def _document(mapping):
    return {
        "images": [image.as_json() for image in mapping],
        "annotations": [
            {
                "id": 1,
                "image_id": 0,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "crater"}],
    }


class CocoEvaluationTests(unittest.TestCase):
    def test_size_diagnostics_match_all_predictions_to_selected_ground_truth(self):
        medium_ground_truth = {
            "bbox": [10, 10, 10, 10],
            "area": 100,
            "image_id": 0,
        }
        large_ground_truth = {
            "bbox": [40, 40, 40, 40],
            "area": 1600,
            "image_id": 0,
        }
        predictions = [
            {**medium_ground_truth, "score": 0.9},
            {**large_ground_truth, "score": 0.8},
        ]

        self.assertEqual(
            _match_counts(
                [medium_ground_truth, large_ground_truth],
                predictions,
                0.5,
                100,
                size="medium",
            ),
            (1, 0, 0),
        )

    def test_evaluates_and_writes_standard_coco_metrics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            mapping = (ManifestImage(0, "a_corrected.png", 64, 64),)
            manifest_path = root / "manifest.json"
            write_manifest(mapping, manifest_path)
            ground_truth = _document(mapping)
            prediction = {
                **ground_truth,
                "annotations": [
                    {
                        **ground_truth["annotations"][0],
                        "score": 1.0,
                    }
                ],
            }
            metadata = json.loads(manifest_path.read_text())
            ground_truth["mapping"] = {
                "manifest": manifest_path.name,
                "fingerprint": metadata["fingerprint"],
                "image_count": 1,
            }
            prediction["mapping"] = ground_truth["mapping"]
            gt_path = root / "ground-truth.json"
            prediction_path = root / "prediction.json"
            gt_path.write_text(json.dumps(ground_truth))
            prediction_path.write_text(json.dumps(prediction))

            report = evaluate_coco_bbox(
                gt_path,
                [("model", prediction_path)],
                manifest_path,
                root / "output",
            )

            self.assertEqual(report["protocol"]["evaluator"], "pycocotools.COCOeval")
            self.assertAlmostEqual(
                report["models"]["model"]["standard_coco"]["AP50"], 1.0
            )
            self.assertAlmostEqual(
                report["models"]["model"]["custom_bbox_area"]["AP50"], 1.0
            )
            self.assertTrue((root / "output" / "coco_bbox_metrics.json").is_file())

    def test_rejects_mapping_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            mapping = (ManifestImage(0, "a_corrected.png", 64, 64),)
            manifest_path = root / "manifest.json"
            write_manifest(mapping, manifest_path)
            document = _document(mapping)
            document["images"][0]["file_name"] = "wrong_corrected.png"
            path = root / "wrong.json"
            path.write_text(json.dumps(document))

            with self.assertRaisesRegex(ValueError, "image records"):
                evaluate_coco_bbox(path, [], manifest_path, root / "output")

    def test_rejects_unclipped_ground_truth_bbox(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            mapping = (ManifestImage(0, "a_corrected.png", 64, 64),)
            manifest_path = root / "manifest.json"
            write_manifest(mapping, manifest_path)
            document = _document(mapping)
            document["annotations"][0]["bbox"] = [10, 50, 20, 20]
            document["mapping"] = {
                "manifest": manifest_path.name,
                "fingerprint": json.loads(manifest_path.read_text())["fingerprint"],
                "image_count": 1,
            }
            path = root / "raw.json"
            path.write_text(json.dumps(document))

            with self.assertRaisesRegex(ValueError, "image-clipped"):
                evaluate_coco_bbox(path, [], manifest_path, root / "output")


if __name__ == "__main__":
    unittest.main()
