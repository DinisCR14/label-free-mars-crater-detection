import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

from crater_detection.fbwr import (
    FBWRConfig,
    _gradient_magnitude,
    fbwr_score,
    fbwr_scores,
)
from crater_detection.filter_coco_fbwr import filter_coco_predictions


class TestFBWR(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((64, 64), dtype=np.uint8)
        self.image[24:40, 24:40] = 255
        self.boxes = [(16, 16, 48, 48), (8, 8, 24, 24)]
        self.config = FBWRConfig(num_angles=4, rim_margin=1)

    def test_batch_scores_match_single_box_scores(self):
        batch_scores = fbwr_scores(self.image, self.boxes, config=self.config)
        single_scores = np.asarray(
            [fbwr_score(self.image, box, config=self.config) for box in self.boxes],
            dtype=np.float32,
        )
        np.testing.assert_allclose(batch_scores, single_scores)

    def test_batch_scoring_computes_gradient_once(self):
        with patch("crater_detection.fbwr._gradient_magnitude", wraps=None) as gradient:
            gradient.side_effect = lambda image: np.zeros_like(image, dtype=np.float32)
            fbwr_scores(self.image, self.boxes, config=self.config)
            gradient.assert_called_once_with(self.image)

    def test_threshold_is_inclusive(self):
        from crater_detection.fbwr import passes_fbwr

        self.assertTrue(passes_fbwr(10.0, 10.0))
        self.assertTrue(passes_fbwr(10.01, 10.0))
        self.assertFalse(passes_fbwr(9.99, 10.0))

    def test_invalid_box_is_rejected(self):
        with self.assertRaises(ValueError):
            fbwr_score(self.image, (1, 2, 1, 4), config=self.config)

    def test_coco_filter_computes_gradient_once_per_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            input_path = root / "input.json"
            output_path = root / "output.json"
            cv2.imwrite(str(image_path), self.image)
            input_path.write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": image_path.name}],
                        "annotations": [
                            {"id": 1, "image_id": 1, "bbox": [16, 16, 32, 32]},
                            {"id": 2, "image_id": 1, "bbox": [8, 8, 16, 16]},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "crater_detection.filter_coco_fbwr._gradient_magnitude",
                wraps=_gradient_magnitude,
            ) as gradient:
                filter_coco_predictions(
                    input_path, root, output_path, threshold=-1.0, device="cpu"
                )
                gradient.assert_called_once()

            output = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("fbwr_score", output["annotations"][0])
            self.assertIn("fbwr_valid_sample_fraction", output["annotations"][0])
            self.assertGreaterEqual(output["annotations"][0]["fbwr_score"], 0.0)
            report = json.loads(
                (root / "output_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["comparison"], ">=")
            self.assertEqual(report["input_annotation_count"], 2)
            self.assertEqual(report["retained_annotation_count"], 2)
            self.assertIn("valid_sample_fraction_summary", report)


if __name__ == "__main__":
    unittest.main()
