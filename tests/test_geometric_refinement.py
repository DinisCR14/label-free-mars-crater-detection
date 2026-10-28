import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from crater_detection.geometric_refinement import refine_geometric_annotations


def _encoded_mask(mask: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


class GeometricRefinementTests(unittest.TestCase):
    def test_keeps_original_bbox_for_edge_touching_circular_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (10, 50), 10, 1, -1)
        annotation = {
            "id": 4,
            "image_id": 0,
            "bbox": [-5, 35, 25, 30],
            "segmentation": _encoded_mask(mask),
            "score": 0.8,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            target = root / "target.json"
            source.write_text(json.dumps({
                "images": [{"id": 0, "width": 100, "height": 100}],
                "annotations": [annotation],
                "categories": [{"id": 1, "name": "crater"}],
            }))

            refine_geometric_annotations(source, target)

            output = json.loads(target.read_text())
            self.assertEqual(output["annotations"][0]["bbox"], annotation["bbox"])
            self.assertEqual(output["annotations"][0]["segmentation"], annotation["segmentation"])

    def test_discards_eccentric_mask_and_invalid_segmentation(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.ellipse(mask, (50, 50), (30, 8), 0, 0, 360, 1, -1)
        annotations = [
            {
                "id": 1,
                "image_id": 0,
                "bbox": [20, 42, 60, 16],
                "segmentation": _encoded_mask(mask),
            },
            {
                "id": 2,
                "image_id": 0,
                "bbox": [0, 0, 1, 1],
                "segmentation": {"size": [100, 100], "counts": "invalid"},
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            target = root / "target.json"
            source.write_text(json.dumps({
                "images": [{"id": 0, "width": 100, "height": 100}],
                "annotations": annotations,
            }))

            refine_geometric_annotations(source, target)

            self.assertEqual(json.loads(target.read_text())["annotations"], [])

    def test_axis_order_does_not_change_acceptance(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.ellipse(mask, (50, 50), (20, 16), 0, 0, 360, 1, -1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            target = root / "target.json"
            source.write_text(json.dumps({
                "images": [{"id": 0, "width": 100, "height": 100}],
                "annotations": [{
                    "id": 1,
                    "image_id": 0,
                    "bbox": [30, 34, 40, 32],
                    "segmentation": _encoded_mask(mask),
                }],
            }))

            refine_geometric_annotations(source, target, ellipse_min_ratio=0.7)

            self.assertEqual(len(json.loads(target.read_text())["annotations"]), 1)


if __name__ == "__main__":
    unittest.main()
