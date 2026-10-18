import json
import tempfile
import unittest
from pathlib import Path


from crater_detection.export_coco import convert_ellipse_labels_with_metadata


class MetadataEllipseConversionTests(unittest.TestCase):
    def test_converts_corrected_filenames_without_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            labels_path = root / "labels.json"
            metadata_path = root / "predictions.json"
            output_path = root / "ground-truth.json"
            labels_path.write_text(json.dumps({
                "lat_-10.0_long_-10.0": [[100.0, 80.0, 20.0, 10.0]],
            }))
            metadata_path.write_text(json.dumps({
                "images": [{
                    "id": 7,
                    "file_name": "lat_-10.0_long_-10.0_corrected.png",
                    "width": 200,
                    "height": 200,
                }],
            }))

            convert_ellipse_labels_with_metadata(labels_path, metadata_path, output_path)

            output = json.loads(output_path.read_text())
            self.assertEqual(output["images"][0]["id"], 7)
            self.assertEqual(output["annotations"][0]["image_id"], 7)
            self.assertEqual(output["annotations"][0]["bbox"], [80.0, 70.0, 40.0, 20.0])


if __name__ == "__main__":
    unittest.main()