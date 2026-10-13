import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import sys


sys.path.insert(0, str(Path(__file__).parents[1] / "tools" / "dataset"))

from generate_robbins_2012_labels import (
    _ellipse_is_fully_black,
    _historical_non_black_percentage,
    _label_center_is_black,
    _tile_labels,
)


class Robbins2012LabelTests(unittest.TestCase):
    def test_generates_lower_left_coordinates_and_boundary_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            for longitude in (-2.0, 0.0):
                (image_dir / f"lat_0.0_long_{longitude:.1f}_original.png").touch()
            catalogue = root / "catalogue.tsv"
            with catalogue.open("w", encoding="latin-1", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "LATITUDE_CIRCLE_IMAGE",
                        "LONGITUDE_CIRCLE_IMAGE",
                        "DIAM_CIRCLE_IMAGE",
                    ],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "LATITUDE_CIRCLE_IMAGE": "1.0",
                        "LONGITUDE_CIRCLE_IMAGE": "0.0",
                        "DIAM_CIRCLE_IMAGE": "10.0",
                    }
                )
                writer.writerow(
                    {
                        "LATITUDE_CIRCLE_IMAGE": "1.0",
                        "LONGITUDE_CIRCLE_IMAGE": "-2.01",
                        "DIAM_CIRCLE_IMAGE": "10.0",
                    }
                )
            labels, report = _tile_labels(image_dir, catalogue)
            self.assertEqual(report["output_tiles"], 2)
            self.assertEqual(report["output_labels"], 3)
            self.assertEqual(len(labels["lat_0.0_long_0.0"]), 1)
            self.assertEqual(len(labels["lat_0.0_long_-2.0"]), 2)
            self.assertTrue(any(label[0] < 0 for label in labels["lat_0.0_long_-2.0"]))

    def test_center_inclusion_excludes_boundary_only_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            (image_dir / "lat_0.0_long_0.0_original.png").touch()
            catalogue = root / "catalogue.tsv"
            catalogue.write_text(
                "LATITUDE_CIRCLE_IMAGE\tLONGITUDE_CIRCLE_IMAGE\tDIAM_CIRCLE_IMAGE\n"
                "1.0\t-0.01\t10.0\n",
                encoding="latin-1",
            )
            labels, report = _tile_labels(
                image_dir, catalogue, inclusion="center"
            )
            self.assertEqual(report["output_labels"], 0)
            self.assertEqual(labels["lat_0.0_long_0.0"], [])

    def test_fully_black_ellipse_is_discarded(self):
        image = Image.new("RGB", (32, 32), (0, 0, 0))
        self.assertTrue(_ellipse_is_fully_black(image, [16.0, 16.0, 4.0, 3.0]))
        image.putpixel((16, 16), (1, 1, 1))
        self.assertFalse(_ellipse_is_fully_black(image, [16.0, 16.0, 4.0, 3.0]))

    def test_black_center_is_discarded_but_outside_center_is_kept(self):
        image = Image.new("L", (32, 32), 1)
        image.putpixel((16, 16), 0)
        self.assertTrue(_label_center_is_black(image, [16.2, 16.8, 4.0, 3.0]))
        self.assertFalse(_label_center_is_black(image, [-1.0, 16.0, 4.0, 3.0]))

    def test_historical_filter_uses_bbox_content_and_minimum_diameter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            image = Image.new("RGB", (512, 512), (255, 255, 255))
            image.save(image_dir / "lat_0.0_long_0.0_original.png")
            catalogue = root / "catalogue.tsv"
            catalogue.write_text(
                "LATITUDE_CIRCLE_IMAGE\tLONGITUDE_CIRCLE_IMAGE\tDIAM_CIRCLE_IMAGE\n"
                "1.0\t0.0\t10.0\n"
                "1.0\t0.0\t0.2\n",
                encoding="latin-1",
            )
            labels, report = _tile_labels(
                image_dir, catalogue, historical_filter=True
            )
            self.assertEqual(report["catalogue_rows_filtered_by_minimum_diameter"], 1)
            self.assertEqual(report["historical_labels_removed"], 0)
            self.assertEqual(report["output_labels"], 1)
            self.assertEqual(
                _historical_non_black_percentage(
                    Image.new("RGB", (16, 16), (0, 0, 0)),
                    [8.0, 8.0, 4.0, 4.0],
                ),
                0.0,
            )


if __name__ == "__main__":
    unittest.main()