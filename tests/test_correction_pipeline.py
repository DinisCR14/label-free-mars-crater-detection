import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "tools" / "dataset"))

from filter_corrected_labels import _deduplicate, _intersects_canvas, filter_labels


class CorrectedLabelTests(unittest.TestCase):
    def test_intersecting_edge_label_is_preserved(self):
        self.assertTrue(_intersects_canvas([-3.0, 100.0, 5.0, 5.0]))
        self.assertFalse(_intersects_canvas([-6.0, 100.0, 5.0, 5.0]))

    def test_near_identical_labels_are_deduplicated(self):
        labels = [
            [100.0, 100.0, 20.0, 18.0],
            [100.5, 100.4, 20.2, 18.1],
            [140.0, 100.0, 20.0, 18.0],
        ]
        filtered, duplicate_count = _deduplicate(labels)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(duplicate_count, 1)

    def test_filter_report_counts_and_preserves_outside_center(self):
        labels = {
            "tile": [
                [-3.0, 100.0, 5.0, 5.0],
                [600.0, 100.0, 5.0, 5.0],
            ]
        }
        filtered, report = filter_labels(labels)
        self.assertEqual(len(filtered["tile"]), 1)
        self.assertEqual(report["summary"]["fully_outside_labels_removed"], 1)
        self.assertEqual(report["tile_reports"][0]["outside_center_labels_preserved"], 1)


if __name__ == "__main__":
    unittest.main()
