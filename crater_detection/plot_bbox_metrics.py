"""Compatibility wrapper for the relocated bbox diagnostic plotting tool."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from tools.evaluation.plot_bbox_diagnostics import main


if __name__ == "__main__":
    main()
