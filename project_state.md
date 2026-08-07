# Project State

## 1. Project Overview

The goal is to build a clean GitHub repository for the IGARSS 2026 paper on label-free Martian crater detection. The repository is being reconstructed from the available paper, notebooks, scripts, and university backup, including the missing `Unsupervised` pipeline folder.

The intended pipeline is:

THEMIS Mars imagery -> Grounded DINO/SAM pseudo-labels -> FBWR filtering and refinement -> Cascade Mask R-CNN training -> iterative self-training -> final standard-loss detector -> evaluation.

## 2. Completed Milestones

- Read and mapped the paper's full crater-detection pipeline.
- Audited the workspace and compared `ExtraCode/` with `university_backup/`.
- Rebuilt `crater_detection/fbwr.py` with:
  - Flexible Black and White Rims scoring.
  - NumPy image scoring.
  - Torch/CUDA image scoring.
  - Threshold and box-filter helpers.
- Rebuilt `crater_detection/grounded_sam.py` with configurable:
  - Grounding DINO model ID.
  - SAM checkpoint and model type.
  - Prompt.
  - Box and text thresholds.
  - Device.
  - Dataset root, image list, and output JSON paths.
- Rebuilt `crater_detection/filter_coco_fbwr.py` to apply FBWR to COCO annotations, converting COCO `[x, y, width, height]` boxes to internal `xyxy` coordinates.
- Rebuilt `crater_detection/export_coco.py` to emit and validate the Detectron2-compatible COCO schema.
- Rebuilt `crater_detection/merge_self_training.py` to:
  - Filter new predictions by confidence.
  - Compare new and previous masks by IoU.
  - Replace old labels when maximum mask IoU is at least `0.5`.
  - Preserve non-overlapping old labels.
  - Reindex annotations and emit COCO output.
- Rebuilt `scripts/run_training_pipeline.py` to orchestrate three self-training rounds and final standard-loss training.
- Confirmed the new modules compile and have no reported VS Code diagnostics.
- Validated the COCO exporter with a dependency-free schema check.
- Validated the training orchestrator with a complete dry run.
- Implemented `crater_detection/geometric_boundary_refinement.py` for the documented post-Grounded-SAM refinement:
  - Exact one-boundary COCO box correction into a square, allowing coordinates outside the image.
  - Removal of boxes touching two or more image boundaries.
  - OpenCV mask ellipse filtering with an x-axis/y-axis ratio in `[0.7, 1.3]`.
  - Preservation of corrected boxes' original SAM masks and mask areas.
  - Retention of masks that cannot produce a valid ellipse.
- Implemented `scripts/run_label_generation.py` to orchestrate Grounded SAM -> geometric/boundary refinement -> FBWR while preserving `grounded_sam.json`, `refined.json`, and `fbwr_filtered.json` intermediate outputs.
- Added a curated `tools/dataset/` layer with external THEMIS source documentation and a portable download script. The legacy supervisor-provided generator remains excluded because it contains hardcoded paths and an existing syntax defect; it is not required by the detector pipeline.
- Added an MIT license for repository-authored code, while retaining separate upstream notices for vendored dependencies and external data.
- Added `CITATION.cff` and `docs/citation.bib` for the IGARSS 2026 paper, marked `forthcoming` until publication after the conference presentation.
- Deliberately left the existing detector-side ROIAlign/Sobel scorer in CutLER unchanged for now.

## 3. Current State & Next Steps

The rebuild has completed the main implementation work through Step 7: integrating the thesis CutLER fork with the clean pipeline.

The recovered CutLER launcher is `third_party/CutLER/cutler/train_net.py`. It imports the thesis-specific CutLER engine, modeling registry, evaluator, and crater dataset registration, and supports the runner's dataset overrides. `scripts/run_training_pipeline.py` defaults its launcher and crater config paths to this vendored fork, exposes it on `PYTHONPATH`, and resolves all filesystem arguments to absolute paths. The final standard-loss config remains explicit because the backup does not contain a distinct standard-loss YAML.

Step 7 layout decision and integration are complete. The verified thesis CutLER fork and bundled Detectron2 source are vendored under `third_party/CutLER/`, excluding datasets, checkpoints, caches, bytecode, and macOS artifacts. The runner retains `--cutler-root` as an override for experiments.

Step 8 packaging is in place: root documentation, dependency/setup files, curated dataset notes, and a combined workflow entry point are present. CPU validation and dry-run verification are the acceptance checks for this environment. The active Python environment has the project-compatible `opencv-python==4.6.0.66`, `numpy==1.24.4`, and `pycocotools==2.0.11`.

The final public repository name is `label-free-mars-crater-detection`. `university_backup/` is archival source material and must not be included in the final repository. The vendored `third_party/CutLER/` tree is the publishable source of truth.

The working project identity is `Label-Free Crater Detection using Grounded SAM and Domain-Guided Self-Training`, with repository slug `label-free-mars-crater-detection`. This terminology is preferred over “unsupervised” because the foundation models provide prior training but no crater-specific labels are used in the project.

`allexceptdata/` is supervisor-provided thesis provenance material, not a dataset copy: the current folder contains only approximately 108 KB of documentation and scripts. The relevant Mars workflow tools (`tools/download-dataset.sh`, `tools/gen_dataset.py`, `tools/gencsv.py`, and `tools/visualize.py`) were used to obtain and prepare the data at the beginning of the thesis, although they were not used by the later detector pipeline. The unrelated MNIST utilities and local artifacts should be excluded during cleanup. The final repository may include the curated Mars tools and documentation, while the downloaded imagery and source crater CSV remain external. README documentation should link to the public THEMIS Image Explorer and NASA Planetary Data System and preserve appropriate source attribution.

The thesis training sequence is confirmed:

1. Initial training uses `cascade_mask_rcnn_R_50_FPN_modified.yaml` with learning rate `0.01` and `4000` iterations.
2. Each self-training round uses `cascade_mask_rcnn_R_50_FPN_self_train_modified.yaml` with learning rate `0.001`, FBWR-Enhanced DropLoss enabled, and `4000` iterations.
3. Final training uses the same self-training configuration family with DropLoss disabled and `40000` iterations.

The final configuration is represented as a derived, documented final-training config, without changing the thesis self-training config used for the three rounds.

Step 8 decisions now established:

- Curate the relevant supervisor-provided Mars preparation tools into `tools/dataset/` rather than publishing all of `allexceptdata/`.
- Add one top-level command for Grounded SAM -> geometric/boundary refinement -> FBWR, while documenting each stage as an independently runnable command.
- Add a dedicated final-training YAML derived from the thesis self-training YAML, with `USE_DROPLOSS: False` and `MAX_ITER: 40000`.
- Extend the training runner to launch the initial training stage before the three self-training rounds.
- Keep datasets, checkpoints, and generated outputs external to the repository.
- Record CutLER provenance as the upstream project `https://github.com/facebookresearch/CutLER`; the upstream page currently shows commit `cca0a27`, but the thesis fork is not assumed to match that commit exactly.

## 4. Technical Context

### Key thresholds and constants

- Grounding DINO prompt: `circle` by default.
- Grounding DINO box threshold: `0.30`.
- Grounding DINO text threshold: `0.25`.
- Initial raw-image FBWR threshold: `2000` for 8-bit imagery.
- Ellipse x-axis/y-axis ratio: `[0.7, 1.3]` for non-boundary-corrected masks.
- Normalized training FBWR threshold: `0.03` for `[0, 1]` imagery.
- Self-training confidence thresholds: `0.70`, `0.65`, `0.60` for rounds 1, 2, and 3.
- Self-training replacement mask-IoU threshold: `0.50`.
- FBWR center-window fraction: `0.10`.
- FBWR rim margin: `5` pixels.
- FBWR angular samples: `12` over `[0, pi)`.
- FBWR sample pairs are skipped when either point is outside the image.

### Coordinate conventions

- COCO JSON annotations use `bbox: [x, y, width, height]`.
- Internal FBWR and Detectron2 box operations use `(x1, y1, x2, y2)`.
- `filter_coco_fbwr.py` performs the explicit COCO-to-`xyxy` conversion.

### Important files

- Core implementation package: `crater_detection/`
- FBWR implementation: `crater_detection/fbwr.py`
- Grounded SAM generation: `crater_detection/grounded_sam.py`
- Initial COCO FBWR filtering: `crater_detection/filter_coco_fbwr.py`
- Geometric and boundary refinement: `crater_detection/geometric_boundary_refinement.py`
- Label-generation orchestrator: `scripts/run_label_generation.py`
- Curated dataset tools: `tools/dataset/`
- COCO schema export: `crater_detection/export_coco.py`
- Self-training merger: `crater_detection/merge_self_training.py`
- Three-round training runner: `scripts/run_training_pipeline.py`
- Original CutLER source backup: `university_backup/CutLER/`
- Vendored CutLER fork: `third_party/CutLER/`
- CutLER dataset registration: `third_party/CutLER/cutler/register_crater_dataset.py`
- CutLER training entry point: `third_party/CutLER/cutler/train_net.py`
- Base crater config: `third_party/CutLER/cutler/model_zoo/configs/CutLER-CraterDataset/cascade_mask_rcnn_R_50_FPN_modified.yaml`
- Self-training crater config: `third_party/CutLER/cutler/model_zoo/configs/CutLER-CraterDataset/cascade_mask_rcnn_R_50_FPN_self_train_modified.yaml`
- Existing detector tensor scorer: `third_party/CutLER/cutler/modeling/roi_heads/custom_cascade_rcnn_fbwr2.py`
- Existing grayscale reconstruction: `third_party/CutLER/cutler/modeling/meta_arch/rcnn_fbwr2.py`
- Source Grounded SAM script: `ExtraCode/rungroundedsam.py`
- Source FBWR notebooks: `ExtraCode/map_filtering.ipynb` and `ExtraCode/test_tensorfbwr2.ipynb`

### Important implementation decisions

- The core implementation package is named `crater_detection`, with runnable orchestration commands under `scripts/`.
- Grounded SAM outputs masks and derives COCO bounding boxes from mask extents.
- The final label-generation output is the FBWR-filtered COCO file and is used as the initial training annotation input.
- New prediction annotations retain their COCO fields; the merger removes transient prediction scores from final training labels.
- The runner calls CutLER's `train_net.py` for model inference/training and calls the clean-repo merger for label updates.
- The runner requires initial annotations and an initial pretrained checkpoint, then launches the initial 4,000-iteration training stage itself. It subsequently performs inference, merging, and self-training rounds, followed by final training.
- The training runner distinguishes the self-training config (`USE_DROPLOSS: True`, `MAX_ITER: 4000`) from the dedicated final config (`USE_DROPLOSS: False`, `MAX_ITER: 40000`) while preserving the thesis learning rates and dataset sequence.
- CutLER crater dataset registration uses `CRATER_DATASET_ROOT`, `CRATER_ANNOTATIONS_DIR`, and `CRATER_INITIAL_ANNOTATIONS` environment variables supplied by the runner, instead of a machine-specific absolute path.
- CPU validation covers data preparation, COCO refinement, FBWR, dry runs, focused tests, and package installation. Grounded DINO/SAM and CutLER execution can be attempted in the available environment according to its runtime and memory limits.
- No formal test suite has been added yet, by explicit agreement. Validation so far is compilation, diagnostics, schema checking, dry-run command inspection, and synthetic refinement checks. The synthetic refinement test passed after installing the project-compatible vision dependencies.

## 5. Working Protocol

Work strictly step-by-step. Start from the nearest concrete file or code path, gather only enough local context to form a falsifiable implementation hypothesis, and make the smallest focused change.

Ask clarification questions before making consequential design decisions. Wait for approval when the user requests a review or plan before code. Keep comments and docstrings written as final repository documentation, without mentioning reconstruction work, notebooks, or temporary history.

Preserve existing user changes and do not revert unrelated files. Avoid broad refactors. After every substantive edit, run the narrowest available executable validation before expanding the scope. Do not commit changes or create branches unless explicitly requested.
