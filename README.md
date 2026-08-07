# Label-Free Martian Crater Detection

Code and configuration for **Label-Free Crater Detection using Grounded SAM and Domain-Guided Self-Training**.

## Abstract

This project investigates crater detection in Martian THEMIS imagery without using crater annotations during pseudo-label generation. Grounded DINO and Segment Anything produce candidate crater masks, which are refined geometrically and filtered with Flexible Black and White Rims (FBWR) criteria. A thesis-specific Cascade Mask R-CNN implementation based on CutLER then performs initial training, three domain-guided self-training rounds, and final standard-loss training.

The repository is intended to support reproducible research experiments. Datasets, model checkpoints, and generated outputs are external artifacts and are not distributed here.

## Method Overview

```text
THEMIS imagery
    -> Grounded DINO + Segment Anything pseudo-labels
    -> geometric and boundary refinement
    -> FBWR filtering
    -> initial Cascade Mask R-CNN training
    -> three self-training rounds
    -> final standard-loss detector
```

The label-generation pipeline does not require crater annotations. The detector training stages use the generated COCO annotations and the model checkpoints produced by preceding stages. Reference crater annotations are reserved for offline evaluation and are not used for training, threshold selection, or pseudo-label generation.

## Experimental Setting

The experiments use the global THEMIS daytime infrared mosaic. Non-overlapping $512 \times 512$ pixel tiles are sampled while excluding latitudes beyond $\pm50^\circ$, resulting in 9,122 tiles split into 80% training, 10% validation, and 10% evaluation data. The evaluation catalog is the Robbins and Hynek Mars crater catalog; it is not required for the label-free training workflow.

The reported Grounded DINO/SAM settings are the `circle` text prompt, box threshold `0.30`, and text threshold `0.25`. Geometric refinement retains ellipse axis ratios in `[0.7, 1.3]`. Initial FBWR filtering uses threshold `2000` on 8-bit imagery; the normalized detector-side FBWR threshold is `0.03`.

## Repository Structure

```text
crater_detection/       Core label-generation and annotation utilities
scripts/                 End-to-end and stage-specific workflow commands
tools/dataset/           THEMIS sources and prepared-dataset documentation
third_party/CutLER/      Vendored thesis-specific CutLER/Detectron2 source
docs/citation.bib       BibTeX citation for the associated paper
```

## Environment Setup

Use an isolated Python environment. Python 3.9 is the reference environment; other versions may require compatible PyTorch, Detectron2, and third-party package versions.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e third_party/CutLER
python -m pip install -e .
```

Install a PyTorch build appropriate for the target machine before running model inference or training. CPU execution is suitable for schema checks, refinement, FBWR evaluation, and workflow dry runs. Foundation-model inference and detector training can require substantial memory and runtime.

## Data Download

THEMIS imagery is available from the [THEMIS Image Explorer](http://viewer.mars.asu.edu/faq) and NASA's [Planetary Data System](http://pds-imaging.jpl.nasa.gov/). The optional download helper stores source data outside the repository:

```bash
bash tools/dataset/download-themis-data.sh /data/mars
```

See [tools/dataset/README.md](tools/dataset/README.md) for the expected prepared layout. You must also provide:

- a text file listing images relative to the dataset root;
- a Segment Anything checkpoint;
- the Grounded DINO model downloaded through Transformers;
- the initial detector checkpoint used by the CutLER configuration.

Do not commit imagery, annotations, checkpoints, or generated runs.

## Step-by-Step Usage

### 1. Generate pseudo-labels

Run Grounded DINO/SAM, geometric and boundary refinement, and FBWR filtering as one stage:

```bash
python scripts/run_label_generation.py \
  --dataset-root /data/mars \
  --image-root /data/mars/images \
  --image-list /data/mars/image-list.txt \
  --sam-checkpoint /models/sam_vit_h_4b8939.pth \
  --output-dir /data/mars-runs/labels
```

The output directory contains `grounded_sam.json`, `refined.json`, and `fbwr_filtered.json`. The last file is the initial COCO annotation input for detector training.

### 2. Run detector training and self-training

The training runner launches initial training, inference and annotation merging for three rounds, and final standard-loss training:

```bash
python scripts/run_training_pipeline.py \
  --dataset-root /data/mars \
  --initial-annotations /data/mars-runs/labels/fbwr_filtered.json \
  --initial-weights /models/dino_RN50_pretrain_d2_format.pkl \
  --annotations-dir /data/mars-runs/annotations \
  --output-root /data/mars-runs/training \
  --standard-loss-config third_party/CutLER/cutler/model_zoo/configs/CutLER-CraterDataset/cascade_mask_rcnn_R_50_FPN_final.yaml
```

### 3. Run the complete workflow

The combined command performs both stages and preserves all intermediate labels and model outputs:

```bash
python scripts/run_full_pipeline.py \
  --dataset-root /data/mars \
  --image-root /data/mars/images \
  --image-list /data/mars/image-list.txt \
  --sam-checkpoint /models/sam_vit_h_4b8939.pth \
  --initial-weights /models/dino_RN50_pretrain_d2_format.pkl \
  --output-root /data/mars-runs \
  --annotations-dir /data/mars-runs/annotations
```

Add `--dry-run` to any workflow command to inspect the generated commands without loading models or starting training.

## Reported Result

In the accompanying paper, the label-free approach improves the zero-shot baseline from 10.1 AP50 to 56.2 AP50, approximately 86% of the reported fully supervised reference performance. These values are included for context and should be interpreted together with the paper's evaluation protocol.

## Configuration

The crater-specific configurations are under `third_party/CutLER/cutler/model_zoo/configs/CutLER-CraterDataset/`:

- `cascade_mask_rcnn_R_50_FPN_modified.yaml`: initial training;
- `cascade_mask_rcnn_R_50_FPN_self_train_modified.yaml`: self-training rounds with FBWR-enhanced DropLoss;
- `cascade_mask_rcnn_R_50_FPN_final.yaml`: final standard-loss training.

Dataset registration is configured through the environment variables set by `scripts/run_training_pipeline.py`. The runner accepts explicit path overrides for experiments.

## Citation

If you use this software, please cite the associated IGARSS 2026 paper. The paper is marked forthcoming until publication after the conference presentation. GitHub citation metadata is available in [CITATION.cff](CITATION.cff), and a BibTeX entry is available in [docs/citation.bib](docs/citation.bib).

## License and Acknowledgments

Original repository code is released under the [MIT License](LICENSE). The `third_party/CutLER/` directory contains source derived from CutLER and Detectron2 and retains its applicable upstream licenses. Grounded DINO, Segment Anything, THEMIS imagery, crater metadata, and model checkpoints remain subject to their respective licenses and terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
