# Label-Free Martian Crater Detection

This repository implements label-free Martian crater detection using Grounded DINO/SAM pseudo-labels, geometric and boundary refinement, FBWR filtering, Cascade Mask R-CNN training, and domain-guided self-training.

## Pipeline

```text
THEMIS imagery -> Grounded DINO/SAM -> geometric and boundary refinement
              -> FBWR filtering -> initial training
              -> three self-training rounds -> final standard-loss model
```

The project uses no crater annotations during label generation. Grounded DINO and SAM are pretrained foundation models; their checkpoints and all datasets must be obtained separately.

## Requirements

The repository can be validated on CPU with dry runs, schema checks, refinement, and FBWR tests. Grounded DINO/SAM and detector training may require substantial runtime and memory; use the available execution environment accordingly.

Install the project dependencies in an isolated environment, then install the vendored CutLER/Detectron2 source in editable mode if required by your environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e third_party/CutLER
```

The implementation was validated with Python 3.9, `numpy==1.24.4`, `opencv-python==4.6.0.66`, and `pycocotools==2.0.11`. Match the PyTorch wheel to the target machine.

## Data and checkpoints

Keep external assets outside this repository. See [tools/dataset/README.md](tools/dataset/README.md) for THEMIS sources and the prepared dataset layout. You also need a SAM checkpoint and the Grounded DINO model download supported by Transformers.

## Run the complete workflow

The combined wrapper preserves `grounded_sam.json`, `refined.json`, `fbwr_filtered.json`, self-training annotations, and model outputs under the selected output directory:

```bash
  python scripts/run_full_pipeline.py \
  --dataset-root /data/crater \
  --image-root /data/crater/images \
  --image-list /data/crater/image-list.txt \
  --sam-checkpoint /models/sam_vit_h_4b8939.pth \
  --initial-weights /models/dino_RN50_pretrain_d2_format.pkl \
  --output-root /data/crater-runs \
  --annotations-dir /data/crater/annotations
```

Use `--dry-run` to inspect every command without loading models or training. The individual entry points are `scripts/run_label_generation.py` and `scripts/run_training_pipeline.py`.

## Provenance

The detector code includes a thesis-specific CutLER fork under `third_party/CutLER/`, based on [Facebook Research CutLER](https://github.com/facebookresearch/CutLER). Detectron2 is vendored with that source. Preserve the upstream licenses and notices when redistributing the repository. Mars imagery remains subject to its source archive's terms.

The repository slug is `label-free-mars-crater-detection`. Publication metadata can be added independently.

## Citation

If you use this software, please cite the associated IGARSS 2026 paper. The citation is currently marked as forthcoming because the paper will be published after the conference presentation. GitHub also reads the metadata in [CITATION.cff](CITATION.cff); a BibTeX entry is available in [docs/citation.bib](docs/citation.bib).
