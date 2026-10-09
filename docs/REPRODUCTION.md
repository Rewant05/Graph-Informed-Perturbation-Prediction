# Reproduce and audit

## 1. Inspect saved evidence (no GPU/data needed)

```bash
python verify_release.py
python make_saved_figures.py
```

Figure regeneration needs NumPy/SciPy/Matplotlib. Results go to `work/saved_figures/`. UMAP regeneration is a separate raw-data step.

## 2. Prepare the exact data and environment

Use the processed H5AD and checksum in `DATA_AND_ENVIRONMENT.md`, not the separate 10k-gene file. Preserve the current working environment; install compatible dependencies rather than blindly upgrading it.

```bash
python prepare_local.py --data /absolute/path/to/perturb_processed.h5ad
python check_dataset.py --data /absolute/path/to/perturb_processed.h5ad
python run_study.py preflight
python code/run_validation_selected.py --self-test
```

Preflight and self-test do not benchmark predictions. Preparation creates local configuration and compatible split files; local paths are not committed.

## 3. Baselines and predictor

```bash
python run_study.py baseline
python run_study.py predictor
```

Compare every pair and fold with the supplied evidence, not only the rounded mean. The uploaded reruns agree exactly.

## 4. Diffusion and graph controls

```bash
python run_study.py diffusion --fold 1 --device cuda:0
python run_study.py diffusion --all-folds --device cuda:0
python run_study.py diffusion-controls --all-folds --device cuda:0
```

Inspect fold-1 reference agreement before proceeding. The uploaded completed reruns reproduce all saved diffusion control scores exactly. Each model uses validation-selected checkpoints; do not select an epoch using held-out test MSE. Saved weights and preprocessing are written locally and are not included in this release.

## 5. GEARS: two different reproduction tasks

Fresh training:

```bash
python run_study.py gears --fold 1 --device cuda:0
python run_study.py gears --all-folds --device cuda:0
```

The wrapper uses 20 epochs, batch 32, hidden size 64, Adam lr 0.001/weight decay 0.0005, the implemented StepLR and native fold seed 20260929+fold. Selection uses validation top-20 DE error. The supplied fresh report is partial (fold 1); a full fresh-training agreement is not claimed.

Original checkpoint rescoring uses `code/gears/gears_pair_resumable.py --score-checkpoint` with the original compatible inputs/checkpoint directory; consult its `--help`. Original checkpoint weights remain with the author and are not bundled. All five were rescored in the supplied audit and match the archived original scores to numerical tolerance.

Signatures include paths, data size/mtime and input hashes. Moving files may invalidate a checkpoint signature. Do not disable these checks or overwrite original checkpoints to manufacture agreement.

Archived GEARS scores are the default comparison reference. To compare newly trained GEARS, explicitly add `--gears-results work/results/gears_retrained.json` to the predictor/diffusion command; compatible report/baseline identifiers are required. Keep original and fresh outputs separate.

## 6. Figures

```bash
python run_study.py figures
python run_study.py data-figures --fold 1
```

The figure guide maps each output to its script. UMAP is descriptive and requires raw data. The heatmap and OLS-only example reuse supplied predictions; formatting changes do not change model results.

Run one GPU training process at a time on an 8 GB GPU to avoid resource contention. Hardware availability does not determine whether a scientific comparison is valid.
