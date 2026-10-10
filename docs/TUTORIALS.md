# Tutorials

## Inspect the supplied study

```bash
python verify_release.py
python make_saved_figures.py
```

Inspect `work/saved_figures/` and the figures already included in the source release. These commands regenerate evidence plots rather than training a model.

## Prepare and run the two models

Use the exact processed H5AD in the data guide. Do not use raw-preprocessing output as a substitute.

```bash
python preprocessing.py benchmark --data /path/to/perturb_processed.h5ad
python prepare_local.py --data /path/to/perturb_processed.h5ad
python run_study.py preflight
python code/run_validation_selected.py --self-test
python run_study.py baseline
python -m models.predictor
python -m models.diffusion --fold 1 --device cuda:0
```

Check actual fold-1 results and agreement reports before continuing:

```bash
python -m models.diffusion --all-folds --device cuda:0
python -m models.diffusion --controls --all-folds --device cuda:0
python run_study.py gears --all-folds --device cuda:0
```

For fresh GEARS comparisons, pass `--gears-results work/results/gears_retrained.json` explicitly to the model entry point. Archived scores are the default reference. Keep fresh and original outputs separate.

## Explore expression structure

Follow the preprocessing guide for PCA/UMAP and optional batch correction. These exploratory outputs do not change the benchmark or its scores. For the existing fold-specific UMAP, use `python run_study.py data-figures --fold 1` after generating the relevant diffusion exports.

## Software checks for the new preprocessing interface

```bash
python -m unittest discover -s tests -v
```

These synthetic tests check data preservation and guards, not biological predictive accuracy. The optional Harmony test skips when harmonypy is unavailable.
