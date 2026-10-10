# Preprocessing: three separate workflows

The existing scores use the distributed processed GEARS matrix. They are not results from a newly QC-filtered/normalized/PCA-corrected input.

## 1. Benchmark audit (no expression transformation)

```bash
python preprocessing.py benchmark --data /path/to/perturb_processed.h5ad
```

This delegates to `check_dataset.py`. Then prepare the frozen study inputs with `prepare_local.py`. It does not normalize, select genes, replace SVD with PCA or correct batches.

## 2. Raw Norman counts (a new dataset)

```bash
python preprocess_norman.py --data /path/to/norman_raw.h5ad --out work/preprocessing/norman_log1p.h5ad --confirm-raw-counts
```

`preprocess_norman.py` is a named wrapper for `preprocessing.py raw`. The raw source must be verified by the user; integer values alone do not establish provenance. If counts are stored in a layer, add `--counts-layer counts`.

The pipeline checks finite nonnegative integer counts, computes mitochondrial/QC metrics, applies 5-MAD count/gene and 3-MAD/8% mitochondrial filters, removes zero-total cells and genes expressed in fewer than three cells, preserves a `counts` layer, and applies total-count normalization to 10,000 followed by log1p. The result is recorded as `log1p_normalized`, not scran normalization. Gene-name/condition/cell annotations are preserved, subject to filtering; the historical A549 label is not rewritten.

Filtering may remove benchmark conditions/genes. These outputs are not compatible replacements for the current scored dataset. A new benchmark requires rebuilding splits, representations and all baselines/models on the new data.

## 3. Descriptive PCA/UMAP and optional Harmony

Processed expression (no normalization/QC repeated):

```bash
python preprocessing.py explore --data /path/to/processed.h5ad --input-kind processed --out work/preprocessing/exploration.h5ad
```

Raw counts plus preprocessing and exploration:

```bash
python preprocessing.py explore --data /path/to/raw.h5ad --input-kind raw-counts --confirm-raw-counts --out work/preprocessing/raw_exploration.h5ad
```

PCA, neighbours and UMAP are fitted on all supplied cells for descriptive analysis. This output is not used by the held-out prediction benchmark. The existing fold-specific figure script remains the path for the supplied training-fit/test-transform UMAP.

Optional batch correction:

```bash
python -m pip install -r requirements-preprocessing.txt
python preprocessing.py explore --data /path/to/processed.h5ad --input-kind processed --batch-correction harmony --batch-key batch --out work/preprocessing/harmony_exploration.h5ad
```

Harmony requires a real observation column with two or more batches and corrects PCA coordinates, not expression X. The reported Norman observation columns contain no batch key, so batch correction is **not performed by default**. Do not invent batches from perturbation labels; examine biological/batch confounding before correction. Correction does not prove better prediction.

Inputs and existing outputs are never overwritten. An audit JSON is written alongside the new H5AD. Integer/count checks help reject processed inputs but are not a substitute for provenance. Full input loading/copying can consume substantial RAM; use an appropriate machine.

## Historical script

`archive/preprocess_norman_original.py` preserves the script retrieved from the old GitHub repo (blob `6a080529a6dee1e787ee6ca3fb021686662b3682`). It is not the current benchmark preparation path. It labelled a total-count-normalized layer `scran_normalization`, used fixed relative paths, and did not include PCA, UMAP or batch correction. It is retained for provenance rather than presented as proof of how the current benchmark file was generated.
