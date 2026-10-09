# Figure guide

The PNG/PDF pairs are alternate formats of the same figure, not duplicate experiments. Use PNG in GitHub/PowerPoint and PDF for vector export. Numerical colours are consistent: GEARS navy, no edges grey, shuffled STRING orange and real STRING teal.

| Figure | Purpose | Recreate |
| --- | --- | --- |
| `architectures.svg` | Exact schematic of the two model paths | Static documentation graphic; numerical algorithms in `docs/ARCHITECTURES.md` |
| `all_method_mse.*` | Additive/nearest, graph predictor, diffusion and original GEARS comparison | `python make_saved_figures.py` |
| `four_method_comparison.*` | Four diffusion/GEARS methods, fold dots and significance brackets | `python make_saved_figures.py` |
| `training_and_validation.*` | DDPM training noise MSE and validation expression MSE | `python make_saved_figures.py` |
| `actual_vs_predicted.*` | Actual/predicted expression-effect agreement | `python make_saved_figures.py` |
| `AHR_KLF1_OLS_only.*` | Example pair with OLS fit; no identity line | `python make_saved_figures.py` |
| `response_heatmap.*` | Similarity-clustered observed/predicted effects | `python make_saved_figures.py` |
| `three_panel_umap.*` | Cells, CRISPRa condition and fold-1 split | `python run_study.py data-figures --fold 1` |

Saved-data regeneration writes `work/saved_figures/`. Raw-data figures write local work outputs. The curated `figures/` files are included for convenience.

## Statistical annotations

Four-method bars show the equal-pair mean. Dots are fold means, not independent experimental replicates. Stars above comparison brackets come from the paired mean sign-flip test (200,000 draws) with Holm correction across three comparisons. GEARS versus STRING diffusion: adjusted p = 0.0157 (`*`); no edges and shuffled STRING: `ns`. The fold-block sensitivity p-values are 0.0625, 0.1250 and 0.0625 respectively. The cohort was repeatedly examined and independence is limited by shared genes/training data. Do not replace these with decorative three/four-star labels.

Stars on validation curves mark selected epochs, not statistical significance. Noise training loss and expression validation MSE are different objectives and should not be interpreted on one scale.

## Scatter and heatmap interpretation

Expression plots use effects relative to controls. Pearson, Spearman and predictive R² are not interchangeable; R² is calculated from predictions, not squared Pearson correlation. The AHR+KLF1 example is illustrative; mean metrics across all 101 pairs are also shown.

The heatmap selects 20 genes by largest observed mean absolute effect. Rows/columns are clustered by correlation similarity with average linkage; the predicted panel reuses the observed ordering. Display clipping at ±3.2 is a visual scale choice. Selection and clustering are descriptive, not an independent differential-expression test or validation set.

UMAP is descriptive. Neutral cell labels preserve the distributed annotation discrepancy. Conditions are CRISPRa perturbations, not drug treatments; the split panel describes fold 1, not all five folds simultaneously.
