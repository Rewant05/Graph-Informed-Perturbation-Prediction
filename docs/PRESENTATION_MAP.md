# Presentation-to-repository map

This maps the research discussion to implementation and saved evidence. It does not add claims beyond the experiments.

| Presentation topic | Implementation/evidence | Figures |
| --- | --- | --- |
| Perturbation and research question | README overview; `docs/EVALUATION.md` | Dataset UMAP |
| Two architectures | `docs/ARCHITECTURES.md`; `code/run_validation_selected.py`; `code/gears_pair_graph_branch_matched.py` | `architectures.svg` |
| Matched evaluation / Hi-C | `evidence/splits/`; `graphs/`; data/evaluation docs | Three-panel UMAP |
| Loss and validation selection | `evidence/visual_export_result.json`; validation checkpoint histories | Training and validation |
| Diffusion versus saved GEARS | `evidence/validation_string_seed42.json`; `evidence/gears_pair_result_go.json`; GEARS audit | Four-method bars; heatmap |
| Graph controls | Diffusion controls and predictor graph/shuffle results | Four-method bars |
| Predictor and additive reference | `code/gears_matched_simple_baseline.py`; `code/gears_pair_graph_branch_matched.py`; saved baseline/predictor JSON | All-method MSE |
| Pearson/R² and example | `code/plot_expression_export.py`; `evidence/expression_export.npz`; `figures/expression_metrics.json` | Scatter; AHR+KLF1 OLS |
| Statistics | `make_saved_figures.py`; `figures/statistics.json` | Calculated significance brackets |
| Future AOPxLink work | README limitations/next steps | Proposed work, no implemented AOP prediction claim |

The original 18.9% reduction is versus the original saved GEARS checkpoints. Fresh GEARS training has a different fold-1 result; the supplied audit makes this visible. Do not describe either architecture as established zero-shot prediction of entirely new intervention genes.
