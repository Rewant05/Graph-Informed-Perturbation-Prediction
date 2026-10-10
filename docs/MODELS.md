# Where are the two models?

## STRING-conditioned latent diffusion

Public source-tree interface: `models/diffusion.py`.

Actual definitions:

- `code/run_validation_selected.py`: `PairConditioner`, `PairDDPM`, `train_pair_variant`, `run_fold`.
- `code/compare_dual_graph_ot_vae_ddpm.py`: `LatentVAE`, `GraphDDPM`, VAE training utilities.
- `code/heldout_conditional_diffusion_2o.py`: intervention-aware graph conditioner and expression projection helper.

Training entry point: `python -m models.diffusion --all-folds --device cuda:0`.

The public interface resolves the exact original class/function objects. Numerical definitions were not moved or rewritten, preserving compatibility with archived source/checkpoint signatures. SVD expression compression and graph conditioning operate in parallel; graph adjacency is not compressed by the expression SVD.

## STRING-assisted residual predictor

Public source-tree interface: `models/predictor.py`.

Actual algorithm: `code/gears_pair_graph_branch_matched.py`, especially `learned_predictions`. Helpers implement single-effect pair features, nearest-residual transfer, STRING response aggregation and RBF kernels. It uses NumPy/scikit-learn rather than neural-network training.

Prediction entry point: `python -m models.predictor`.

The final mean response is the additive measured-single prediction plus 0.75 nearest residual and 0.25 kernel residual. Fit residuals use training pair outcomes only. The complete runner performs matched evaluation and returns stored original GEARS scores as a comparison reference.

## What is bundled?

Source, frozen graphs/splits, results, histories and expression exports are supplied. Trained weights and expression datasets are not bundled. These interfaces do not promise deployment-ready inference for an arbitrary new intervention pair. The algorithms' full component-level description is in the architecture guide.
