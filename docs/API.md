# Component API

These interfaces are used from a source checkout. They are not an installed package or a complete arbitrary-pair inference service.

## Diffusion

```python
from models.diffusion import LatentVAE, PairConditioner, PairDDPM
```

`LatentVAE` and `GraphDDPM` resolve to `code/compare_dual_graph_ot_vae_ddpm.py`; the pair-specific classes resolve to `code/run_validation_selected.py`.

- `PairConditioner(feature_dim, pair_nodes)`: combines intervention-aware target embeddings symmetrically.
- `PairDDPM(feat_dim, latent_dim, steps, pair_nodes)`: conditional latent noise predictor with a pair graph conditioner.
- `LatentVAE`: encoder/decoder used for latent representation; consult the preserved class definition for constructor/input dimensions.
- `train_pair_variant` and `run_fold`: low-level training/evaluation routines; the full command-line runner supplies their configuration and inputs.

Do not pass a gene adjacency matrix to the VAE as if it were the expression matrix. Their roles differ.

## Deterministic predictor

```python
from models.predictor import learned_predictions
```

`learned_predictions(base_features, graph_features, train_residual, n_train, seed=SEED)` returns named residual estimates. Feature rows must be ordered training rows first, followed by query rows. `train_residual` contains only training outcomes. The returned residuals need the additive measured-single response to produce predicted pair means; use the complete runner for matched scoring.

## Preprocessing

`preprocessing.validate_counts(matrix)` checks finite nonnegative integer-valued input.

`preprocessing.preprocess_counts(adata, target_sum=10000, min_cells=3, qc=True)` returns a copied/QC-filtered/normalized AnnData; it does not overwrite its input. Raw provenance must be established by the caller.

`preprocessing.embed_exploratory(adata, n_pcs=50, neighbors=15, seed=42, batch_correction='none', batch_key=None)` returns descriptive PCA/neighbours/UMAP. It fits all supplied cells and is not used to validate the held-out model. Harmony is optional and requires actual batch metadata.
