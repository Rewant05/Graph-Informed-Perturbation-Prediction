# Earlier diffusion ablations

Historical exploratory experiment: 101 held-out combinations,
five custom study folds, 5,045 expression genes, seed 42.
SVD: 64 components; VAE: 24 dimensions and eight training epochs;
diffusion: fixed 12 training epochs. Lower MSE is better.

| Configuration | Mean expression MSE |
| --- | ---: |
| no_edges:ot | 0.00576689 |
| shuffled:ot | 0.00514442 |
| string:ot | 0.00486964 |
| hic:ot | 0.00566612 |
| dual:ot | 0.00487668 |
| dual:no_ot | 0.00463742 |

`dual` means STRING + Hi-C; `shuffled` randomizes both relations.
`ot` denotes optimal transport; `no_ot` disables it.

Adding Hi-C to STRING did not improve MSE with OT enabled.
Disabling OT improved the combined-graph configuration in this run.

The later diffusion protocol uses up to 40 epochs and validation
checkpoint selection. Differences between protocols cannot be
attributed solely to removing Hi-C or OT.

## Related evidence

- [Later diffusion results](../validation_string_seed42.json)
- [Predictor graph ablations](../gears_pair_graph_branch_matched.json)
- [Earlier experiment code](../../code/compare_pair_ot_vae_ddpm_gears.py)

These checks verify saved-score consistency, not fresh GPU training.
