# GEARS checkpoint scoring and fresh training

These are distinct checks and must not be combined into one reproducibility claim.

## Original checkpoint rescoring

The author rescored the five original saved validation-selected checkpoints without further training. All 101 pairs were scored.

| Fold | Original checkpoint MSE (rounded) |
| --- | ---: |
| 1 | 0.00473009 |
| 2 | 0.00459141 |
| 3 | 0.00454864 |
| 4 | 0.00415177 |
| 5 | 0.00458519 |

The equal-pair aggregate, not an unweighted average of rounded fold means, is:

- Archived original: 0.004523486637704874
- Rescored original: 0.0045234866097634795
- Maximum absolute per-pair difference: 2.1365637236855917e-9

The original checkpoint result is reproduced to numerical tolerance. The audit JSON is retained in `evidence/gears_original_allfold_checkpoint_audit.json`.

## Fresh training: partial result

| Fold 1 method | MSE |
| --- | ---: |
| GEARS, original checkpoint | 0.00473009 |
| GEARS, fresh training | 0.00391132 |
| STRING diffusion, same fold | 0.00420399 |

The uploaded fresh GEARS report contains 21 of 101 pairs. It is not a complete five-fold result. The old/new fold-1 native split memberships and order were checked equal, and the three GEARS wrapper files were checked byte-identical. Original and fresh best validation DE MSE differ (0.261627 versus 0.244600), showing different trained outcomes. Differing signature hashes alone do not prove differing split membership.

The exact cause of training variation remains unresolved. Environment/backend differences, data-loader/control sampling and nondeterministic operations are audit candidates, not confirmed causes. Do not alter training or checkpoint selection to force a desired test score.

The 18.9% diffusion advantage refers to the original five-fold saved GEARS run. It is not established across fresh GEARS training. On the available fresh fold 1, GEARS has lower MSE than diffusion. Never compare that single-fold value with diffusion's five-fold mean.

## Other reproduction evidence

The uploaded baseline, predictor and diffusion reports reproduce their reference per-pair scores exactly, including all three diffusion graph controls. The computed comparisons are in `evidence/reproduction/audit_summary.json`; fresh reports are preserved beside it.

Original GEARS weights remain with the author and are not redistributed in this source package. Checkpoint rescoring provenance is supplied; running fresh training from the code is a separate experiment.
