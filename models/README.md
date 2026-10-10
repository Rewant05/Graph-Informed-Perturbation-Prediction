# The two models

These modules expose the existing implementations with clear public names. They do not duplicate/retrain the models or rewrite the numerical source.

| Model | Public interface | Actual definitions |
| --- | --- | --- |
| STRING latent diffusion | `models/diffusion.py` | `code/run_validation_selected.py` (`PairDDPM`, `PairConditioner`); `code/compare_dual_graph_ot_vae_ddpm.py` (`LatentVAE`, `GraphDDPM`); intervention conditioner in `code/heldout_conditional_diffusion_2o.py` |
| STRING residual predictor | `models/predictor.py` | `code/gears_pair_graph_branch_matched.py` (`learned_predictions`), with neighbour/kernel helpers in `code/` |

From the repository root, after preparing the benchmark inputs:

```bash
python -m models.diffusion --fold 1 --device cuda:0
python -m models.diffusion --all-folds --device cuda:0
python -m models.diffusion --controls --all-folds --device cuda:0
python -m models.predictor
```

Example Python imports (require the corresponding scientific/model dependencies):

```python
from models.diffusion import LatentVAE, PairConditioner, PairDDPM
from models.predictor import learned_predictions
```

They resolve to the original class/function objects. These are research-level components; complete evaluation should use the runners rather than inventing inputs to a component. No standalone arbitrary-pair inference endpoint or trained weights are bundled.
