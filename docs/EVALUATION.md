# Evaluation protocol and supported claims

The included frozen cohort has101distinct test pairs across folds21/20/20/20/20.
It derives from103discovery pairs; two RHOXF2 pairs lacked support, and26 earlier
opened reserved pairs were excluded. These are **custom study folds** exported
for the official GEARS processed Norman data, not its published default split.
All methods use the same test pair assignment/expression gene axis and scale.

Diffusion trains expression representation on training-pair/control cells;
validation means select epochs; test means score generated predictions afterward.
Dataset loading may load the expression matrix into memory: the claim is about
which rows affect fitting/selection, not that test bytes are never loaded.
Graph topology and target identities are available across the cohort as priors.
This protocol does not establish an inductive unseen-gene or edge-masked claim.

The deterministic baseline/predictor uses measured singles and only training-pair
residual outcomes. GEARS's supplied native training split includes singles and
controls. Diffusion does not train on single outcomes. Equal-information/model
selection-budget comparisons require further work.

Score: mean over genes of squared error between observed and predicted
condition-mean expression effects, then unweighted mean across held-out pairs.
Adding the same control mean to both sides preserves this MSE. Correlations and
predictive R² are calculated per pair across effects; predictive R² is not the
square of Pearson or the fit quality of an OLS recalibration line.

The heatmap selects20genes by largest observed mean absolute effect; both orders
are learned from observed response correlation distances using average linkage,
then reused for predictions. Shared fixed±3.2 colour limits may clip extremes.
This is descriptive held-out visualisation, not a differential-expression test
or evidence of successful response generation for every gene.

Stats: paired two-sided mean sign-flip Monte Carlo200000, seeds20261008+j,
Holm adjustment acrossGEARS/no-edge/shuffled comparisons. Expected adjusted
p≈.0157/.0734/.2413. Pair exchangeability unverified because pairs share genes;
exact five-fold weighted sign-flip sensitivity gives.0625/.125/.0625. Folds share
training data, so even this sensitivity check is not independent replication.
Five-block minimum two-sided p=.0625. No graph-specific significance claim.

Current conclusion: lower observed STRING diffusion MSE than the saved GEARS run
on this exploratory matched task; simpler baselines stronger; biological graph
benefit requires replication and untouched evaluation. TxPert/Squidiff are not
implemented or compared here. No distribution-level superiority is established.
