# Architectures and algorithm

## STRING-conditioned latent diffusion

Expression branch: training-pair/control RNA across 5045 genes → uncentered
TruncatedSVD64 fitted on training data → control centering/component scaling →
VAE24. The graph branch runs in parallel: STRING connectivity, control-derived
node features and intervention target markers → conditioner → symmetric pair
condition48. Denoiser inputs: noisy latent24, control anchor24, condition48 and
time embedding32. MLP hidden192/SiLU → predicted noise24.

VAE is trained separately: reconstruction MSE + 0.02 KL to a standard-normal
prior. Diffusion objective is MSE(actual noise, predicted noise). VAE8 epochs;
diffusion max40 epochs, validation every2, patience6; best validation all-gene
checkpoint restored. Generate128 latent samples with32 sampling steps and decode
their mean effect to the expression axis. Sampling uses the supplied code, not
an assertion that every DDPM sampling step is used. SVD/VAE necessity or retained
biological information is not established by these experiments.

The actual training mask contains training pairs and controls, not measured
single-gene profiles. Single responses are read after generation to verify the
saved additive baseline; they are not the diffusion training input. GEARS's
custom training split includes singles. This difference must remain visible.

## STRING-assisted interaction-transfer predictor

Single effect ΔA = μA−μ0. Additive pair effect = ΔA+ΔB.
Training interaction residual = observed pair effect−additive pair effect.
Single effects compressed with SVD16 produce symmetric pair features.
10 Euclidean nearest training pairs transfer inverse-distance-weighted residuals.
STRING neighbour aggregation supplies graph response features. The residual
kernel branch combines response and graph RBF kernels (graph kernel weight0.5),
ridge penalty1 and rank12 residual representation. Final predicted mean expression:

μhatAB = μA+μB−μ0 + 0.75 rhatnearest + 0.25 rhatkernel.

RBF is Radial Basis Function. This is kernel ridge/nearest-neighbour residual
transfer, not a random forest, VAE, diffusion model or trained GNN.

STRING-only graph: one gene node type, one weighted undirected association
relation. STRING+Hi-C: one gene node type with separate relations. The matched
generative graph support is125nodes/170STRINGedges/14HiCedges;50 of65targets
connected bySTRING,9 byHiC. These counts are not the number of predicted genes.
