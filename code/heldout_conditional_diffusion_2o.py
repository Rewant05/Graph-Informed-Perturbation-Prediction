r"""Target-conditioned graph/latent-diffusion prototype for unseen Norman CRISPRa.

The prediction inputs are training control expression, a target gene present in
the gene graph, and fresh Gaussian noise. Held-out perturbed expression is read
only AFTER predictions have been made, for scoring. A TruncatedSVD learned on
training cells compresses expression; the model generates a distribution of
effects relative to the mean of unperturbed controls. Cells are unpaired, so
this predicts an unpaired response distribution, not each cell's counterfactual.

The gene-aligned graph here uses control coexpression, NOT the original
unaligned STRING and Hi-C CSV files. This is a repaired research prototype,
not a reproduction of the originally reported denoising results.

PowerShell (from the directory containing this file and norman_processed.h5ad):
  python .\heldout_conditional_diffusion_2o.py .\norman_processed.h5ad --split balanced1 --epochs 15 --device cpu
"""

import argparse

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

import torch
from torch import nn
from torch.nn import functional as F

import gene_graph_response_2o as original
from count_balanced_graph_check_2o import predefined_splits, valid_targets


VERSION = "2026-09-29-heldout-conditional-latent-diffusion-v1"


class InterventionConditioner(nn.Module):
    """Control-gene graph -> marked target -> propagated intervention context."""

    def __init__(self, n_features, width=48):
        super().__init__()
        self.pre_self = nn.Linear(n_features, width)
        self.pre_pos = nn.Linear(n_features, width, bias=False)
        self.pre_neg = nn.Linear(n_features, width, bias=False)
        self.intervene = nn.Linear(width, width)
        self.post_self = nn.Linear(width, width)
        self.post_pos = nn.Linear(width, width, bias=False)
        self.post_neg = nn.Linear(width, width, bias=False)
        self.head = nn.Sequential(nn.LayerNorm(width * 2),
                                  nn.Linear(width * 2, width), nn.SiLU())

    def forward(self, features, pos, neg, targets):
        h = F.silu(self.pre_self(features) + pos @ self.pre_pos(features)
                   + neg @ self.pre_neg(features))
        base_pre = self.post_self(h) + pos @ self.post_pos(h) + neg @ self.post_neg(h)
        # Only the marked node changes. Propagate its effect over post-layer
        # graph edges without building a B x N x N tensor.
        marker = self.intervene(h[targets])
        self_change = F.linear(marker, self.post_self.weight, bias=None)
        positive_change = F.linear(marker, self.post_pos.weight, bias=None)
        negative_change = F.linear(marker, self.post_neg.weight, bias=None)
        addition = (pos[:, targets].T.unsqueeze(-1) * positive_change[:, None]
                    + neg[:, targets].T.unsqueeze(-1) * negative_change[:, None])
        rows = torch.arange(len(targets), device=targets.device)
        addition = addition + F.one_hot(targets, features.shape[0]).to(
            dtype=features.dtype).unsqueeze(-1) * self_change[:, None, :]
        response = F.silu(base_pre.unsqueeze(0) + addition) - F.silu(base_pre).unsqueeze(0)
        at_target = h[targets] + F.silu(base_pre[targets] + addition[rows, targets])
        spread = response.sum(dim=1)
        return self.head(torch.cat((at_target, spread), dim=-1))


class ConditionalDiffusion(nn.Module):
    """Predict forward-process noise in a low-dimensional expression effect."""

    def __init__(self, n_features, latent_dim, diffusion_steps):
        super().__init__()
        self.graph = InterventionConditioner(n_features)
        self.time = nn.Embedding(diffusion_steps, 32)
        self.denoiser = nn.Sequential(
            nn.LayerNorm(latent_dim + 48 + 32),
            nn.Linear(latent_dim + 48 + 32, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, latent_dim))

    def forward(self, noisy_effect, steps, targets, features, pos, neg):
        condition = self.graph(features, pos, neg, targets)
        return self.denoiser(torch.cat((noisy_effect, condition, self.time(steps)), dim=-1))


def get_split(conditions, genes, split):
    if split == "top20":
        return original.split_conditions(conditions, genes, 20)
    index = int(split.removeprefix("balanced")) - 1
    selected = list(predefined_splits(conditions, genes, index + 1))[index]
    _, heldout, train, test = selected
    return heldout, train, test


def make_training_data(matrix, conditions, train_mask, train_targets,
                       control_indices, max_per_target, latent_dim, seed):
    """Fit SVD exclusively on controls and seen single-target training cells."""
    rng = np.random.default_rng(seed)
    chosen = []
    labels = []
    all_seen = []
    for node, target in enumerate(train_targets):
        indices = np.flatnonzero(train_mask & (conditions == target))
        all_seen.extend(indices)
        if len(indices) > max_per_target:
            indices = rng.choice(indices, max_per_target, replace=False)
        chosen.extend(indices)
        labels.extend([node] * len(indices))
    fit_indices = np.concatenate((control_indices, np.asarray(all_seen, dtype=int)))
    fit_matrix = matrix[fit_indices]
    dimensions = min(fit_matrix.shape)
    if dimensions <= latent_dim:
        raise ValueError(f"Need SVD dimensions < {dimensions}; lower --latent-dim")
    svd = TruncatedSVD(n_components=latent_dim, n_iter=4, random_state=seed)
    print(f"Fitting {latent_dim}-component SVD on {len(fit_indices)} training cells...",
          flush=True)
    svd.fit(fit_matrix)
    ctrl_mean = original.row_mean(matrix[control_indices])
    ctrl_latent = svd.transform(ctrl_mean[None, :])[0]
    effect = svd.transform(matrix[chosen]).astype(np.float32) - ctrl_latent
    spread = np.maximum(effect.std(axis=0), 1e-3).astype(np.float32)
    standardized = (effect / spread).astype(np.float32)
    return svd, ctrl_mean, spread, standardized, np.asarray(labels, dtype=np.int64)


def train_model(model, features, pos, neg, values, labels, alpha_bar, epochs,
                batch_size, device, seed):
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(values, device=device)
    target = torch.as_tensor(labels, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = len(alpha_bar)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for indices in np.array_split(rng.permutation(len(values)),
                                      int(np.ceil(len(values) / batch_size))):
            batch = torch.as_tensor(indices, dtype=torch.long, device=device)
            t = torch.randint(0, steps, (len(batch),), device=device)
            noise = torch.randn_like(x[batch])
            alpha = alpha_bar[t].unsqueeze(-1)
            noisy = alpha.sqrt() * x[batch] + (1 - alpha).sqrt() * noise
            optimizer.zero_grad(set_to_none=True)
            estimate = model(noisy, t, target[batch], features, pos, neg)
            loss = F.mse_loss(estimate, noise)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch}/{epochs}: noise MSE={np.mean(losses):.5f}", flush=True)


@torch.no_grad()
def sample_effects(model, features, pos, neg, alpha_bar, nodes, n_samples,
                   latent_dim, seed, device, common_noise=False):
    """DDIM with eta=0: fresh seeded noise, no observed test expression."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    output = []
    shared = (torch.randn((n_samples, latent_dim), generator=generator,
                          device=device) if common_noise else None)
    for node in nodes:
        z = (shared.clone() if common_noise else
             torch.randn((n_samples, latent_dim), generator=generator, device=device))
        targets = torch.full((n_samples,), int(node), dtype=torch.long, device=device)
        for t in range(len(alpha_bar) - 1, -1, -1):
            steps = torch.full((n_samples,), t, dtype=torch.long, device=device)
            eps = model(z, steps, targets, features, pos, neg)
            alpha = alpha_bar[t]
            estimate_x0 = ((z - (1 - alpha).sqrt() * eps) / alpha.sqrt()).clamp(-8, 8)
            if t > 0:
                previous = alpha_bar[t - 1]
                z = previous.sqrt() * estimate_x0 + (1 - previous).sqrt() * eps
            else:
                z = estimate_x0
        output.append(z.mean(dim=0).cpu().numpy())
    return np.stack(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="Norman preprocessed .h5ad")
    parser.add_argument("--split", choices=("balanced1", "balanced2", "balanced3", "top20"),
                        default="balanced1")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-per-target", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnose", action="store_true",
                        help="Also generate selected SEEN targets to check whether conditioning works")
    parser.add_argument("--save-checkpoint", default=None,
                        help="Optional .pt file for reusing trained weights in later diagnostics")
    args = parser.parse_args()
    if min(args.epochs, args.latent_dim, args.steps, args.samples,
           args.batch_size, args.max_per_target) < 1 or args.latent_dim < 2 or args.steps < 24:
        parser.error("Training/sampling sizes must be positive; latent-dim >= 2; steps >= 24")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; omit --device cuda to run on CPU")
    try:
        import scanpy as sc
    except ImportError as exc:
        parser.error(f"Scanpy is required: {exc}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    adata = sc.read_h5ad(args.data)
    if "perturbation" not in adata.obs:
        parser.error("Missing obs['perturbation']")
    genes = (adata.var["gene_symbol"].astype(str).tolist()
             if "gene_symbol" in adata.var else adata.var_names.astype(str).tolist())
    conditions = np.asarray(adata.obs["perturbation"].astype(str).to_numpy(), dtype=str)
    heldout, train_mask, test_mask = get_split(conditions, genes, args.split)
    counts = valid_targets(conditions, genes)
    train_targets = sorted(name for name in counts if np.any(train_mask & (conditions == name)))
    if not train_targets or np.any(train_mask & test_mask):
        raise ValueError("Training/test targets overlap or no single-target training conditions")
    controls = np.flatnonzero(train_mask & np.isin(np.char.lower(conditions), list(original.CONTROLS)))
    if len(controls) < 2:
        raise ValueError("Need at least two unperturbed training controls")
    matrix = adata.X
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
        if not np.isfinite(matrix.data).all():
            raise ValueError("Expression contains nonfinite values")
    elif not np.isfinite(matrix).all():
        raise ValueError("Expression contains nonfinite values")
    print(f"Version: {VERSION}; split={args.split}; device={device}", flush=True)
    print(f"Held-out ({len(heldout)}): {', '.join(heldout)}", flush=True)
    print(f"Training single targets={len(train_targets)}; training controls={len(controls)}; "
          f"excluded held-out combos={np.count_nonzero(~train_mask & ~test_mask)}",
          flush=True)
    graph_targets = train_targets + list(heldout)
    ctrl_matrix = matrix[controls]
    feature_np, pos_np, neg_np = original.graph_from_controls(
        ctrl_matrix, genes, graph_targets)
    svd, ctrl_mean, spread, values, labels = make_training_data(
        matrix, conditions, train_mask, train_targets, controls,
        args.max_per_target, args.latent_dim, args.seed)
    print(f"Graph nodes={len(graph_targets)}; edges={np.count_nonzero(pos_np) + np.count_nonzero(neg_np)}; "
          f"sampled training cells={len(values)}; output genes={len(genes)}", flush=True)

    features = torch.as_tensor(feature_np, device=device)
    pos = torch.as_tensor(pos_np, device=device)
    neg = torch.as_tensor(neg_np, device=device)
    # At the last step alpha_bar is near zero, so the reverse sampler can
    # legitimately start from fresh N(0, I) noise instead of a test cell.
    beta = torch.linspace(1e-4, min(0.4, 12.0 / args.steps),
                          args.steps, device=device)
    alpha_bar = torch.cumprod(1 - beta, dim=0)
    model = ConditionalDiffusion(features.shape[1], args.latent_dim, args.steps).to(device)
    train_model(model, features, pos, neg, values, labels, alpha_bar,
                args.epochs, args.batch_size, device, args.seed)
    if args.save_checkpoint:
        torch.save({
            "version": VERSION, "split": args.split, "seed": args.seed,
            "model_state": {name: value.detach().cpu() for name, value
                            in model.state_dict().items()},
            "features": features.cpu(), "pos": pos.cpu(), "neg": neg.cpu(),
            "alpha_bar": alpha_bar.cpu(), "spread": torch.from_numpy(spread),
            "components": torch.from_numpy(svd.components_.copy()),
            "control_mean": torch.from_numpy(ctrl_mean.copy()),
            "train_targets": list(train_targets),
            "heldout": tuple(str(name) for name in heldout),
        }, args.save_checkpoint)
        print(f"Saved trained checkpoint: {args.save_checkpoint}", flush=True)
    latent_predictions = sample_effects(
        model, features, pos, neg, alpha_bar,
        range(len(train_targets), len(graph_targets)), args.samples,
        args.latent_dim, args.seed + 1, device)
    # Inverting the training-only SVD yields a predicted gene-expression EFFECT.
    predictions = (latent_predictions * spread) @ svd.components_

    train_matrix = matrix[train_mask]
    train_cond = conditions[train_mask]
    mean_response = np.stack([
        original.row_mean(train_matrix[train_cond == name]) - ctrl_mean
        for name in train_targets]).mean(axis=0)
    # Read held-out perturbed rows only now, after all predictions exist.
    test_matrix = matrix[test_mask]
    test_cond = conditions[test_mask]
    print("target\ttest_cells\tno_change_MSE\tmean_response_MSE\tdiffusion_MSE\teffect_Pearson")
    metrics = []
    for name, prediction in zip(heldout, predictions):
        rows = test_cond == name
        actual = original.row_mean(test_matrix[rows]) - ctrl_mean
        error = float(np.mean((actual - prediction) ** 2))
        null = float(np.mean(actual ** 2))
        mean = float(np.mean((actual - mean_response) ** 2))
        corr = (float(np.corrcoef(actual, prediction)[0, 1])
                if np.std(actual) > 0 and np.std(prediction) > 0 else float("nan"))
        metrics.append((null, mean, error))
        print(f"{name}\t{int(rows.sum())}\t{null:.6f}\t{mean:.6f}\t{error:.6f}\t{corr:.4f}", flush=True)
    aggregate = np.mean(metrics, axis=0)
    print(f"Macro MSE: no_change={aggregate[0]:.6f}; mean_response={aggregate[1]:.6f}; "
          f"diffusion={aggregate[2]:.6f}")
    if args.diagnose:
        # This training-target test is deliberately in-sample. Reuse identical
        # initial noise so differences between predictions reflect the target
        # condition rather than variation in the Monte Carlo draws.
        selected = np.linspace(0, len(train_targets) - 1,
                               num=min(10, len(train_targets)), dtype=int)
        seen_latent = sample_effects(
            model, features, pos, neg, alpha_bar, selected,
            min(128, args.samples), args.latent_dim, args.seed + 2,
            device, common_noise=True)
        seen_predictions = (seen_latent * spread) @ svd.components_
        seen_actual = np.stack([
            original.row_mean(train_matrix[train_cond == train_targets[int(node)]])
            - ctrl_mean for node in selected])
        seen_error = float(np.mean(np.mean((seen_predictions - seen_actual) ** 2,
                                           axis=1)))
        seen_mean_error = float(np.mean(np.mean((seen_actual - mean_response) ** 2,
                                                axis=1)))
        predicted_variance = float(np.var(seen_predictions, axis=0).mean())
        observed_variance = float(np.var(seen_actual, axis=0).mean())
        print("IN-SAMPLE TARGET DIAGNOSTIC (10 training targets; NOT an unseen result):")
        print(f"seen_diffusion_MSE={seen_error:.6f}; "
              f"seen_mean_response_MSE={seen_mean_error:.6f}")
        print(f"between_target_variance: generated={predicted_variance:.6f}; "
              f"observed={observed_variance:.6f}")
        print("Identical initial noise was used for each seen target in this diagnostic.")
    print("Fixed training settings; no test-selected checkpoint. This is a control-coexpression "
          "graph, NOT validated STRING/Hi-C, and the splits have already been examined.")


if __name__ == "__main__":
    main()
