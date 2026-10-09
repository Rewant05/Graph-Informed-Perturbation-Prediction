"""Prototype with a control graph encoder, an intervention, and a graph decoder.

GraphSAGE-style block 1 encodes each candidate target gene using unperturbed
control expression and control-derived signed coexpression edges. An explicit
CRISPRa marker is applied ONLY to the selected target node. GraphSAGE-style
block 2 propagates the marker before a decoder predicts the mean response of
all measured genes. No held-out perturbed cells are used at inference.

The graph has candidate target genes as nodes, not all output genes. It uses
neither the unaligned 128-node STRING/Hi-C CSVs nor a diffusion process.
"""

import argparse
import re

import numpy as np
from scipy import sparse

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


CONTROLS = frozenset({"control", "ctrl", "unperturbed"})
VERSION = "2026-09-28-prepost-intervention-v1"


def components(name, gene_set):
    """A measured single-gene name takes precedence over combo separators."""
    upper = str(name).upper()
    if upper in gene_set:
        return (upper,)
    return tuple(part.strip().upper() for part in re.split(r"[+_]", upper))


def split_conditions(conditions, gene_names, n_heldout):
    """Keep held-out singles and all their combinations out of training."""
    conditions = np.asarray(conditions, dtype=str)
    gene_set = {str(g).upper() for g in gene_names}
    if n_heldout < 1:
        raise ValueError("--heldout must be positive")
    names, counts = np.unique(conditions, return_counts=True)
    eligible = [(int(n), name) for name, n in zip(names, counts)
                if name.lower() not in CONTROLS and "+" not in name
                and "_" not in name and name.upper() in gene_set]
    if len(eligible) <= n_heldout:
        raise ValueError("Need more measured single-gene targets than held-out targets")
    heldout = tuple(name for _, name in sorted(eligible, reverse=True)[:n_heldout])
    hidden = {name.upper() for name in heldout}
    train = np.asarray([not any(part in hidden for part in components(c, gene_set))
                        for c in conditions], dtype=bool)
    test = np.isin(conditions, heldout)
    return heldout, train, test


def row_mean(matrix):
    return np.asarray(matrix.mean(axis=0)).ravel().astype(np.float32)


def graph_from_controls(control, gene_names, candidate_targets, k=8,
                        n_anchors=16):
    """Align every graph node to an expression-gene column by symbol.

    Returns fixed node features and row-normalized positive/negative edges.
    The only expression rows accessed here are unperturbed training controls.
    """
    upper = [str(g).upper() for g in gene_names]
    if len(set(upper)) != len(upper):
        raise ValueError("Expression gene symbols must be unique")
    gene_index = {g: i for i, g in enumerate(upper)}
    if len(set(candidate_targets)) != len(candidate_targets):
        raise ValueError("Target names must be unique")
    target_cols = [gene_index[g.upper()] for g in candidate_targets]
    if control.shape[0] < 2 or len(target_cols) < 2:
        raise ValueError("Need at least two controls and two candidate targets")
    means = row_mean(control)
    second = (row_mean(control.multiply(control)) if sparse.issparse(control)
              else np.mean(np.square(control), axis=0))
    variances = np.maximum(second - np.square(means), 0)
    anchor_cols = np.argsort(-variances, kind="stable")[:min(n_anchors, len(upper))]

    def standardize(columns):
        piece = control[:, columns]
        piece = piece.toarray() if sparse.issparse(piece) else np.asarray(piece)
        return ((piece.astype(np.float32) - means[columns]) /
                np.maximum(np.sqrt(variances[columns]), 1e-6))

    target_profiles = standardize(target_cols)
    anchor_profiles = standardize(anchor_cols)
    corr = (target_profiles.T @ target_profiles) / control.shape[0]
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 0)
    anchors = (target_profiles.T @ anchor_profiles) / control.shape[0]
    features = np.column_stack((
        np.log1p(np.maximum(means[target_cols], 0)),
        np.log1p(variances[target_cols]),
        np.clip(anchors, -1, 1)
    )).astype(np.float32)
    # Shared, fixed feature scaling: no held-out perturbed outcomes involved.
    features = (features - features.mean(axis=0, keepdims=True)) / np.maximum(
        features.std(axis=0, keepdims=True), 1e-4)
    features = np.clip(features, -5, 5)

    n = len(target_cols)
    pos = np.zeros((n, n), dtype=np.float32)
    neg = np.zeros((n, n), dtype=np.float32)
    for node in range(n):
        choices = np.argsort(-np.abs(corr[node]), kind="stable")
        choices = [j for j in choices if j != node and abs(corr[node, j]) > 1e-6][:k]
        for neighbor in choices:
            weight = float(corr[node, neighbor])
            (pos if weight > 0 else neg)[node, neighbor] = abs(weight)
    # Symmetric gene relationships, then average messages over each neighborhood.
    pos = np.maximum(pos, pos.T)
    neg = np.maximum(neg, neg.T)
    pos /= np.maximum(pos.sum(axis=1, keepdims=True), 1e-8)
    neg /= np.maximum(neg.sum(axis=1, keepdims=True), 1e-8)
    return features, pos, neg


if nn is not None:
    class GeneGraphResponse(nn.Module):
        """Encode controls, inject a target intervention, and propagate it."""

        def __init__(self, n_features, n_output_genes, width=32):
            super().__init__()
            self.pre_self = nn.Linear(n_features, width)
            self.pre_pos = nn.Linear(n_features, width, bias=False)
            self.pre_neg = nn.Linear(n_features, width, bias=False)
            # Shared intervention rule; held-out genes have no learned ID vector.
            self.intervene = nn.Linear(width, width)
            self.post_self = nn.Linear(width, width)
            self.post_pos = nn.Linear(width, width, bias=False)
            self.post_neg = nn.Linear(width, width, bias=False)
            self.head = nn.Sequential(nn.LayerNorm(2 * width),
                                      nn.Linear(2 * width, width),
                                      nn.GELU(), nn.Linear(width, n_output_genes))

        def forward(self, features, pos, neg, target_nodes):
            # Block 1 is identical for every target: its inputs are controls.
            base = torch.nn.functional.gelu(
                self.pre_self(features) + pos @ self.pre_pos(features) +
                neg @ self.pre_neg(features))
            batch_size, num_nodes = target_nodes.numel(), features.shape[0]
            marker = torch.nn.functional.one_hot(target_nodes, num_nodes).to(
                dtype=features.dtype).unsqueeze(-1)
            # Different target IDs activate different *nodes*; there is no
            # parameter table keyed by the held-out perturbation names.
            activated = base.unsqueeze(0) + marker * self.intervene(base).unsqueeze(0)

            def second_stage(h):
                return h + torch.nn.functional.gelu(
                    self.post_self(h) + pos @ self.post_pos(h) +
                    neg @ self.post_neg(h))

            # Block 2 propagates an explicit intervention to neighboring genes.
            post = second_stage(activated)
            no_intervention = second_stage(base)
            spread = (post - no_intervention.unsqueeze(0)).sum(dim=1)
            at_target = post[torch.arange(batch_size, device=target_nodes.device),
                             target_nodes]
            return self.head(torch.cat([at_target, spread], dim=-1))


def fit_graph(features, pos, neg, y_train, train_nodes, max_epochs=120,
              seed=42, device="cpu"):
    """Choose epoch and output shrinkage on seen-target validation only."""
    if torch is None:
        raise ImportError("PyTorch is required; install requirements.txt")
    if len(train_nodes) < 6:
        raise ValueError("At least six single-gene training targets are required")
    if max_epochs < 1:
        raise ValueError("--epochs must be positive")
    dev = torch.device(device)
    x = torch.as_tensor(features, dtype=torch.float32, device=dev)
    p = torch.as_tensor(pos, dtype=torch.float32, device=dev)
    m = torch.as_tensor(neg, dtype=torch.float32, device=dev)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=dev)
    nodes = torch.as_tensor(train_nodes, dtype=torch.long, device=dev)
    perm = np.random.default_rng(seed).permutation(len(train_nodes))
    n_val = max(2, int(round(0.2 * len(train_nodes))))
    tr = torch.as_tensor(perm[n_val:], dtype=torch.long, device=dev)
    va = torch.as_tensor(perm[:n_val], dtype=torch.long, device=dev)

    def fresh_model():
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        return GeneGraphResponse(x.shape[1], y.shape[1]).to(dev)

    def step(model, optimizer, sample):
        model.train()
        optimizer.zero_grad()
        prediction = model(x, p, m, nodes[sample])
        loss = torch.mean((prediction - y[sample]) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model = fresh_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best = (float("inf"), 0, 0.0)
    scales = (0.0, 0.25, 0.5, 0.75, 1.0)
    null_val = float(torch.mean(y[va] ** 2).item())
    for epoch in range(1, max_epochs + 1):
        step(model, optimizer, tr)
        if epoch == 1 or epoch % 5 == 0 or epoch == max_epochs:
            model.eval()
            with torch.no_grad():
                predicted = model(x, p, m, nodes[va])
                for scale in scales:
                    error = float(torch.mean((scale * predicted - y[va]) ** 2).item())
                    if (error, epoch, scale) < best:
                        best = (error, epoch, scale)
        if epoch % 20 == 0:
            print(f"Validation epoch {epoch}/{max_epochs}: "
                  f"best MSE={best[0]:.6f}", flush=True)
        if epoch > best[1] + 35:
            break

    _, selected_epoch, selected_scale = best
    # Reinitialize and fit all 82 observed single targets for the chosen epoch.
    final = fresh_model()
    optimizer = torch.optim.AdamW(final.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(selected_epoch):
        step(final, optimizer, torch.arange(len(nodes), device=dev))
    final.eval()
    return final, (x, p, m), selected_scale, selected_epoch, best[0], null_val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="Preprocessed Norman .h5ad with obs['perturbation']")
    parser.add_argument("--heldout", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if torch is None:
        parser.error("PyTorch is missing. Install the project requirements first.")
    if args.neighbors < 1:
        parser.error("--neighbors must be positive")
    try:
        import scanpy as sc
    except ImportError as exc:
        parser.error(f"Scanpy is missing: {exc}")

    adata = sc.read_h5ad(args.data)
    if "perturbation" not in adata.obs:
        parser.error("Missing obs['perturbation']")
    conditions = adata.obs["perturbation"].astype(str).to_numpy()
    genes = (adata.var["gene_symbol"].astype(str).tolist()
             if "gene_symbol" in adata.var else adata.var_names.astype(str).tolist())
    heldout, train_mask, test_mask = split_conditions(conditions, genes, args.heldout)
    conditions_train = conditions[train_mask]
    control_mask = np.asarray([c.lower() in CONTROLS for c in conditions_train])
    if not control_mask.any():
        parser.error("Training needs control cells")
    gene_set = {g.upper() for g in genes}
    train_targets = sorted({c for c in conditions_train
                            if c.lower() not in CONTROLS and "+" not in c
                            and "_" not in c and c.upper() in gene_set})
    targets = train_targets + list(heldout)
    if len(train_targets) < 6:
        parser.error("Need at least six measured single-gene training targets")

    print(f"Model version: {VERSION}", flush=True)
    print(f"Held-out: {', '.join(heldout)}", flush=True)
    print(f"Training targets: {len(train_targets)}; train cells: {train_mask.sum()}; "
          f"excluded held-out combinations: {np.count_nonzero(~train_mask & ~test_mask)}",
          flush=True)
    matrix = adata.X
    control = matrix[train_mask][control_mask]
    control_mean = row_mean(control)
    graph_features, pos, neg = graph_from_controls(
        control, genes, targets, k=args.neighbors)
    print(f"Graph: {len(targets)} measured gene nodes; "
          f"{np.count_nonzero(pos) + np.count_nonzero(neg)} signed directed edges; "
          f"{len(genes)} output genes", flush=True)

    # Mean differential effects from *only seen single-gene* training cells.
    training_matrix = matrix[train_mask]
    responses = np.stack([
        row_mean(training_matrix[conditions_train == target]) - control_mean
        for target in train_targets
    ])
    model, tensors, scale, epochs, val_mse, val_zero = fit_graph(
        graph_features, pos, neg, responses, list(range(len(train_targets))),
        max_epochs=args.epochs, device=args.device)
    print(f"Seen-target validation: intervention_graph={val_mse:.6f}, "
          f"no_change={val_zero:.6f}; "
          f"selected epochs={epochs}, scale={scale:g}", flush=True)
    print("target\ttest_cells\tno_change_MSE\tmean_response_MSE\tintervention_graph_MSE\teffect_Pearson")
    mean_response = responses.mean(axis=0)
    test_matrix = matrix[test_mask]
    test_conditions = conditions[test_mask]
    with torch.no_grad():
        target_nodes = torch.arange(len(train_targets), len(targets),
                                    device=tensors[0].device)
        predictions = (scale * model(*tensors, target_nodes)).cpu().numpy()
    results = []
    for target, prediction in zip(heldout, predictions):
        rows = test_conditions == target
        observed = row_mean(test_matrix[rows]) - control_mean
        zero_mse = float(np.mean(observed ** 2))
        mean_mse = float(np.mean((observed - mean_response) ** 2))
        graph_mse = float(np.mean((observed - prediction) ** 2))
        corr = (float(np.corrcoef(observed, prediction)[0, 1])
                if np.std(observed) > 0 and np.std(prediction) > 0 else float("nan"))
        results.append((zero_mse, mean_mse, graph_mse))
        print(f"{target}\t{rows.sum()}\t{zero_mse:.6f}\t{mean_mse:.6f}\t"
              f"{graph_mse:.6f}\t{corr:.4f}")
    zero_avg, mean_avg, graph_avg = np.mean(results, axis=0)
    print(f"Macro MSE across held-out targets: no_change={zero_avg:.6f}; "
          f"mean_response={mean_avg:.6f}; intervention_graph={graph_avg:.6f}")
    print("Intervention injected between two signed mean-aggregation graph blocks.")


if __name__ == "__main__":
    main()
