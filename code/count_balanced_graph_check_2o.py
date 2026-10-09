"""Predefined count-balanced sensitivity check for unseen Norman targets.

Runs no change, mean response, no-edge graph, original graph, and the explicit
intervention graph on the SAME target splits. Uses the existing standalone
gene_graph_response_2o.py and intervention_graphsage_2o.py modules. Neither
held-out response values nor held-out combination cells enter model fitting.

The original top-count 20 have already been examined, so they are eligible for
training here but excluded from the new evaluation targets. These new splits
are exploratory sensitivity checks, not a wholly untouched external test.
"""

import argparse

import numpy as np

import gene_graph_response_2o as original
import intervention_graphsage_2o as intervention


SEEDS = (20260928, 20260929, 20260930)
VERSION = "2026-09-28-count-balanced-predeclared-v1"


def valid_targets(conditions, gene_names):
    """Measured single-gene conditions, with their cell counts."""
    gene_set = {str(g).upper() for g in gene_names}
    names, counts = np.unique(np.asarray(conditions, dtype=str),
                              return_counts=True)
    return {str(name): int(count) for name, count in zip(names, counts)
            if name.lower() not in original.CONTROLS and "+" not in name
            and "_" not in name and name.upper() in gene_set}


def predefined_splits(conditions, gene_names, n_splits=3):
    """Pick five unseen targets per count quartile, excluding prior top 20."""
    if not 1 <= n_splits <= len(SEEDS):
        raise ValueError(f"Choose between 1 and {len(SEEDS)} predefined splits")
    counts = valid_targets(conditions, gene_names)
    if len(counts) < 40:
        raise ValueError("Need at least 40 measured single-gene targets")
    ranking = sorted(counts, key=lambda name: (counts[name], name))
    previous_top20 = set(ranking[-20:])
    strata = np.array_split(np.asarray(ranking, dtype=object), 4)
    pools = [[name for name in stratum if name not in previous_top20]
             for stratum in strata]
    if any(len(pool) < 5 for pool in pools):
        raise ValueError("Not enough previously unevaluated genes in each count quartile")
    gene_set = {str(g).upper() for g in gene_names}
    conditions = np.asarray(conditions, dtype=str)
    for seed in SEEDS[:n_splits]:
        rng = np.random.default_rng(seed)
        heldout = tuple(sorted(str(name)
                               for pool in pools
                               for name in rng.choice(pool, size=5, replace=False)))
        hidden = {name.upper() for name in heldout}
        train = np.asarray([
            not any(part in hidden for part in original.components(c, gene_set))
            for c in conditions
        ], dtype=bool)
        test = np.isin(conditions, heldout)
        assert not np.any(train & test)
        yield seed, heldout, train, test


def score(actual, prediction):
    """Macro-average MSE over conditions, then genes."""
    return float(np.mean(np.mean((actual - prediction) ** 2, axis=1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="Processed Norman .h5ad")
    parser.add_argument("--splits", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if original.torch is None or intervention.torch is None:
        parser.error("PyTorch is missing")
    if args.epochs < 1 or args.neighbors < 1:
        parser.error("--epochs and --neighbors must be positive")
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
    matrix = adata.X
    is_control = np.asarray([c.lower() in original.CONTROLS for c in conditions])
    if not is_control.any():
        parser.error("Need control cells")
    controls = matrix[is_control]
    control_mean = original.row_mean(controls)
    counts = valid_targets(conditions, genes)
    print(f"Protocol: {VERSION}; {len(counts)} measured single-gene targets; "
          f"{len(SEEDS)} fixed seeds; testing the first {args.splits}", flush=True)

    results = []
    for run, (seed, heldout, train_mask, test_mask) in enumerate(
            predefined_splits(conditions, genes, args.splits), 1):
        print(f"\nSplit {run}/{args.splits}; seed={seed}; "
              f"held-out={', '.join(heldout)}", flush=True)
        print(f"Train cells={train_mask.sum()}; held-out combination cells "
              f"excluded={np.count_nonzero(~train_mask & ~test_mask)}", flush=True)
        train_conditions = conditions[train_mask]
        train_targets = sorted({c for c in train_conditions if c in counts})
        targets = train_targets + list(heldout)
        if len(train_targets) < 6:
            raise ValueError("Fewer than six training targets")
        train_matrix = matrix[train_mask]
        responses = np.stack([
            original.row_mean(train_matrix[train_conditions == name]) - control_mean
            for name in train_targets
        ])
        features, pos, neg = original.graph_from_controls(
            controls, genes, targets, k=args.neighbors)
        test_matrix = matrix[test_mask]
        test_conditions = conditions[test_mask]
        actual = np.stack([
            original.row_mean(test_matrix[test_conditions == name]) - control_mean
            for name in heldout
        ])
        target_nodes = original.torch.arange(len(train_targets), len(targets),
                                             device=args.device)

        def run_architecture(label, train_function, positive, negative):
            print(f"Training {label}...", flush=True)
            model, tensors, scale, epochs, val, null_val = train_function(
                features, positive, negative, responses,
                list(range(len(train_targets))), max_epochs=args.epochs,
                device=args.device)
            with original.torch.no_grad():
                pred = (scale * model(*tensors, target_nodes)).cpu().numpy()
            outcome = score(actual, pred)
            print(f"{label}: validation MSE={val:.6f} "
                  f"(no change={null_val:.6f}); epochs={epochs}; "
                  f"scale={scale:g}; held-out MSE={outcome:.6f}", flush=True)
            return outcome

        empty = np.zeros_like(pos)
        null_mse = score(actual, np.zeros_like(actual))
        mean_mse = score(actual, np.broadcast_to(responses.mean(axis=0), actual.shape))
        no_edges = run_architecture("no_edges", original.fit_graph, empty, empty)
        graph = run_architecture("graph", original.fit_graph, pos, neg)
        marked = run_architecture("intervention", intervention.fit_graph, pos, neg)
        print(f"Split {run} macro MSE: no_change={null_mse:.6f}; "
              f"mean_response={mean_mse:.6f}; no_edges={no_edges:.6f}; "
              f"graph={graph:.6f}; intervention={marked:.6f}", flush=True)
        results.append((null_mse, mean_mse, no_edges, graph, marked))

    values = np.asarray(results)
    print("\nAverage of predefined splits (descriptive; targets may recur):")
    for name, value in zip(("no_change", "mean_response", "no_edges",
                            "graph", "intervention"), values.mean(axis=0)):
        print(f"{name}={value:.6f}")
    for name, column in (("no_edges", 2), ("graph", 3), ("intervention", 4)):
        print(f"{name} beat no_change on "
              f"{np.count_nonzero(values[:, column] < values[:, 0])}/{len(values)} splits")
    print("Sensitivity check only; prior training already used some of these targets.")


if __name__ == "__main__":
    main()
