r"""Exploratory K562 pair predictor with a learned STRING/Hi-C residual branch.

PowerShell from the project folder:
  python .\k562_pair_graph_residual_branch_cv.py .\norman_log1p_10k_v2.h5ad .\k562_pair_signal.split.json .\string_balanced1.json .\k562_hic_mapping_v2.json --out .\k562_pair_graph_residual_branch_cv.json

Uses 103 discovery pairs, five fixed pair folds, measured single-gene
responses, and the existing fixed nearest-pair predictor. A regularized
low-rank kernel branch learns pair interaction residuals from the training
pairs only. It uses the original single-response pair features plus a
relation-specific graph aggregation of those responses. The graph-free
branch, real STRING/Hi-C, and node-shuffled networks share the same model,
weight and folds. The 26 reserved pair outcomes are never read here.

This is an exploratory pair-level graph test, NOT GraphSAGE or DDPM, and
does not establish biological graph benefit without beating shuffles.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold

from k562_pair_graph_context_cv import graph_pair_features
from k562_pair_interaction_cv import load_manifest, pair_features, predict_residual, score
from k562_pair_signal_audit import collect_groups, mean_rows


VERSION = "2026-10-07-k562-pair-graph-residual-branch-v1"
SEED = 20260929
K = 10
LATENT = 12
RIDGE = 1.0
GRAPH_WEIGHT = 0.5
BRANCH_WEIGHT = 0.25


def rbf_kernels(train_features, query_features):
    """Training-fold standardization and bandwidth, with no outcome access."""
    center = train_features.mean(axis=0)
    spread = np.maximum(train_features.std(axis=0), 1e-6)
    train = (train_features - center) / spread
    query = (query_features - center) / spread
    if not np.any(train):
        # An absent graph must contribute zero rather than a constant kernel.
        return (np.zeros((len(train), len(train))),
                np.zeros((len(query), len(train))))
    train_d2 = cdist(train, train, metric="sqeuclidean")
    query_d2 = cdist(query, train, metric="sqeuclidean")
    positive = train_d2[train_d2 > 1e-8]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    return np.exp(-train_d2 / bandwidth), np.exp(-query_d2 / bandwidth)


def score_fold(base_features, graph_features, residual, pairs, gene_index,
               train_idx, query_idx, seed=SEED):
    """Fit all branches on exactly the same donor pair outcomes."""
    base_train = base_features[train_idx]
    base_query = base_features[query_idx]
    train_residual = residual[train_idx]
    nearest = predict_residual(base_train, train_residual, base_query,
                               ("nearest", K, 1.0))

    # Outcome projection is fitted on training pair labels only.
    residual_center = train_residual.mean(axis=0)
    n_components = min(LATENT, len(train_idx) - 1,
                       train_residual.shape[1] - 1)
    projector = TruncatedSVD(n_components=n_components, random_state=seed)
    latent_train = projector.fit_transform(train_residual - residual_center)
    base_kernel, base_query_kernel = rbf_kernels(base_train, base_query)

    estimates = {"additive": np.zeros_like(nearest),
                 "nearest": nearest}
    for name, features in [("no_graph", None), *graph_features.items()]:
        if features is None:
            kernel, query_kernel = base_kernel, base_query_kernel
        else:
            extra, extra_query = rbf_kernels(features[train_idx],
                                              features[query_idx])
            kernel = base_kernel + GRAPH_WEIGHT * extra
            query_kernel = base_query_kernel + GRAPH_WEIGHT * extra_query
        weights = np.linalg.solve(kernel + RIDGE * np.eye(len(train_idx)),
                                  latent_train)
        branch = ((query_kernel @ weights) @ projector.components_ +
                  residual_center)
        estimates[name] = ((1 - BRANCH_WEIGHT) * nearest +
                           BRANCH_WEIGHT * branch)
    return {name: score(residual[query_idx], estimate,
                        [pairs[i] for i in query_idx], gene_index)
            for name, estimate in estimates.items()}


def evaluate(base_features, graph_features, residual, pairs, gene_index,
             seed=SEED):
    if len(pairs) <= K + 5:
        raise ValueError("Not enough discovery pairs for fixed k=10 five-fold CV")
    if any(len(value) != len(pairs) for value in graph_features.values()):
        raise ValueError("Graph feature rows must match pair order")
    folds = list(KFold(n_splits=5, shuffle=True,
                       random_state=seed).split(base_features))
    names = ["additive", "nearest", "no_graph", *graph_features]
    first = {name: np.empty(len(pairs)) for name in names}
    second = {name: np.empty(len(pairs)) for name in names}
    for number, (train, valid) in enumerate(folds, 1):
        result = score_fold(base_features, graph_features, residual, pairs,
                            gene_index, train, valid, seed)
        for name, (primary, no_targets) in result.items():
            first[name][valid] = primary
            second[name][valid] = no_targets
        print(f"Scored fold {number}/5; training={len(train)}, test={len(valid)}",
              flush=True)
    metrics = {name: {
        "mse_all": float(first[name].mean()),
        "mse_excluding_targets": float(second[name].mean()),
        "pairs_better_than_additive": int((first[name] < first["additive"]).sum())
    } for name in names}
    return metrics, {"pairs": ["+".join(pair) for pair in pairs],
                     "mse_all": {name: first[name].tolist() for name in names},
                     "mse_excluding_targets": {
                         name: second[name].tolist() for name in names}}


def run(data, split, string_file, hic_file, seed=SEED, min_cells=50,
        chunk_size=256):
    import scanpy as sc
    with open(string_file, encoding="utf-8") as handle:
        string = json.load(handle)
    with open(hic_file, encoding="utf-8") as handle:
        hic = json.load(handle)
    if (string.get("species") != 9606 or string.get("split") != "balanced1" or
            hic.get("version") != "2026-09-29-k562-hic-gene-mapping-v2"):
        raise ValueError("Expected human balanced1 STRING and strict K562 Hi-C v2")

    adata = sc.read_h5ad(data, backed="r")
    try:
        if (adata.X is None or adata.uns.get("normalization_protocol") !=
                "2026-09-29-norman-cp10k-log1p-v2"):
            raise ValueError("Expected verified normalized Norman data")
        genes = (adata.var["gene_symbol"].astype(str).tolist()
                 if "gene_symbol" in adata.var else adata.var_names.astype(str).tolist())
        if (hashlib.sha256("\n".join(genes).encode()).hexdigest() !=
                string.get("gene_sha256") or
                len(genes) != hic.get("measured_genes") or
                len(set(g.upper() for g in genes)) != len(genes)):
            raise ValueError("Expression genes and graph files do not align")
        labels = adata.obs["perturbation"].astype(str).to_numpy()
        controls, singles, pairs, _ = collect_groups(labels, genes, min_cells)
        discovery, reserved = load_manifest(split, pairs)
        print(f"{VERSION}: {len(discovery)} discovery pairs; "
              f"{len(reserved)} reserved pair outcomes UNREAD", flush=True)
        control = mean_rows(adata.X, controls, chunk_size)[0]
        targets = sorted({g for pair in discovery + reserved for g in pair})
        single_effects = {g: mean_rows(adata.X, singles[g], chunk_size)[0] -
                          control for g in targets}
        residuals = []
        for n, (a, b) in enumerate(discovery, 1):
            observed = mean_rows(adata.X, pairs[a, b], chunk_size)[0]
            residuals.append(observed - control - single_effects[a] -
                             single_effects[b])
            if n % 25 == 0:
                print(f"Read {n}/{len(discovery)} discovery pair means; "
                      "reserved pair expression unread", flush=True)
        residual = np.stack(residuals)
        base_features = pair_features(single_effects, discovery, seed=seed)
        gene_index = {g.upper(): i for i, g in enumerate(genes)}
    finally:
        adata.file.close()

    networks = {
        "string": (string["edges"], None),
        "hic_strict": (hic["strict_promoter_promoter"]["edges"], None),
        "shuffled_string": (string["edges"], seed + 37),
        "shuffled_hic": (hic["strict_promoter_promoter"]["edges"], seed + 37),
    }
    features, coverage = {}, {}
    for name, (edges, shuffled_seed) in networks.items():
        features[name], connected, kept = graph_pair_features(
            single_effects, discovery, genes, edges, seed, shuffled_seed)
        coverage[name] = {"measured_genes_connected": connected,
                          "unique_edges": kept}
    features["string_plus_hic"] = np.concatenate(
        [features["string"], features["hic_strict"]], axis=1)
    features["shuffled_both"] = np.concatenate(
        [features["shuffled_string"], features["shuffled_hic"]], axis=1)
    metrics, per_pair = evaluate(base_features, features, residual, discovery,
                                 gene_index, seed)
    print("method\tall_MSE\tMSE_excl_targets\tpairs_better_than_additive")
    for name, row in metrics.items():
        print(f"{name}\t{row['mse_all']:.8f}\t"
              f"{row['mse_excluding_targets']:.8f}\t"
              f"{row['pairs_better_than_additive']}")
    print("Only a graph method that beats the matched no-graph branch and "
          "its shuffled control offers evidence for graph specificity.")
    print("Exploratory repeated discovery-pair CV; do not present a selected "
          "variant as an independent test.")
    return {"version": VERSION, "protocol": "K562 normalized discovery pairs",
            "n_discovery": len(discovery),
            "n_reserved_pair_outcomes_unread": len(reserved),
            "n_genes": len(genes), "seed": seed, "k": K,
            "rank": LATENT, "ridge": RIDGE,
            "graph_kernel_weight": GRAPH_WEIGHT,
            "branch_blend_weight": BRANCH_WEIGHT,
            "graph_coverage": coverage, "metrics": metrics,
            "per_pair": per_pair}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data")
    parser.add_argument("split")
    parser.add_argument("string")
    parser.add_argument("hic")
    parser.add_argument("--out", default="k562_pair_graph_residual_branch_cv.json")
    parser.add_argument("--min-cells", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    if args.min_cells < 4 or args.chunk_size < 1:
        parser.error("Need min-cells >= 4 and chunk-size >= 1")
    for path in (args.data, args.split, args.string, args.hic):
        if not Path(path).is_file():
            parser.error(f"File not found: {path}")
    result = run(args.data, args.split, args.string, args.hic,
                 args.seed, args.min_cells, args.chunk_size)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
