r"""Does graph-neighbor aggregation of measured single responses help K562 pairs?

PowerShell (from the project folder):
  python .\k562_pair_graph_context_cv.py .\norman_log1p_10k_v2.h5ad .\k562_pair_signal.split.json .\string_balanced1.json .\k562_hic_mapping_v2.json --out .\k562_pair_graph_context_cv.json

For every measured single-gene response (a vector across expression genes),
one fixed graph aggregation averages the responses at each gene's neighbors.
Pair features from these aggregated responses augment the existing no-graph
nearest-residual predictor. Graph edges are built without pair outcomes;
five fixed CV folds read only DISCOVERY pair outcomes. The 26 reserved pair
outcomes are not read. Shuffled-node controls check graph specificity.

This is a fixed, one-step graph-message probe, NOT a learned GraphSAGE or DDPM.
Do not choose a setting based on the previously opened reserved pairs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold

from k562_pair_signal_audit import collect_groups, mean_rows
from k562_pair_interaction_cv import load_manifest, pair_features, score


VERSION = "2026-09-29-k562-pair-graph-context-v1"
K = 10  # Frozen from prior discovery-pair CV.
WEIGHTS = (0.25, 0.5, 1.0)


def graph_operator(genes, edges, shuffled_seed=None):
    """Sparse, row-normalized neighbor aggregation; no self edges."""
    names = {name.upper(): col for col, name in enumerate(genes)}
    if len(names) != len(genes):
        raise ValueError("Expression gene names must be unique ignoring case")
    if shuffled_seed is None:
        permutation = np.arange(len(genes))
    else:
        permutation = np.random.default_rng(shuffled_seed).permutation(len(genes))
    row, col, val = [], [], []
    for left, right, weight in edges:
        a, b = names.get(str(left).upper()), names.get(str(right).upper())
        if a is None or b is None:
            raise ValueError(f"Graph edge refers to missing gene: {left}, {right}")
        a, b = int(permutation[a]), int(permutation[b])
        if a == b:
            continue
        w = float(weight)
        if w <= 0 or not np.isfinite(w):
            raise ValueError("Graph weights must be finite and positive")
        row.extend((a, b))
        col.extend((b, a))
        val.extend((w, w))
    n = len(genes)
    graph = sparse.coo_matrix((val, (row, col)), shape=(n, n),
                              dtype=np.float64).tocsr()
    graph.sum_duplicates()
    degree = np.asarray(graph.sum(axis=1)).ravel()
    inverse = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    operator = sparse.diags(inverse) @ graph
    return operator, int((degree > 0).sum()), int(graph.nnz // 2)


def graph_pair_features(single_effects, pairs, genes, edges, seed,
                        shuffled_seed=None):
    """Gene neighbors carry measured SINGLE response features, never pair labels."""
    names = sorted(single_effects)
    effect_matrix = np.stack([single_effects[name] for name in names])
    operator, connected, kept = graph_operator(genes, edges, shuffled_seed)
    # X[target, gene] -> neighbor average at each expression-gene node.
    aggregated = (operator @ effect_matrix.T).T
    if not np.any(aggregated):
        return np.zeros((len(pairs), 32)), connected, kept
    profiles = dict(zip(names, aggregated))
    return pair_features(profiles, pairs, seed=seed), connected, kept


def predict_fold(base_features, graph_features, residual,
                 train_idx, valid_idx, strength):
    """Same 10-neighbor response learner for every graph condition."""
    def standardized(features):
        train, valid = features[train_idx], features[valid_idx]
        mu = train.mean(axis=0)
        std = np.maximum(train.std(axis=0), 1e-6)
        return (train - mu) / std, (valid - mu) / std

    base_train, base_valid = standardized(base_features)
    base_distance = cdist(base_valid, base_train) / np.sqrt(base_train.shape[1])
    if strength:
        graph_train, graph_valid = standardized(graph_features)
        graph_distance = cdist(graph_valid, graph_train) / np.sqrt(graph_train.shape[1])
        distances = np.sqrt(base_distance**2 + (strength * graph_distance)**2)
    else:
        distances = base_distance
    indices = np.argsort(distances, axis=1, kind="stable")[:, :K]
    selected = np.take_along_axis(distances, indices, axis=1)
    weights = 1 / np.maximum(selected, 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("ij,ijk->ik", weights, residual[train_idx][indices])


def evaluate(base_features, graph_features, residual, pairs, gene_index,
             seed=20260929):
    if len(pairs) < K + 6:
        raise ValueError("Not enough discovery pairs for five-fold CV with k=10")
    folds = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(base_features))
    output = []
    for method, features, weights in [
        ("no_graph", None, (0.0,)),
        *[(name, values, WEIGHTS) for name, values in graph_features.items()],
    ]:
        for strength in weights:
            errors = np.empty(len(pairs))
            for train, valid in folds:
                predicted = predict_fold(base_features, features, residual,
                                         train, valid, strength)
                fold_mse, _ = score(residual[valid], predicted,
                                    [pairs[i] for i in valid], gene_index)
                errors[valid] = fold_mse
            output.append({"method": method, "graph_weight": strength,
                           "cv_macro_mse": float(errors.mean()),
                           "pairs_improved_vs_additive": int((errors <
                               np.mean(residual**2, axis=1)).sum())})
    return output


def run(data, split, string_file, hic_file, seed=20260929,
        min_cells=50, chunk_size=256):
    import scanpy as sc
    with open(string_file, encoding="utf-8") as handle:
        string = json.load(handle)
    with open(hic_file, encoding="utf-8") as handle:
        hic = json.load(handle)
    if string.get("species") != 9606 or string.get("split") != "balanced1":
        raise ValueError("Expected human balanced1 STRING graph")
    if hic.get("version") != "2026-09-29-k562-hic-gene-mapping-v2":
        raise ValueError("Expected the v2 Hi-C graph mapping")

    adata = sc.read_h5ad(data, backed="r")
    try:
        if adata.X is None or adata.uns.get("normalization_protocol") != \
                "2026-09-29-norman-cp10k-log1p-v2":
            raise ValueError("Expected verified normalized Norman data")
        genes = (adata.var["gene_symbol"].astype(str).tolist()
                 if "gene_symbol" in adata.var else adata.var_names.astype(str).tolist())
        fingerprint = hashlib.sha256("\n".join(genes).encode()).hexdigest()
        if fingerprint != string.get("gene_sha256") or len(genes) != hic.get("measured_genes"):
            raise ValueError("Inconsistent expression genes and graph inputs")
        labels = adata.obs["perturbation"].astype(str).to_numpy()
        controls, singles, pairs, _ = collect_groups(labels, genes, min_cells)
        discovery, reserved = load_manifest(split, pairs)
        print(f"Version={VERSION}; discovery={len(discovery)}; "
              f"reserved outcomes UNREAD={len(reserved)}", flush=True)
        control_mean = mean_rows(adata.X, controls, chunk_size)[0]
        needed = sorted({g for pair in discovery + reserved for g in pair})
        single_effects = {name: mean_rows(adata.X, singles[name], chunk_size)[0] -
                          control_mean for name in needed}
        residual = []
        for n, (a, b) in enumerate(discovery, 1):
            observed = mean_rows(adata.X, pairs[a, b], chunk_size)[0]
            residual.append(observed - control_mean -
                            single_effects[a] - single_effects[b])
            if n % 25 == 0:
                print(f"Read {n}/{len(discovery)} discovery pairs; "
                      "zero reserved pair-expression rows read", flush=True)
        residual = np.stack(residual)
        base_features = pair_features(single_effects, discovery, seed=seed)
        gene_index = {g.upper(): i for i, g in enumerate(genes)}
    finally:
        adata.file.close()

    hic_edges = hic["strict_promoter_promoter"]["edges"]
    string_edges = string["edges"]
    variants = {
        "string": (string_edges, None),
        "hic_strict": (hic_edges, None),
        "shuffled_string": (string_edges, seed + 37),
        "shuffled_hic": (hic_edges, seed + 37),
    }
    graph_features, coverage = {}, {}
    for name, (edges, shuffle) in variants.items():
        print(f"Aggregating single-response features with {name}...", flush=True)
        features, connected, kept = graph_pair_features(
            single_effects, discovery, genes, edges, seed, shuffle)
        graph_features[name] = features
        coverage[name] = {"measured_genes_with_neighbor": connected,
                          "unique_edges": kept}
    graph_features["string_plus_hic"] = np.concatenate(
        [graph_features["string"], graph_features["hic_strict"]], axis=1)
    scores = evaluate(base_features, graph_features, residual,
                      discovery, gene_index, seed)
    print("method\tweight\tCV_MSE\tpairs_improved_vs_additive")
    for row in scores:
        print(f"{row['method']}\t{row['graph_weight']}\t"
              f"{row['cv_macro_mse']:.8f}\t{row['pairs_improved_vs_additive']}")
    print(f"Additive={float(np.mean(residual**2)):.8f}; "
          f"no_graph={scores[0]['cv_macro_mse']:.8f}")
    print("Exploratory repeated discovery-pair CV only. Fixed one-step graph "
          "aggregation, not GraphSAGE/DDPM. Reserved pair outcomes remain unread.")
    return {"version": VERSION, "n_discovery": len(discovery),
            "n_reserved_outcomes_unread": len(reserved),
            "additive_cv_mse": float(np.mean(residual**2)),
            "graph_coverage": coverage, "results": scores}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data")
    p.add_argument("split")
    p.add_argument("string")
    p.add_argument("hic")
    p.add_argument("--out", default="k562_pair_graph_context_cv.json")
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--min-cells", type=int, default=50)
    p.add_argument("--chunk-size", type=int, default=256)
    args = p.parse_args()
    if args.min_cells < 4 or args.chunk_size < 1:
        p.error("--min-cells must be >=4 and --chunk-size must be positive")
    for path in (args.data, args.split, args.string, args.hic):
        if not Path(path).is_file():
            p.error(f"Missing file: {path}")
    result = run(args.data, args.split, args.string, args.hic,
                 args.seed, args.min_cells, args.chunk_size)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
