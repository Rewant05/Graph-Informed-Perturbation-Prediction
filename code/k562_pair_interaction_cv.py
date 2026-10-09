r"""Test whether K562 two-gene interaction signatures can be predicted.

PowerShell (from the project folder):
  python .\k562_pair_interaction_cv.py .\norman_log1p_10k_v2.h5ad .\k562_pair_signal.split.json --out .\k562_pair_interaction_cv.json

Uses observed control and single-gene responses and ONLY the discovery pairs
listed in the saved manifest. Scores additive, mean-interaction, and a small
nearest-pair residual learner in fixed pair-level cross-validation. Does not
read reserved pair-expression rows. This is a feasibility model, not a claim
of graph or diffusion novelty, and it does not use STRING or Hi-C.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold

from k562_pair_signal_audit import VERSION as MANIFEST_VERSION
from k562_pair_signal_audit import collect_groups, mean_rows


VERSION = "2026-09-29-k562-pair-interaction-cv-v1"
SCALES = (0.25, 0.5, 1.0)
K_VALUES = (1, 3, 5, 10)


def pair_name(pair):
    return "+".join(pair)


def load_manifest(path, eligible_pairs):
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError("Split manifest was not produced by the matching pair audit")
    discovery = manifest.get("discovery_pairs")
    reserved = manifest.get("reserved_pairs")
    if not isinstance(discovery, list) or not isinstance(reserved, list):
        raise ValueError("Manifest needs discovery_pairs and reserved_pairs lists")
    expected = {pair_name(pair) for pair in eligible_pairs}
    if (not discovery or not reserved or
            len(set(discovery + reserved)) != len(discovery) + len(reserved) or
            set(discovery + reserved) != expected):
        raise ValueError("Manifest pair lists do not match eligible pairs in this dataset")
    by_name = {pair_name(pair): pair for pair in eligible_pairs}
    return [by_name[name] for name in discovery], [by_name[name] for name in reserved]


def pair_features(single_effects, discovery_pairs, n_components=16, seed=20260929):
    """Fit a fixed projection on measured singles, without pair outcomes."""
    names = sorted(single_effects)
    matrix = np.stack([single_effects[name] for name in names])
    components = min(n_components, len(names) - 1, matrix.shape[1] - 1)
    if components < 2:
        raise ValueError("Need more single effects and genes for pair features")
    embedding = TruncatedSVD(n_components=components,
                             random_state=seed).fit_transform(matrix)
    embedding = ((embedding - embedding.mean(axis=0)) /
                 np.maximum(embedding.std(axis=0), 1e-8))
    vectors = dict(zip(names, embedding))
    # Addition and absolute difference make the descriptor invariant to A/B order.
    return np.stack([np.concatenate((vectors[a] + vectors[b],
                                     np.abs(vectors[a] - vectors[b])))
                     for a, b in discovery_pairs])


def configurations(max_train):
    configs = [("additive", 0, 0.0)]
    configs += [("mean", 0, strength) for strength in SCALES]
    configs += [("nearest", k, strength) for k in K_VALUES
                if k <= max_train for strength in SCALES]
    return configs


def predict_residual(features_train, residual_train, features_query, config):
    method, neighbors, strength = config
    if method == "additive":
        return np.zeros((len(features_query), residual_train.shape[1]), dtype=np.float64)
    if method == "mean":
        return np.broadcast_to(strength * residual_train.mean(axis=0),
                               (len(features_query), residual_train.shape[1])).copy()
    # Transform geometry with discovery-training features in each fold only.
    mu = features_train.mean(axis=0)
    sigma = np.maximum(features_train.std(axis=0), 1e-6)
    train = (features_train - mu) / sigma
    query = (features_query - mu) / sigma
    distances = cdist(query, train)
    indices = np.argsort(distances, axis=1, kind="stable")[:, :neighbors]
    picked_distances = np.take_along_axis(distances, indices, axis=1)
    weights = 1 / np.maximum(picked_distances, 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    return strength * np.einsum("ij,ijk->ik", weights, residual_train[indices])


def score(true_residual, predicted_residual, pairs, gene_index):
    """Full-gene primary MSE; exclude targeted columns as a QC secondary MSE."""
    error2 = (true_residual - predicted_residual) ** 2
    primary = np.mean(error2, axis=1)
    secondary = []
    for i, (a, b) in enumerate(pairs):
        excluded = {gene_index[a], gene_index[b]}
        secondary.append((error2[i].sum() - error2[i, list(excluded)].sum()) /
                         (error2.shape[1] - len(excluded)))
    return np.asarray(primary), np.asarray(secondary)


def cross_validate(features, residual, pairs, gene_index, seed=20260929, folds=5):
    """Select a method exclusively using outcomes of discovery pairs."""
    if len(pairs) < folds + 1:
        raise ValueError("Not enough discovery pairs for five-fold validation")
    fold_splits = list(KFold(n_splits=folds, shuffle=True,
                            random_state=seed).split(features))
    configs = configurations(min(len(train) for train, _ in fold_splits))
    evaluations = []
    for config in configs:
        primary = np.empty(len(pairs), dtype=np.float64)
        secondary = np.empty(len(pairs), dtype=np.float64)
        for train, valid in fold_splits:
            estimate = predict_residual(features[train], residual[train],
                                        features[valid], config)
            first, second = score(residual[valid], estimate,
                                  [pairs[i] for i in valid], gene_index)
            primary[valid] = first
            secondary[valid] = second
        method, neighbors, strength = config
        evaluations.append({"method": method, "neighbors": neighbors,
                            "strength": strength,
                            "mse_all_genes": float(primary.mean()),
                            "mse_excluding_targets": float(secondary.mean()),
                            "pairs_improved_vs_additive": int((primary <
                                np.mean(residual**2, axis=1)).sum())})
    # List starts with additive, so an exact tie retains the simpler baseline.
    selected = min(evaluations, key=lambda r: r["mse_all_genes"])
    return evaluations, selected


def run(data, split, n_components=16, min_cells=50,
        seed=20260929, chunk_size=256):
    import scanpy as sc

    adata = sc.read_h5ad(data, backed="r")
    try:
        if adata.X is None or "perturbation" not in adata.obs:
            raise ValueError("Expected X and obs['perturbation']")
        if adata.uns.get("normalization_protocol") != "2026-09-29-norman-cp10k-log1p-v2":
            raise ValueError("Use the verified norman_log1p_10k_v2.h5ad, not raw counts")
        labels = adata.obs["perturbation"].astype(str).to_numpy()
        genes = (adata.var["gene_symbol"].astype(str).to_numpy()
                 if "gene_symbol" in adata.var else
                 adata.var_names.astype(str).to_numpy())
        upper_genes = [str(g).upper() for g in genes]
        if len(set(upper_genes)) != len(upper_genes):
            raise ValueError("Duplicate expression gene symbols; resolve before scoring")
        gene_index = {name: col for col, name in enumerate(upper_genes)}
        controls, singles, pairs, _ = collect_groups(labels, genes, min_cells)
        discovery, reserved = load_manifest(split, pairs)
        print(f"Model={VERSION}; discovery={len(discovery)}; reserved={len(reserved)}; "
              f"genes={adata.n_vars}", flush=True)
        print("Loading control and individual-gene means, then DISCOVERY pairs...",
              flush=True)
        control_mean = mean_rows(adata.X, controls, chunk_size)[0]
        needed = sorted({name for pair in discovery + reserved for name in pair})
        effects = {name: mean_rows(adata.X, singles[name], chunk_size)[0] -
                   control_mean for name in needed}
        residuals = []
        for n, pair in enumerate(discovery, 1):
            observed = mean_rows(adata.X, pairs[pair], chunk_size)[0]
            residuals.append(observed - control_mean - effects[pair[0]] -
                             effects[pair[1]])
            if n % 25 == 0:
                print(f"Read {n}/{len(discovery)} discovery pairs; "
                      "zero reserved pair-expression rows read", flush=True)
        residual = np.stack(residuals)
        features = pair_features(effects, discovery, n_components, seed)
        evaluations, best = cross_validate(features, residual, discovery,
                                           gene_index, seed)
        result = {"version": VERSION, "source": str(Path(data).resolve()),
                  "manifest": str(Path(split).resolve()),
                  "n_discovery": len(discovery), "n_reserved_unread": len(reserved),
                  "n_genes": adata.n_vars, "seed": seed, "cv_folds": 5,
                  "n_components": min(n_components, len(effects)-1, adata.n_vars-1),
                  "additive_cv_mse": evaluations[0]["mse_all_genes"],
                  "selected": best, "candidates": evaluations}
        return result
    finally:
        adata.file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="Norman normalized log1p .h5ad")
    parser.add_argument("split", help="k562_pair_signal.split.json from the audit")
    parser.add_argument("--out", default="k562_pair_interaction_cv.json")
    parser.add_argument("--min-cells", type=int, default=50)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    if args.min_cells < 4 or args.latent_dim < 2 or args.chunk_size < 1:
        parser.error("Need min-cells >= 4, latent-dim >= 2, chunk-size >= 1")
    for filename in (args.data, args.split):
        if not Path(filename).is_file():
            parser.error(f"File not found: {filename}")
    if not Path(args.out).parent.is_dir():
        parser.error(f"Output folder does not exist: {Path(args.out).parent}")
    try:
        result = run(args.data, args.split, args.latent_dim, args.min_cells,
                     args.seed, args.chunk_size)
    except ImportError as exc:
        parser.error(f"Install requirements.txt first: {exc}")
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("method\tk\tstrength\tCV_MSE\tCV_MSE_without_target_genes")
    for row in result["candidates"]:
        print(f"{row['method']}\t{row['neighbors']}\t{row['strength']:g}\t"
              f"{row['mse_all_genes']:.8f}\t"
              f"{row['mse_excluding_targets']:.8f}")
    best = result["selected"]
    print(f"SELECTED FROM DISCOVERY-PAIR CV: {best['method']} "
          f"k={best['neighbors']} strength={best['strength']:g} "
          f"MSE={best['mse_all_genes']:.8f}; "
          f"additive={result['additive_cv_mse']:.8f}")
    print(f"Saved CV results to {args.out}. Reserved pairs remain unread. "
          "This is a non-graph feasibility predictor, not a conference result.")


if __name__ == "__main__":
    main()
