r"""Score a graph residual branch on the SAME 101 pairs scored by GEARS.

PowerShell after extracting the matched graph package into the project root:
  python .\matched_graph\gears_pair_graph_branch_matched.py .\gears_data\norman\perturb_processed.h5ad .\gears_pair_splits_go\coverage_report.json .\gears_pair_baseline_go.json .\gears_pair_result_go.json .\string_balanced1.json .\k562_hic_mapping_v2.json --out .\gears_pair_graph_branch_matched.json

The method uses the previously fixed nearest, low-rank kernel and blend
settings. No tuning on GEARS scores or pair test outcomes. It checks that
the original nearest and additive pair scores reproduce exactly before
reporting any graph result. Same fold test pair, genes, normalization and
condition-mean all-gene MSE as official GEARS. Graph edges are name-matched
to the GEARS gene axis; strict Hi-C only. Previously explored Norman data.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD

from gears_pair_result_audit import summarize
from k562_pair_graph_context_cv import graph_pair_features
from k562_pair_graph_residual_branch_cv import (
    BRANCH_WEIGHT, GRAPH_WEIGHT, K, LATENT, RIDGE, rbf_kernels,
)
from k562_pair_interaction_cv import pair_features, predict_residual
from k562_pair_signal_audit import mean_rows
from prepare_gears_pair_splits import VERSION as SPLIT_VERSION, SEED
from prepare_gears_pair_splits import gears_conditions


VERSION = "2026-10-07-gears-pair-matched-graph-branch-v1"


def pair_key(name):
    parts = tuple(sorted(name.upper().split("+")))
    if len(parts) != 2 or parts[0] == parts[1]:
        raise ValueError(f"Invalid pair name {name!r}")
    return parts


def project_edges(edges, genes):
    """Retain only both-endpoint edges with symbols on GEARS expression axis."""
    allowed = {g.upper() for g in genes}
    filtered = [[a, b, w] for a, b, w in edges
                if str(a).upper() in allowed and
                str(b).upper() in allowed and str(a).upper() != str(b).upper()]
    return filtered


def learned_predictions(base_features, graph_features, train_residual,
                        n_train, seed=SEED):
    """All branches fit using ONLY the named training pair labels."""
    x_train, x_test = base_features[:n_train], base_features[n_train:]
    nearest = predict_residual(x_train, train_residual, x_test,
                               ("nearest", K, 1.0))
    center = train_residual.mean(axis=0)
    projector = TruncatedSVD(
        n_components=min(LATENT, n_train-1, train_residual.shape[1]-1),
        random_state=seed)
    latent = projector.fit_transform(train_residual-center)
    k_train, k_test = rbf_kernels(x_train, x_test)
    estimates = {"nearest": nearest,
                 "additive": np.zeros_like(nearest)}
    for method, features in [("no_graph", None), *graph_features.items()]:
        if features is None:
            kernel, query = k_train, k_test
        else:
            extra, extra_query = rbf_kernels(features[:n_train],
                                              features[n_train:])
            kernel = k_train + GRAPH_WEIGHT*extra
            query = k_test + GRAPH_WEIGHT*extra_query
        weights = np.linalg.solve(kernel + RIDGE*np.eye(n_train), latent)
        branch = (query @ weights) @ projector.components_ + center
        estimates[method] = (1-BRANCH_WEIGHT)*nearest + BRANCH_WEIGHT*branch
    return estimates


def run(data, report_path, baseline_path, gears_path, string_path, hic_path,
        chunk_size=256):
    import scanpy as sc
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    gears = json.loads(Path(gears_path).read_text(encoding="utf-8"))
    audit, matched = summarize(gears, baseline)
    if audit["status"] != "complete" or audit["n_scored"] != 101:
        raise ValueError("Only a complete 101-pair GEARS run can be compared")
    if (report.get("version") != SPLIT_VERSION or
            report.get("n_comparison_pairs") != 101 or
            not report.get("go_filter_note") or
            Path(report["gears_data"]).resolve() != Path(data).resolve() or
            Path(baseline["source"]).resolve() != Path(data).resolve() or
            Path(baseline["split_report"]).resolve() != Path(report_path).resolve() or
            not [f["fold"] for f in report["folds"]] == [1, 2, 3, 4, 5]):
        raise ValueError("GEARS report, baseline, folds and expression file differ")
    with open(string_path, encoding="utf-8") as handle:
        string = json.load(handle)
    with open(hic_path, encoding="utf-8") as handle:
        hic = json.load(handle)
    if (string.get("species") != 9606 or string.get("split") != "balanced1" or
            hic.get("version") != "2026-09-29-k562-hic-gene-mapping-v2"):
        raise ValueError("Expected the original human STRING and strict Hi-C mapping")

    pair_names = sorted({p for fold in report["folds"] for p in
                         fold["train_pairs"]+fold["validation_pairs"]+
                         fold["test_pairs"]})
    if set(pair_names) != {row["pair"] for row in matched}:
        raise ValueError("Matched GEARS pairs differ from report pair cohort")
    pairs = {name: pair_key(name) for name in pair_names}
    targets = sorted({g for pair in pairs.values() for g in pair})
    adata = sc.read_h5ad(data, backed="r")
    try:
        if (adata.X is None or "condition" not in adata.obs or
                "gene_name" not in adata.var):
            raise ValueError("Expected official GEARS Norman h5ad")
        genes = adata.var["gene_name"].astype(str).tolist()
        if len(genes) != baseline["n_genes"] or \
                len({g.upper() for g in genes}) != len(genes):
            raise ValueError("GEARS expression gene axis differs from saved baseline")
        gene_index = {g.upper(): i for i, g in enumerate(genes)}
        by_label = {}
        for i, label in enumerate(adata.obs["condition"].astype(str)):
            by_label.setdefault(label, []).append(i)
        native = gears_conditions(by_label)

        def indices(condition):
            if condition not in native:
                raise ValueError(f"Missing official condition {condition}")
            return np.asarray(sorted(index for alias in native[condition]
                                     for index in by_label[alias]), dtype=np.int64)

        control = mean_rows(adata.X, indices(("control",)), chunk_size)[0]
        effects = {g: mean_rows(adata.X, indices(("single", g)),
                                chunk_size)[0] - control for g in targets}
        pair_mean = {}
        for i, name in enumerate(pair_names, 1):
            pair_mean[name] = mean_rows(
                adata.X, indices(("pair", *pairs[name])), chunk_size)[0]
            if i % 25 == 0:
                print(f"Averaged {i}/{len(pair_names)} pair conditions", flush=True)
    finally:
        adata.file.close()
    s_edges = project_edges(string["edges"], genes)
    h_edges = project_edges(hic["strict_promoter_promoter"]["edges"], genes)
    if not s_edges or not h_edges:
        print("Warning: a relation has zero mapped edges on the official GEARS "
              "gene axis. Its branch will equal no_graph and cannot support a "
              "graph-specific conclusion.", flush=True)
    network = {
        "string": (s_edges, None), "hic_strict": (h_edges, None),
        "shuffled_string": (s_edges, SEED+37),
        "shuffled_hic": (h_edges, SEED+37),
    }
    coverage = {}
    for label, (edges, _) in network.items():
        connected_targets = {str(g).upper() for a, b, _ in edges
                             for g in (a, b)} & set(targets)
        coverage[label] = {"retained_edges": len(edges),
                           "targets_with_an_edge": len(connected_targets),
                           "targets_in_cohort": len(targets)}

    expected_rows = {row["pair"]: row for row in matched}
    scored = []
    for fold in report["folds"]:
        train_names, val_names, test_names = (
            fold["train_pairs"], fold["validation_pairs"], fold["test_pairs"])
        if (set(train_names) & set(val_names+test_names) or
                set(val_names) & set(test_names)):
            raise ValueError("Overlapping pair train/validation/test split")
        with open(fold["pickle"], "rb") as handle:
            native_split = pickle.load(handle)
        for section, names in [("train", train_names), ("val", val_names),
                               ("test", test_names)]:
            for name in names:
                if not set(native[("pair", *pairs[name])]).issubset(
                        native_split[section]):
                    raise ValueError(f"Native aliases of {name} cross folds")
        ordered_names = train_names + test_names
        ordered_pairs = [pairs[name] for name in ordered_names]
        base = pair_features(effects, ordered_pairs, seed=SEED)
        graph_features = {}
        for method, (edges, shuffle) in network.items():
            graph_features[method], _, _ = graph_pair_features(
                effects, ordered_pairs, genes, edges, SEED, shuffle)
        graph_features["string_plus_hic"] = np.concatenate(
            [graph_features["string"], graph_features["hic_strict"]], axis=1)
        graph_features["shuffled_both"] = np.concatenate(
            [graph_features["shuffled_string"],
             graph_features["shuffled_hic"]], axis=1)
        residual = lambda name: (pair_mean[name] - control -
                                 effects[pairs[name][0]] - effects[pairs[name][1]])
        train_residual = np.stack([residual(name) for name in train_names])
        test_residual = np.stack([residual(name) for name in test_names])
        estimates = learned_predictions(base, graph_features, train_residual,
                                         len(train_names), SEED)
        for i, name in enumerate(test_names):
            row = {"pair": name, "fold": fold["fold"], "scores": {}}
            cols = [gene_index[g] for g in pairs[name] if g in gene_index]
            keep = np.ones(len(genes), dtype=bool)
            keep[cols] = False
            for method, prediction in estimates.items():
                squared = (test_residual[i] - prediction[i])**2
                row["scores"][method] = {
                    "mse_all": float(squared.mean()),
                    "mse_excluding_targets": float(squared[keep].mean())}
            old = expected_rows[name]
            for method, old_key in [("additive", "additive_mse"),
                                     ("nearest", "nearest_mse")]:
                new = row["scores"][method]["mse_all"]
                if not np.isclose(new, old[old_key], rtol=2e-5, atol=2e-8):
                    raise ValueError(f"Pair {name}: {method} score does not "
                                     f"reproduce saved baseline: {new} vs {old[old_key]}")
            row["scores"]["gears"] = {"mse_all": old["gears_mse"]}
            scored.append(row)
        print(f"Validated fold {fold['fold']}/5 against the saved nearest "
              "and additive scores", flush=True)
    if len(scored) != 101 or len({r["pair"] for r in scored}) != 101:
        raise ValueError("Matched 101-pair cohort incomplete")
    methods = list(scored[0]["scores"])
    macro = {method: {"mse_all": float(np.mean([
                 row["scores"][method]["mse_all"] for row in scored]))}
             for method in methods}
    for method in methods:
        if method != "gears":
            macro[method]["pairs_better_than_gears"] = sum(
                row["scores"][method]["mse_all"] < row["scores"]["gears"]["mse_all"]
                for row in scored)
        if method in ("string", "hic_strict", "string_plus_hic",
                      "shuffled_string", "shuffled_hic", "shuffled_both"):
            macro[method]["pairs_better_than_no_graph"] = sum(
                row["scores"][method]["mse_all"] <
                row["scores"]["no_graph"]["mse_all"] for row in scored)
    print("Same official GEARS test pairs, gene axis, folds, expression scale; "
          "condition-mean all-gene MSE (lower is better):")
    for method, result in macro.items():
        print(f"  {method}: {result['mse_all']:.8f}", flush=True)
    print("Real biological graph contribution requires improvement against "
          "BOTH no_graph and the corresponding shuffle. These 101 Norman "
          "pairs have been examined before; comparisons are exploratory.")
    return {"version": VERSION, "n_pairs": len(scored),
            "n_genes": len(genes), "coverage": coverage,
            "graph_settings": {"rank": LATENT, "ridge": RIDGE,
                               "graph_weight": GRAPH_WEIGHT,
                               "branch_weight": BRANCH_WEIGHT, "nearest_k": K},
            "macro": macro, "per_pair": scored,
            "limitations": [
                "Previously examined Norman cohort; no independent model test.",
                "GEARS checkpoint was selected by validation DE20; the primary "
                "matched comparison here is all-gene pair-mean MSE.",
                "STRING/Hi-C were mapped by gene symbol onto official genes; "
                "low retained coverage limits interpretation.",
                "This graph residual branch is kernel regression, not "
                "a trained dual GraphSAGE, OT, VAE, or DDPM." ]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data")
    p.add_argument("split_report")
    p.add_argument("baseline")
    p.add_argument("gears")
    p.add_argument("string")
    p.add_argument("hic")
    p.add_argument("--out", default="gears_pair_graph_branch_matched.json")
    p.add_argument("--chunk-size", type=int, default=256)
    args = p.parse_args()
    if args.chunk_size < 1:
        p.error("--chunk-size must be positive")
    for path in (args.data, args.split_report, args.baseline,
                 args.gears, args.string, args.hic):
        if not Path(path).is_file():
            p.error(f"Missing {path}")
    outcome = run(args.data, args.split_report, args.baseline, args.gears,
                  args.string, args.hic, args.chunk_size)
    Path(args.out).write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
