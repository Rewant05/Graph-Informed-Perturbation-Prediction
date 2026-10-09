r"""Refit the original pair predictor on official GEARS Norman expression.

PowerShell after prepare_gears_pair_splits.py --allow-intersection:
  python .\gears_setup_v3\gears_matched_simple_baseline.py .\gears_data\norman\perturb_processed.h5ad .\gears_pair_splits_v3\coverage_report.json --out .\gears_pair_baseline_v3.json

For a CPU-only partner-disjoint diagnostic, add --partner-disjoint and write
to a new output JSON. Each test pair A+B then excludes from its training
donors every pair containing A or B, while measured singles remain available.
This is the same Norman biology and is not an independent GEARS model result.

Predicts held-out pair condition MEANS using only the training pair residuals
in the five exported folds. Uses GEARS's measured gene axis and expression
scale. The validation pairs are not used in model fitting. This baseline is
NOT a GEARS result, and the 26 old reserved pairs are excluded.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from k562_pair_interaction_cv import pair_features, predict_residual
from k562_pair_signal_audit import mean_rows
from prepare_gears_pair_splits import VERSION as SPLIT_VERSION
from prepare_gears_pair_splits import gears_conditions, SEED


VERSION = "2026-09-30-gears-expression-simple-pair-baseline-v1"


def key_for_pair(name):
    parts = name.upper().split("+")
    if len(parts) != 2 or parts[0] == parts[1]:
        raise ValueError(f"Expected one two-gene pair, got {name!r}")
    return tuple(sorted(parts))


def partner_disjoint_predictions(train_features, train_residual, train_pairs,
                                 query_features, query_pairs, k=10):
    """Predict each query without donor pairs sharing either query gene."""
    if (len(train_features) != len(train_residual) or
            len(train_features) != len(train_pairs) or
            len(query_features) != len(query_pairs)):
        raise ValueError("Features, residuals, and pair names must align")
    predictions, counts = [], []
    for query, pair in zip(query_features, query_pairs):
        forbidden = set(pair)
        eligible = np.asarray([i for i, donor in enumerate(train_pairs)
                               if forbidden.isdisjoint(donor)], dtype=np.int64)
        if len(eligible) < k:
            raise ValueError(f"Only {len(eligible)} partner-disjoint donors "
                             f"for {pair}; need at least {k}")
        predictions.append(predict_residual(
            train_features[eligible], train_residual[eligible],
            query[np.newaxis, :], ("nearest", k, 1.0))[0])
        counts.append(len(eligible))
    return np.stack(predictions), counts


def evaluate(data, report_path, chunk_size=256, partner_disjoint=False):
    import scanpy as sc

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if (report.get("version") != SPLIT_VERSION or
            not report.get("split_pickles_created") or
            not report.get("folds") or
            Path(report.get("gears_data", "")).resolve() != Path(data).resolve()):
        raise ValueError("Pass the v3 GEARS coverage report for this same h5ad")
    folds = report["folds"]
    pair_names = sorted(set(p for fold in folds for field in
                            ("train_pairs", "validation_pairs", "test_pairs")
                            for p in fold[field]))
    expected = set(p for fold in folds for p in fold["test_pairs"])
    if len(expected) != report["n_comparison_pairs"]:
        raise ValueError("Test folds do not cover the comparison cohort")
    pairs = {name: key_for_pair(name) for name in pair_names}
    needed = sorted(set(g for pair in pairs.values() for g in pair))
    adata = sc.read_h5ad(data, backed="r")
    try:
        if adata.X is None or "condition" not in adata.obs or \
                "gene_name" not in adata.var:
            raise ValueError("Expected official GEARS Norman h5ad")
        genes = adata.var["gene_name"].astype(str).tolist()
        gene_index = {name.upper(): i for i, name in enumerate(genes)}
        if len(gene_index) != len(genes):
            raise ValueError("Duplicate gene symbols in GEARS expression matrix")
        labels = adata.obs["condition"].astype(str).to_numpy()
        by_label = {}
        for row, label in enumerate(labels):
            by_label.setdefault(label, []).append(row)
        native = gears_conditions(by_label)

        def rows_for(key):
            if key not in native:
                raise ValueError(f"Official GEARS data lacks condition {key}")
            return np.asarray(sorted(row for alias in native[key]
                                     for row in by_label[alias]), dtype=np.int64)

        control = mean_rows(adata.X, rows_for(("control",)), chunk_size)[0]
        single = {g: mean_rows(adata.X, rows_for(("single", g)), chunk_size)[0]
                  for g in needed}
        effects = {g: single[g] - control for g in needed}
        # Pair truth is loaded for scoring, and only train pair truth is passed
        # into each fold's predictor. Validation truth is never fitted/scored.
        pair_mean = {}
        for i, name in enumerate(pair_names, 1):
            pair_mean[name] = mean_rows(
                adata.X, rows_for(("pair", *pairs[name])), chunk_size)[0]
            if i % 25 == 0:
                print(f"Averaged {i}/{len(pair_names)} official pair conditions...",
                      flush=True)
    finally:
        adata.file.close()

    folds_result = []
    test_rows = []
    for fold in folds:
        train_names, val_names, test_names = (
            fold["train_pairs"], fold["validation_pairs"], fold["test_pairs"])
        if set(train_names) & (set(val_names) | set(test_names)) or \
                set(val_names) & set(test_names) or len(train_names) < 10:
            raise ValueError("Fold pair lists overlap or too few training pairs")
        with open(fold["pickle"], "rb") as handle:
            native_split = pickle.load(handle)
        for section, names in (("train", train_names), ("val", val_names),
                               ("test", test_names)):
            for name in names:
                if not set(native[("pair", *pairs[name])]).issubset(
                        native_split[section]):
                    raise ValueError(f"Fold {fold['fold']} does not group "
                                     f"native aliases of {name} in {section}")
        ordered_names = train_names + test_names
        ordered_pairs = [pairs[name] for name in ordered_names]
        features = pair_features(effects, ordered_pairs, n_components=16,
                                 seed=SEED)
        residual = lambda name: (pair_mean[name] - single[pairs[name][0]] -
                                 single[pairs[name][1]] + control)
        train_residual = np.stack([residual(name) for name in train_names])
        test_residual = np.stack([residual(name) for name in test_names])
        prediction = predict_residual(
            features[:len(train_names)], train_residual,
            features[len(train_names):], ("nearest", 10, 1.0))
        if partner_disjoint:
            cold_prediction, donor_counts = partner_disjoint_predictions(
                features[:len(train_names)], train_residual,
                [pairs[name] for name in train_names],
                features[len(train_names):], [pairs[name] for name in test_names])
            cold_error = np.mean((test_residual - cold_prediction)**2, axis=1)
        error2 = (test_residual - prediction)**2
        nearest_error = np.mean(error2, axis=1)
        nearest_excl = np.asarray([
            np.mean(np.delete(error2[i], [gene_index[g] for g in pairs[name]
                                          if g in gene_index]))
            for i, name in enumerate(test_names)])
        mean_error = np.mean((test_residual - train_residual.mean(axis=0))**2,
                             axis=1)
        additive_error = np.mean(test_residual**2, axis=1)
        no_change_error = np.mean(np.stack(
            [pair_mean[name] - control for name in test_names])**2, axis=1)
        rows = [{"pair": name, "fold": fold["fold"],
                 "no_change_mse": float(no_change_error[i]),
                 "additive_mse": float(additive_error[i]),
                 "mean_residual_mse": float(mean_error[i]),
                 "nearest_mse": float(nearest_error[i]),
                 "nearest_mse_excluding_target_genes": float(nearest_excl[i])}
                for i, name in enumerate(test_names)]
        if partner_disjoint:
            for i, row in enumerate(rows):
                row["partner_disjoint_mse"] = float(cold_error[i])
                row["partner_disjoint_donors"] = donor_counts[i]
        test_rows.extend(rows)
        folds_result.append({"fold": fold["fold"],
                             "n_train_pairs": len(train_names),
                             "n_validation_pairs": len(val_names),
                             "n_test_pairs": len(test_names),
                             "additive_mse": float(additive_error.mean()),
                             "nearest_mse": float(nearest_error.mean())})
        if partner_disjoint:
            folds_result[-1]["partner_disjoint_mse"] = float(cold_error.mean())
        print(f"Fold {fold['fold']}: train={len(train_names)}, "
              f"val={len(val_names)}, test={len(test_names)}; "
              f"additive={additive_error.mean():.8f}; "
              f"nearest={nearest_error.mean():.8f}", flush=True)
    if len(test_rows) != len(expected) or \
            set(row["pair"] for row in test_rows) != expected:
        raise ValueError("Test pairs missing or duplicated across folds")
    metrics = {key: float(np.mean([row[key] for row in test_rows]))
               for key in ("no_change_mse", "additive_mse",
                           "mean_residual_mse", "nearest_mse",
                           "nearest_mse_excluding_target_genes")}
    if partner_disjoint:
        metrics["partner_disjoint_mse"] = float(np.mean(
            [row["partner_disjoint_mse"] for row in test_rows]))
    print(f"{len(test_rows)}-PAIR INTERSECTION macro MSE on official GEARS gene/scale: "
          f"additive={metrics['additive_mse']:.8f}; "
          f"nearest={metrics['nearest_mse']:.8f}; "
          f"pairs improved={sum(row['nearest_mse'] < row['additive_mse'] for row in test_rows)}/"
          f"{len(test_rows)}")
    if partner_disjoint:
        print(f"PARTNER-DISJOINT (training donors sharing either query gene removed): "
              f"MSE={metrics['partner_disjoint_mse']:.8f}; "
              f"improved={sum(row['partner_disjoint_mse'] < row['additive_mse'] for row in test_rows)}/"
              f"{len(test_rows)}; minimum donors="
              f"{min(row['partner_disjoint_donors'] for row in test_rows)}")
    print("This is the simple predictor on GEARS's expression data. "
          "No GEARS model has been trained, and previous 103-pair scores "
          "are on a DIFFERENT cohort and expression scale.")
    return {"version": VERSION, "source": str(Path(data).resolve()),
            "split_report": str(Path(report_path).resolve()),
            "n_test_pairs": len(test_rows), "n_genes": len(genes),
            "measured_target_genes": len(set(needed) & set(gene_index)),
            "frozen_method": "nearest, k=10, strength=1, SVD=16",
            "partner_disjoint": partner_disjoint,
            "metrics": metrics, "folds": folds_result, "per_pair": test_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gears_data")
    parser.add_argument("split_report")
    parser.add_argument("--out", default="gears_pair_baseline_v3.json")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--partner-disjoint", action="store_true",
                        help="Exclude training pairs containing either test-pair gene")
    args = parser.parse_args()
    for path in (args.gears_data, args.split_report):
        if not Path(path).is_file():
            parser.error(f"File not found: {path}")
    if args.chunk_size < 1 or not Path(args.out).parent.is_dir():
        parser.error("Check chunk size and output directory")
    result = evaluate(args.gears_data, args.split_report, args.chunk_size,
                      args.partner_disjoint)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
