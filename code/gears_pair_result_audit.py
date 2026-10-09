r"""Audit the matched Norman 101-pair comparison, without loading cell data.

PowerShell from the project root, after all five GEARS folds have finished:
  python .\benchmark_audit\gears_pair_result_audit.py .\gears_pair_result_go.json .\gears_pair_baseline_go.json --disjoint .\gears_pair_partner_disjoint_go.json --out .\gears_pair_audit_go.json --csv .\gears_pair_audit_go.csv

If training is still running, add --allow-partial for a clearly provisional
summary. This script never fits models or reads expression matrices.
"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


VERSION = "2026-10-06-gears-pair-audit-v1"
KEYS = ("additive_mse", "nearest_mse", "gears_mse")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unique_rows(payload, label):
    rows = payload.get("per_pair", [])
    if not rows:
        raise ValueError(f"{label} has no per-pair scores")
    if any(not isinstance(row.get("pair"), str) or
           not isinstance(row.get("fold"), int) for row in rows):
        raise ValueError(f"{label} has invalid pair/fold labels")
    names = [row["pair"] for row in rows]
    duplicates = [pair for pair, n in Counter(names).items() if n > 1]
    if duplicates:
        raise ValueError(f"{label} duplicates pairs: {duplicates[:5]}")
    return {row["pair"]: row for row in rows}


def number(row, key, label):
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)) or \
            not math.isfinite(value) or value < 0:
        raise ValueError(f"{label}: {key} must be a finite, nonnegative number")
    return float(value)


def mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def aggregate(rows, methods):
    scores = {key: mean(rows, key) for key in methods}
    for key in methods:
        if key != "additive_mse":
            scores[key + "_reduction_vs_additive_pct"] = (
                100 * (1 - scores[key] / scores["additive_mse"])
                if scores["additive_mse"] else None)
            scores[key + "_pairs_better_than_additive"] = sum(
                row[key] < row["additive_mse"] for row in rows)
    return scores


def summarize(gears, baseline, disjoint=None, allow_partial=False):
    if baseline.get("partner_disjoint"):
        raise ValueError("Pass the original nearest baseline as the second argument")
    if gears.get("split") and baseline.get("split_report") and \
            gears["split"] != baseline["split_report"]:
        raise ValueError("GEARS result and baseline refer to different split reports")
    b = unique_rows(baseline, "baseline")
    g = unique_rows(gears, "GEARS")
    expected = baseline.get("n_test_pairs")
    if expected != len(b) or expected != 101:
        raise ValueError("Expected the 101-pair official GEARS baseline")
    if gears.get("partial") and not allow_partial:
        raise ValueError("GEARS scored a checkpoint early; use --allow-partial for provisional reporting")
    if len(g) != expected and not allow_partial:
        raise ValueError(f"Only {len(g)}/{expected} GEARS pairs scored; wait for all five folds or use --allow-partial")
    if not set(g).issubset(b):
        raise ValueError("GEARS has pair names absent from the baseline")
    if len(g) > expected or not set(gears.get("folds", {})) == \
            {str(row["fold"]) for row in g.values()}:
        raise ValueError("GEARS fold metadata disagrees with per-pair rows")
    if not gears.get("folds") or any(not 1 <= row["fold"] <= 5 for row in g.values()):
        raise ValueError("Invalid GEARS fold IDs")
    d = unique_rows(disjoint, "partner-disjoint") if disjoint else None
    if d is not None:
        if not disjoint.get("partner_disjoint") or set(d) != set(b) or \
                disjoint.get("n_test_pairs") != expected:
            raise ValueError("Partner-disjoint baseline does not cover all 101 pairs")
        if (disjoint.get("source") != baseline.get("source") or
                disjoint.get("split_report") != baseline.get("split_report") or
                disjoint.get("n_genes") != baseline.get("n_genes")):
            raise ValueError("Partner-disjoint baseline has different data, split, or genes")
    rows = []
    for pair, model_row in sorted(g.items(), key=lambda item: (item[1]["fold"], item[0])):
        base = b[pair]
        if base["fold"] != model_row["fold"]:
            raise ValueError(f"Fold mismatch: {pair}")
        for key in ("additive_mse", "nearest_mse"):
            if not math.isclose(number(base, key, pair), number(model_row, key, pair),
                                rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"Copied baseline score disagrees for {pair}: {key}")
        row = {"pair": pair, "fold": base["fold"],
               "test_cells": model_row.get("test_cells")}
        for key in KEYS:
            row[key] = number(model_row, key, pair)
        if d is not None:
            if d[pair]["fold"] != base["fold"]:
                raise ValueError(f"Partner-disjoint fold mismatch: {pair}")
            for key in ("additive_mse", "nearest_mse"):
                if not math.isclose(number(d[pair], key, pair), number(base, key, pair),
                                    rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(f"Partner-disjoint baseline mismatch: {pair}")
            row["partner_disjoint_mse"] = number(d[pair], "partner_disjoint_mse", pair)
            row["partner_disjoint_donors"] = d[pair].get("partner_disjoint_donors")
        rows.append(row)
    for fold, summary in gears["folds"].items():
        subset = [r for r in rows if r["fold"] == int(fold)]
        if not subset:
            raise ValueError(f"Fold {fold} has no scored rows")
        for key in KEYS:
            if not math.isclose(mean(subset, key), number(summary, key, fold),
                                rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError(f"Fold {fold} summary disagrees for {key}")
    complete = (len(rows) == expected and len(gears["folds"]) == 5 and
                not gears.get("partial"))
    if not complete and not allow_partial:
        raise ValueError("GEARS run is incomplete")
    methods = KEYS + (("partner_disjoint_mse",) if d else ())
    macro = aggregate(rows, methods)
    macro["gears_minus_nearest_mse"] = macro["gears_mse"] - macro["nearest_mse"]
    macro["nearest_pairs_better_than_gears"] = sum(
        r["nearest_mse"] < r["gears_mse"] for r in rows)
    folds = {str(f): {"n": sum(r["fold"] == f for r in rows),
                      **aggregate([r for r in rows if r["fold"] == f], methods)}
             for f in sorted({r["fold"] for r in rows})}
    # Leave all pairs involving each gene out in turn. This is a sensitivity
    # analysis, not a confidence interval: pairs overlap in their targets.
    genes = sorted({gene for row in rows for gene in row["pair"].split("+")})
    leave_gene_out = []
    for gene in genes:
        remain = [r for r in rows if gene not in r["pair"].split("+")]
        if remain:
            leave_gene_out.append({"gene": gene, "n_remaining": len(remain),
                                   "nearest_minus_gears_mse":
                                   mean(remain, "nearest_mse") - mean(remain, "gears_mse"),
                                   "nearest_reduction_vs_additive_pct":
                                   100 * (1 - mean(remain, "nearest_mse") /
                                          mean(remain, "additive_mse"))})
    report = {
        "version": VERSION, "status": "complete" if complete else "provisional_partial",
        "n_scored": len(rows), "n_expected": expected,
        "n_folds": len(folds), "n_genes_scored": baseline.get("n_genes"),
        "metrics": macro, "folds": folds,
        "leave_one_gene_out": leave_gene_out,
        "worst_for_nearest_vs_gears": sorted(
            rows, key=lambda r: r["nearest_mse"] - r["gears_mse"], reverse=True)[:10],
        "worst_for_gears_vs_nearest": sorted(
            rows, key=lambda r: r["gears_mse"] - r["nearest_mse"], reverse=True)[:10],
        "limitations": [
            "Norman comparison was explored during method development; this is not independent validation.",
            "All-gene mean MSE differs from the validation top-20 DE metric used to select GEARS checkpoints.",
            "GEARS and ordinary nearest share a matched pair split. The partner-disjoint column excludes donors only for the simple predictor; it is NOT a partner-disjoint GEARS comparison.",
            "Pairs share genes and cells; folds are not independent biological experiments.",
        ],
    }
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gears_result")
    parser.add_argument("baseline")
    parser.add_argument("--disjoint")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--out", default="gears_pair_audit_go.json")
    parser.add_argument("--csv", default="gears_pair_audit_go.csv")
    args = parser.parse_args()
    gears = load(args.gears_result)
    if gears.get("baseline") and Path(gears["baseline"]).resolve() != \
            Path(args.baseline).resolve():
        parser.error("GEARS result names a different baseline JSON file")
    report, rows = summarize(gears, load(args.baseline),
                             load(args.disjoint) if args.disjoint else None,
                             args.allow_partial)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    with Path(args.csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    score = report["metrics"]
    print(f"{report['status']}: {report['n_scored']}/{report['n_expected']} pairs, "
          f"{report['n_folds']}/5 folds; all-gene pair-mean MSE")
    print(f"additive={score['additive_mse']:.8f}; "
          f"nearest={score['nearest_mse']:.8f}; "
          f"GEARS={score['gears_mse']:.8f}; "
          f"nearest better than GEARS on "
          f"{score['nearest_pairs_better_than_gears']}/{report['n_scored']} pairs")
    if "partner_disjoint_mse" in score:
        print(f"Simple predictor, partner-disjoint donors: "
              f"{score['partner_disjoint_mse']:.8f} (not a disjoint GEARS score)")
    print(f"Saved {args.out} and {args.csv}")


if __name__ == "__main__":
    main()
