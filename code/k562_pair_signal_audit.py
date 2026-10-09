r"""Measure reproducible, non-additive K562 CRISPRa pair responses.

PowerShell:
  python .\k562_pair_signal_audit.py .\norman_log1p_10k_v2.h5ad --out .\k562_pair_signal.csv

This is a diagnostic, NOT a held-out prediction model. It deliberately reads
each measured pair to ask whether its deviation from the sum of its measured
single-gene responses is larger than within-condition sampling variation.
The entire expression matrix is never converted to a dense array.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy import sparse


VERSION = "2026-09-29-k562-pair-signal-v1"
CONTROLS = frozenset({"control", "ctrl", "unperturbed"})


def collect_groups(conditions, gene_names, min_cells=50):
    """Find observed singles and pairs; prefer an exact gene over separators."""
    if min_cells < 4:
        raise ValueError("min_cells must be at least 4 for half-split checks")
    known_genes = {str(g).upper() for g in gene_names}
    by_label = {}
    for row, label in enumerate(conditions):
        by_label.setdefault(str(label).strip().upper(), []).append(row)

    controls = np.asarray(sorted(i for label, rows in by_label.items()
                                 if label.lower() in CONTROLS for i in rows),
                          dtype=np.int64)
    singles = {label: np.asarray(rows, dtype=np.int64)
               for label, rows in by_label.items()
               if label in known_genes and label.lower() not in CONTROLS
               and len(rows) >= min_cells}
    pair_rows = {}
    pair_labels = {}
    for label, rows in by_label.items():
        if label in known_genes:  # A real gene symbol may contain '_'.
            continue
        names = re.split(r"[+_]", label)
        if (len(names) != 2 or names[0] == names[1] or
                names[0] not in singles or names[1] not in singles):
            continue
        key = tuple(sorted(names))
        pair_rows.setdefault(key, []).extend(rows)
        pair_labels.setdefault(key, []).append(label)
    pairs = {key: np.asarray(sorted(rows), dtype=np.int64)
             for key, rows in pair_rows.items() if len(rows) >= min_cells}
    if len(controls) < min_cells:
        raise ValueError(f"Expected at least {min_cells} unperturbed controls; found {len(controls)}")
    if not pairs:
        raise ValueError("No eligible two-gene conditions found; inspect perturbation labels")
    return controls, singles, pairs, pair_labels


def mean_rows(matrix, indices, chunk_size=256):
    """Read sorted row chunks; return mean and sum of squares per gene."""
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        raise ValueError("Cannot average zero cells")
    total = np.zeros(matrix.shape[1], dtype=np.float64)
    squares = np.zeros(matrix.shape[1], dtype=np.float64)
    for offset in range(0, len(indices), chunk_size):
        block = matrix[np.sort(indices[offset:offset + chunk_size])]
        if sparse.issparse(block):
            total += np.asarray(block.sum(axis=0), dtype=np.float64).ravel()
            squares += np.asarray(block.multiply(block).sum(axis=0),
                                  dtype=np.float64).ravel()
        else:
            values = np.asarray(block, dtype=np.float64)
            total += values.sum(axis=0)
            squares += np.square(values).sum(axis=0)
    return total / len(indices), squares


def group_statistics(matrix, indices, rng, chunk_size=256):
    """Full mean, two disjoint half-means and sampling variance by gene."""
    if len(indices) < 4:
        raise ValueError("At least four cells are required per condition")
    first, second = np.array_split(rng.permutation(indices), 2)
    first_mean, first_squares = mean_rows(matrix, first, chunk_size)
    second_mean, second_squares = mean_rows(matrix, second, chunk_size)
    n = len(indices)
    full = (len(first) * first_mean + len(second) * second_mean) / n
    # Sample variance / n estimates sampling variance of each full mean.
    squared = first_squares + second_squares
    variance = np.maximum(squared - n * full**2, 0) / (n * (n - 1))
    return full, first_mean, second_mean, variance


def correlation(a, b):
    if not np.std(a) or not np.std(b):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def score_pair(pair, stats, n_cells, labels):
    """Compare the measured double response to its two measured singles."""
    a, b = pair
    mu_ab, ab1, ab2, var_ab = stats[pair]
    mu_a, a1, a2, var_a = stats[a]
    mu_b, b1, b2, var_b = stats[b]
    mu_c, c1, c2, var_c = stats["CONTROL"]
    interaction = mu_ab - mu_a - mu_b + mu_c
    residual1 = ab1 - a1 - b1 + c1
    residual2 = ab2 - a2 - b2 + c2
    energy = float(np.mean(interaction ** 2))
    sampling = float(np.mean(var_ab + var_a + var_b + var_c))
    # Additive prediction is mu_a + mu_b - mu_c; no-change predicts mu_c.
    no_change = float(np.mean((mu_ab - mu_c) ** 2))
    return {
        "pair": "+".join(pair), "labels": ";".join(sorted(labels)),
        "pair_cells": n_cells[pair], "a_cells": n_cells[a],
        "b_cells": n_cells[b], "control_cells": n_cells["CONTROL"],
        "no_change_mse": no_change, "additive_mse": energy,
        "sampling_mse_est": sampling,
        "excess_mse_est": max(0.0, energy - sampling),
        "interaction_to_sampling": energy / sampling if sampling else float("inf"),
        "split_half_interaction_corr": correlation(residual1, residual2),
    }


def reserve_pairs(pairs, seed, fraction=0.2):
    """Deterministically hide pair-expression rows from this diagnostic."""
    if not 0 <= fraction < 1:
        raise ValueError("reserve fraction must be in [0, 1)")
    ranking = sorted(pairs, key=lambda pair: hashlib.sha256(
        f"{seed}:{'+'.join(pair)}".encode("utf-8")).hexdigest())
    n_reserved = min(round(len(ranking) * fraction), len(ranking) - 1)
    return set(ranking[:n_reserved])


def audit(matrix, conditions, gene_names, min_cells=50, seed=20260929,
          chunk_size=256, reserve_fraction=0.2):
    controls, singles, pairs, labels = collect_groups(conditions, gene_names,
                                                       min_cells)
    reserved = reserve_pairs(pairs, seed, reserve_fraction)
    discovery_pairs = {pair: rows for pair, rows in pairs.items()
                       if pair not in reserved}
    rng = np.random.default_rng(seed)
    needed = {name for pair in discovery_pairs for name in pair}
    indices = {"CONTROL": controls}
    indices.update({name: singles[name] for name in sorted(needed)})
    indices.update(discovery_pairs)
    stats = {}
    for count, (name, rows) in enumerate(indices.items(), 1):
        stats[name] = group_statistics(matrix, rows, rng, chunk_size)
        if count % 25 == 0:
            print(f"Averaged {count}/{len(indices)} condition groups...", flush=True)
    counts = {name: len(rows) for name, rows in indices.items()}
    results = [score_pair(pair, stats, counts, labels[pair])
               for pair in sorted(discovery_pairs)]
    return results, len(controls), len(singles), len(pairs), sorted(reserved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="Norman log1p CP10k .h5ad")
    parser.add_argument("--out", default="k562_pair_signal.csv",
                        help="CSV output path")
    parser.add_argument("--min-cells", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--reserve-fraction", type=float, default=0.2,
                        help="Fraction of pair outcomes not read by this diagnostic")
    args = parser.parse_args()
    if args.min_cells < 4 or args.chunk_size < 1:
        parser.error("--min-cells must be >= 4 and --chunk-size >= 1")
    if not 0 <= args.reserve_fraction < 1:
        parser.error("--reserve-fraction must be in [0, 1)")
    if not Path(args.data).is_file():
        parser.error(f"Input file not found: {args.data}")
    if not Path(args.out).parent.is_dir():
        parser.error(f"Output folder does not exist: {Path(args.out).parent}")
    try:
        import scanpy as sc
    except ImportError as exc:
        parser.error(f"Install scanpy first: {exc}")

    adata = sc.read_h5ad(args.data, backed="r")
    try:
        if adata.X is None or "perturbation" not in adata.obs:
            parser.error("AnnData needs X and obs['perturbation']")
        if "log1p" not in adata.uns:
            parser.error("Input lacks uns['log1p']; pass norman_log1p_10k_v2.h5ad, not raw counts")
        conditions = adata.obs["perturbation"].astype(str).to_numpy()
        gene_names = (adata.var["gene_symbol"].astype(str).to_numpy()
                      if "gene_symbol" in adata.var else
                      adata.var_names.astype(str).to_numpy())
        print(f"Version={VERSION}; K562 CRISPRa diagnostic; "
              f"{adata.n_obs} cells x {adata.n_vars} genes", flush=True)
        print(f"Normalization={adata.uns.get('normalization_protocol', 'log1p (unspecified)')}; "
              f"seed={args.seed}; min_cells={args.min_cells}", flush=True)
        rows, n_controls, n_singles, n_pairs, reserved = audit(
            adata.X, conditions, gene_names, args.min_cells,
            args.seed, args.chunk_size, args.reserve_fraction)
    finally:
        adata.file.close()

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = str(Path(args.out).with_suffix(".split.json"))
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({"version": VERSION, "seed": args.seed,
                   "reserve_fraction": args.reserve_fraction,
                   "discovery_pairs": [row["pair"] for row in rows],
                   "reserved_pairs": ["+".join(pair) for pair in reserved]},
                  handle, indent=2)
    print(f"Controls={n_controls}; eligible singles={n_singles}; "
          f"eligible pairs={n_pairs}; diagnostic pairs={len(rows)}; "
          f"reserved pairs={len(reserved)}")
    print("pair\tcells\tadditive_MSE\tsampling_MSE_est\tratio\thalf_corr")
    for row in sorted(rows, key=lambda item: item["excess_mse_est"], reverse=True)[:12]:
        print(f"{row['pair']}\t{row['pair_cells']}\t"
              f"{row['additive_mse']:.6g}\t{row['sampling_mse_est']:.6g}\t"
              f"{row['interaction_to_sampling']:.2f}\t"
              f"{row['split_half_interaction_corr']:.3f}")
    print(f"Macro MSE: additive={np.mean([r['additive_mse'] for r in rows]):.6g}; "
          f"no_change={np.mean([r['no_change_mse'] for r in rows]):.6g}")
    print(f"Saved exploratory diagnostics to {args.out}; "
          f"fixed split to {manifest_path}")
    print("DIAGNOSTIC ONLY: this reads observed discovery-pair answers; it is "
          "NOT a held-out predictor. Reserved pairs are untouched by THIS "
          "script, but previous Norman experiments used combinations, so "
          "they are not an independent external test. Half-split variation "
          "is not a biological replicate or batch-adjusted noise ceiling. "
          "No STRING, Hi-C, or diffusion is used.")


if __name__ == "__main__":
    main()
