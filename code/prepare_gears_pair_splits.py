r"""Audit label compatibility and export K562 pair folds for official GEARS.

PowerShell after obtaining GEARS's official Norman processed h5ad:
  python .\gears_setup\prepare_gears_pair_splits.py .\norman_log1p_10k_v2.h5ad .\k562_pair_signal.split.json .\gears_data\norman\perturb_processed.h5ad --out-dir .\gears_pair_splits

Metadata only: never reads expression from either dataset. Refuses to write
split pickle files unless all 103 discovery pairs and all measured singles
needed by the current predictor exist in the official GEARS Norman dataset.
This is preparation, NOT a GEARS result or a directly comparable score.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from k562_pair_interaction_cv import load_manifest
from k562_pair_signal_audit import collect_groups


VERSION = "2026-09-30-gears-pair-fold-export-v3"
SEED = 20260929


def gears_conditions(values):
    """Map all native condition orientations to one biological condition.

    GEARS may contain both FEV+ctrl and ctrl+FEV. Both labels must travel
    together into train, validation, or test; arbitrarily choosing one would
    discard cells, and splitting them would leak a measured response.
    """
    mapped = {}
    for label in values:
        parts = str(label).strip().upper().split("+")
        if parts == ["CTRL"]:
            key = ("control",)
        elif len(parts) == 2 and parts.count("CTRL") == 1:
            key = ("single", next(g for g in parts if g != "CTRL"))
        elif len(parts) == 2 and "CTRL" not in parts and parts[0] != parts[1]:
            key = ("pair", *sorted(parts))
        else:
            continue
        mapped.setdefault(key, [])
        if label not in mapped[key]:
            mapped[key].append(label)
    return mapped


def flatten(groups):
    return [condition for group in groups for condition in group]


def export(our_data, manifest_path, gears_data, out_dir,
           allow_intersection=False):
    import scanpy as sc

    our = sc.read_h5ad(our_data, backed="r")
    official = sc.read_h5ad(gears_data, backed="r")
    try:
        if (our.X is None or "perturbation" not in our.obs or
                our.uns.get("normalization_protocol") !=
                "2026-09-29-norman-cp10k-log1p-v2"):
            raise ValueError("First file must be the verified normalized Norman file")
        if "condition" not in official.obs or "gene_name" not in official.var:
            raise ValueError("Third file must have GEARS obs['condition'] "
                             "and var['gene_name']")
        genes = (our.var["gene_symbol"].astype(str).tolist()
                 if "gene_symbol" in our.var else our.var_names.astype(str).tolist())
        labels = our.obs["perturbation"].astype(str).to_numpy()
        _ctrl, singles, pairs, _ = collect_groups(labels, genes, 50)
        discovery, reserved = load_manifest(manifest_path, pairs)
        original_singles = sorted({g for pair in discovery + reserved for g in pair})
        native = gears_conditions(official.obs["condition"].astype(str).unique())
        missing_pairs = ["+".join(p) for p in discovery
                         if ("pair", *p) not in native]
        missing_singles = [g for g in original_singles
                           if ("single", g) not in native]
        compatible = [p for p in discovery if ("pair", *p) in native
                      and all(("single", gene) in native for gene in p)]
        excluded = ["+".join(p) for p in discovery if p not in compatible]
        selected = compatible if allow_intersection else discovery
        needed_singles = sorted({g for pair in selected for g in pair})
        our_genes = set(g.upper() for g in genes)
        official_genes = set(g.upper() for g in
                             official.var["gene_name"].astype(str))
        report = {
            "version": VERSION, "our_data": str(Path(our_data).resolve()),
            "gears_data": str(Path(gears_data).resolve()),
            "n_discovery_pairs": len(discovery),
            "n_comparison_pairs": len(selected),
            "intersection_requested": bool(allow_intersection),
            "excluded_discovery_pairs": excluded,
            "n_previously_opened_reserved_pairs_excluded": len(reserved),
            "needed_singles": len(needed_singles),
            "original_needed_singles": len(original_singles),
            "missing_discovery_pairs": missing_pairs,
            "missing_measured_singles": missing_singles,
            "our_genes": len(our_genes), "gears_genes": len(official_genes),
            "common_gene_names": len(our_genes & official_genes),
            "official_contains_control": ("control",) in native,
            "multiple_native_labels": {
                "+".join(key): labels for key, labels in native.items()
                if len(labels) > 1 and (key == ("control",) or
                   (key[0] == "single" and key[1] in needed_singles) or
                   (key[0] == "pair" and key[1:] in discovery))},
            "split_pickles_created": False,
        }
    finally:
        our.file.close()
        official.file.close()

    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    if ((missing_pairs or missing_singles) and not allow_intersection or
            ("control",) not in native or len(selected) < 10):
        (folder / "coverage_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(f"GEARS label coverage: {len(discovery)-len(missing_pairs)}/{len(discovery)} "
              f"discovery pairs; {len(original_singles)-len(missing_singles)}/"
              f"{len(original_singles)} measured singles; "
              f"{report['common_gene_names']} common genes")
        print(f"Missing examples: pairs={missing_pairs[:10]}; "
              f"singles={missing_singles[:10]}")
        print("No split pickles created: requested conditions are absent, "
              "control is missing, or fewer than 10 pairs remain. "
              f"Coverage report: {folder / 'coverage_report.json'}")
        return report

    if excluded:
        print(f"EXPLICIT INTERSECTION: removed {len(excluded)} discovery pairs "
              f"with no official GEARS pair and/or measured singles: "
              f"{', '.join(excluded)}. This defines a new {len(selected)}-pair "
              "comparison; prior 103-pair scores cannot be copied here.")

    single_names = flatten(native[("single", g)] for g in needed_singles)
    discovery_native = [native[("pair", *p)] for p in selected]
    all_used = native[("control",)] + single_names + flatten(discovery_native)
    if len(set(all_used)) != len(all_used):
        raise ValueError("Condition labels collided during official mapping")
    folds = []
    for fold, (trainval, test) in enumerate(
            KFold(n_splits=5, shuffle=True, random_state=SEED).split(selected), 1):
        rng = np.random.default_rng(SEED + fold)
        val_count = max(1, round(len(trainval) * 0.1))
        val = set(rng.choice(trainval, size=val_count, replace=False))
        train = [int(i) for i in trainval if i not in val]
        val = sorted(int(i) for i in val)
        test = [int(i) for i in test]
        conditions = {
            "train": native[("control",)] + single_names +
                     flatten(discovery_native[i] for i in train),
            "val": flatten(discovery_native[i] for i in val),
            "test": flatten(discovery_native[i] for i in test),
        }
        sets = [set(values) for values in conditions.values()]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise AssertionError("GEARS fold has overlapping condition labels")
        path = folder / f"gears_custom_fold{fold}.pkl"
        with open(path, "wb") as handle:
            pickle.dump(conditions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        folds.append({"fold": fold, "pickle": str(path.resolve()),
                      "train_pairs": ["+".join(selected[i]) for i in train],
                      "validation_pairs": ["+".join(selected[i]) for i in val],
                      "test_pairs": ["+".join(selected[i]) for i in test]})
    report.update({"split_pickles_created": True, "seed": SEED,
                   "folds": folds,
                   "warning": "Use the identical GEARS gene axis, normalization, "
                              "train/val/test pair lists and metric when "
                              "rerunning the simple predictor; earlier CV "
                              "numbers are not directly comparable."})
    (folder / "coverage_report.json").write_text(json.dumps(report, indent=2),
                                                   encoding="utf-8")
    print(f"GEARS label coverage complete: {len(selected)} comparison pairs; "
          f"{len(needed_singles)} singles; {report['common_gene_names']} "
          f"common genes; {len(report['multiple_native_labels'])} conditions "
          f"with multiple native labels; wrote 5 custom split pickles to {folder}")
    print("METADATA ONLY: GEARS has not been trained or scored. The 26 "
          "previously opened reserve outcomes were not read.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("our_data", help="Verified normalized Norman .h5ad")
    parser.add_argument("split", help="Original pair split JSON")
    parser.add_argument("gears_data", help="Official GEARS Norman perturb_processed.h5ad")
    parser.add_argument("--out-dir", default="gears_pair_splits")
    parser.add_argument("--allow-intersection", action="store_true",
                        help="Explicitly exclude missing discovery pairs/genes")
    args = parser.parse_args()
    for path in (args.our_data, args.split, args.gears_data):
        if not Path(path).is_file():
            parser.error(f"File not found: {path}")
    export(args.our_data, args.split, args.gears_data, args.out_dir,
           args.allow_intersection)


if __name__ == "__main__":
    main()
