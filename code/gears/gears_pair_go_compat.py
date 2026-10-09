r"""Filter matched GEARS pair folds by GEARS's own GO gene availability.

PowerShell (from repository root, after installing cell-gears):
  python .\gears_run\gears_pair_go_compat.py .\gears_data\norman .\gears_pair_splits_v3\coverage_report.json --out-dir .\gears_pair_splits_go

Reads metadata and GEARS gene-to-GO resources only, never perturbation
expression. Keeps the previous pair fold assignment; removes conditions GEARS
itself would remove during load. Recompute the simple baseline on the NEW
coverage report before comparing GEARS on the same cohort.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


REPORT_VERSION = "2026-09-30-gears-pair-fold-export-v3"


def label_genes(label):
    return tuple(part for part in str(label).split("+") if part != "ctrl")


def pair_name(label):
    genes = label_genes(label)
    if len(genes) != 2 or genes[0] == genes[1]:
        raise ValueError(f"Not a two-gene condition: {label!r}")
    return "+".join(sorted(genes))


def official_go_genes(dataset_dir):
    """Replicate GEARS.load's set_pert_genes step without cell graph creation."""
    import scanpy as sc
    from gears import PertData

    dataset = Path(dataset_dir).resolve()
    adata = sc.read_h5ad(dataset / "perturb_processed.h5ad", backed="r")
    try:
        pert_data = PertData(str(dataset.parent.as_posix()))
        pert_data.adata = adata
        pert_data.set_pert_genes()
        return set(pert_data.pert_names.astype(str))
    finally:
        adata.file.close()


def coverage(report, go_genes, official_filter=None):
    """Identify unsupported pairs across all five folds by native labels."""
    if report.get("version") != REPORT_VERSION or \
            not report.get("split_pickles_created"):
        raise ValueError("Expected v3 GEARS split coverage_report.json")
    pair_labels = {}
    for fold in report["folds"]:
        with open(fold["pickle"], "rb") as handle:
            split = pickle.load(handle)
        for section in ("train", "val", "test"):
            for label in split[section]:
                genes = label_genes(label)
                if len(genes) == 2:
                    pair_labels.setdefault(pair_name(label), set()).add(label)
    all_pairs = {p for f in report["folds"] for p in
                 f["train_pairs"] + f["validation_pairs"] + f["test_pairs"]}
    if set(pair_labels) != all_pairs:
        raise ValueError("Native pair conditions differ from the split report")
    if official_filter is None:
        official_filter = lambda label, genes: set(label_genes(label)) <= genes
    excluded = sorted(p for p in all_pairs
                      if any(not official_filter(label, go_genes)
                             for label in pair_labels[p]))
    return excluded


def write_filtered(report, excluded, out_dir):
    """Prune unavailable conditions, preserving the five original fold IDs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    omitted = set(excluded)
    updated = dict(report)
    updated["folds"] = []
    for fold in report["folds"]:
        with open(fold["pickle"], "rb") as handle:
            old_split = pickle.load(handle)
        new_fold = dict(fold)
        for section, field in (("train", "train_pairs"),
                               ("val", "validation_pairs"),
                               ("test", "test_pairs")):
            new_fold[field] = [p for p in fold[field] if p not in omitted]
        if not new_fold["validation_pairs"] or not new_fold["test_pairs"]:
            raise ValueError("Pruning removed all validation/test pairs in a fold")
        needed = {g for f in report["folds"] for p in
                  (q for field in ("train_pairs", "validation_pairs", "test_pairs")
                   for q in f[field] if q not in omitted) for g in p.split("+")}
        new_split = {}
        for section, field in (("train", "train_pairs"),
                               ("val", "validation_pairs"),
                               ("test", "test_pairs")):
            pair_set = set(new_fold[field])
            native = []
            for label in old_split[section]:
                genes = label_genes(label)
                if len(genes) == 0:
                    if section == "train":
                        native.append(label)
                elif len(genes) == 1:
                    if section == "train" and genes[0] in needed:
                        native.append(label)
                elif len(genes) == 2 and pair_name(label) in pair_set:
                    native.append(label)
            new_split[section] = native
            found_pairs = {pair_name(label) for label in native
                           if len(label_genes(label)) == 2}
            if found_pairs != pair_set:
                raise ValueError(f"Missing native labels for fold {fold['fold']}, {section}")
        path = out / f"gears_custom_fold{fold['fold']}.pkl"
        with path.open("wb") as handle:
            pickle.dump(new_split, handle, protocol=pickle.HIGHEST_PROTOCOL)
        new_fold["pickle"] = str(path.resolve())
        updated["folds"].append(new_fold)
    remaining = {p for f in updated["folds"] for p in f["test_pairs"]}
    if len(remaining) != sum(len(f["test_pairs"]) for f in updated["folds"]):
        raise ValueError("Some retained pairs lack exactly one test fold")
    updated["n_comparison_pairs"] = len(remaining)
    updated["needed_singles"] = len({g for p in remaining for g in p.split("+")})
    updated["excluded_discovery_pairs"] = sorted(set(
        report.get("excluded_discovery_pairs", [])) | omitted)
    updated["go_unsupported_pairs"] = excluded
    updated["go_filter_note"] = (
        "Pairs filtered using GEARS perturbation GO gene availability, "
        "without expression scores. Refit every baseline on this new report.")
    report_path = out / "coverage_report.json"
    report_path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
    return report_path, updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", help="Official GEARS norman directory")
    parser.add_argument("split_report", help="v3 coverage_report.json")
    parser.add_argument("--out-dir", default="gears_pair_splits_go")
    args = parser.parse_args()
    report = json.loads(Path(args.split_report).read_text(encoding="utf-8"))
    if Path(report["gears_data"]).resolve() != (Path(args.dataset_dir).resolve() /
                                                   "perturb_processed.h5ad"):
        parser.error("The report and official GEARS directory must match")
    from gears.utils import filter_pert_in_go
    go_genes = np.asarray(sorted(official_go_genes(args.dataset_dir)))
    excluded = coverage(report, go_genes, filter_pert_in_go)
    path, updated = write_filtered(report, excluded, args.out_dir)
    print(f"GEARS GO metadata supports {updated['n_comparison_pairs']} of "
          f"{report['n_comparison_pairs']} comparison pairs; excluded="
          f"{excluded or 'none'}", flush=True)
    print(f"Saved {path}. Recompute the baseline on THIS report; previous "
          "101-pair scores are not directly comparable.", flush=True)


if __name__ == "__main__":
    main()
