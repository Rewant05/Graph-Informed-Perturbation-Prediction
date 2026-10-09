r"""Train official GEARS on the v3 custom pair folds and score condition means.

PowerShell from the repository root (after installing official cell-gears):
  python .\gears_run\gears_pair_train_and_score.py .\gears_data\norman .\gears_pair_splits_v3\coverage_report.json .\gears_pair_baseline_v3.json --fold 1 --device cuda:0 --out .\gears_pair_result.json
  python .\gears_run\gears_pair_train_and_score.py .\gears_data\norman .\gears_pair_splits_v3\coverage_report.json .\gears_pair_baseline_v3.json --all-folds --device cuda:0 --out .\gears_pair_result.json

The official training code selects its checkpoint from validation data. The
evaluation compares pair-mean MSE on the same official cells and gene axis as
the saved additive and nearest-pair scores. This script does not tune on test.
"""

import argparse
from contextlib import contextmanager
import json
import pickle
import random
from pathlib import Path

import numpy as np


VERSION = "2026-09-30-official-gears-matched-pair-mean-v5"
REPORT_VERSION = "2026-09-30-gears-pair-fold-export-v3"
BASELINE_VERSION = "2026-09-30-gears-expression-simple-pair-baseline-v1"


def biological_pair(label):
    parts = str(label).upper().split("+")
    if len(parts) != 2 or any(not part or part == "CTRL" for part in parts):
        raise ValueError(f"Expected two-gene native GEARS condition: {label!r}")
    return "+".join(sorted(parts))


def validate_inputs(dataset_dir, report, baseline, report_path=None):
    data = Path(dataset_dir) / "perturb_processed.h5ad"
    if not data.is_file():
        raise FileNotFoundError(f"Missing official GEARS data: {data}")
    if report.get("version") != REPORT_VERSION or not report.get("split_pickles_created"):
        raise ValueError("Expected the v3 coverage report with five exported folds")
    if not report.get("go_filter_note"):
        raise ValueError("First run gears_pair_go_compat.py and use its GO-compatible report")
    if baseline.get("version") != BASELINE_VERSION:
        raise ValueError("Expected the v3 baseline JSON on official GEARS expression")
    if Path(report["gears_data"]).resolve() != data.resolve() or \
            Path(baseline["source"]).resolve() != data.resolve():
        raise ValueError("The report, baseline and GEARS data must refer to the same h5ad")
    if report_path is not None and \
            Path(baseline["split_report"]).resolve() != Path(report_path).resolve():
        raise ValueError("Recompute the simple baseline on THIS GO-compatible report")
    folds = report["folds"]
    if [f["fold"] for f in folds] != [1, 2, 3, 4, 5]:
        raise ValueError("Expected exactly five sequential folds")
    baseline_rows = {row["pair"]: row for row in baseline["per_pair"]}
    test_pairs = [name for fold in folds for name in fold["test_pairs"]]
    if len(test_pairs) != len(set(test_pairs)) or \
            set(test_pairs) != set(baseline_rows) or \
            len(test_pairs) != report["n_comparison_pairs"]:
        raise ValueError("GEARS test pairs do not match baseline comparison pairs")
    for fold in folds:
        path = Path(fold["pickle"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing custom split: {path}")
        with path.open("rb") as handle:
            split = pickle.load(handle)
        if set(split) != {"train", "val", "test"}:
            raise ValueError(f"Invalid custom split keys in {path}")
        if any(set(split[a]) & set(split[b]) for a, b in
               (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError(f"Native condition labels overlap in {path}")
        if {biological_pair(x) for x in split["test"]} != set(fold["test_pairs"]):
            raise ValueError(f"Test pairs do not match custom split in {path}")
        if {biological_pair(x) for x in split["val"]} != set(fold["validation_pairs"]):
            raise ValueError(f"Validation pairs do not match custom split in {path}")
        for name in fold["test_pairs"]:
            if baseline_rows[name]["fold"] != fold["fold"]:
                raise ValueError(f"Baseline fold differs for {name}")
    if baseline["n_genes"] <= 0:
        raise ValueError("Baseline gene count is invalid")
    return folds, baseline_rows


def environment_preflight(device):
    """Check the actual training imports and requested accelerator up front."""
    try:
        import torch
        import torch_geometric
        import pandas as pd
        from importlib.metadata import version
        from gears import GEARS, PertData
        from gears.inference import evaluate
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing Python module {exc.name!r} in this interpreter. "
            "GEARS requires PyTorch, PyTorch Geometric and the 'cell-gears' package. "
            "Install them with 'python -m pip' for the SAME Python used to "
            "run this script, then repeat --preflight.") from exc
    except ImportError as exc:
        raise SystemExit(
            f"GEARS dependency import failed: {exc}. Check the GEARS/PyG "
            "installation in this same Python environment.") from exc
    if int(pd.__version__.split(".")[0]) >= 3 and version("cell-gears") == "0.1.2":
        raise SystemExit(
            "Official cell-gears 0.1.2 indexes a Pandas Series with [0], "
            "which fails under Pandas 3. Run: python -m pip install 'pandas>=2.2,<3' "
            "(in PowerShell use double quotes), then repeat --preflight.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            f"Requested --device {device}, but this PyTorch cannot access "
            "CUDA. Use --device cpu or install a compatible CUDA PyTorch build.")
    if device != "cpu" and not device.startswith("cuda"):
        raise SystemExit("--device must be cpu or cuda:<GPU index>")
    print(f"Environment ready: PyTorch={torch.__version__}; "
          f"PyG={torch_geometric.__version__}; CUDA={torch.cuda.is_available()}; "
          f"device={device}", flush=True)


def score_predictions(test_result, fold, baseline_rows, adata, native_test):
    pred = np.asarray(test_result["pred"], dtype=np.float64)
    truth = np.asarray(test_result["truth"], dtype=np.float64)
    native = np.asarray(test_result["pert_cat"], dtype=str)
    if pred.shape != truth.shape or pred.ndim != 2 or \
            len(native) != pred.shape[0] or pred.shape[1] != adata.n_vars or \
            not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("GEARS evaluate output has wrong shape or nonfinite values")
    if set(native) != set(native_test):
        raise ValueError("GEARS test loader labels differ from the custom split")
    dataset_labels = adata.obs["condition"].astype(str).to_numpy()
    actual_pair = np.asarray([biological_pair(x) for x in native])
    rows = []
    for name in fold["test_pairs"]:
        mask = actual_pair == name
        if not mask.any():
            raise ValueError(f"No test cells for {name}")
        pair_aliases = [x for x in native_test if biological_pair(x) == name]
        h5_mask = np.isin(dataset_labels, pair_aliases)
        if int(mask.sum()) != int(h5_mask.sum()):
            raise ValueError(f"GEARS dropped test cells for {name}; comparison invalid")
        reference = np.asarray(adata.X[h5_mask].mean(axis=0)).ravel()
        observed = truth[mask].mean(axis=0)
        if not np.allclose(observed, reference, atol=2e-5, rtol=2e-5):
            raise ValueError(f"GEARS test truth differs from official h5ad for {name}")
        prediction = pred[mask].mean(axis=0)
        mse = float(np.mean((prediction - reference) ** 2))
        row = baseline_rows[name]
        rows.append({"pair": name, "fold": fold["fold"],
                     "test_cells": int(mask.sum()),
                     "additive_mse": float(row["additive_mse"]),
                     "nearest_mse": float(row["nearest_mse"]),
                     "gears_mse": mse})
    if sum(row["test_cells"] for row in rows) != len(native):
        raise ValueError("Some GEARS test predictions were not scored")
    return rows


@contextmanager
def pandas_sparse_mask_compat():
    """Bridge GEARS' Series mask through SciPy's index validator.

    AnnData may return a different sparse object on each ``adata.X`` access.
    Adapt the *mask class* briefly rather than a particular CSR instance.
    Only boolean Series without missing values are accepted.
    """
    import pandas as pd
    from scipy import sparse

    if hasattr(pd.Series, "nonzero"):
        raise RuntimeError("Pandas Series.nonzero already exists; cannot safely patch it")

    def boolean_series_nonzero(series):
        if not pd.api.types.is_bool_dtype(series.dtype) or series.isna().any():
            raise TypeError("SciPy sparse indexing requires a complete boolean mask")
        return series.to_numpy(dtype=bool).nonzero()

    pd.Series.nonzero = boolean_series_nonzero
    try:
        # Exact failure path from GEARS 0.1.2, checked before expensive model
        # construction. This does not touch the experiment's expression data.
        example = sparse.csr_matrix([[1], [2]])
        if example[pd.Series([True, False])].toarray().tolist() != [[1]]:
            raise RuntimeError("SciPy sparse/Pandas boolean-mask probe failed")
        print("GEARS sparse-mask compatibility check passed", flush=True)
        yield
    finally:
        del pd.Series.nonzero


def train_fold(dataset_dir, fold, baseline_rows, n_genes, device, epochs, batch_size):
    import torch
    from gears import GEARS, PertData
    from gears.inference import evaluate

    seed = 20260929 + fold["fold"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
        torch.cuda.manual_seed_all(seed)

    # GEARS derives its cache filename from (dataset, split, seed, train size).
    # Different fold seeds keep training-only coexpression graphs distinct.
    pert_data = PertData(str(Path(dataset_dir).parent.resolve().as_posix()))
    pert_data.load(data_path=str(Path(dataset_dir).resolve().as_posix()))
    if pert_data.adata.n_vars != n_genes:
        raise ValueError("Official GEARS gene axis differs from baseline JSON")
    with Path(fold["pickle"]).open("rb") as handle:
        split = pickle.load(handle)
    missing = set(split["train"] + split["val"] + split["test"]) - \
        set(pert_data.dataset_processed)
    if missing:
        raise ValueError("Official GEARS filtered native conditions, including: " +
                         ", ".join(sorted(missing)[:8]))
    pert_data.prepare_split(split="custom", seed=seed,
                            split_dict_path=str(Path(fold["pickle"]).resolve()))
    # In official GEARS custom prepare_split returns before setting this
    # attribute. It is only used as a graph-cache key for custom splits.
    pert_data.train_gene_set_size = 0.75
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=batch_size)

    # Do not call GEARS.train's internal test and deeper_analysis. The saved
    # validation-selected best_model is scored once below on the heldout cells.
    test_loader = pert_data.dataloader.pop("test_loader")
    print(f"Fold {fold['fold']}: training official GEARS; "
          f"train pairs={len(fold['train_pairs'])}, "
          f"val pairs={len(fold['validation_pairs'])}, "
          f"test pairs={len(fold['test_pairs'])}; seed={seed}", flush=True)
    with pandas_sparse_mask_compat():
        model = GEARS(pert_data, device=device, weight_bias_track=False)
        model.model_initialize(hidden_size=64)
        model.train(epochs=epochs)
        result = evaluate(test_loader, model.best_model, False, device)
        rows = score_predictions(result, fold, baseline_rows, pert_data.adata,
                                 split["test"])
    print(f"Fold {fold['fold']} pair-mean MSE: additive="
          f"{np.mean([x['additive_mse'] for x in rows]):.8f}; nearest="
          f"{np.mean([x['nearest_mse'] for x in rows]):.8f}; GEARS="
          f"{np.mean([x['gears_mse'] for x in rows]):.8f}", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset_dir", help="Official GEARS norman/ directory")
    p.add_argument("split_report", help="v3 coverage_report.json")
    p.add_argument("baseline", help="gears_pair_baseline_v3.json")
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument("--fold", type=int, choices=range(1, 6))
    choice.add_argument("--all-folds", action="store_true")
    p.add_argument("--device", default="cpu", help="cpu or cuda:0")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default="gears_pair_result.json")
    p.add_argument("--preflight", action="store_true",
                   help="Validate files and splits without importing GEARS")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error("epochs and batch-size must be positive")
    report = json.loads(Path(args.split_report).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    folds, baseline_rows = validate_inputs(args.dataset_dir, report, baseline,
                                           args.split_report)
    selected = folds if args.all_folds else [folds[args.fold - 1]]
    print(f"Validated matched GEARS folds. Selected {[f['fold'] for f in selected]}; "
          f"gene count={baseline['n_genes']}. Checking model environment...", flush=True)
    environment_preflight(args.device)
    if args.preflight:
        print("Preflight passed: files, official GEARS import, and device are ready.",
              flush=True)
        return
    out = Path(args.out)
    if not out.parent.is_dir():
        p.error(f"Output directory does not exist: {out.parent}")
    result = {"version": VERSION, "dataset": str(Path(args.dataset_dir).resolve()),
              "split_report": str(Path(args.split_report).resolve()),
              "baseline": str(Path(args.baseline).resolve()),
              "device": args.device, "epochs": args.epochs,
              "batch_size": args.batch_size, "folds": {}, "per_pair": []}
    # Save after each fold so a later run can resume; a full rerun may be
    # requested explicitly by deleting/renaming the output JSON.
    if out.is_file():
        old = json.loads(out.read_text(encoding="utf-8"))
        if any(old.get(k) != result[k] for k in
               ("version", "dataset", "split_report", "baseline", "device",
                "epochs", "batch_size")):
            p.error(f"Existing {out} has different settings; select a new --out")
        result = old
    for fold in selected:
        key = str(fold["fold"])
        if key in result["folds"]:
            print(f"Fold {key} already saved; skipping", flush=True)
            continue
        rows = train_fold(args.dataset_dir, fold, baseline_rows, baseline["n_genes"],
                          args.device, args.epochs, args.batch_size)
        result["folds"][key] = {"n_test_pairs": len(rows),
                                "additive_mse": float(np.mean([r["additive_mse"] for r in rows])),
                                "nearest_mse": float(np.mean([r["nearest_mse"] for r in rows])),
                                "gears_mse": float(np.mean([r["gears_mse"] for r in rows]))}
        result["per_pair"].extend(rows)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved {out}", flush=True)
    if len(result["folds"]) == 5:
        rows = result["per_pair"]
        if len(rows) != report["n_comparison_pairs"] or \
                len({r["pair"] for r in rows}) != len(rows):
            raise ValueError("Final result contains missing or duplicate test pairs")
        print(f"All {len(rows)} pair-mean MSE: additive="
              f"{np.mean([r['additive_mse'] for r in rows]):.8f}; nearest="
              f"{np.mean([r['nearest_mse'] for r in rows]):.8f}; GEARS="
              f"{np.mean([r['gears_mse'] for r in rows]):.8f}", flush=True)
        print("Descriptive comparison on previously inspected Norman pairs; "
              "not an independent biological replication.")


if __name__ == "__main__":
    main()
