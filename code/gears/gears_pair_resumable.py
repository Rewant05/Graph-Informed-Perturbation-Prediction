r"""Train matched official GEARS with validation-selected, per-epoch checkpoints.

PowerShell, from the project root:
  python .\gears_run\gears_pair_resumable.py .\gears_data\norman .\gears_pair_splits_go\coverage_report.json .\gears_pair_baseline_go.json --fold 1 --epochs 20 --device cuda:0 --out .\gears_pair_result_go.json --checkpoint-dir .\gears_checkpoints_go

After a GPU crash, run the SAME command to resume the last completed epoch.
If CUDA remains unstable, add --score-checkpoint --device cpu and choose a
different --out to score the best completed validation checkpoint on CPU.

This uses official GEARS architecture, loss, Adam settings, learning-rate
schedule, and best validation top-20 DE selection. It omits GEARS' redundant
training-set evaluation at each epoch. Test outcomes are opened only at final
scoring or when --score-checkpoint is explicitly requested.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from gears_pair_train_and_score import (
    environment_preflight, pandas_sparse_mask_compat, score_predictions,
    validate_inputs,
)


VERSION = "2026-09-30-official-gears-epoch-checkpoints-v1"


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def signature(dataset_dir, fold, baseline_path, batch_size, epochs):
    dataset = Path(dataset_dir).resolve() / "perturb_processed.h5ad"
    dataset_stat = dataset.stat()
    return {
        "version": VERSION,
        "dataset": str(dataset), "dataset_size": dataset_stat.st_size,
        "dataset_mtime_ns": dataset_stat.st_mtime_ns,
        "fold": fold["fold"], "split_hash": sha256_file(fold["pickle"]),
        "baseline_hash": sha256_file(baseline_path), "batch_size": batch_size,
        "planned_epochs": epochs, "hidden_size": 64, "lr": 1e-3,
        "weight_decay": 5e-4, "seed": 20260929 + fold["fold"],
    }


def cpu_state_dict(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def save_checkpoint(path, payload):
    import torch
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)
    print(f"Checkpoint saved: {path} (epoch {payload['epoch']})", flush=True)


def read_checkpoint(path, expected):
    import torch
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint at {path}")
    # Only load checkpoints produced by this script in your own project.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("signature") != expected:
        raise ValueError("Checkpoint settings/data differ. Use the original "
                         "command or choose another --checkpoint-dir")
    if not 1 <= saved.get("epoch", 0) <= expected["planned_epochs"]:
        raise ValueError("Invalid checkpoint epoch")
    return saved


def restore_rng(checkpoint, torch, device):
    random.setstate(checkpoint["python_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    if device.startswith("cuda") and checkpoint.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])


def run_fold(args, fold, baseline_rows, n_genes):
    import pickle
    import torch
    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR
    from gears import GEARS, PertData
    from gears.inference import evaluate, compute_metrics
    from gears.utils import loss_fct

    expected = signature(args.dataset_dir, fold, args.baseline, args.batch_size,
                         args.epochs)
    seed = expected["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    checkpoint_path = Path(args.checkpoint_dir) / f"fold_{fold['fold']}.pt"
    checkpoint = (read_checkpoint(checkpoint_path, expected)
                  if checkpoint_path.exists() else None)
    if args.score_checkpoint and checkpoint is None:
        raise FileNotFoundError(f"No completed epoch to score at {checkpoint_path}")

    pert_data = PertData(str(Path(args.dataset_dir).parent.resolve().as_posix()))
    pert_data.load(data_path=str(Path(args.dataset_dir).resolve().as_posix()))
    if pert_data.adata.n_vars != n_genes:
        raise ValueError("Official GEARS gene axis differs from baseline JSON")
    with open(fold["pickle"], "rb") as handle:
        split = pickle.load(handle)
    missing = set(split["train"] + split["val"] + split["test"]) - \
        set(pert_data.dataset_processed)
    if missing:
        raise ValueError("GEARS dropped custom-split conditions: " +
                         ", ".join(sorted(missing)[:8]))
    pert_data.prepare_split(split="custom", seed=seed,
                            split_dict_path=str(Path(fold["pickle"]).resolve()))
    # Official custom prepare_split returns before setting this cache-key field.
    pert_data.train_gene_set_size = 0.75
    pert_data.get_dataloader(batch_size=args.batch_size,
                             test_batch_size=args.batch_size)
    test_loader = pert_data.dataloader.pop("test_loader")
    train_loader = pert_data.dataloader["train_loader"]
    val_loader = pert_data.dataloader["val_loader"]
    print(f"Fold {fold['fold']}: GEARS on {len(fold['train_pairs'])} training, "
          f"{len(fold['validation_pairs'])} validation, "
          f"{len(fold['test_pairs'])} test pairs", flush=True)

    with pandas_sparse_mask_compat():
        gears = GEARS(pert_data, device=args.device, weight_bias_track=False)
        gears.model_initialize(hidden_size=64)
        net = gears.model
        if gears.config["uncertainty"]:
            raise ValueError("This matched run expects GEARS uncertainty=False")
        optimizer = Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        best_metric = float("inf")
        best_state = None
        start_epoch = 0

        if checkpoint is not None:
            net.load_state_dict(checkpoint["latest_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            best_state = checkpoint["best_state"]
            best_metric = float(checkpoint["best_val_de_mse"])
            start_epoch = int(checkpoint["epoch"])
            restore_rng(checkpoint, torch, args.device)
            print(f"Resumed fold {fold['fold']} after epoch {start_epoch}; "
                  f"best validation top-20 DE MSE={best_metric:.6f}", flush=True)

        if not args.score_checkpoint:
            for epoch in range(start_epoch, args.epochs):
                net.train()
                for step, batch in enumerate(train_loader):
                    batch.to(args.device)
                    optimizer.zero_grad()
                    pred = net(batch)
                    loss = loss_fct(pred, batch.y, batch.pert,
                                    ctrl=gears.ctrl_expression,
                                    dict_filter=gears.dict_filter,
                                    direction_lambda=gears.config["direction_lambda"])
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)
                    optimizer.step()
                    if step % 250 == 0:
                        print(f"Epoch {epoch+1}/{args.epochs}, batch {step+1}: "
                              f"loss={loss.item():.4f}", flush=True)
                scheduler.step()
                # GEARS chooses its best model by validation top-20 DE MSE.
                # Train-set evaluation is not needed for that choice.
                val_result = evaluate(val_loader, net, False, args.device)
                val_metrics, _ = compute_metrics(val_result)
                current = float(val_metrics["mse_de"])
                if not np.isfinite(current):
                    raise ValueError("Validation DE MSE is nonfinite; no checkpoint saved")
                if current < best_metric:
                    best_metric = current
                    best_state = cpu_state_dict(net.state_dict())
                print(f"Epoch {epoch+1}: validation all-gene MSE="
                      f"{val_metrics['mse']:.6f}; top-20 DE MSE={current:.6f}; "
                      f"best={best_metric:.6f}", flush=True)
                # Save the complete continuation state. An interrupted epoch
                # restarts from the last fully validated checkpoint.
                save_checkpoint(checkpoint_path, {
                    "signature": expected, "epoch": epoch + 1,
                    "latest_state": cpu_state_dict(net.state_dict()),
                    "best_state": best_state, "best_val_de_mse": best_metric,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": (torch.cuda.get_rng_state_all()
                                 if args.device.startswith("cuda") else None),
                })
            epochs_completed = args.epochs
        else:
            epochs_completed = start_epoch
            print("Scoring the best saved validation model without further training. "
                  "This opens held-out outcomes and fixes the evaluated epoch.",
                  flush=True)

        if best_state is None:
            raise RuntimeError("No validated model available to score")
        net.load_state_dict(best_state)
        result = evaluate(test_loader, net, False, args.device)
        rows = score_predictions(result, fold, baseline_rows, pert_data.adata,
                                 split["test"])
        print(f"Fold {fold['fold']}, {epochs_completed} completed epochs: "
              f"additive={np.mean([r['additive_mse'] for r in rows]):.8f}; "
              f"nearest={np.mean([r['nearest_mse'] for r in rows]):.8f}; "
              f"GEARS={np.mean([r['gears_mse'] for r in rows]):.8f}", flush=True)
    return rows, epochs_completed, best_metric


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset_dir")
    p.add_argument("split_report")
    p.add_argument("baseline")
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument("--fold", type=int, choices=range(1, 6))
    choice.add_argument("--all-folds", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--checkpoint-dir", default="gears_checkpoints_go")
    p.add_argument("--out", default="gears_pair_result_go.json")
    p.add_argument("--score-checkpoint", action="store_true",
                   help="Score best checkpoint so far, including a partial run")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error("epochs and batch-size must be positive")
    report = json.loads(Path(args.split_report).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    folds, baseline_rows = validate_inputs(args.dataset_dir, report, baseline,
                                           args.split_report)
    selected = folds if args.all_folds else [folds[args.fold - 1]]
    environment_preflight(args.device)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    if not out.parent.is_dir():
        p.error(f"Output directory does not exist: {out.parent}")
    payload = {"version": VERSION, "data": str(Path(args.dataset_dir).resolve()),
               "split": str(Path(args.split_report).resolve()),
               "baseline": str(Path(args.baseline).resolve()),
               "device": args.device, "planned_epochs": args.epochs,
               "batch_size": args.batch_size, "partial": args.score_checkpoint,
               "folds": {}, "per_pair": []}
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        keys = ("version", "data", "split", "baseline", "device",
                "planned_epochs", "batch_size", "partial")
        if any(old.get(key) != payload[key] for key in keys):
            p.error(f"Existing {out} has different settings; choose a new --out")
        payload = old
    for fold in selected:
        if str(fold["fold"]) in payload["folds"]:
            print(f"Fold {fold['fold']} already scored in {out}; skipping",
                  flush=True)
            continue
        rows, epoch, best = run_fold(args, fold, baseline_rows, baseline["n_genes"])
        payload["folds"][str(fold["fold"])] = {
            "epochs_completed": epoch, "best_val_de_mse": best,
            "additive_mse": float(np.mean([x["additive_mse"] for x in rows])),
            "nearest_mse": float(np.mean([x["nearest_mse"] for x in rows])),
            "gears_mse": float(np.mean([x["gears_mse"] for x in rows])),
        }
        payload["per_pair"].extend(rows)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved matched pair scores to {out}", flush=True)
    print("Exploratory Norman comparison; not an independent biological test.",
          flush=True)


if __name__ == "__main__":
    main()
