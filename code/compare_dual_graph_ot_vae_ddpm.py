"""Exploratory unseen-SINGLE-target K562 CRISPRa graph/VAE/OT/DDPM ablation.

All variants use one training-only SVD and one training-only VAE. STRING and
strict promoter--promoter Hi-C are separate nonnegative edge relations. A
conditional transport head is trained with entropic Sinkhorn divergence on
UNPAIRED control/seen-target cell samples. A DDPM then models the encoded
perturbed-cell distribution, conditioned on graph target and transport output.
No held-out perturbed rows are read before predictions for all variants exist.

This is a research prototype. It does not reuse the pair predictor because
that predictor requires measured single responses for the query genes, while
this protocol deliberately hides those responses.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import gene_graph_response_2o as graph
import heldout_conditional_diffusion_2o as old_ddpm
from count_balanced_graph_check_2o import valid_targets

VERSION = "2026-10-07-dual-graph-ot-vae-ddpm-prototype-v1"
DEFAULT_VARIANTS = ("no_edges:ot", "shuffled:ot", "string:ot", "hic:ot",
                    "dual:ot", "dual:no_ot")


def load_graphs(string_path, hic_path, genes, split, train_targets, heldout,
                max_nodes=512, seed=42):
    """Name-aligned, bounded graph; no expression outcomes enter selection."""
    with open(string_path, encoding="utf-8") as f:
        string = json.load(f)
    with open(hic_path, encoding="utf-8") as f:
        hic = json.load(f)
    digest = hashlib.sha256("\n".join(genes).encode()).hexdigest()
    if (string.get("gene_sha256") != digest or string.get("split") != split or
            string.get("train_targets") != train_targets or
            string.get("heldout") != list(heldout)):
        raise ValueError("STRING manifest differs from data or split; rebuild it")
    if (hic.get("assembly_asserted") != "GRCh38" or
            hic.get("measured_genes") != len(genes) or
            hic.get("string_node_count") != len(string["nodes"])):
        raise ValueError("Hi-C mapping metadata differs from the STRING/data input")
    targets = train_targets + list(heldout)
    if len(set(targets)) != len(targets) or max_nodes < len(targets):
        raise ValueError("Duplicate targets or --max-nodes smaller than target count")
    measured = set(genes)
    s_edges = string["edges"]
    h_edges = hic["strict_promoter_promoter"]["edges"]
    if any(a not in measured or b not in measured for a, b, _ in s_edges + h_edges):
        raise ValueError("Graph references a gene absent from expression data")
    target_set = set(targets)
    # Rank one-hop neighbors by topology alone, with deterministic ties.
    rank = {}
    for edges in (s_edges, h_edges):
        for a, b, weight in edges:
            if a in target_set and b not in target_set:
                rank[b] = rank.get(b, 0.0) + 1.0
            if b in target_set and a not in target_set:
                rank[a] = rank.get(a, 0.0) + 1.0
    extra = sorted(rank, key=lambda name: (-rank[name], name))[:max_nodes-len(targets)]
    nodes = targets + extra
    index = {g: i for i, g in enumerate(nodes)}

    def make(edges, is_hic):
        raw = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
        for a, b, weight in edges:
            if a not in index or b not in index or a == b:
                continue
            value = math.log1p(float(weight)) if is_hic else float(weight)
            if value < 0 or not math.isfinite(value):
                raise ValueError("Graph has an invalid edge weight")
            i, j = index[a], index[b]
            raw[i, j] = raw[j, i] = max(raw[i, j], value)
        return raw

    string_raw, hic_raw = make(s_edges, False), make(h_edges, True)
    rng = np.random.default_rng(seed + 77)
    # Preserve target/context partitions and edge-count distribution; break
    # biological identity, leaving the control-derived node features unchanged.
    perm = np.arange(len(nodes))
    perm[:len(targets)] = rng.permutation(len(targets))
    perm[len(targets):] = len(targets) + rng.permutation(len(extra))
    shuffled = (string_raw[np.ix_(perm, perm)], hic_raw[np.ix_(perm, perm)])
    zero = np.zeros_like(string_raw)

    def row_normalize(raw):
        return raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-8)

    modes = {
        "no_edges": (zero, zero), "shuffled": shuffled,
        "string": (string_raw, zero), "hic": (zero, hic_raw),
        "dual": (string_raw, hic_raw),
    }
    modes = {k: tuple(row_normalize(a) for a in pair)
             for k, pair in modes.items()}
    coverage = {k: {"string_targets": int(np.count_nonzero(a[:len(targets)].sum(axis=1))),
                    "hic_targets": int(np.count_nonzero(b[:len(targets)].sum(axis=1))),
                    "string_edges": int(np.count_nonzero(a) // 2),
                    "hic_edges": int(np.count_nonzero(b) // 2)}
                for k, (a, b) in modes.items()}
    return nodes, modes, coverage


class LatentVAE(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, 128), nn.SiLU(),
                                     nn.Linear(128, 128), nn.SiLU())
        self.mu = nn.Linear(128, z_dim)
        self.logvar = nn.Linear(128, z_dim)
        self.decoder = nn.Sequential(nn.Linear(z_dim, 128), nn.SiLU(),
                                     nn.Linear(128, 128), nn.SiLU(),
                                     nn.Linear(128, in_dim))

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(-8, 8)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(z), mu, logvar


def train_vae(values, controls, in_dim, z_dim, epochs, batch_size, device, seed):
    torch.manual_seed(seed)
    model = LatentVAE(in_dim, z_dim).to(device)
    x = torch.as_tensor(np.concatenate((values, controls)), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        errors = []
        for ix in np.array_split(rng.permutation(len(x)), math.ceil(len(x)/batch_size)):
            pred, mu, logvar = model(x[ix])
            reconstruction = F.mse_loss(pred, x[ix])
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
            loss = reconstruction + 0.02 * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            errors.append(reconstruction.item())
        print(f"VAE epoch {epoch+1}/{epochs}: recon MSE={np.mean(errors):.5f}", flush=True)
    model.eval()
    with torch.no_grad():
        target_z = model.encode(torch.as_tensor(values, device=device))[0].detach()
        ctrl_z = model.encode(torch.as_tensor(controls, device=device))[0].detach()
    return model, target_z, ctrl_z


def sinkhorn_cost(x, y, epsilon=0.2, iterations=20):
    """Differentiable entropic OT cost for equal-weight, unpaired minibatches."""
    cost = torch.cdist(x, y).square() / x.shape[1]
    log_kernel = -cost / epsilon
    log_a = x.new_full((len(x),), -math.log(len(x)))
    log_b = x.new_full((len(y),), -math.log(len(y)))
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(iterations):
        u = log_a - torch.logsumexp(log_kernel + v[None, :], dim=1)
        v = log_b - torch.logsumexp(log_kernel + u[:, None], dim=0)
    coupling = torch.exp(log_kernel + u[:, None] + v[None, :])
    return torch.sum(coupling * cost)


def sinkhorn_divergence(x, y):
    return sinkhorn_cost(x, y) - 0.5 * sinkhorn_cost(x, x) - 0.5 * sinkhorn_cost(y, y)


class TransportHead(nn.Module):
    def __init__(self, feature_dim, z_dim):
        super().__init__()
        # The inherited `pos`/`neg` slots represent STRING/Hi-C relations,
        # respectively; neither adjacency contains signed negative edges.
        self.graph = old_ddpm.InterventionConditioner(feature_dim)
        self.delta = nn.Sequential(nn.Linear(z_dim + 48, 128), nn.SiLU(),
                                   nn.Linear(128, z_dim))

    def forward(self, ctrl_z, target, features, string_adj, hic_adj):
        context = self.graph(features, string_adj, hic_adj, target)
        delta = self.delta(torch.cat((ctrl_z, context), dim=1))
        return ctrl_z + delta


class GraphDDPM(nn.Module):
    def __init__(self, feature_dim, z_dim, steps):
        super().__init__()
        self.graph = old_ddpm.InterventionConditioner(feature_dim)
        self.time = nn.Embedding(steps, 32)
        self.denoiser = nn.Sequential(nn.LayerNorm(z_dim*2 + 48 + 32),
                                     nn.Linear(z_dim*2 + 48 + 32, 192), nn.SiLU(),
                                     nn.Linear(192, 192), nn.SiLU(),
                                     nn.Linear(192, z_dim))

    def forward(self, noisy, step, target, anchor, features, string_adj, hic_adj):
        condition = self.graph(features, string_adj, hic_adj, target)
        return self.denoiser(torch.cat((noisy, anchor, condition, self.time(step)), 1))


def train_variant(mode, use_ot, features, s_adj, h_adj, target_z, ctrl_z,
                  labels, n_targets, cfg, device):
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    donor = [np.flatnonzero(labels == j) for j in range(n_targets)]
    if min(map(len, donor)) == 0:
        raise ValueError("A training target has no sampled cells")
    transport = TransportHead(features.shape[1], cfg.vae_dim).to(device)
    if use_ot:
        opt = torch.optim.AdamW(transport.parameters(), lr=7e-4)
        transport.train()
        for step in range(cfg.ot_steps):
            j = step % n_targets
            subset = rng.choice(donor[j], size=cfg.ot_batch, replace=True)
            ctrl_ix = rng.integers(len(ctrl_z), size=cfg.ot_batch)
            ctrl = ctrl_z[torch.as_tensor(ctrl_ix, dtype=torch.long, device=device)]
            goal = target_z[torch.as_tensor(subset, dtype=torch.long, device=device)]
            targets = torch.full((len(ctrl),), j, dtype=torch.long, device=device)
            moved = transport(ctrl, targets, features, s_adj, h_adj)
            loss = (sinkhorn_divergence(moved, goal) +
                    cfg.transport_cost * (moved - ctrl).square().mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(transport.parameters(), 1.0)
            opt.step()
        print(f"{mode}: OT head last minibatch loss={loss.item():.5f}", flush=True)
    transport.eval()
    for p in transport.parameters():
        p.requires_grad_(False)

    beta = torch.linspace(1e-4, min(0.4, 12/cfg.steps), cfg.steps, device=device)
    alpha_bar = torch.cumprod(1-beta, dim=0)
    model = GraphDDPM(features.shape[1], cfg.vae_dim, cfg.steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    label_t = torch.as_tensor(labels, dtype=torch.long, device=device)
    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for ix in np.array_split(rng.permutation(len(labels)),
                                 math.ceil(len(labels)/cfg.batch_size)):
            ix_t = torch.as_tensor(ix, dtype=torch.long, device=device)
            target, truth = label_t[ix_t], target_z[ix_t]
            ctrl_ix = torch.as_tensor(rng.integers(len(ctrl_z), size=len(ix)),
                                      dtype=torch.long, device=device)
            ctrl = ctrl_z[ctrl_ix]
            with torch.no_grad():
                anchor = (transport(ctrl, target, features, s_adj, h_adj)
                          if use_ot else ctrl)
            t = torch.randint(cfg.steps, (len(ix),), device=device)
            noise = torch.randn_like(truth)
            a = alpha_bar[t, None]
            noisy = a.sqrt()*truth + (1-a).sqrt()*noise
            pred = model(noisy, t, target, anchor, features, s_adj, h_adj)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        print(f"{mode} DDPM epoch {epoch+1}/{cfg.epochs}: noise MSE={np.mean(losses):.5f}",
              flush=True)
    return model, transport, alpha_bar


@torch.no_grad()
def generate(model, transport, use_ot, features, s_adj, h_adj, ctrl_z,
             alpha_bar, nodes, n_samples, seed, device):
    model.eval()
    out = []
    rng = np.random.default_rng(seed)
    # Same control/noise draws for each variant; interventions differ.
    generator = torch.Generator(device=device).manual_seed(seed)
    initial = torch.randn((n_samples, ctrl_z.shape[1]), device=device, generator=generator)
    ctrl_indices = torch.as_tensor(rng.integers(len(ctrl_z), size=n_samples),
                                   dtype=torch.long, device=device)
    for node in nodes:
        ctrl = ctrl_z[ctrl_indices]
        targets = torch.full((n_samples,), int(node), dtype=torch.long, device=device)
        anchor = (transport(ctrl, targets, features, s_adj, h_adj) if use_ot else ctrl)
        latent = initial.clone()
        for t in range(len(alpha_bar)-1, -1, -1):
            steps = torch.full((n_samples,), t, dtype=torch.long, device=device)
            eps = model(latent, steps, targets, anchor, features, s_adj, h_adj)
            a = alpha_bar[t]
            clean = ((latent - (1-a).sqrt()*eps) / a.sqrt()).clamp(-8, 8)
            if t:
                prev = alpha_bar[t-1]
                latent = prev.sqrt()*clean + (1-prev).sqrt()*eps
            else:
                latent = clean
        out.append(latent.cpu().numpy())
    return np.stack(out)


def preflight(data, string_path, hic_path, split, max_nodes, seed):
    import scanpy as sc
    adata = sc.read_h5ad(data, backed="r")
    try:
        if adata.uns.get("normalization_protocol") != "2026-09-29-norman-cp10k-log1p-v2":
            raise ValueError("Expected verified CP10k log1p Norman file")
        genes = (adata.var["gene_symbol"].astype(str).tolist()
                 if "gene_symbol" in adata.var else adata.var_names.astype(str).tolist())
        if len(set(genes)) != len(genes):
            raise ValueError("Duplicate measured gene names")
        # pandas often returns an object array even after astype(str); NumPy's
        # string ufuncs (used for control matching below) require string dtype.
        conditions = np.asarray(adata.obs["perturbation"].astype(str).to_numpy(),
                                dtype=str)
        heldout, train_mask, test_mask = old_ddpm.get_split(conditions, genes, split)
        counts = valid_targets(conditions, genes)
        train_targets = sorted(n for n in counts if np.any(train_mask & (conditions == n)))
        if np.any(train_mask & test_mask) or len(train_targets) < 10:
            raise ValueError("Invalid unseen-target split")
        nodes, modes, coverage = load_graphs(string_path, hic_path, genes, split,
                                             train_targets, heldout, max_nodes, seed)
        return genes, conditions, heldout, train_mask, test_mask, train_targets, nodes, modes, coverage
    finally:
        adata.file.close()


def self_test():
    """Exercise VAE, differentiable OT, DDPM, and unseen-node sampling."""
    object_labels = np.array(["control", "FEV"], dtype=object)
    string_labels = np.asarray(object_labels, dtype=str)
    assert np.char.lower(string_labels)[0] == "control"
    device = torch.device("cpu")
    rng = np.random.default_rng(7)
    values = rng.normal(size=(24, 8)).astype(np.float32)
    controls = rng.normal(size=(12, 8)).astype(np.float32)
    vae, target_z, ctrl_z = train_vae(values, controls, 8, 4, 1, 8, device, 7)
    features = torch.as_tensor(rng.normal(size=(4, 5)).astype(np.float32))
    raw = np.zeros((4, 4), dtype=np.float32)
    raw[0, 1] = raw[1, 0] = raw[1, 3] = raw[3, 1] = 1
    string_adj = torch.as_tensor(raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-8))
    hic_adj = torch.zeros_like(string_adj)
    cfg = argparse.Namespace(seed=7, vae_dim=4, ot_steps=6, ot_batch=8,
                             transport_cost=0.01, steps=24, epochs=1, batch_size=8)
    model, transport, alpha = train_variant(
        "self_test", True, features, string_adj, hic_adj, target_z, ctrl_z,
        np.repeat(np.arange(3), 8), 3, cfg, device)
    generated = generate(model, transport, True, features, string_adj, hic_adj,
                         ctrl_z, alpha, [3], 4, 9, device)
    with torch.no_grad():
        decoded = vae.decoder(torch.as_tensor(generated[0]))
    if generated.shape != (1, 4, 4) or decoded.shape != (4, 8) or \
            not np.isfinite(generated).all():
        raise AssertionError("Generated latent samples have invalid shape or values")
    print("SELF-TEST PASSED: VAE, Sinkhorn OT, graph DDPM, unseen-node sampling")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", nargs="?", help="Verified Norman CP10k log1p .h5ad")
    p.add_argument("string", nargs="?", help="Aligned STRING manifest for balanced1")
    p.add_argument("hic", nargs="?", help="K562 Hi-C strict-promoter mapping v2")
    p.add_argument("--split", default="balanced1", choices=("balanced1",))
    p.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    p.add_argument("--max-nodes", type=int, default=512)
    p.add_argument("--vae-dim", type=int, default=24)
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--vae-epochs", type=int, default=8)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--ot-steps", type=int, default=160)
    p.add_argument("--ot-batch", type=int, default=32)
    p.add_argument("--transport-cost", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-per-target", type=int, default=256)
    p.add_argument("--max-controls", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--preflight", action="store_true", help="Metadata and graph checks only")
    p.add_argument("--self-test", action="store_true", help="Small synthetic CPU training test")
    p.add_argument("--out", default="dual_graph_ot_vae_ddpm_result.json")
    cfg = p.parse_args()
    if cfg.self_test:
        self_test()
        return
    if not all((cfg.data, cfg.string, cfg.hic)):
        p.error("data, string and hic paths are required except with --self-test")
    if (min(cfg.max_nodes, cfg.vae_dim, cfg.svd_dim, cfg.vae_epochs, cfg.epochs,
            cfg.ot_steps, cfg.ot_batch, cfg.steps, cfg.samples, cfg.batch_size,
            cfg.max_per_target, cfg.max_controls) <= 0 or cfg.steps < 24 or
            cfg.vae_dim >= cfg.svd_dim or cfg.transport_cost < 0):
        p.error("Invalid training dimensions or counts; steps >= 24, VAE dim < SVD dim")
    variants = cfg.variants.split(",")
    if not variants or len(set(variants)) != len(variants) or any(
            v not in [f"{mode}:{kind}" for mode in ("no_edges", "shuffled", "string", "hic", "dual")
                      for kind in ("ot", "no_ot")] for v in variants):
        p.error("--variants must be unique mode:ot or mode:no_ot entries")
    genes, conditions, heldout, train_mask, test_mask, train_targets, nodes, modes, coverage = (
        preflight(cfg.data, cfg.string, cfg.hic, cfg.split, cfg.max_nodes, cfg.seed))
    print(f"Version={VERSION}; targets train={len(train_targets)}, heldout={len(heldout)}; "
          f"graph nodes={len(nodes)}", flush=True)
    print("Graph coverage:", json.dumps(coverage, sort_keys=True), flush=True)
    if cfg.preflight:
        print("PREFLIGHT ONLY: zero expression rows read; no model trained or scored.")
        return
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA unavailable; use --device cpu")
    device = torch.device(cfg.device)
    import scanpy as sc
    adata = sc.read_h5ad(cfg.data)
    matrix = adata.X.tocsr() if hasattr(adata.X, "tocsr") else adata.X
    ctrl_indices = np.flatnonzero(train_mask & np.isin(
        np.char.lower(conditions), list(graph.CONTROLS)))
    if len(ctrl_indices) < 2:
        raise ValueError("No eligible training controls")
    rng = np.random.default_rng(cfg.seed)
    chosen_ctrl = (rng.choice(ctrl_indices, cfg.max_controls, replace=False)
                   if len(ctrl_indices) > cfg.max_controls else ctrl_indices)
    features_np, _, _ = graph.graph_from_controls(matrix[chosen_ctrl], genes, nodes)
    svd, ctrl_mean, spread, values, labels = old_ddpm.make_training_data(
        matrix, conditions, train_mask, train_targets, ctrl_indices,
        cfg.max_per_target, cfg.svd_dim, cfg.seed)
    ctrl_latent = svd.transform(ctrl_mean[None, :])[0]
    controls = ((svd.transform(matrix[chosen_ctrl]) - ctrl_latent) / spread).astype(np.float32)
    vae, target_z, ctrl_z = train_vae(values, controls, cfg.svd_dim, cfg.vae_dim,
                                      cfg.vae_epochs, cfg.batch_size, device, cfg.seed)
    features = torch.as_tensor(features_np, device=device)
    latent_predictions = {}
    for name in variants:
        mode, ot = name.split(":")
        s_np, h_np = modes[mode]
        s_adj, h_adj = torch.as_tensor(s_np, device=device), torch.as_tensor(h_np, device=device)
        print(f"Training {name}; STRING and Hi-C are distinct relations", flush=True)
        model, transport, alpha = train_variant(name, ot == "ot", features, s_adj,
            h_adj, target_z, ctrl_z, labels, len(train_targets), cfg, device)
        latent_predictions[name] = generate(model, transport, ot == "ot",
            features, s_adj, h_adj, ctrl_z, alpha,
            range(len(train_targets), len(train_targets)+len(heldout)),
            cfg.samples, cfg.seed+1, device)
        del model, transport
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("All predictions frozen; reading held-out target expression NOW.", flush=True)
    mean_response = np.stack([
        graph.row_mean(matrix[train_mask & (conditions == g)]) - ctrl_mean
        for g in train_targets]).mean(axis=0)
    result = {"version": VERSION, "split": cfg.split, "variants": variants,
              "heldout_targets": list(heldout), "graph_coverage": coverage,
              "parameters": {key: value for key, value in vars(cfg).items()
                             if key not in ("data", "string", "hic", "out")},
              "per_target": []}
    for j, gene in enumerate(heldout):
        rows = np.flatnonzero(test_mask & (conditions == gene))
        truth = graph.row_mean(matrix[rows]) - ctrl_mean
        row = {"target": gene, "n_cells": len(rows),
               "no_change_mse": float(np.mean(truth**2)),
               "mean_response_mse": float(np.mean((truth-mean_response)**2)),
               "models": {}}
        # Encode held-out cells ONLY for scoring latent distribution distance.
        heldout_svd = ((svd.transform(matrix[rows]) - ctrl_latent) / spread).astype(np.float32)
        with torch.no_grad():
            heldout_z = vae.encode(torch.as_tensor(heldout_svd, device=device))[0]
        n_compare = min(64, len(rows), cfg.samples)
        sample_ix = np.random.default_rng(cfg.seed + j).choice(len(rows), n_compare,
                                                               replace=False)
        for name, draws in latent_predictions.items():
            with torch.no_grad():
                decoded = vae.decoder(torch.as_tensor(draws[j], device=device)).cpu().numpy()
            predicted = (decoded.mean(axis=0) * spread) @ svd.components_
            with torch.no_grad():
                distance = sinkhorn_divergence(
                    torch.as_tensor(draws[j][:n_compare], device=device),
                    heldout_z[torch.as_tensor(sample_ix, dtype=torch.long,
                                              device=device)]).item()
            row["models"][name] = {"mean_effect_mse": float(np.mean((truth-predicted)**2)),
                                   "latent_sinkhorn": float(distance)}
        result["per_target"].append(row)
    summary = {"no_change_mse": float(np.mean([r["no_change_mse"] for r in result["per_target"]])),
               "mean_response_mse": float(np.mean([r["mean_response_mse"] for r in result["per_target"]]))}
    for name in variants:
        summary[name] = {metric: float(np.mean([
            r["models"][name][metric] for r in result["per_target"]]))
            for metric in ("mean_effect_mse", "latent_sinkhorn")}
    result["macro"] = summary
    Path(cfg.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {cfg.out}. Same previously explored split; prototype only. "
          "Strict Hi-C only; no enhancer-to-gene direction is claimed.")


if __name__ == "__main__":
    main()
