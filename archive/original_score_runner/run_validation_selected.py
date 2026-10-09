r"""Validation-selected STRING VAE/DDPM controls on the matched GEARS cohort.

The original dual-graph model predicts unseen SINGLES on 33,694 genes. This
adapts its pair conditioner to the 101 official GEARS pair folds/5,045 genes.
This is a NEW architecture, trained from scratch; its scores cannot be copied
from the old single-target run. Run --preflight before requesting GPU training.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.decomposition import TruncatedSVD

import compare_dual_graph_ot_vae_ddpm as proto
import gene_graph_response_2o as graph
import heldout_conditional_diffusion_2o as old
from gears_pair_graph_branch_matched import pair_key
from gears_pair_result_audit import summarize
from prepare_gears_pair_splits import VERSION as SPLIT_VERSION, gears_conditions


VERSION = "2026-10-07-validation-selected-string-ddpm-v1"
VARIANTS = ("no_edges:no_ot", "shuffled_string:no_ot", "string:no_ot")


def read_inputs(data, report_path, baseline_path, gears_path, string_path, hic_path):
    import scanpy as sc
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    gears = json.loads(Path(gears_path).read_text(encoding="utf-8"))
    audit, rows = summarize(gears, baseline)
    if (audit["status"] != "complete" or audit["n_scored"] != 101 or
            report.get("version") != SPLIT_VERSION or
            not report.get("go_filter_note") or
            report.get("n_comparison_pairs") != 101 or
            Path(report["gears_data"]).resolve() != Path(data).resolve() or
            Path(baseline["source"]).resolve() != Path(data).resolve() or
            Path(baseline["split_report"]).resolve() != Path(report_path).resolve()):
        raise ValueError("Need the complete matched 101-pair GO-compatible GEARS run")
    string = json.loads(Path(string_path).read_text(encoding="utf-8"))
    hic = json.loads(Path(hic_path).read_text(encoding="utf-8"))
    if (string.get("species") != 9606 or string.get("split") != "balanced1" or
            hic.get("version") != "2026-09-29-k562-hic-gene-mapping-v2"):
        raise ValueError("Expected original human STRING balanced1 and K562 Hi-C v2")
    adata = sc.read_h5ad(data, backed="r")
    try:
        genes = adata.var["gene_name"].astype(str).tolist()
        native = gears_conditions(adata.obs["condition"].astype(str).unique())
        if len(genes) != baseline["n_genes"] or len({g.upper() for g in genes}) != len(genes):
            raise ValueError("Official gene axis differs from saved baseline")
        all_pairs = {p for f in report["folds"] for p in
                     f["train_pairs"] + f["validation_pairs"] + f["test_pairs"]}
        if all_pairs != {r["pair"] for r in rows} or len(all_pairs) != 101:
            raise ValueError("Report pair cohort differs from GEARS score cohort")
        for p in all_pairs:
            if ("pair", *pair_key(p)) not in native:
                raise ValueError(f"Official data lacks pair {p}")
    finally:
        adata.file.close()
    return report, {r["pair"]: r for r in rows}, genes, native, string, hic


def graph_inputs(genes, pairs, string, hic, max_nodes, seed):
    """Keep target nodes even when a target is outside the 5,045-gene axis."""
    targets = sorted({g for p in pairs for g in pair_key(p)})
    if len(targets) > max_nodes:
        raise ValueError("--max-nodes smaller than the pair target gene count")
    measured = {g.upper() for g in genes}
    target_set = set(targets)
    allowed = measured | target_set
    edges = []
    for original in (string["edges"], hic["strict_promoter_promoter"]["edges"]):
        edges.append([(str(a).upper(), str(b).upper(), float(w)) for a,b,w in original
                      if str(a).upper() in allowed and str(b).upper() in allowed and
                      str(a).upper() != str(b).upper()])
    neighbors = {}
    for relation in edges:
        for a,b,_ in relation:
            if a in target_set and b not in target_set and b in measured:
                neighbors[b] = neighbors.get(b, 0) + 1
            if b in target_set and a not in target_set and a in measured:
                neighbors[a] = neighbors.get(a, 0) + 1
    extra = sorted(neighbors, key=lambda g: (-neighbors[g], g))[:max_nodes-len(targets)]
    nodes = targets + extra
    index = {g:i for i,g in enumerate(nodes)}
    raw = []
    for relation, is_hic in zip(edges, (False, True)):
        adj = np.zeros((len(nodes),len(nodes)),dtype=np.float32)
        for a,b,w in relation:
            if a not in index or b not in index: continue
            if w <= 0 or not np.isfinite(w): raise ValueError("Invalid graph weight")
            value = math.log1p(w) if is_hic else w
            i,j = index[a], index[b]
            adj[i,j] = adj[j,i] = max(adj[i,j], value)
        raw.append(adj)
    rng = np.random.default_rng(seed+77)
    perm = np.arange(len(nodes))
    perm[:len(targets)] = rng.permutation(len(targets))
    perm[len(targets):] = len(targets) + rng.permutation(len(extra))
    zero = np.zeros_like(raw[0])
    variants = {"no_edges":(zero,zero), "shuffled":tuple(a[np.ix_(perm,perm)] for a in raw),
                "string":(raw[0],zero), "hic":(zero,raw[1]), "dual":tuple(raw)}
    variants["shuffled_string"] = (variants["shuffled"][0], zero)
    modes = {k:tuple(a / np.maximum(a.sum(1,keepdims=True),1e-8) for a in v)
             for k,v in variants.items()}
    support = {"n_nodes":len(nodes), "target_genes":len(targets),
               "target_genes_on_expression_axis":len(target_set & measured),
               "string_edges":int(np.count_nonzero(raw[0])//2),
               "hic_edges":int(np.count_nonzero(raw[1])//2),
               "string_targets_connected":int(np.count_nonzero(raw[0][:len(targets)].sum(1))),
               "hic_targets_connected":int(np.count_nonzero(raw[1][:len(targets)].sum(1)))}
    return nodes, modes, support


def control_features(ctrl_rows, genes, nodes):
    measured = {g.upper() for g in genes}
    selected = [g for g in nodes if g in measured]
    if len(selected) < 2:
        raise ValueError("Fewer than two nodes have measured control expression")
    feat, _, _ = graph.graph_from_controls(ctrl_rows, genes, selected)
    by_name = dict(zip(selected, feat))
    filler = np.zeros(feat.shape[1],dtype=np.float32)
    return np.stack([np.r_[by_name.get(g,filler), float(g in measured)]
                     for g in nodes]).astype(np.float32)


class PairConditioner(nn.Module):
    def __init__(self, feature_dim, pair_nodes):
        super().__init__()
        self.single = old.InterventionConditioner(feature_dim)
        self.register_buffer("pair_nodes", torch.as_tensor(pair_nodes,dtype=torch.long))
        self.combine = nn.Sequential(nn.LayerNorm(144),nn.Linear(144,48),nn.SiLU())

    def forward(self, features, string_adj, hic_adj, pair_ids):
        a,b = self.pair_nodes[pair_ids].unbind(dim=1)
        left = self.single(features,string_adj,hic_adj,a)
        right = self.single(features,string_adj,hic_adj,b)
        return self.combine(torch.cat((left+right,(left-right).abs(),left*right),1))


class PairTransport(proto.TransportHead):
    def __init__(self, feat_dim, latent_dim, pair_nodes):
        super().__init__(feat_dim,latent_dim)
        self.graph = PairConditioner(feat_dim,pair_nodes)


class PairDDPM(proto.GraphDDPM):
    def __init__(self, feat_dim, latent_dim, steps, pair_nodes):
        super().__init__(feat_dim,latent_dim,steps)
        self.graph = PairConditioner(feat_dim,pair_nodes)


def train_pair_variant(name, features, s_adj, h_adj, target_z, ctrl_z,
                       labels, n_train, pair_nodes, cfg, device, validate=None):
    """The original OT and denoising losses with a two-gene conditioner."""
    torch.manual_seed(cfg.seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    use_ot = name.endswith(":ot")
    transport = PairTransport(features.shape[1],cfg.vae_dim,pair_nodes).to(device)
    if use_ot:
        donors = [np.flatnonzero(labels==i) for i in range(n_train)]
        if any(len(x)==0 for x in donors): raise ValueError("Empty training pair")
        optimizer = torch.optim.AdamW(transport.parameters(),lr=7e-4)
        for step in range(cfg.ot_steps):
            j = step % n_train
            ix = rng.choice(donors[j],size=cfg.ot_batch,replace=True)
            ctrl_ix = rng.integers(len(ctrl_z),size=cfg.ot_batch)
            source = ctrl_z[torch.as_tensor(ctrl_ix,dtype=torch.long,device=device)]
            goal = target_z[torch.as_tensor(ix,dtype=torch.long,device=device)]
            pair_ids = torch.full((len(source),),j,dtype=torch.long,device=device)
            moved = transport(source,pair_ids,features,s_adj,h_adj)
            loss = (proto.sinkhorn_divergence(moved,goal)+
                    cfg.transport_cost*(moved-source).square().mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(transport.parameters(),1.0)
            optimizer.step()
        print(f"{name} OT final minibatch loss={loss.item():.5f}",flush=True)
    transport.eval()
    for p in transport.parameters(): p.requires_grad_(False)
    beta = torch.linspace(1e-4,min(0.4,12/cfg.steps),cfg.steps,device=device)
    alpha_bar = torch.cumprod(1-beta,dim=0)
    model = PairDDPM(features.shape[1],cfg.vae_dim,cfg.steps,pair_nodes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    label_t = torch.as_tensor(labels,dtype=torch.long,device=device)
    best_state = None
    best_score = float("inf")
    best_epoch = None
    history = []
    stale = 0
    for epoch in range(cfg.epochs):
        model.train()
        losses=[]
        for ix in np.array_split(rng.permutation(len(labels)),
                                 math.ceil(len(labels)/cfg.batch_size)):
            ix_t=torch.as_tensor(ix,dtype=torch.long,device=device)
            pair_ids,truth=label_t[ix_t],target_z[ix_t]
            ctrl_ix=torch.as_tensor(rng.integers(len(ctrl_z),size=len(ix)),
                                    dtype=torch.long,device=device)
            source=ctrl_z[ctrl_ix]
            with torch.no_grad():
                anchor=(transport(source,pair_ids,features,s_adj,h_adj)
                        if use_ot else source)
            times=torch.randint(cfg.steps,(len(ix),),device=device)
            noise=torch.randn_like(truth)
            a=alpha_bar[times,None]
            noisy=a.sqrt()*truth+(1-a).sqrt()*noise
            pred=model(noisy,times,pair_ids,anchor,features,s_adj,h_adj)
            loss=F.mse_loss(pred,noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            losses.append(loss.item())
        print(f"{name} DDPM epoch {epoch+1}/{cfg.epochs}: noise MSE={np.mean(losses):.5f}",flush=True)
        if validate is not None and ((epoch+1) % cfg.eval_every == 0 or epoch+1 == cfg.epochs):
            score = float(validate(model, transport, alpha_bar))
            if not np.isfinite(score):
                raise ValueError("Non-finite validation expression MSE")
            history.append({"epoch": epoch+1, "validation_mse": score})
            print(f"{name} epoch {epoch+1}: validation expression MSE={score:.8f}", flush=True)
            if score < best_score:
                best_score, best_epoch = score, epoch+1
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= cfg.patience:
                break
    if validate is not None:
        if best_state is None:
            raise RuntimeError("No validation checkpoint selected")
        model.load_state_dict(best_state)
        print(f"Selected {name} epoch {best_epoch} using validation only", flush=True)
    model.selection_info = {"selected_epoch":best_epoch, "best_validation_mse":best_score if history else None,
                            "validation_history":history}
    return model,transport,alpha_bar


def self_test():
    torch.manual_seed(8)
    device=torch.device("cpu")
    features=torch.randn(4,19)
    pair_nodes=np.asarray([[0,1],[1,2],[0,3]],dtype=np.int64)
    adjacency=torch.eye(4)
    model=PairDDPM(19,4,24,pair_nodes)
    transport=PairTransport(19,4,pair_nodes)
    pairs=torch.as_tensor([0,1,2],dtype=torch.long)
    output=model(torch.randn(3,4),torch.zeros(3,dtype=torch.long),pairs,
                 torch.randn(3,4),features,adjacency,adjacency)
    moved=transport(torch.randn(3,4),pairs,features,adjacency,adjacency)
    if output.shape!=(3,4) or moved.shape!=(3,4) or not torch.isfinite(output).all():
        raise AssertionError("Invalid pair-conditioned model shapes")
    cfg=argparse.Namespace(seed=8,vae_dim=4,ot_steps=3,ot_batch=4,
                           transport_cost=0.01,steps=24,epochs=2,batch_size=4,eval_every=1,patience=2)
    calls=[]
    def synthetic_validation(model, transport, alpha):
        calls.append({k:v.detach().clone() for k,v in model.state_dict().items()})
        return float(len(calls)) # Epoch 1 must win; check restoration below.
    fitted,ot,alpha=train_pair_variant("self_test:no_ot",features,adjacency,
        torch.zeros_like(adjacency),torch.randn(8,4),torch.randn(6,4),
        np.asarray([0,0,0,0,1,1,1,1]),2,pair_nodes,cfg,device,validate=synthetic_validation)
    assert fitted.selection_info["selected_epoch"] == 1
    assert all(torch.equal(v, calls[0][k]) for k,v in fitted.state_dict().items())
    predicted=proto.generate(fitted,ot,False,features,adjacency,
        torch.zeros_like(adjacency),torch.randn(6,4),alpha,[2],3,9,device)
    if predicted.shape!=(1,3,4) or not np.isfinite(predicted).all():
        raise AssertionError("Pair-conditioned DDPM sampling failed")
    print("SELF-TEST PASSED: pair graph, no-OT DDPM, validation selection/restoration and generation")


def run_fold(cfg,fold,baseline_rows,genes,native,modes,nodes,support):
    import scanpy as sc
    data=sc.read_h5ad(cfg.data)
    try:
        matrix=data.X.tocsr() if hasattr(data.X,"tocsr") else data.X
        labels=np.asarray(data.obs["condition"].astype(str).to_numpy(),dtype=str)
        mapping={alias: key for key,aliases in native.items() for alias in aliases}
        names=np.asarray(["+".join(mapping.get(label,("unused",))[1:])
                          if mapping.get(label,("unused",))[0]=="pair"
                          else "ctrl" if mapping.get(label,("unused",))[0]=="control" else "unused"
                          for label in labels],dtype=str)
        train=fold["train_pairs"]
        test=fold["test_pairs"]
        valid=fold["validation_pairs"]
        ordered=train+valid+test
        if not valid or len(set(ordered)) != len(ordered):
            raise ValueError("Fold overlaps")
        control_ix=np.flatnonzero(names=="ctrl")
        if len(control_ix)<2: raise ValueError("No controls")
        rng=np.random.default_rng(cfg.seed+fold["fold"])
        ctrl_select=rng.choice(control_ix,min(len(control_ix),cfg.max_controls),replace=False)
        features_np=control_features(matrix[ctrl_select],genes,nodes)
        train_mask=np.isin(names,train) | (names=="ctrl")
        # Both VAE and SVD fit exclusively to training-pair/control cells.
        svd,control_mean,spread,values,train_labels=old.make_training_data(
            matrix,names,train_mask,train,control_ix,cfg.max_per_pair,cfg.svd_dim,cfg.seed)
        origin=svd.transform(control_mean[None,:])[0]
        ctrl_values=((svd.transform(matrix[ctrl_select])-origin)/spread).astype(np.float32)
        device=torch.device(cfg.device)
        vae,target_z,ctrl_z=proto.train_vae(values,ctrl_values,cfg.svd_dim,
             cfg.vae_dim,cfg.vae_epochs,cfg.batch_size,device,cfg.seed)
        node_index={g:i for i,g in enumerate(nodes)}
        pair_nodes=np.asarray([[node_index[g] for g in pair_key(p)]
                               for p in ordered],dtype=np.int64)
        features=torch.as_tensor(features_np,device=device)
        # Only validation labels are used for model selection. No test means here.
        valid_truth = np.stack([graph.row_mean(matrix[np.flatnonzero(names==p)])-control_mean for p in valid])
        if not np.isfinite(valid_truth).all():
            raise ValueError("Invalid validation means")
        validation_ids = range(len(train), len(train)+len(valid))
        rows=[]
        cached={}
        selections={}
        checkpoint_folder=Path(cfg.checkpoint_dir)
        checkpoint_folder.mkdir(parents=True,exist_ok=True)
        signature_values={k:v for k,v in vars(cfg).items() if k not in
                          ("out","variants","fold","all_folds","preflight",
                           "self_test","checkpoint_dir")}
        signature_values["input_metadata"]={name:{
            "size":Path(getattr(cfg,name)).stat().st_size,
            "mtime_ns":Path(getattr(cfg,name)).stat().st_mtime_ns}
            for name in ("data","report","baseline","gears","string","hic")}
        signature_values["version"] = VERSION
        signature_values["code_sha256"] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in Path(__file__).parent.glob("*.py")}
        signature=hashlib.sha256(json.dumps(signature_values,sort_keys=True).encode()).hexdigest()
        for variant in cfg.variants:
            checkpoint=checkpoint_folder/f"fold_{fold['fold']}_{variant.replace(':','_')}.json"
            if checkpoint.exists():
                record=json.loads(checkpoint.read_text(encoding="utf-8"))
                if record.get("signature")!=signature or set(record.get("pair_mse",{}))!=set(test):
                    raise ValueError(f"Stale/mismatched variant checkpoint: {checkpoint}")
                cached[variant]=record["pair_mse"]
                selections[variant]=record["selection"]
                print(f"Fold {fold['fold']}: reused completed {variant}",flush=True)
                continue
            mode,ot=variant.split(":")
            s,h=modes[mode]
            s_adj=torch.as_tensor(s,device=device)
            h_adj=torch.as_tensor(h,device=device)
            print(f"Fold {fold['fold']}: training pair-conditioned {variant}",flush=True)
            def validate(model, transport, alpha):
                with torch.no_grad():
                    draws=proto.generate(model,transport,False,features,s_adj,h_adj,
                        ctrl_z,alpha,validation_ids,cfg.samples,cfg.seed+10000+fold["fold"],device)
                    errors=[]
                    for j in range(len(valid)):
                        decoded=vae.decoder(torch.as_tensor(draws[j],device=device)).cpu().numpy()
                        predicted=(decoded.mean(axis=0)*spread)@svd.components_
                        errors.append(float(np.mean((valid_truth[j]-predicted)**2)))
                    return float(np.mean(errors))
            model,transport,alpha=train_pair_variant(variant,features,s_adj,h_adj,
                target_z,ctrl_z,train_labels,len(train),pair_nodes,cfg,device,validate=validate)
            selections[variant]=model.selection_info
            draws=proto.generate(model,transport,ot=="ot",features,s_adj,h_adj,
                ctrl_z,alpha,range(len(train)+len(valid),len(ordered)),cfg.samples,
                cfg.seed+fold["fold"],device)
            if not rows:
                print("Predictions frozen; now reading test-pair expression",flush=True)
                single_means={g:graph.row_mean(matrix[np.isin(labels,native[("single",g)])])
                              for p in test for g in pair_key(p)}
                for name in test:
                    ix=np.flatnonzero(names==name)
                    if not len(ix): raise ValueError(f"No cells for pair {name}")
                    observed=graph.row_mean(matrix[ix])-control_mean
                    a,b=pair_key(name)
                    additive=single_means[a]+single_means[b]-2*control_mean
                    saved=baseline_rows[name]["additive_mse"]
                    check=float(np.mean((observed-additive)**2))
                    if not np.isclose(check,saved,rtol=2e-5,atol=2e-8):
                        raise ValueError(f"Pair {name}: additive {check} != saved {saved}")
                    rows.append({"pair":name,"cells":len(ix),"truth_effect":observed,
                                 "scores":{}})
            for j,row in enumerate(rows):
                with torch.no_grad():
                    decoded=vae.decoder(torch.as_tensor(draws[j],device=device)).cpu().numpy()
                predicted=(decoded.mean(axis=0)*spread)@svd.components_
                error=float(np.mean((row["truth_effect"]-predicted)**2))
                row["scores"][variant]=error
            pair_scores={row["pair"]:row["scores"][variant] for row in rows}
            checkpoint.write_text(json.dumps({"signature":signature,
                "fold":fold["fold"],"variant":variant,"pair_mse":pair_scores,
                "selection":selections[variant],
                "n_cells":{row["pair"]:row["cells"] for row in rows}},indent=2),
                encoding="utf-8")
            print(f"Completed {variant}; saved {checkpoint}",flush=True)
            del model,transport,draws
            if device.type=="cuda": torch.cuda.empty_cache()
        if not rows:
            # All variants were already scored, so no test expression is needed.
            first=json.loads((checkpoint_folder/f"fold_{fold['fold']}_{cfg.variants[0].replace(':','_')}.json").read_text(encoding="utf-8"))
            rows=[{"pair":name,"cells":first["n_cells"][name],"scores":{}}
                  for name in test]
        for row in rows:
            for variant,pair_scores in cached.items():
                row["scores"][variant]=pair_scores[row["pair"]]
        result=[]
        for row in rows:
            old_row=baseline_rows[row["pair"]]
            result.append({"pair":row["pair"],"fold":fold["fold"],
                           "n_cells":row["cells"],
                           "additive_mse":old_row["additive_mse"],
                           "nearest_mse":old_row["nearest_mse"],
                           "gears_mse":old_row["gears_mse"],
                           "variants":row["scores"], "selection":selections})
        return result
    finally:
        data.file.close() if data.isbacked else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("data","report","baseline","gears","string","hic"):
        p.add_argument(arg,nargs="?")
    p.add_argument("--fold",type=int,default=1)
    p.add_argument("--all-folds",action="store_true")
    p.add_argument("--variants",default=",".join(VARIANTS))
    p.add_argument("--max-nodes",type=int,default=512)
    p.add_argument("--svd-dim",type=int,default=64)
    p.add_argument("--vae-dim",type=int,default=24)
    p.add_argument("--vae-epochs",type=int,default=8)
    p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--eval-every",type=int,default=2)
    p.add_argument("--patience",type=int,default=6)
    p.add_argument("--ot-steps",type=int,default=160)
    p.add_argument("--ot-batch",type=int,default=32)
    p.add_argument("--transport-cost",type=float,default=0.01)
    p.add_argument("--steps",type=int,default=32)
    p.add_argument("--samples",type=int,default=128)
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--max-per-pair",type=int,default=128)
    p.add_argument("--max-controls",type=int,default=2048)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--checkpoint-dir",default="validation_string_checkpoints_v1")
    p.add_argument("--out",default="validation_string_result.json")
    cfg=p.parse_args()
    if cfg.self_test:
        self_test()
        return
    if not all(getattr(cfg,name) for name in ("data","report","baseline","gears","string","hic")):
        p.error("Six input paths required, or use --self-test")
    cfg.variants=cfg.variants.split(",")
    if (not cfg.variants or len(set(cfg.variants))!=len(cfg.variants) or
            set(cfg.variants)-set(VARIANTS) or
            not 1<=cfg.fold<=5 or cfg.steps<24 or cfg.vae_dim>=cfg.svd_dim or
            min(cfg.eval_every,cfg.patience,cfg.max_nodes,cfg.svd_dim,cfg.vae_dim,cfg.vae_epochs,cfg.epochs,
                cfg.ot_steps,cfg.ot_batch,cfg.steps,cfg.samples,cfg.batch_size,
                cfg.max_per_pair,cfg.max_controls)<=0):
        p.error("Invalid variants, fold, or positive hyperparameters")
    report,old_rows,genes,native,string,hic=read_inputs(
        cfg.data,cfg.report,cfg.baseline,cfg.gears,cfg.string,cfg.hic)
    all_pairs=sorted(old_rows)
    nodes,modes,support=graph_inputs(genes,all_pairs,string,hic,cfg.max_nodes,cfg.seed)
    print(f"Matched official GEARS pair cohort: {len(all_pairs)} pairs, "
          f"{len(genes)} genes, 5 folds; graph support={json.dumps(support)}")
    if cfg.preflight:
        print("PREFLIGHT ONLY: no pair expression read; no training or scoring")
        return
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA unavailable; use --device cpu")
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    selected=report["folds"] if cfg.all_folds else [report["folds"][cfg.fold-1]]
    if [f["fold"] for f in report["folds"]]!=[1,2,3,4,5]:
        raise ValueError("Unexpected fold numbering")
    all_rows=[]
    for fold in selected:
        all_rows.extend(run_fold(cfg,fold,old_rows,genes,native,modes,nodes,support))
    summary={name:float(np.mean([row["variants"][name] for row in all_rows]))
             for name in cfg.variants}
    summary.update({name:float(np.mean([row[name] for row in all_rows]))
                    for name in ("additive_mse","nearest_mse","gears_mse")})
    result={"version":VERSION,"complete_101_pairs":len(all_rows)==101,
            "folds":[f["fold"] for f in selected],"n_scored":len(all_rows),
            "n_genes":len(genes),"graph_support":support,
            "settings":{k:v for k,v in vars(cfg).items() if k not in
                        ("data","report","baseline","gears","string","hic","out")},
            "macro":summary,"per_pair":all_rows,
            "limitations":["Pair adaptation of the original single-target OT/VAE/DDPM, not its earlier model weights.",
                           "Previously examined Norman pairs; exploratory only.",
                           "DDPM epoch selected on validation all-gene expression MSE; VAE fixed at 8 epochs by default.",
                           "Training budget differs from earlier fixed 12-epoch results; compare controls in this run.",
                           "No GEARS checkpoint was retrained; stored matched pair scores were reused.",
                           "Official GEARS checkpoint selected on validation DE20, but comparison here is pair-mean all-gene MSE."]}
    Path(cfg.out).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    print(f"Saved {cfg.out}; n_scored={len(all_rows)}/101, complete={len(all_rows)==101}")


if __name__=="__main__":
    main()
