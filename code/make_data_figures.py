"""Fold-specific 3-panel UMAP and exploratory cell-level response volcano.
No treatment/cell-type labels fabricated. UMAP fits on sampled training/control cells.
"""
import argparse,json
from pathlib import Path
import numpy as np
import scanpy as sc
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from prepare_gears_pair_splits import gears_conditions
p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('report');p.add_argument('--exports',default='visual_expression_exports');p.add_argument('--fold',type=int,default=1);p.add_argument('--out',default='inference_figures');cfg=p.parse_args()
out=Path(cfg.out);out.mkdir(parents=True,exist_ok=True)
export=Path(cfg.exports)/f'fold_{cfg.fold}_string_no_ot.npz'
q=np.load(export,allow_pickle=False);fitted=joblib.load(export.with_suffix('.joblib'))
a=sc.read_h5ad(cfg.data);assert np.array_equal(a.var['gene_name'].astype(str).to_numpy(),q['gene_names']), 'Gene axis differs from export';X=a.X.tocsr() if hasattr(a.X,'tocsr') else a.X
fold=json.loads(Path(cfg.report).read_text())['folds'][cfg.fold-1];native=gears_conditions(a.obs['condition'].astype(str).unique());mapping={alias:key for key,aliases in native.items() for alias in aliases}
labels=a.obs['condition'].astype(str).to_numpy();names=np.array(['+'.join(mapping.get(x,('unused',))[1:]) if mapping.get(x,('unused',))[0]=='pair' else 'ctrl' if mapping.get(x,('unused',))[0]=='control' else 'unused' for x in labels])
rng=np.random.default_rng(42);train=set(fold['train_pairs']);valid=set(fold['validation_pairs']);test=set(fold['test_pairs']);eligible=np.flatnonzero(np.isin(names,list(train|valid|test)+['ctrl']))
chosen=[]
for c in sorted(set(names[eligible])):
 ix=np.flatnonzero(names==c);chosen.extend(rng.choice(ix,min(len(ix),40),replace=False))
chosen=np.asarray(chosen);split=np.array(['Control' if names[i]=='ctrl' else 'Training' if names[i] in train else 'Validation' if names[i] in valid else 'Held-out test' for i in chosen]);train_mask=np.isin(split,['Control','Training'])
svd=fitted['svd'];origin=svd.transform(fitted['control_mean'][None,:])[0];features=(svd.transform(X[chosen])-origin)/fitted['spread']
try:
 import umap
 embedder=umap.UMAP(n_components=2,random_state=42,n_neighbors=15,min_dist=.3)
 coords=np.empty((len(chosen),2));coords[train_mask]=embedder.fit_transform(features[train_mask]);coords[~train_mask]=embedder.transform(features[~train_mask])
 cellcol=next((c for c in ['cell_type','celltype','cell_type_annotation'] if c in a.obs),None)
 celllabels=np.repeat('Cells',len(chosen)) # Neutral display; raw annotations remain in the h5ad.
 # Retain six explicit perturbation labels plus control; all remaining labels grouped visibly.
 examples=sorted(test)[:3]+sorted(train)[:3];conditions=np.array([names[i] if names[i] in examples or names[i]=='ctrl' else 'Other pair conditions' for i in chosen])
 fig,axes=plt.subplots(1,3,figsize=(15,5))
 for ax,values,title in zip(axes,[celllabels,conditions,split],['Cells in processed benchmark','CRISPRa conditions (six named examples)',f'Fold {cfg.fold} split']):
  for j,v in enumerate(sorted(set(values))):
   mask=values==v;ax.scatter(coords[mask,0],coords[mask,1],s=5,alpha=.65,label=v,color=plt.get_cmap('tab10')(j%10))
  ax.set_title(title);ax.set_xlabel('UMAP 1');ax.set_ylabel('UMAP 2');ax.legend(fontsize=6,markerscale=2)
 fig.suptitle('Same embedding, fitted on sampled training/control cells; descriptive visualization');fig.tight_layout();fig.savefig(out/'three_panel_umap.png',dpi=200);fig.savefig(out/'three_panel_umap.pdf');plt.close(fig)
except ImportError:
 print('UMAP skipped: install umap-learn in your existing environment if desired; other figures continue.')
# Transparent example: first alphabetically among this fold's held-out pairs.
example=sorted(q['pair_names'].astype(str))[0];ix=np.flatnonzero(names==example);ctrl=np.flatnonzero(names=='ctrl')
ix=rng.choice(ix,min(500,len(ix)),replace=False);ctrl=rng.choice(ctrl,min(500,len(ctrl)),replace=False)
def dense(m):return m.toarray() if hasattr(m,'toarray') else np.asarray(m)
u=dense(X[ix]);v=dense(X[ctrl]);effect=u.mean(0)-v.mean(0)
# Exploratory cell-level Welch test: cells are not biological replicates.
with np.errstate(all='ignore'):pv=ttest_ind(u,v,axis=0,equal_var=False).pvalue
pv=np.where(np.isfinite(pv),pv,1.);order=np.argsort(pv);adj=np.empty_like(pv);adj[order]=np.minimum(1,np.minimum.accumulate((pv[order]*len(pv)/np.arange(1,len(pv)+1))[::-1])[::-1])
fig,ax=plt.subplots(figsize=(8,6));ax.scatter(effect,-np.log10(np.maximum(adj,1e-300)),s=7,alpha=.45,color='#008B83');ax.set_xlabel('Observed mean expression difference on processed scale');ax.set_ylabel('−log10 exploratory cell-level BH-adjusted p');ax.set_title(f'{example} versus control: observed-cell response')
fig.text(.1,.01,'Cells are not biological replicates. Descriptive exploration; no population-level significance claim.',fontsize=9);fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(out/'exploratory_observed_volcano.png',dpi=200);fig.savefig(out/'exploratory_observed_volcano.pdf');plt.close(fig)
np.savez_compressed(out/'observed_de_example.npz',gene_names=q['gene_names'],effect=effect,cell_level_p=pv,cell_level_bh=adj,pair=np.asarray(example))
print('Data figures completed. Volcano is observed-cell analysis, not model-inference significance.')
