"""Usage: python plot_expression_export.py expression_export.npz --out expression_figures
Required: observed_effect[N,G], predicted_effect[N,G], pair_names[N], gene_names[G].
All arrays must be aligned, held-out predictions, same scale/control reference.
"""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import r2_score
p=argparse.ArgumentParser();p.add_argument('export');p.add_argument('--out',default='expression_figures');cfg=p.parse_args()
f=np.load(cfg.export,allow_pickle=False);y=f['observed_effect'];pred=f['predicted_effect'];names=f['pair_names'].astype(str);genes=f['gene_names'].astype(str)
assert y.shape==pred.shape and y.ndim==2 and len(names)==len(y) and len(genes)==y.shape[1]
assert np.isfinite(y).all() and np.isfinite(pred).all()
out=Path(cfg.out);out.mkdir(parents=True,exist_ok=True)
rows=[]
for i in range(len(y)):
 rows.append({'pair':names[i],'pearson':float(pearsonr(y[i],pred[i]).statistic) if np.std(y[i])*np.std(pred[i]) else None,'spearman':float(spearmanr(y[i],pred[i]).statistic) if np.std(y[i])*np.std(pred[i]) else None,'predictive_r2':float(r2_score(y[i],pred[i])),'mse':float(np.mean((y[i]-pred[i])**2))})
(out/'expression_metrics.json').write_text(json.dumps({'unit':'per held-out pair across genes; expression changes relative to control','per_pair':rows},indent=2))
# Examples fixed by lexicographic order, never by best score.
ix=np.argsort(names)[:3];fig,axes=plt.subplots(1,len(ix),figsize=(5*len(ix),4.5),squeeze=False)
for ax,i in zip(axes.ravel(),ix):
 ax.scatter(y[i],pred[i],s=5,alpha=.2,color='#008B83');lo=min(y[i].min(),pred[i].min());hi=max(y[i].max(),pred[i].max());ax.plot([lo,hi],[lo,hi],ls='--',color='#627387',label='Identity')
 if np.std(y[i]):
  slope,intercept=np.polyfit(y[i],pred[i],1);ax.plot([lo,hi],slope*np.array([lo,hi])+intercept,color='#C17A27',label='OLS')
 r=rows[i];pearson_label='undefined' if r['pearson'] is None else f"{r['pearson']:.3f}";ax.set_title(f"{names[i]}\nPearson={pearson_label}; predictive R²={r['predictive_r2']:.3f}");ax.set_xlabel('Observed expression change');ax.set_ylabel('Predicted expression change');ax.legend()
fig.tight_layout();fig.savefig(out/'actual_vs_predicted.png',dpi=200);fig.savefig(out/'actual_vs_predicted.pdf');plt.close(fig)
# Largest observed effects across pairs; descriptive, ground-truth-informed selection.
j=np.argsort(np.mean(abs(y),axis=0))[-20:];order=np.argsort(names);a=y[order][:,j].T;b=pred[order][:,j].T;lim=max(abs(a).max(),abs(b).max())
fig,axs=plt.subplots(1,2,figsize=(15,7),sharey=True)
for ax,values,title in zip(axs,[a,b],['Observed effects','Predicted effects']):
 im=ax.imshow(values,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim);ax.set_title(title);ax.set_xlabel('Held-out pairs, alphabetical');ax.set_yticks(range(len(j)),genes[j],fontsize=8)
fig.colorbar(im,ax=axs.ravel().tolist(),label='Expression change');fig.suptitle('Descriptive response heatmap: genes selected by largest observed mean absolute effect')
fig.savefig(out/'response_heatmap.png',dpi=200,bbox_inches='tight');fig.savefig(out/'response_heatmap.pdf',bbox_inches='tight')
print('Saved per-pair metrics, fixed-example scatters, and descriptive heatmap. No DEG p-values or volcano plot inferred.')
