"""Held-out effect metrics, OLS-only scatters, observed-order clustered heatmap."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,spearmanr
from scipy.cluster.hierarchy import linkage,dendrogram,leaves_list
from scipy.spatial.distance import pdist
from sklearn.metrics import r2_score
p=argparse.ArgumentParser();p.add_argument('export');p.add_argument('--out',default='work/figures');cfg=p.parse_args()
q=np.load(cfg.export,allow_pickle=False);y=q['observed_effect'];pred=q['predicted_effect'];names=q['pair_names'].astype(str);genes=q['gene_names'].astype(str)
if y.shape!=pred.shape or y.ndim!=2 or len(names)!=len(y) or len(genes)!=y.shape[1] or len(set(names))!=len(names) or not np.isfinite(y).all() or not np.isfinite(pred).all():raise ValueError('Invalid aligned expression export')
out=Path(cfg.out);out.mkdir(parents=True,exist_ok=True)
rows=[]
for i in range(len(y)):
 rows.append({'pair':names[i],'pearson':float(pearsonr(y[i],pred[i]).statistic) if np.std(y[i])*np.std(pred[i]) else None,'spearman':float(spearmanr(y[i],pred[i]).statistic) if np.std(y[i])*np.std(pred[i]) else None,'predictive_r2':float(r2_score(y[i],pred[i])),'mse':float(np.mean((y[i]-pred[i])**2))})
summary={k:{'mean':float(np.mean([r[k] for r in rows if r[k] is not None])),'median':float(np.median([r[k] for r in rows if r[k] is not None]))} for k in ['pearson','spearman','predictive_r2','mse']}
(out/'expression_metrics.json').write_text(json.dumps({'unit':'Per pair across genes; effects relative to control','summary':summary,'per_pair':rows},indent=2))
def scatter(ax,i):
 ax.scatter(y[i],pred[i],s=5,alpha=.2,color='#008B83');lo=min(y[i].min(),pred[i].min());hi=max(y[i].max(),pred[i].max())
 if np.std(y[i]):
  slope,intercept=np.polyfit(y[i],pred[i],1);ax.plot([lo,hi],slope*np.array([lo,hi])+intercept,color='#C17A27',label='OLS fit');ax.legend()
 r=rows[i];pr='undefined' if r['pearson'] is None else f"{r['pearson']:.3f}"
 ax.set_title(f"{names[i]}\nPearson={pr}; predictive R²={r['predictive_r2']:.3f}");ax.set_xlabel('Observed expression change');ax.set_ylabel('Predicted expression change')
ix=np.argsort(names)[:3];fig,axs=plt.subplots(1,len(ix),figsize=(5*len(ix),4.5),squeeze=False)
for ax,i in zip(axs.ravel(),ix):scatter(ax,i)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'actual_vs_predicted.{ext}',dpi=220)
plt.close(fig)
example='AHR+KLF1'
if example in names:
 fig,ax=plt.subplots(figsize=(6,5));scatter(ax,int(np.flatnonzero(names==example)[0]));fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/f'AHR_KLF1_OLS_only.{ext}',dpi=220)
 plt.close(fig)
idx=np.argsort(np.mean(np.abs(y),axis=0))[-min(20,len(genes)):];a=y[:,idx].T;b=pred[:,idx].T
def cluster(v):
 d=pdist(v,metric='correlation')
 if not np.isfinite(d).all():raise ValueError('Undefined correlation distances; constant response profile')
 return linkage(np.clip(d,0,2),method='average',optimal_ordering=True)
rl=cluster(a);cl=cluster(a.T);ri=leaves_list(rl);ci=leaves_list(cl)
fig=plt.figure(figsize=(15,7.8));gs=fig.add_gridspec(2,4,height_ratios=[1,5],width_ratios=[.65,5,5,.2],hspace=.045,wspace=.42)
for j,t in [(1,'Observed effects'),(2,'Predicted effects')]:
 ax=fig.add_subplot(gs[0,j]);dendrogram(cl,ax=ax,no_labels=True,color_threshold=0,above_threshold_color='#506b75');ax.axis('off');ax.set_title(t,fontsize=15,pad=8)
ax=fig.add_subplot(gs[1,0]);dendrogram(rl,ax=ax,orientation='left',no_labels=True,color_threshold=0,above_threshold_color='#506b75');ax.invert_yaxis();pos=ax.get_position();ax.set_position([pos.x0,pos.y0,pos.width*.65,pos.height]);ax.axis('off')
for j,m in [(1,a),(2,b)]:
 ax=fig.add_subplot(gs[1,j]);im=ax.imshow(m[ri][:,ci],aspect='auto',cmap='RdBu_r',vmin=-3.2,vmax=3.2,interpolation='nearest');ax.set_yticks(range(len(idx)));ax.set_yticklabels(genes[idx][ri] if j==1 else [],fontsize=10);ax.tick_params(axis='y',length=0);ticks=np.unique(np.linspace(0,len(names)-1,min(6,len(names)),dtype=int));ax.set_xticks(ticks,ticks+1);ax.set_xlabel('Held-out pairs, clustered order')
cax=fig.add_subplot(gs[1,3]);fig.colorbar(im,cax=cax,label='Expression change relative to control')
fig.suptitle('Observed and predicted responses: genes and pairs grouped by similarity',fontsize=17,y=.98)
fig.text(.5,.018,'20 genes selected by observed mean absolute effect. Average-linkage clustering with correlation distance.\nBoth panels use observed-derived orders and a fixed shared ±3.2 colour scale; extremes may be clipped. Descriptive analysis.',ha='center',fontsize=9)
fig.subplots_adjust(top=.91,bottom=.14,left=.03,right=.93)
for ext in ['png','pdf']:fig.savefig(out/f'response_heatmap.{ext}',dpi=220)
(out/'heatmap_order.json').write_text(json.dumps({'genes':genes[idx][ri].tolist(),'pairs':names[ci].tolist(),'selection':'Top observed mean absolute effect','distance':'correlation','linkage':'average','colour_limit':3.2},indent=2))
print(json.dumps(summary,indent=2))
