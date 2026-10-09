"""Regenerate benchmark bars, tests, histories and effect plots from saved evidence."""
from pathlib import Path
import json,itertools,subprocess,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;out=ROOT/'work/saved_figures';out.mkdir(parents=True,exist_ok=True)
d=json.loads((ROOT/'evidence/validation_string_seed42.json').read_text());r=sorted(d['per_pair'],key=lambda x:x['pair']);fold=np.array([x['fold'] for x in r]);real=np.array([x['variants']['string:no_ot'] for x in r])
methods=['GEARS','no_edges:no_ot','shuffled_string:no_ot','string:no_ot'];labels=['GEARS','No edges','Shuffled STRING','Real STRING'];colors=['#173642','#627387','#C17A27','#008B83']
scores={v:np.array([x['gears_mse'] if v=='GEARS' else x['variants'][v] for x in r]) for v in methods}
results=[]
for j,v in enumerate(methods[:3]):
 delta=scores[v]-real;obs=abs(delta.mean());rng=np.random.default_rng(20261008+j);ex=0;N=200000
 for _ in range(N//2000):ex+=int((abs((rng.choice([-1,1],size=(2000,len(delta)))*delta).mean(1))>=obs).sum())
 fd=np.array([delta[fold==f].sum()/len(delta) for f in range(1,6)]);per=np.array(list(itertools.product([-1,1],repeat=5)))@fd
 results.append({'comparison':v,'pair_sign_flip_p':(ex+1)/(N+1),'fold_block_sensitivity_p':float(np.mean(abs(per)>=obs-1e-15)),'mean_error_reduction':float(delta.mean())})
order=np.argsort([x['pair_sign_flip_p'] for x in results]);running=0
for rank,i in enumerate(order):running=max(running,(3-rank)*results[i]['pair_sign_flip_p']);results[i]['holm_adjusted_p']=min(1,running)
(out/'statistics.json').write_text(json.dumps({'n_pairs':101,'tests':'Exploratory mean sign-flip; Holm correction (3 tests); weighted fold-block sensitivity','comparisons':results,'limitations':['Pair exchangeability unverified; shared genes/training folds','Repeatedly examined cohort,one training seed','Fold dots are not independent replicates']},indent=2))
fig,ax=plt.subplots(figsize=(10,6));means=[scores[v].mean() for v in methods];ax.bar(range(4),means,color=colors,width=.6)
for i,m in enumerate(means):
 label_y=max(scores[methods[i]][fold==f].mean() for f in range(1,6))+.00012
 ax.text(i,label_y,f'{m:.8f}',ha='center',fontsize=11,zorder=10,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
for f,offset in zip(range(1,6),np.linspace(-.16,.16,5)):
 vals=[scores[v][fold==f].mean() for v in methods];ax.plot(np.arange(4)+offset,vals,color='#555',alpha=.18,lw=.8);ax.scatter(np.arange(4)+offset,vals,color='#444',s=22,zorder=3)
for i,y in [(2,.0051),(1,.0058),(0,.0065)]:
 p=results[i]['holm_adjusted_p'];star='*' if p<.05 else 'ns';ax.plot([i,i,3,3],[y-.0001,y,y,y-.0001],color='black',lw=1);ax.text((i+3)/2,y+.00006,f'{star}, Holm p = {p:.3f}',ha='center',fontsize=11)
ax.set_xticks(range(4),labels);ax.set_ylim(0,.0072);ax.set_ylabel('All-gene pair-mean MSE (lower is better)');ax.set_title('Matched 101-pair comparison');ax.spines[['top','right']].set_visible(False)
fig.text(.08,.035,'Exploratory paired sign-flip tests; Holm correction (3 tests). * p<0.05; ns p≥0.05.\nDots: five fold means, not independent replicates. Pair exchangeability unverified; fold-block checks all p>0.05.',fontsize=9);fig.tight_layout(rect=[0,.12,1,1])
for ext in ['png','pdf']:fig.savefig(out/f'four_method_comparison.{ext}',dpi=220)
plt.close(fig)
pr=json.loads((ROOT/'evidence/gears_pair_graph_branch_matched.json').read_text())['macro']
names=['Additive','Nearest residual','No-graph predictor','Shuffled-STRING predictor','STRING predictor','STRING diffusion','GEARS'];values=[d['macro']['additive_mse'],d['macro']['nearest_mse'],pr['no_graph']['mse_all'],pr['shuffled_string']['mse_all'],pr['string']['mse_all'],real.mean(),scores['GEARS'].mean()]
fig,ax=plt.subplots(figsize=(10,5.5));bars=ax.barh(names,values,color=['#75679A','#929393','#627387','#C17A27','#008B83','#008B83','#173642']);ax.invert_yaxis()
for b,v in zip(bars,values):ax.text(v+.00005,b.get_y()+b.get_height()/2,f'{v:.8f}',va='center')
ax.set_xlim(0,.0054);ax.set_xlabel('All-gene pair-mean MSE');ax.set_title('All study methods; predictor uses measured singles');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'all_method_mse.{ext}',dpi=220)
plt.close(fig)
visual=json.loads((ROOT/'evidence/visual_export_result.json').read_text());fig,axes=plt.subplots(1,2,figsize=(12,5))
for f in range(1,6):
 item=next(x for x in visual['per_pair'] if x['fold']==f);s=item['selection']['string:no_ot'];h=s.get('training_history',[]);v=s['validation_history']
 if h:axes[0].plot([x['epoch'] for x in h],[x['noise_mse'] for x in h],label=f'Fold {f}')
 axes[1].plot([x['epoch'] for x in v],[x['validation_mse'] for x in v],label=f'Fold {f}');axes[1].scatter(s['selected_epoch'],s['best_validation_mse'],marker='*',s=80)
for ax,title,yl in zip(axes,['Training objective','Validation prediction error'],['Noise MSE','Expression MSE']):ax.set_title(title);ax.set_ylabel(yl);ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.15)
fig.suptitle('Stars mark selected checkpoints, not significance');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'training_and_validation.{ext}',dpi=220)
plt.close(fig)
export=ROOT/'evidence/expression_export.npz'
if export.exists():
 q=np.load(export,allow_pickle=False);names=q['pair_names'].astype(str);errors=np.mean((q['observed_effect']-q['predicted_effect'])**2,axis=1);saved={x['pair']:x['variants']['string:no_ot'] for x in r}
 if len(names)!=101 or set(names)!=set(saved) or not all(np.isclose(e,saved[n],rtol=1e-5,atol=1e-8) for n,e in zip(names,errors)):raise ValueError('Expression export does not match savedSTRING scores')
 subprocess.run([sys.executable,str(ROOT/'code/plot_expression_export.py'),str(export),'--out',str(out)],check=True)
 print('Expression export verified against101 saved per-pair MSEs.')
print('Saved figures:',out)
