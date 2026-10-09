"""Assemble exported held-out predictions and generate scatter/heatmap/learning curves."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--exports',default='visual_expression_exports');p.add_argument('--checkpoints',default='visual_training_checkpoints');p.add_argument('--out',default='inference_figures');p.add_argument('--variant',default='string_no_ot');cfg=p.parse_args()
out=Path(cfg.out);out.mkdir(parents=True,exist_ok=True)
files=sorted(Path(cfg.exports).glob(f'fold_*_{cfg.variant}.npz'))
if not files:raise SystemExit('No expression exports. Run export training first.')
parts=[np.load(f,allow_pickle=False) for f in files]
genes=parts[0]['gene_names'];assert all(np.array_equal(q['gene_names'],genes) for q in parts)
names=np.concatenate([q['pair_names'] for q in parts]);assert len(set(names.tolist()))==len(names)
export=out/'expression_export.npz';np.savez_compressed(export,observed_effect=np.concatenate([q['observed_effect'] for q in parts]),predicted_effect=np.concatenate([q['predicted_effect'] for q in parts]),pair_names=names,gene_names=genes)
subprocess.run([sys.executable,str(Path(__file__).with_name('plot_expression_export.py')),str(export),'--out',str(out)],check=True)
fig,axes=plt.subplots(1,2,figsize=(12,5))
for f in sorted(Path(cfg.checkpoints).glob(f'fold_*_{cfg.variant}.json')):
 r=json.loads(f.read_text());s=r['selection'];h=s.get('training_history',[]);v=s['validation_history'];label=f"Fold {r['fold']}"
 if h:axes[0].plot([x['epoch'] for x in h],[x['noise_mse'] for x in h],label=label)
 axes[1].plot([x['epoch'] for x in v],[x['validation_mse'] for x in v],label=label)
 axes[1].scatter(s['selected_epoch'],s['best_validation_mse'],marker='*',s=80)
for ax,title,yl in zip(axes,['Training objective','Validation prediction error'],['Noise MSE','Expression MSE']):ax.set_title(title);ax.set_ylabel(yl);ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.15)
fig.tight_layout();fig.savefig(out/'training_and_validation.png',dpi=200);fig.savefig(out/'training_and_validation.pdf')
agreements=[]
for f in files:
 a=f.with_suffix('.agreement.json');agreements.append(json.loads(a.read_text()) if a.exists() else {'matches_saved_pair_mse':None,'export':str(f)})
(out/'run_agreement.json').write_text(json.dumps({'n_pairs':len(names),'complete_101_pairs':len(names)==101,'agreements':agreements},indent=2))
print(f'Generated figures for {len(names)}/101 pairs. Read run_agreement.json before attaching these to the original score.')
