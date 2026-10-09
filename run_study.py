"""Run one of the four study components from a prepared folder."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('task',choices=['preflight','baseline','predictor','diffusion','diffusion-controls','gears','figures','data-figures']);p.add_argument('--fold',type=int,choices=range(1,6),default=1);p.add_argument('--all-folds',action='store_true');p.add_argument('--device',default='cuda:0');p.add_argument('--gears-results',help='Optional newly retrained GEARS JSON on the same local report');cfg=p.parse_args()
conf=ROOT/'work/local_config.json'
if not conf.exists():raise SystemExit('Run prepare_local.py --data PATH first.')
data=Path(json.loads(conf.read_text())['data']);local=ROOT/'work/inputs';results=ROOT/'work/results';graphs=ROOT/'graphs'
common=[str(data),str(local/'coverage_report.json'),str(local/'baseline.json'),str(Path(cfg.gears_results).resolve() if cfg.gears_results else local/'gears_saved.json'),str(graphs/'string_balanced1.json'),str(graphs/'k562_hic_mapping_v2.json')]
fold=['--all-folds'] if cfg.all_folds else ['--fold',str(cfg.fold)]
def run(script,args):subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],check=True,cwd=ROOT)
if cfg.task=='preflight':run('code/run_validation_selected.py',common+['--preflight'])
elif cfg.task=='baseline':run('code/gears_matched_simple_baseline.py',[data,local/'coverage_report.json','--out',results/'additive_nearest.json'])
elif cfg.task=='predictor':run('code/gears_pair_graph_branch_matched.py',common+['--out',results/'predictor.json'])
elif cfg.task in ['diffusion','diffusion-controls']:
 variants='string:no_ot' if cfg.task=='diffusion' else 'no_edges:no_ot,shuffled_string:no_ot,string:no_ot'
 # Separate outputs avoid confusing control runs with STRING-only figure exports.
 tag=cfg.task
 run('code/run_validation_selected.py',common+fold+['--variants',variants,'--device',cfg.device,'--seed','42','--epochs','40','--vae-epochs','8','--eval-every','2','--patience','6','--checkpoint-dir',ROOT/f'work/checkpoints/{tag}','--export-dir',ROOT/f'work/exports/{tag}','--reference-dir',ROOT/'evidence/validation_checkpoints','--out',results/f'{tag}.json'])
elif cfg.task=='gears':run('code/gears/gears_pair_resumable.py',[data.parent,local/'coverage_report.json',local/'baseline.json',*fold,'--epochs','20','--batch-size','32','--device',cfg.device,'--checkpoint-dir',ROOT/'work/checkpoints/gears','--out',results/'gears_retrained.json'])
elif cfg.task=='figures':run('code/make_inference_figures.py',['--exports',ROOT/'work/exports/diffusion','--checkpoints',ROOT/'work/checkpoints/diffusion','--out',ROOT/'work/figures'])
elif cfg.task=='data-figures':run('code/make_data_figures.py',[data,local/'coverage_report.json','--exports',ROOT/'work/exports/diffusion','--fold',cfg.fold,'--out',ROOT/'work/figures'])
