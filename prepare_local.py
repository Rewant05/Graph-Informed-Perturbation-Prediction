"""Rebase only path metadata into ignored work/; preserve original evidence."""
import argparse,json,pickle,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--data',required=True,help='Existing perturb_processed.h5ad');cfg=p.parse_args()
data=Path(cfg.data).resolve()
if not data.is_file():raise SystemExit(f'Missing dataset: {data}')
work=ROOT/'work';local=work/'inputs';local.mkdir(parents=True,exist_ok=True)
for d in ['results','checkpoints','exports','figures']:(work/d).mkdir(exist_ok=True)
report=json.loads((ROOT/'evidence/splits/coverage_report.json').read_text());report['gears_data']=str(data)
for fold in report['folds']:
 f=fold['fold'];native=json.loads((ROOT/f'evidence/splits/gears_custom_fold{f}.json').read_text())
 path=local/f'gears_custom_fold{f}.pkl';path.write_bytes(pickle.dumps(native,protocol=4));fold['pickle']=str(path.resolve())
rp=local/'coverage_report.json';rp.write_text(json.dumps(report,indent=2))
b=json.loads((ROOT/'evidence/gears_pair_baseline_go.json').read_text());b['source']=str(data);b['split_report']=str(rp.resolve());bp=local/'baseline.json';bp.write_text(json.dumps(b,indent=2))
g=json.loads((ROOT/'evidence/gears_pair_result_go.json').read_text());g['data']=str(data.parent);g['split']=str(rp.resolve());g['baseline']=str(bp.resolve());(local/'gears_saved.json').write_text(json.dumps(g,indent=2))
h=hashlib.sha256()
with data.open('rb') as f:
 for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
(work/'local_config.json').write_text(json.dumps({'data':str(data),'dataset_sha256':h.hexdigest(),'dataset_size':data.stat().st_size},indent=2))
print('Local paths prepared; original evidence unchanged. Dataset fingerprint recorded.')
