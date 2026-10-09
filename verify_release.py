"""Verify source closure, frozen splits and saved evidence without training."""
import ast,json,math,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def require(ok,message):
 if not ok:raise ValueError(message)
def rows(d):
 r=d['per_pair'];require(len(r)==101,'Expected101pairs');require(len({x['pair'] for x in r})==101,'Duplicate pairs');return {x['pair']:x for x in r}
def close(a,b):return math.isclose(a,b,rel_tol=2e-10,abs_tol=1e-12)
def main():
 for p in ROOT.rglob('*.py'):
  if 'work' in p.relative_to(ROOT).parts:continue
  ast.parse(p.read_text(encoding='utf-8-sig'))
 report=load('evidence/splits/coverage_report.json');test={}
 for f in report['folds']:
  tr,va,te=map(set,[f['train_pairs'],f['validation_pairs'],f['test_pairs']]);require(not(tr&va or tr&te or va&te),'Overlapping pair outcomes')
  require(len(tr|va|te)==101,'Fold cohort differs')
  for p in te:require(p not in test,'Repeated test pair');test[p]=f['fold']
  native=load(f"evidence/splits/gears_custom_fold{f['fold']}.json")
  require(set(native)=={'train','val','test'},'Native split keys')
  require(not(set(native['train'])&set(native['val']) or set(native['train'])&set(native['test']) or set(native['val'])&set(native['test'])),'Native overlap')
  canonical=lambda labels:{'+'.join(sorted(x.split('+'))) for x in labels}
  require(canonical(native['test'])==te and canonical(native['val'])==va,'Native pair labels disagree')
 require(len(test)==101,'Incomplete test coverage')
 b=rows(load('evidence/gears_pair_baseline_go.json'));g=rows(load('evidence/gears_pair_result_go.json'));d=load('evidence/validation_string_seed42.json');dr=rows(d);pr=rows(load('evidence/gears_pair_graph_branch_matched.json'));ex=rows(load('evidence/visual_export_result.json'))
 for name,r in [('baseline',b),('GEARS',g),('diffusion',dr),('predictor',pr),('export rerun',ex)]:
  require(set(r)==set(test),f'{name}cohort mismatch');require(all(x['fold']==test[p] for p,x in r.items()),f'{name}fold mismatch')
 for p in test:
  for k in ['additive_mse','nearest_mse']:require(close(b[p][k],g[p][k]) and close(g[p][k],dr[p][k]),f'Baseline mismatch:{p}')
  require(close(dr[p]['gears_mse'],g[p]['gears_mse']),'GEARS mismatch')
  require(close(dr[p]['variants']['string:no_ot'],ex[p]['variants']['string:no_ot']),'Export rerun differs')
 for v in d['settings']['variants']:
  require(close(sum(r['variants'][v] for r in dr.values())/101,d['macro'][v]),f'Bad saved macro:{v}')
  for fold in range(1,6):
   cp=load(f"evidence/validation_checkpoints/fold_{fold}_{v.replace(':','_')}.json")
   require(set(cp['pair_mse'])=={p for p,f in test.items() if f==fold},'Checkpoint test set mismatch')
   require(all(close(score,dr[p]['variants'][v]) for p,score in cp['pair_mse'].items()),'Checkpoint score mismatch')
 expected={'string':.0012224435893630667,'no_graph':.0012333526632344301,'shuffled_string':.0012312186179153888}
 for method,value in expected.items():require(close(sum(r['scores'][method]['mse_all'] for r in pr.values())/101,value),f'Predictor:{method}')
 # Source-local dependency closure; third-party dependencies checked in the owner's environment.
 local={p.stem for p in (ROOT/'code').glob('*.py')}|{p.stem for p in (ROOT/'code/gears').glob('*.py')}
 for p in (ROOT/'code').rglob('*.py'):
  for node in ast.walk(ast.parse(p.read_text())):
   mods=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else []
   for m in mods:
    if m in local:require((p.parent/(m+'.py')).exists(),f'Missing local import {m} from {p}')
 # Uploaded fresh reports are checked against frozen references, per pair.
 comparisons=[('additive_nearest.json',b),('predictor.json',pr),('diffusion.json',dr),('diffusion-controls.json',dr)]
 for filename,ref in comparisons:
  fresh=rows(load('evidence/reproduction/'+filename))
  require(set(fresh)==set(ref),f'Reproduction cohort mismatch:{filename}')
  for pair,record in fresh.items():
   require(record['fold']==ref[pair]['fold'],f'Reproduction fold mismatch:{filename}')
   if filename=='predictor.json':
    for method,score in record['scores'].items():
     for metric,value in score.items():
      if isinstance(value,(int,float)):require(value==ref[pair]['scores'][method][metric],f'Predictor rerun differs:{pair}')
   elif filename.startswith('diffusion'):
    for variant,value in record['variants'].items():require(value==ref[pair]['variants'][variant],f'Diffusion rerun differs:{pair}')
   else:
    for key in ['additive_mse','nearest_mse']:require(record[key]==ref[pair][key],f'Baseline rerun differs:{pair}')
 audit=rows(load('evidence/gears_original_allfold_checkpoint_audit.json'))
 require(set(audit)==set(g),'GEARS audit cohort mismatch')
 max_difference=max(abs(audit[p]['gears_mse']-g[p]['gears_mse']) for p in g)
 require(max_difference<1e-8,'Original GEARS checkpoint rescoring differs')
 summary=load('evidence/reproduction/audit_summary.json')
 require(close(max_difference,summary['gears_max_pair_difference']),'Audit summary mismatch')
 for figure in ['architectures.svg','all_method_mse.png','four_method_comparison.png','training_and_validation.png','actual_vs_predicted.png','AHR_KLF1_OLS_only.png','response_heatmap.png','three_panel_umap.png']:
  require((ROOT/'figures'/figure).is_file() and (ROOT/'figures'/figure).stat().st_size>0,f'Missing figure:{figure}')
 if (ROOT/'MANIFEST.json').exists():
  manifest=load('MANIFEST.json')
  for name,sha in manifest['sha256'].items():require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,f'Changed release file:{name}')
 print('PASS: source/imports, 101 pairs/five folds, saved scores, exact uploaded baseline/predictor/diffusion reruns, original GEARS rescoring tolerance, figures and manifest.')
 print('Saved evidence check only: GPU training, raw dataset identity and dependency environment are not verified by this command.')
if __name__=='__main__':main()
