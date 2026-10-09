"""Read-only metadata/gene-axis check against the archived expression export."""
import argparse,json
from pathlib import Path
import anndata as ad
import numpy as np
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--data',required=True);cfg=p.parse_args()
a=ad.read_h5ad(cfg.data,backed='r')
try:
 if 'gene_name' not in a.var or 'condition' not in a.obs:raise ValueError('Required gene/condition metadata missing')
 genes=a.var['gene_name'].astype(str).to_numpy();q=np.load(ROOT/'evidence/expression_export.npz',allow_pickle=False)
 if not np.array_equal(genes,q['gene_names']):raise ValueError('Dataset gene names/order differ from saved expression export')
 from collections import Counter
 conditions=Counter(a.obs['condition'].astype(str));cellcols=[c for c in a.obs if 'cell' in c.lower() or 'type' in c.lower()]
 annotations={c:a.obs[c].astype(str).value_counts().head(10).to_dict() for c in cellcols}
 result={'shape':list(a.shape),'n_conditions':len(conditions),'gene_axis_matches_saved_export':True,'cell_annotations':annotations,'note':'Metadata match does not independently establish biological dataset identity. Preserve raw annotations.'}
 print(json.dumps(result,indent=2))
finally:a.file.close()
