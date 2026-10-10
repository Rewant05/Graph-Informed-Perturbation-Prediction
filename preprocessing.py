"""Separate frozen benchmark auditing from exploratory single-cell preprocessing.

The matched benchmark uses the distributed processed expression matrix unchanged.
Raw-count preprocessing creates a NEW exploratory dataset, not the scored input.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def validate_counts(matrix):
    """Reject negative, nonfinite or noninteger values before normalization."""
    import numpy as np
    from scipy import sparse
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Raw counts must be finite and nonnegative.')
    if not np.allclose(values, np.rint(values), rtol=0, atol=1e-6):
        raise ValueError('Noninteger values found; do not normalize processed/log expression as raw counts.')

def preprocess_counts(adata, *, target_sum=10000, min_cells=3, qc=True):
    """Copy, optionally QC/filter, preserve counts, and apply total-count/log1p.

    The caller must establish that the matrix contains raw counts. Integer values
    alone do not establish raw-count provenance.
    """
    import numpy as np
    import scanpy as sc
    from scipy.stats import median_abs_deviation
    validate_counts(adata.X)
    a = adata.copy()
    symbols = next((k for k in ('gene_name','gene_symbol') if k in a.var), None)
    names = a.var[symbols].astype(str) if symbols else a.var_names.astype(str)
    a.var['mt'] = names.str.upper().str.startswith('MT-')
    sc.pp.calculate_qc_metrics(a, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    if qc:
        def outlier(v, n):
            v = np.asarray(v); mad = median_abs_deviation(v)
            return np.abs(v - np.median(v)) > n * mad if mad > 0 else np.zeros(len(v),bool)
        reject = (outlier(np.log1p(a.obs.total_counts),5) |
                  outlier(np.log1p(a.obs.n_genes_by_counts),5) |
                  outlier(a.obs.pct_counts_mt,3) | (a.obs.pct_counts_mt > 8))
        a = a[~reject].copy()
    a = a[a.obs.total_counts > 0].copy()
    sc.pp.filter_genes(a, min_cells=min_cells)
    if min(a.shape) < 2:
        raise ValueError('Too few cells/genes remain after filtering.')
    a.layers['counts'] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=target_sum)
    sc.pp.log1p(a)
    a.layers['log1p_normalized'] = a.X.copy()
    a.uns['study_preprocessing'] = dict(
        purpose='exploratory_new_dataset_not_matched_benchmark',
        normalization='total_count_then_log1p', target_sum=float(target_sum),
        min_cells=int(min_cells), qc=bool(qc), raw_annotations_preserved=True)
    return a

def embed_exploratory(adata, *, n_pcs=50, neighbors=15, seed=42,
                      batch_correction='none', batch_key=None):
    """PCA/neighbours/UMAP for descriptive analysis; fits on all supplied cells.

    Optional Harmony corrects PCA coordinates, not expression X. This function
    is not used by the held-out benchmark or its train-only representations.
    """
    import scanpy as sc
    a = adata.copy()
    if batch_correction not in ('none','harmony'):
        raise ValueError('Batch correction must be none or harmony.')
    if batch_correction == 'harmony':
        if not batch_key or batch_key not in a.obs:
            raise ValueError('Harmony requires an existing --batch-key annotation.')
        if a.obs[batch_key].isna().any() or a.obs[batch_key].nunique() < 2:
            raise ValueError('Harmony requires at least two nonmissing batch labels.')
    components = min(n_pcs, a.n_obs-1, a.n_vars-1)
    if components < 2 or a.n_obs < 4:
        raise ValueError('Exploration requires at least four cells and three genes.')
    sc.pp.pca(a, n_comps=components, random_state=seed)
    representation = 'X_pca'
    if batch_correction == 'harmony':
        try:
            import harmonypy  # noqa: F401
        except ImportError as exc:
            raise ImportError('Install requirements-preprocessing.txt for Harmony.') from exc
        sc.external.pp.harmony_integrate(a, key=batch_key, nclust=max(2,min(50,a.n_obs//30)))
        representation = 'X_pca_harmony'
    sc.pp.neighbors(a, n_neighbors=min(neighbors,a.n_obs-1), use_rep=representation,
                    random_state=seed)
    sc.tl.umap(a, random_state=seed)
    a.uns['study_exploration'] = dict(
        purpose='descriptive_not_prediction_validation', fit_scope='all_supplied_cells',
        n_pcs=int(components), seed=int(seed), batch_correction=batch_correction,
        batch_key=batch_key or '', expression_matrix_corrected=False)
    return a

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    benchmark = sub.add_parser('benchmark',help='Audit existing matched input without normalization.')
    benchmark.add_argument('--data',required=True)
    for command in ('raw','explore'):
        q = sub.add_parser(command)
        q.add_argument('--data',required=True); q.add_argument('--out',required=True)
        q.add_argument('--counts-layer',help='Raw-count layer; available only with raw counts.')
        q.add_argument('--confirm-raw-counts',action='store_true')
        q.add_argument('--target-sum',type=float,default=10000)
        q.add_argument('--min-cells',type=int,default=3)
        q.add_argument('--skip-qc',action='store_true')
        if command == 'explore':
            q.add_argument('--input-kind',choices=['processed','raw-counts'],required=True)
            q.add_argument('--n-pcs',type=int,default=50)
            q.add_argument('--neighbors',type=int,default=15)
            q.add_argument('--seed',type=int,default=42)
            q.add_argument('--batch-correction',choices=['none','harmony'],default='none')
            q.add_argument('--batch-key')
    args = p.parse_args(argv)
    if args.command == 'benchmark':
        return subprocess.run([sys.executable,str(ROOT/'check_dataset.py'),'--data',args.data],check=True).returncode
    data, out = Path(args.data).resolve(), Path(args.out).resolve()
    if data == out or out.exists():
        p.error('Use a new output path; input/existing outputs are never overwritten.')
    raw = args.command == 'raw' or args.input_kind == 'raw-counts'
    if raw and not args.confirm_raw_counts:
        p.error('Raw processing requires --confirm-raw-counts and verified source provenance.')
    if not raw and (args.counts_layer or args.confirm_raw_counts or args.skip_qc or
                    args.target_sum != 10000 or args.min_cells != 3):
        p.error('Raw-count options are not applicable to processed expression.')
    if args.target_sum <= 0 or args.min_cells < 1:
        p.error('target-sum and min-cells must be positive.')
    if args.command == 'explore' and (args.n_pcs < 2 or args.neighbors < 2):
        p.error('n-pcs and neighbors must be at least two.')
    import scanpy as sc
    a = sc.read_h5ad(data)
    initial_shape = list(a.shape)
    if raw:
        if args.counts_layer:
            if args.counts_layer not in a.layers:p.error('Counts layer not found.')
            a.X = a.layers[args.counts_layer].copy()
        a = preprocess_counts(a,target_sum=args.target_sum,min_cells=args.min_cells,qc=not args.skip_qc)
    if args.command == 'explore':
        a = embed_exploratory(a,n_pcs=args.n_pcs,neighbors=args.neighbors,seed=args.seed,
                             batch_correction=args.batch_correction,batch_key=args.batch_key)
    out.parent.mkdir(parents=True,exist_ok=True)
    a.write_h5ad(out)
    report = dict(command=args.command,source=str(data),output=str(out),initial_shape=initial_shape,
                  final_shape=list(a.shape),benchmark_input_unchanged=True,
                  eligible_for_existing_benchmark=False,
                  preprocessing=dict(a.uns.get('study_preprocessing',{})),
                  exploration=dict(a.uns.get('study_exploration',{})))
    out.with_suffix('.preprocessing.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
