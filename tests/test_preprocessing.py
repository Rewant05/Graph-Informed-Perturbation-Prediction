"""Input safety and preservation tests; no GPU or biological claims."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad
from preprocessing import validate_counts, preprocess_counts, embed_exploratory

class PreprocessingTests(unittest.TestCase):
    def counts(self):
        values = np.random.default_rng(12).poisson(4, (40,12)).astype(float)
        return ad.AnnData(sparse.csr_matrix(values),
            obs=pd.DataFrame({'condition':['ctrl']*20+['A+B']*20, 'cell_type':['A549']*40},index=[f'c{i}' for i in range(40)]),
            var=pd.DataFrame({'gene_name':[f'G{i}' for i in range(12)]},index=[f'g{i}' for i in range(12)]))
    def test_reject_processed_counts(self):
        for bad in [np.array([[.2,1.1]]),np.array([[-1,2]]),np.array([[np.nan,1]])]:
            with self.assertRaises(ValueError): validate_counts(bad)
    def test_normalization_keeps_raw_and_annotations(self):
        a=self.counts(); before=a.X.toarray().copy(); b=preprocess_counts(a,qc=False)
        np.testing.assert_array_equal(a.X.toarray(),before)
        np.testing.assert_array_equal(b.layers['counts'].toarray(),before)
        self.assertTrue(b.obs['cell_type'].eq('A549').all())
        np.testing.assert_allclose(np.expm1(b.X.toarray()).sum(1),10000,rtol=1e-5)
        self.assertEqual(b.uns['study_preprocessing']['normalization'],'total_count_then_log1p')
    def test_exploration_preserves_expression(self):
        a=preprocess_counts(self.counts(),qc=False); before=a.X.toarray().copy()
        b=embed_exploratory(a,n_pcs=4,neighbors=5)
        np.testing.assert_array_equal(b.X.toarray(),before)
        self.assertEqual(b.obsm['X_umap'].shape,(40,2))
        self.assertEqual(b.uns['study_exploration']['fit_scope'],'all_supplied_cells')
    def test_batch_correction_requires_real_metadata(self):
        a=self.counts()
        with self.assertRaises(ValueError): embed_exploratory(a,batch_correction='harmony',batch_key='batch')
        a.obs['batch']='one'
        with self.assertRaises(ValueError): embed_exploratory(a,batch_correction='harmony',batch_key='batch')

    def test_harmony_keeps_expression_and_corrects_coordinates(self):
        import importlib.util
        if importlib.util.find_spec('harmonypy') is None:
            self.skipTest('Optional harmonypy not installed')
        a=preprocess_counts(self.counts(),qc=False)
        a.obs['batch']=['batch1','batch2']*20
        before=a.X.toarray().copy()
        b=embed_exploratory(a,n_pcs=4,neighbors=5,batch_correction='harmony',batch_key='batch')
        np.testing.assert_array_equal(b.X.toarray(),before)
        self.assertEqual(b.obsm['X_pca_harmony'].shape,(40,4))

    def test_raw_cli_roundtrip_and_overwrite_guard(self):
        import tempfile, contextlib, io, json, hashlib
        from preprocessing import main
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'raw.h5ad';out=Path(td)/'processed.h5ad'
            self.counts().write_h5ad(src);digest=hashlib.sha256(src.read_bytes()).hexdigest()
            args=['raw','--data',str(src),'--out',str(out),'--confirm-raw-counts','--skip-qc']
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(main(args),0)
            self.assertEqual(hashlib.sha256(src.read_bytes()).hexdigest(),digest)
            result=ad.read_h5ad(out)
            self.assertTrue(result.obs['cell_type'].eq('A549').all())
            report=json.loads(out.with_suffix('.preprocessing.json').read_text())
            self.assertFalse(report['eligible_for_existing_benchmark'])
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):main(args)

if __name__ == '__main__': unittest.main()
