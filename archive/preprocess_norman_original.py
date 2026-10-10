import scanpy as sc
import numpy as np
from scipy.stats import median_abs_deviation
import os

def preprocess_norman(input_path, output_path):
    print(f"Loading raw Norman dataset from {input_path}...")
    adata = sc.read_h5ad(input_path)
    
    print("Initial shape:", adata.shape)

    # 1. Basic QC Metrics
    print("Calculating QC metrics...")
    # Depending on the dataset, mitochondrial genes might be labeled differently.
    # We assume 'gene_symbol' or 'index' contains the names.
    if 'gene_symbol' in adata.var:
        adata.var["mt"] = adata.var["gene_symbol"].str.upper().str.startswith("MT-")
    else:
        adata.var["mt"] = adata.var.index.str.upper().str.startswith("MT-")
        
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    # Calculate log1p metrics for filtering
    adata.obs["log1p_total_counts"] = np.log1p(adata.obs["total_counts"])
    adata.obs["log1p_n_genes_by_counts"] = np.log1p(adata.obs["n_genes_by_counts"])

    # 2. MAD-Based Filtering (Simplified for speed)
    print("Applying MAD-based filtering...")
    def is_outlier(metric, nmads):
        M = adata.obs[metric]
        med = np.median(M)
        mad = median_abs_deviation(M)
        # Avoid division by zero if mad is 0
        if mad == 0:
            return np.zeros_like(M, dtype=bool)
        return (M < med - nmads * mad) | (M > med + nmads * mad)

    outlier = (
        is_outlier("log1p_total_counts", 5)
        | is_outlier("log1p_n_genes_by_counts", 5)
    )
    
    mt_outlier = (is_outlier("pct_counts_mt", 3) | (adata.obs["pct_counts_mt"] > 8))
    
    keep_mask = ~(outlier | mt_outlier)
    adata = adata[keep_mask].copy()
    print("Shape after cell filtering:", adata.shape)

    # 3. Gene Filtering
    print("Filtering lowly expressed genes...")
    sc.pp.filter_genes(adata, min_cells=3)
    print("Shape after gene filtering:", adata.shape)

    # 4. Normalization & Log1p
    print("Normalizing and applying log1p...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 5. Highly Variable Genes
    print("Selecting highly variable genes...")
    # Store the normalized matrix
    adata.layers["scran_normalization"] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, layer="scran_normalization")

    print(f"Final preprocessed shape: {adata.shape}")
    
    # Save the processed data
    print(f"Saving to {output_path}...")
    adata.write_h5ad(output_path)
    print("Done!")

if __name__ == "__main__":
    in_file = "../norman_raw.h5ad"
    out_file = "../norman_processed.h5ad"
    
    if os.path.exists(in_file):
        preprocess_norman(in_file, out_file)
    else:
        print(f"Error: {in_file} not found. Please run download_norman.py first.")
