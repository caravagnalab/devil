#!/usr/bin/env python3
import subprocess
import os
import numpy as np
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
import patsy
import devil

# create a design matrix
def create_design_matrix_patsy(adata):
    """Create design matrix using patsy (R-like formula interface)."""
    
    # Extract metadata 
    metadata = adata.obs.copy()
    
    # Create design matrix using patsy formula
    # "~ label" creates intercept + dummy variables for label
    design_matrix = patsy.dmatrix("~ label", data=metadata, return_type='dataframe')
    
    print("Design matrix shape:", design_matrix.shape)
    print("Design matrix columns:", design_matrix.columns.tolist())
    
    return design_matrix.values

# function to run `git` to find the base path of the repo
def get_repo_base_path():
    # run `git rev-parse --show-toplevel`
    result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True)
    return result.stdout.strip()

def main():
    # read data
    print("Reading data...")
    base_path = get_repo_base_path()
    infile = os.path.join(base_path, "python/notebooks/tmp/baron_pancreas_full.h5ad")
    adata = sc.read_h5ad(infile)
    ## data stats
    print(f"Number of genes: {adata.n_vars}")
    print(f"Number of cells: {adata.n_obs}")

    # Filter top 3 cell types
    print("Filtering top 3 cell types...")
    top_3_ct = adata.obs['label'].value_counts().head(3).index.tolist()
    cell_filter = adata.obs['label'].isin(top_3_ct)
    adata = adata[cell_filter, :].copy()
    adata.write_h5ad("tmp/baron_pancreas_full_top3.h5ad")
    
    # Filter low-expression genes
    print("Filtering low-expression genes...")
    gene_counts = np.array(adata.X.sum(axis=0)).flatten()
    gene_filter = gene_counts > 500
    adata = adata[:, gene_filter].copy()
    ## data stats
    print(f"Number of genes: {adata.n_vars}")
    print(f"Number of cells: {adata.n_obs}")

    # Create design matrix
    print("Creating design matrix...")
    design_matrix = create_design_matrix_patsy(adata)

    # fit devil
    print("Fitting devil...")
    fit = devil.fit_devil(
        adata,
        design_matrix,
        overdispersion = True,
        size_factors = True, 
        init_overdispersion = 100,
        verbose = True,
        n_jobs = 1, 
        offset = 1e-6
    )
    print(fit)

    
if __name__ == "__main__":
    main()
    