"""Module for differential expression testing."""

from typing import Optional, Union, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from tqdm import tqdm

from .variance import compute_hessian, compute_sandwich_estimator


def test_de(
    devil_fit: Dict[str, Any],
    contrast: Union[np.ndarray, List[float]],
    pval_adjust_method: str = "fdr_bh",
    max_lfc: float = 10.0,
    clusters: Optional[Union[np.ndarray, List[Any]]] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Test for differential expression using fitted devil model.
    
    Performs statistical testing with support for standard and robust (clustered)
    variance estimation, multiple testing correction, and fold change thresholding.
    
    Args:
        devil_fit: Fitted model dictionary from fit_devil().
        contrast: Contrast vector specifying comparison of interest.
            Length must match number of coefficients. For example,
            [0, 1, -1] tests difference between second and third coefficient.
        pval_adjust_method: Method for p-value adjustment. Options:
            'fdr_bh' (Benjamini-Hochberg), 'bonferroni', 'holm', 'holm-sidak',
            'simes-hochberg', 'hommel', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'.
        max_lfc: Maximum absolute log2 fold change to report.
            Larger values are capped at ±max_lfc.
        clusters: Sample cluster assignments for robust variance estimation.
            If provided, uses sandwich estimator. Length must match n_samples.
        n_jobs: Number of parallel jobs. If None, uses all available cores.
        verbose: Whether to print progress messages.
        
    Returns:
        DataFrame with columns:
            - gene: Gene identifiers
            - pval: Raw p-values
            - padj: Adjusted p-values
            - lfc: Log2 fold changes (capped at ±max_lfc)
            - se: Standard errors
            - stat: Test statistics
            
    Raises:
        ValueError: If contrast length doesn't match coefficient dimensions.
        
    Examples:
        >>> # Basic differential expression test
        >>> results = test_de(fit, contrast=[0, 1, -1])
        
        >>> # With sample clustering for robust inference
        >>> results = test_de(fit, contrast=[0, 1, -1], clusters=patient_ids)
    """
    # Extract necessary information
    beta = devil_fit["beta"]
    overdispersion = devil_fit["overdispersion"]
    design_matrix = devil_fit["design_matrix"]
    count_matrix = devil_fit["count_matrix"]
    size_factors = devil_fit["size_factors"]
    gene_names = devil_fit["gene_names"]
    n_genes = devil_fit["n_genes"]
    n_samples = devil_fit["n_samples"]
    
    # Validate contrast
    contrast = np.asarray(contrast)
    if contrast.shape[0] != beta.shape[1]:
        raise ValueError(
            f"Contrast length ({contrast.shape[0]}) must match "
            f"number of coefficients ({beta.shape[1]})"
        )
    
    # Calculate log fold changes
    lfcs = beta @ contrast
    
    # Handle clusters if provided
    if clusters is not None:
        clusters = np.asarray(clusters)
        if len(clusters) != n_samples:
            raise ValueError(
                f"Clusters length ({len(clusters)}) must match "
                f"number of samples ({n_samples})"
            )
        # Convert to numeric if needed
        if not np.issubdtype(clusters.dtype, np.number):
            _, clusters = np.unique(clusters, return_inverse=True)
            clusters = clusters + 1  # 1-indexed like R
    
    # Set up parallel processing
    if n_jobs is None:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
    
    # Define test function for single gene
    def test_gene(gene_idx):
        """Perform statistical test for single gene."""
        lfc = lfcs[gene_idx]
        
        if clusters is not None:
            # Use sandwich estimator for clustered data
            H = compute_sandwich_estimator(
                design_matrix,
                count_matrix[gene_idx, :],
                beta[gene_idx, :],
                overdispersion[gene_idx],
                size_factors,
                clusters
            )
        else:
            # Use standard Hessian
            H = compute_hessian(
                beta[gene_idx, :],
                1.0 / overdispersion[gene_idx] if overdispersion[gene_idx] > 0 else 1e6,
                count_matrix[gene_idx, :],
                design_matrix,
                size_factors
            )
        
        # Calculate variance
        variance = contrast.T @ H @ contrast
        se = np.sqrt(variance)
        
        # Calculate test statistic and p-value
        stat = lfc / se if se > 0 else 0
        # Use t-distribution with n_samples - rank(design_matrix) degrees of freedom
        df = n_samples - np.linalg.matrix_rank(design_matrix)
        pval = 2 * stats.t.sf(np.abs(stat), df)
        
        return pval, se, stat
    
    # Run tests in parallel
    if verbose:
        print(f"Testing {n_genes} genes using {n_jobs} cores...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(test_gene)(i) for i in tqdm(range(n_genes), desc="Testing genes")
        )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(test_gene)(i) for i in range(n_genes)
        )
    
    # Extract results
    pvals = np.array([r[0] for r in results])
    ses = np.array([r[1] for r in results])
    stats = np.array([r[2] for r in results])
    
    # Adjust p-values
    if verbose:
        print(f"Adjusting p-values using {pval_adjust_method} method...")
    _, padjs, _, _ = multipletests(pvals, method=pval_adjust_method)
    
    # Convert to log2 fold changes
    log2_lfcs = lfcs / np.log(2)
    
    # Apply fold change capping
    log2_lfcs = np.clip(log2_lfcs, -max_lfc, max_lfc)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        "gene": gene_names,
        "pval": pvals,
        "padj": padjs,
        "lfc": log2_lfcs,
        "se": ses,
        "stat": stats
    })
    
    # Sort by adjusted p-value
    results_df = results_df.sort_values("padj")
    
    # Report summary
    if verbose:
        n_sig = np.sum(padjs < 0.05)
        n_up = np.sum((padjs < 0.05) & (log2_lfcs > 0))
        n_down = np.sum((padjs < 0.05) & (log2_lfcs < 0))
        print(f"\nDifferential expression summary:")
        print(f"  Total significant genes (padj < 0.05): {n_sig}")
        print(f"  Upregulated: {n_up}")
        print(f"  Downregulated: {n_down}")
    
    return results_df