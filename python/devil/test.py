"""Module for differential expression testing with GPU support."""

from typing import Optional, Union, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

from .variance import compute_hessian, compute_sandwich_estimator
from .gpu import is_gpu_available, GPUMemoryManager, get_gpu_memory_info
from .variance_gpu import compute_variance_batch_gpu


def test_de(
    devil_fit: Dict[str, Any],
    contrast: Union[np.ndarray, List[float]],
    pval_adjust_method: str = "fdr_bh",
    max_lfc: float = 10.0,
    clusters: Optional[Union[np.ndarray, List[Any]]] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    use_gpu: Optional[bool] = None,
    gpu_batch_size: Optional[int] = None,
    gpu_dtype: str = "float32",
) -> pd.DataFrame:
    """
    Test for differential expression with optional GPU acceleration.
    
    Performs statistical testing with support for standard and robust (clustered)
    variance estimation, multiple testing correction, and GPU acceleration.
    
    Args:
        devil_fit: Fitted model dictionary from fit_devil().
        contrast: Contrast vector specifying comparison of interest.
        pval_adjust_method: Method for p-value adjustment.
        max_lfc: Maximum absolute log2 fold change to report.
        clusters: Sample cluster assignments for robust variance estimation.
        n_jobs: Number of parallel CPU jobs.
        verbose: Whether to print progress messages.
        use_gpu: Whether to use GPU acceleration. If None, uses same as fit_devil.
        gpu_batch_size: Batch size for GPU processing. If None, uses same as fit_devil.
        gpu_dtype: Data type for GPU computation.
        
    Returns:
        DataFrame with differential expression results.
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
    
    # Use GPU settings from fit_devil if not specified
    if use_gpu is None:
        use_gpu = devil_fit.get("use_gpu", False)
    
    if gpu_batch_size is None:
        gpu_batch_size = devil_fit.get("gpu_batch_size", 1024)
    
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
            clusters = clusters + 1  # 1-indexed
    
    # Check GPU feasibility for testing
    gpu_feasible = False
    if use_gpu:
        gpu_feasible = is_gpu_available()
        if gpu_feasible and verbose:
            free_mem, total_mem = get_gpu_memory_info()
            print(f"Using GPU for testing. Memory: {free_mem/1e9:.1f}GB free")
        elif not gpu_feasible and verbose:
            print("GPU not available, using CPU for testing")
    
    if verbose:
        print(f"Testing {n_genes} genes using {'GPU' if gpu_feasible else 'CPU'}")
        if clusters is not None:
            print(f"Using clustered variance estimation with {len(np.unique(clusters))} clusters")
    
    # Perform statistical testing
    if gpu_feasible and use_gpu:
        pvals, ses, stats_vals = _test_de_gpu(
            devil_fit, contrast, clusters, gpu_batch_size,
            gpu_dtype, verbose
        )
    else:
        pvals, ses, stats_vals = _test_de_cpu(
            devil_fit, contrast, clusters, n_jobs, verbose
        )
    
    # Adjust p-values
    if verbose:
        print(f"Adjusting p-values using {pval_adjust_method} method...")
    _, padjs, _, _ = multipletests(pvals, method=pval_adjust_method)
    
    # Convert to log2 fold changes and apply capping
    log2_lfcs = lfcs / np.log(2)
    log2_lfcs = np.clip(log2_lfcs, -max_lfc, max_lfc)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        "gene": gene_names,
        "pval": pvals,
        "padj": padjs,
        "lfc": log2_lfcs,
        "se": ses,
        "stat": stats_vals
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


def _test_de_gpu(
    devil_fit: Dict[str, Any],
    contrast: np.ndarray,
    clusters: Optional[np.ndarray],
    batch_size: int,
    gpu_dtype: str,
    verbose: bool
) -> tuple:
    """Perform differential expression testing using GPU acceleration."""
    n_genes = devil_fit["n_genes"]
    n_samples = devil_fit["n_samples"]
    lfcs = devil_fit["beta"] @ contrast
    
    # Convert dtype
    dtype = np.float32 if gpu_dtype == "float32" else np.float64
    
    # Process in batches
    pvals_results = []
    ses_results = []
    stats_results = []
    
    with GPUMemoryManager():
        for start_idx in tqdm(range(0, n_genes, batch_size),
                             desc="GPU testing", disable=not verbose):
            end_idx = min(start_idx + batch_size, n_genes)
            gene_indices = np.arange(start_idx, end_idx)
            
            try:
                # Compute variances for batch
                variances, standard_errors = compute_variance_batch_gpu(
                    devil_fit, gene_indices, contrast, clusters, dtype
                )
                
                # Calculate test statistics and p-values
                batch_lfcs = lfcs[start_idx:end_idx]
                test_stats = batch_lfcs / standard_errors
                
                # Use t-distribution
                df = n_samples - np.linalg.matrix_rank(devil_fit["design_matrix"])
                batch_pvals = 2 * stats.t.sf(np.abs(test_stats), df)
                
                pvals_results.append(batch_pvals)
                ses_results.append(standard_errors)
                stats_results.append(test_stats)
                
            except Exception as e:
                if verbose:
                    print(f"GPU testing batch {start_idx}-{end_idx} failed: {e}")
                    print("Falling back to CPU for this batch")
                
                # CPU fallback for this batch
                batch_pvals, batch_ses, batch_stats = _test_genes_cpu_batch(
                    devil_fit, gene_indices, contrast, clusters
                )
                
                pvals_results.append(batch_pvals)
                ses_results.append(batch_ses)
                stats_results.append(batch_stats)
    
    # Combine results
    pvals = np.concatenate(pvals_results)
    ses = np.concatenate(ses_results)
    stats_vals = np.concatenate(stats_results)
    
    return pvals, ses, stats_vals


def _test_de_cpu(
    devil_fit: Dict[str, Any],
    contrast: np.ndarray,
    clusters: Optional[np.ndarray],
    n_jobs: Optional[int],
    verbose: bool
) -> tuple:
    """Perform differential expression testing using CPU parallelization."""
    n_genes = devil_fit["n_genes"]
    n_samples = devil_fit["n_samples"]
    
    # Set up parallel processing
    if n_jobs is None:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
    
    # Define test function for single gene
    def test_gene(gene_idx):
        """Perform statistical test for single gene."""
        lfc = (devil_fit["beta"][gene_idx] @ contrast)
        
        if clusters is not None:
            # Use sandwich estimator for clustered data
            H = compute_sandwich_estimator(
                devil_fit["design_matrix"],
                devil_fit["count_matrix"][gene_idx, :],
                devil_fit["beta"][gene_idx, :],
                devil_fit["overdispersion"][gene_idx],
                devil_fit["size_factors"],
                clusters
            )
        else:
            # Use standard Hessian
            precision = 1.0 / devil_fit["overdispersion"][gene_idx] if devil_fit["overdispersion"][gene_idx] > 0 else 1e6
            H = compute_hessian(
                devil_fit["beta"][gene_idx, :],
                precision,
                devil_fit["count_matrix"][gene_idx, :],
                devil_fit["design_matrix"],
                devil_fit["size_factors"]
            )
        
        # Calculate variance and standard error
        variance = contrast.T @ H @ contrast
        se = np.sqrt(np.maximum(variance, 1e-12))
        
        # Calculate test statistic and p-value
        stat = lfc / se if se > 0 else 0
        df = n_samples - np.linalg.matrix_rank(devil_fit["design_matrix"])
        pval = 2 * stats.t.sf(np.abs(stat), df)
        
        return pval, se, stat
    
    # Run tests in parallel
    if verbose:
        results = Parallel(n_jobs=n_jobs)(
            delayed(test_gene)(i) for i in tqdm(range(n_genes), desc="CPU testing")
        )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(test_gene)(i) for i in range(n_genes)
        )
    
    # Extract results
    pvals = np.array([r[0] for r in results])
    ses = np.array([r[1] for r in results])
    stats_vals = np.array([r[2] for r in results])
    
    return pvals, ses, stats_vals


def _test_genes_cpu_batch(
    devil_fit: Dict[str, Any],
    gene_indices: np.ndarray,
    contrast: np.ndarray,
    clusters: Optional[np.ndarray]
) -> tuple:
    """Test a batch of genes using CPU (fallback for GPU failures)."""
    n_samples = devil_fit["n_samples"]
    batch_size = len(gene_indices)
    
    pvals = np.zeros(batch_size)
    ses = np.zeros(batch_size)
    stats_vals = np.zeros(batch_size)
    
    for i, gene_idx in enumerate(gene_indices):
        lfc = (devil_fit["beta"][gene_idx] @ contrast)
        
        if clusters is not None:
            H = compute_sandwich_estimator(
                devil_fit["design_matrix"],
                devil_fit["count_matrix"][gene_idx, :],
                devil_fit["beta"][gene_idx, :],
                devil_fit["overdispersion"][gene_idx],
                devil_fit["size_factors"],
                clusters
            )
        else:
            precision = 1.0 / devil_fit["overdispersion"][gene_idx] if devil_fit["overdispersion"][gene_idx] > 0 else 1e6
            H = compute_hessian(
                devil_fit["beta"][gene_idx, :],
                precision,
                devil_fit["count_matrix"][gene_idx, :],
                devil_fit["design_matrix"],
                devil_fit["size_factors"]
            )
        
        variance = contrast.T @ H @ contrast
        se = np.sqrt(np.maximum(variance, 1e-12))
        stat = lfc / se if se > 0 else 0
        
        df = n_samples - np.linalg.matrix_rank(devil_fit["design_matrix"])
        pval = 2 * stats.t.sf(np.abs(stat), df)
        
        pvals[i] = pval
        ses[i] = se
        stats_vals[i] = stat
    
    return pvals, ses, stats_vals


def test_de_memory_efficient(
    devil_fit: Dict[str, Any],
    contrast: Union[np.ndarray, List[float]],
    gene_subset: Optional[Union[List[str], np.ndarray]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Memory-efficient version of test_de for very large datasets.
    
    Processes genes in smaller chunks to avoid memory issues.
    
    Args:
        devil_fit: Fitted model dictionary.
        contrast: Contrast vector.
        gene_subset: Optional subset of genes to test (by name or index).
        **kwargs: Other arguments passed to test_de.
        
    Returns:
        DataFrame with differential expression results.
    """
    if gene_subset is not None:
        # Create subset of the fit object
        if isinstance(gene_subset[0], str):
            # Gene names provided
            gene_mask = np.isin(devil_fit["gene_names"], gene_subset)
        else:
            # Gene indices provided
            gene_mask = np.zeros(devil_fit["n_genes"], dtype=bool)
            gene_mask[gene_subset] = True
        
        # Create subset fit object
        subset_fit = {
            "beta": devil_fit["beta"][gene_mask],
            "overdispersion": devil_fit["overdispersion"][gene_mask],
            "design_matrix": devil_fit["design_matrix"],
            "count_matrix": devil_fit["count_matrix"][gene_mask],
            "size_factors": devil_fit["size_factors"],
            "gene_names": devil_fit["gene_names"][gene_mask],
            "n_genes": np.sum(gene_mask),
            "n_samples": devil_fit["n_samples"],
        }
        
        # Copy other metadata if present
        for key in ["use_gpu", "gpu_batch_size"]:
            if key in devil_fit:
                subset_fit[key] = devil_fit[key]
        
        return test_de(subset_fit, contrast, **kwargs)
    else:
        return test_de(devil_fit, contrast, **kwargs)
