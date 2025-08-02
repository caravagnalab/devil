"""Module for differential expression testing with GPU support."""

# Prevent pytest from collecting functions in this module as tests.
__test__ = False

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


def _create_one_vs_rest_contrasts(
    n_coeffs: int,
    n_groups: int,
    group_variable_start_index: int = 1
) -> Dict[str, np.ndarray]:
    """
    Create one-vs-rest contrast vectors for categorical variables.
    
    Args:
        n_coeffs: Total number of coefficients in the model.
        n_groups: Total number of groups (including reference).
        group_variable_start_index: Index where group coefficients start (usually 1, after intercept).
        
    Returns:
        Dictionary mapping group names to contrast vectors.
    """
    contrasts = {}
    
    for target_group in range(n_groups):
        contrast = np.zeros(n_coeffs)
        
        if target_group == 0:
            # Reference group vs average of all others
            # Reference coefficient is implicit 0, so we set others negative
            other_weight = -1.0 / (n_groups - 1)
            for j in range(1, n_groups):
                if group_variable_start_index + j - 1 < n_coeffs:
                    contrast[group_variable_start_index + j - 1] = other_weight
                    
        else:
            # Non-reference group vs average of all others
            contrast[group_variable_start_index + target_group - 1] = 1.0
            
            # All other non-reference groups get negative weight
            other_weight = -1.0 / (n_groups - 1)
            for j in range(1, n_groups):
                if j != target_group and group_variable_start_index + j - 1 < n_coeffs:
                    contrast[group_variable_start_index + j - 1] = other_weight
        
        contrasts[f'group_{target_group}'] = contrast
    
    return contrasts


def _infer_categorical_structure(devil_fit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Infer the structure of categorical variables from the design matrix.
    
    This is a simple heuristic that assumes the most common case:
    - First coefficient is intercept
    - Remaining coefficients represent levels of a categorical variable
    
    Args:
        devil_fit: Fitted model dictionary.
        
    Returns:
        Dictionary with categorical structure information.
    """
    n_coeffs = devil_fit['beta'].shape[1]
    
    # Simple heuristic: assume single categorical variable after intercept
    return {
        'n_groups': n_coeffs,  # intercept + (n_groups - 1) coefficients
        'group_variable_start_index': 1,
        'structure': 'single_categorical'
    }


def test_de(
    devil_fit: Dict[str, Any],
    contrast: Union[np.ndarray, List[float], str] = None,
    pval_adjust_method: str = "fdr_bh",
    max_lfc: float = 10.0,
    clusters: Optional[Union[np.ndarray, List[Any]]] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    use_gpu: Optional[bool] = None,
    gpu_batch_size: Optional[int] = None,
    gpu_dtype: str = "float32",
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Test for differential expression with optional GPU acceleration.
    
    Supports both manual contrast specification and automatic one-vs-rest testing
    for categorical variables.
    
    Args:
        devil_fit: Fitted model dictionary from fit_devil().
        contrast: Either:
            - Array/list of contrast coefficients (traditional usage)
            - String "one-vs-rest" for automatic one-vs-rest comparisons
            - None (defaults to first coefficient if available)
        pval_adjust_method: Method for p-value adjustment.
        max_lfc: Maximum absolute log2 fold change to report.
        clusters: Sample cluster assignments for robust variance estimation.
        n_jobs: Number of parallel CPU jobs.
        verbose: Whether to print progress messages.
        use_gpu: Whether to use GPU acceleration. If None, uses same as fit_devil.
        gpu_batch_size: Batch size for GPU processing.
        gpu_dtype: Data type for GPU computation.
        
    Returns:
        If contrast is array/list: Single DataFrame with DE results.
        If contrast is "one-vs-rest": Dictionary with group names as keys and 
        DataFrames as values.
    """
    # Validate inputs
    if contrast is None:
        raise ValueError("Must provide either a contrast vector or 'one-vs-rest'")
    
    # Handle one-vs-rest case
    if isinstance(contrast, str) and contrast.lower() == "one-vs-rest":
        return _test_de_one_vs_rest(
            devil_fit=devil_fit,
            pval_adjust_method=pval_adjust_method,
            max_lfc=max_lfc,
            clusters=clusters,
            n_jobs=n_jobs,
            verbose=verbose,
            use_gpu=use_gpu,
            gpu_batch_size=gpu_batch_size,
            gpu_dtype=gpu_dtype
        )
    
    # Handle traditional contrast vector case
    contrast = np.asarray(contrast, dtype=np.float64)
    
    # Validate contrast length
    n_coeffs = devil_fit["beta"].shape[1]
    if len(contrast) != n_coeffs:
        raise ValueError(
            f"Contrast length ({len(contrast)}) must match number of "
            f"coefficients ({n_coeffs})"
        )
    
    # Determine GPU usage
    if use_gpu is None:
        use_gpu = devil_fit.get("use_gpu", False)
    
    if use_gpu and not is_gpu_available():
        warnings.warn("GPU requested but not available, falling back to CPU")
        use_gpu = False
    
    # Run the test
    if use_gpu:
        return _test_de_gpu(
            devil_fit, contrast, pval_adjust_method, max_lfc, clusters,
            gpu_batch_size, gpu_dtype, verbose
        )
    else:
        return _test_de_cpu(
            devil_fit, contrast, pval_adjust_method, max_lfc, clusters,
            n_jobs, verbose
        )


def _test_de_one_vs_rest(
    devil_fit: Dict[str, Any],
    pval_adjust_method: str = "fdr_bh",
    max_lfc: float = 10.0,
    clusters: Optional[Union[np.ndarray, List[Any]]] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    use_gpu: Optional[bool] = None,
    gpu_batch_size: Optional[int] = None,
    gpu_dtype: str = "float32",
) -> Dict[str, pd.DataFrame]:
    """
    Perform one-vs-rest differential expression testing.
    
    Args:
        devil_fit: Fitted model dictionary.
        pval_adjust_method: Method for p-value adjustment.
        max_lfc: Maximum absolute log2 fold change to report.
        clusters: Sample cluster assignments for robust variance estimation.
        n_jobs: Number of parallel CPU jobs.
        verbose: Whether to print progress messages.
        use_gpu: Whether to use GPU acceleration.
        gpu_batch_size: Batch size for GPU processing.
        gpu_dtype: Data type for GPU computation.
        
    Returns:
        Dictionary mapping group names to DataFrames with DE results.
    """
    # Infer categorical structure
    cat_info = _infer_categorical_structure(devil_fit)
    n_coeffs = devil_fit["beta"].shape[1]
    
    if verbose:
        print(f"Detected {cat_info['n_groups']} groups for one-vs-rest testing")
    
    # Create all contrast vectors
    contrasts = _create_one_vs_rest_contrasts(
        n_coeffs=n_coeffs,
        n_groups=cat_info['n_groups'],
        group_variable_start_index=cat_info['group_variable_start_index']
    )
    
    if verbose:
        print(f"Created {len(contrasts)} one-vs-rest contrasts")
        for name, contrast_vec in contrasts.items():
            print(f"  {name}: {contrast_vec}")
    
    # Run DE testing for each contrast
    results = {}
    
    for group_name, contrast_vector in tqdm(contrasts.items(), 
                                           desc="Testing groups", 
                                           disable=not verbose):
        if verbose:
            print(f"Testing {group_name} vs rest...")
        
        # Run single contrast test
        de_result = test_de(
            devil_fit=devil_fit,
            contrast=contrast_vector,
            pval_adjust_method=pval_adjust_method,
            max_lfc=max_lfc,
            clusters=clusters,
            n_jobs=n_jobs,
            verbose=False,  # Suppress individual test output
            use_gpu=use_gpu,
            gpu_batch_size=gpu_batch_size,
            gpu_dtype=gpu_dtype
        )
        
        results[group_name] = de_result
    
    if verbose:
        print(f"Completed one-vs-rest testing for {len(results)} groups")
        
        # Print summary statistics
        for group_name, df in results.items():
            n_sig = np.sum(df['padj'] < 0.05)
            n_up = np.sum((df['padj'] < 0.05) & (df['lfc'] > 0))
            n_down = np.sum((df['padj'] < 0.05) & (df['lfc'] < 0))
            print(f"  {group_name}: {n_sig} significant genes ({n_up} up, {n_down} down)")
    
    return results


def _test_de_cpu(
    devil_fit: Dict[str, Any],
    contrast: np.ndarray,
    pval_adjust_method: str,
    max_lfc: float,
    clusters: Optional[Union[np.ndarray, List[Any]]],
    n_jobs: Optional[int],
    verbose: bool
) -> pd.DataFrame:
    """CPU implementation of differential expression testing."""
    
    # Validate clusters if provided
    if clusters is not None:
        clusters = np.asarray(clusters)
        if len(clusters) != devil_fit["n_samples"]:
            raise ValueError(
                f"Clusters length ({len(clusters)}) must match number of "
                f"samples ({devil_fit['n_samples']})"
            )
        
        # Convert string clusters to numeric
        if clusters.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object
            unique_clusters = np.unique(clusters)
            cluster_map = {cluster: i for i, cluster in enumerate(unique_clusters)}
            clusters = np.array([cluster_map[c] for c in clusters])
    
    # Set up parallel processing
    if n_jobs is None:
        n_jobs = -1  # Use all available cores
    
    n_genes = devil_fit["n_genes"]
    gene_indices = np.arange(n_genes)
    
    # Calculate batch size for parallel processing
    if n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
    
    batch_size = max(1, n_genes // (n_jobs * 4))  # 4 batches per job
    batches = [gene_indices[i:i + batch_size] for i in range(0, n_genes, batch_size)]
    
    if verbose:
        print(f"Processing {n_genes} genes in {len(batches)} batches using {n_jobs} jobs")
    
    # Process batches in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_test_genes_cpu_batch)(devil_fit, contrast, batch, clusters)
        for batch in tqdm(batches, desc="Processing batches", disable=not verbose)
    )
    
    # Combine results
    all_pvals = np.concatenate([r[0] for r in results])
    all_ses = np.concatenate([r[1] for r in results])
    all_stats = np.concatenate([r[2] for r in results])
    
    # Calculate log fold changes
    lfcs = (devil_fit["beta"] @ contrast) / np.log(2)  # Convert to log2
    
    # Apply LFC capping
    lfcs = np.clip(lfcs, -max_lfc, max_lfc)
    
    # Multiple testing correction
    _, padj, _, _ = multipletests(all_pvals, method=pval_adjust_method)
    
    # Ensure all arrays are 1D for DataFrame construction
    gene_names = np.asarray(devil_fit["gene_names"]).flatten()
    lfcs = np.asarray(lfcs).flatten()
    all_ses = np.asarray(all_ses).flatten()
    all_stats = np.asarray(all_stats).flatten()
    all_pvals = np.asarray(all_pvals).flatten()
    padj = np.asarray(padj).flatten()
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'gene': gene_names,
        'lfc': lfcs,
        'se': all_ses,
        'stat': all_stats,
        'pval': all_pvals,
        'padj': padj
    })
    
    # Sort by adjusted p-value
    results_df = results_df.sort_values('padj').reset_index(drop=True)
    
    return results_df


def _test_de_gpu(
    devil_fit: Dict[str, Any],
    contrast: np.ndarray,
    pval_adjust_method: str,
    max_lfc: float,
    clusters: Optional[Union[np.ndarray, List[Any]]],
    gpu_batch_size: Optional[int],
    gpu_dtype: str,
    verbose: bool
) -> pd.DataFrame:
    """GPU implementation of differential expression testing (placeholder)."""
    
    # For now, fall back to CPU implementation
    # TODO: Implement actual GPU version
    warnings.warn("GPU implementation not yet available, using CPU")
    return _test_de_cpu(
        devil_fit, contrast, pval_adjust_method, max_lfc, clusters, None, verbose
    )


def _test_genes_cpu_batch(
    devil_fit: Dict[str, Any],
    contrast: np.ndarray,
    gene_indices: np.ndarray,
    clusters: Optional[np.ndarray]
) -> tuple:
    """Process a batch of genes for differential expression testing."""
    
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

        if subset_fit["n_genes"] == 0:
            return pd.DataFrame(columns=["gene", "pval", "padj", "lfc", "se", "stat"])
        
        # Copy other metadata if present
        for key in ["use_gpu", "gpu_batch_size"]:
            if key in devil_fit:
                subset_fit[key] = devil_fit[key]
        
        return test_de(subset_fit, contrast, **kwargs)
    else:
        return test_de(devil_fit, contrast, **kwargs)


test_de_memory_efficient.__test__ = False