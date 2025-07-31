"""Main module for fitting the devil statistical model with GPU support."""

from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad
from joblib import Parallel, delayed
import warnings
from tqdm import tqdm

from .beta import init_beta, fit_beta_coefficients
from .overdispersion import estimate_initial_dispersion, fit_dispersion
from .size_factors import calculate_size_factors, compute_offset_vector
from .utils import handle_input_data, validate_inputs
from .gpu import (
    is_gpu_available, check_gpu_requirements, estimate_batch_size,
    GPUMemoryManager, get_gpu_memory_info
)
from .beta_gpu import (
    init_beta_gpu, fit_beta_coefficients_gpu_batch,
    fit_beta_coefficients_gpu_vectorized, select_gpu_implementation
)
from .overdispersion_gpu import (
    estimate_initial_dispersion_gpu, fit_dispersion_gpu_batch,
    select_dispersion_implementation
)


def fit_devil(
    adata: Union[ad.AnnData, np.ndarray, sparse.spmatrix],
    design_matrix: Optional[np.ndarray] = None,
    design_formula: Optional[str] = None,
    overdispersion: bool = True,
    init_overdispersion: Optional[float] = None,
    do_cox_reid_adjustment: bool = True,
    offset: float = 1e-6,
    size_factors: bool = True,
    verbose: bool = False,
    max_iter: int = 100,
    tolerance: float = 1e-3,
    n_jobs: Optional[int] = None,
    layer: Optional[str] = None,
    use_gpu: Optional[bool] = None,
    gpu_batch_size: Optional[int] = None,
    gpu_dtype: str = "float32",
) -> Dict[str, Any]:
    """
    Fit statistical model for count data with optional GPU acceleration.
    
    Fits a negative binomial regression model with support for overdispersion estimation,
    size factor normalization, and parallel processing on CPU or GPU.
    
    Args:
        adata: Input data as AnnData object, numpy array (genes × samples), or sparse matrix.
        design_matrix: Design matrix of predictor variables (samples × predictors).
        design_formula: Formula string for design matrix construction (e.g., "~ condition + batch").
        overdispersion: Whether to estimate overdispersion parameter.
        init_overdispersion: Initial value for overdispersion.
        do_cox_reid_adjustment: Whether to apply Cox-Reid adjustment.
        offset: Value added to counts to avoid numerical issues.
        size_factors: Whether to compute normalization factors.
        verbose: Whether to print progress messages.
        max_iter: Maximum iterations for optimization.
        tolerance: Convergence criterion.
        n_jobs: Number of parallel CPU jobs.
        layer: If adata is AnnData, which layer to use.
        use_gpu: Whether to use GPU acceleration. If None, auto-detects.
        gpu_batch_size: Batch size for GPU processing. If None, auto-estimates.
        gpu_dtype: Data type for GPU computation ('float32' or 'float64').
        
    Returns:
        Dictionary containing fitted model parameters and metadata.
    """
    # Handle input data
    count_matrix, gene_names, sample_names, obs_df = handle_input_data(
        adata, layer=layer
    )
    
    # Construct design matrix if needed
    if design_matrix is None:
        if design_formula is None:
            raise ValueError("Must provide either design_matrix or design_formula")
        if obs_df is None:
            raise ValueError("design_formula requires AnnData input with .obs")
        
        import patsy
        design_matrix = patsy.dmatrix(design_formula, obs_df, return_type="numpy")
        if verbose:
            print(f"Created design matrix from formula: {design_formula}")
    
    # Validate inputs
    validate_inputs(count_matrix, design_matrix)
    
    n_genes, n_samples = count_matrix.shape
    n_features = design_matrix.shape[1]
    
    # Determine GPU usage
    if use_gpu is None:
        use_gpu = is_gpu_available()
        if use_gpu and verbose:
            print("GPU detected and will be used automatically")
    
    # Check GPU requirements if requested
    gpu_feasible = False
    if use_gpu:
        gpu_feasible, gpu_message = check_gpu_requirements(
            n_genes, n_samples, n_features, verbose=verbose
        )
        if not gpu_feasible:
            if verbose:
                print(f"GPU not feasible: {gpu_message}")
                print("Falling back to CPU computation")
            use_gpu = False
        elif verbose:
            print(f"GPU computation: {gpu_message}")
    
    # Set up GPU parameters
    if use_gpu and gpu_feasible:
        gpu_dtype_np = np.float32 if gpu_dtype == "float32" else np.float64
        
        if gpu_batch_size is None:
            gpu_batch_size = estimate_batch_size(
                n_genes, n_samples, n_features, 
                dtype=gpu_dtype_np, memory_fraction=0.7
            )
        
        if verbose:
            print(f"GPU batch size: {gpu_batch_size}")
            free_mem, total_mem = get_gpu_memory_info()
            print(f"GPU memory: {free_mem/1e9:.1f}GB free / {total_mem/1e9:.1f}GB total")
    
    if verbose:
        print(f"Fitting model for {n_genes} genes and {n_samples} samples")
        print(f"Design matrix has {n_features} features")
        print(f"Using {'GPU' if use_gpu else 'CPU'} computation")
    
    # Compute size factors
    if size_factors:
        if verbose:
            print("Computing size factors...")
        if use_gpu:
            try:
                sf = calculate_size_factors(count_matrix, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"GPU size factor computation failed: {e}")
                    print("Falling back to CPU")
                sf = calculate_size_factors(count_matrix, verbose=verbose)
        else:
            sf = calculate_size_factors(count_matrix, verbose=verbose)
    else:
        sf = np.ones(n_samples)
    
    # Calculate offset vector
    offset_vector = compute_offset_vector(offset, n_samples, sf)
    
    # Initialize overdispersion
    if overdispersion:
        if init_overdispersion is None:
            if verbose:
                print("Estimating initial overdispersion...")
            if use_gpu:
                try:
                    with GPUMemoryManager():
                        dispersion_init = estimate_initial_dispersion_gpu(
                            count_matrix, offset_vector, dtype=gpu_dtype_np
                        )
                except Exception as e:
                    if verbose:
                        print(f"GPU dispersion estimation failed: {e}")
                        print("Falling back to CPU")
                    dispersion_init = estimate_initial_dispersion(count_matrix, offset_vector)
            else:
                dispersion_init = estimate_initial_dispersion(count_matrix, offset_vector)
        else:
            dispersion_init = np.full(n_genes, init_overdispersion)
    else:
        dispersion_init = np.zeros(n_genes)
    
    # Initialize beta coefficients
    if verbose:
        print("Initializing beta coefficients...")
    
    if use_gpu:
        try:
            with GPUMemoryManager():
                beta_init = init_beta_gpu(
                    count_matrix, design_matrix, offset_vector, dtype=gpu_dtype_np
                )
        except Exception as e:
            if verbose:
                print(f"GPU beta initialization failed: {e}")
                print("Falling back to CPU")
            beta_init = init_beta(count_matrix, design_matrix, offset_vector)
    else:
        beta_init = init_beta(count_matrix, design_matrix, offset_vector)
    
    # Fit beta coefficients
    if verbose:
        print("Fitting beta coefficients...")
    
    if use_gpu:
        beta, iterations, converged = _fit_beta_gpu(
            count_matrix, design_matrix, beta_init, offset_vector,
            dispersion_init, max_iter, tolerance, gpu_batch_size,
            gpu_dtype_np, verbose
        )
    else:
        beta, iterations, converged = _fit_beta_cpu(
            count_matrix, design_matrix, beta_init, offset_vector,
            dispersion_init, max_iter, tolerance, n_jobs, verbose
        )
    
    # Set gene names
    if beta.ndim == 1:
        beta = beta.reshape(1, -1)
    
    # Fit overdispersion if requested
    if overdispersion:
        if verbose:
            print("Fitting overdispersion parameters...")
        
        if use_gpu:
            theta = _fit_overdispersion_gpu(
                beta, design_matrix, count_matrix, offset_vector,
                tolerance, max_iter, do_cox_reid_adjustment,
                gpu_batch_size, gpu_dtype_np, verbose
            )
        else:
            theta = _fit_overdispersion_cpu(
                beta, design_matrix, count_matrix, offset_vector,
                tolerance, max_iter, do_cox_reid_adjustment, n_jobs, verbose
            )
    else:
        theta = np.zeros(n_genes)
    
    # Report convergence
    if verbose:
        n_converged = np.sum(converged)
        print(f"Optimization converged for {n_converged}/{n_genes} genes")
        if n_converged < n_genes:
            warnings.warn(
                f"{n_genes - n_converged} genes did not converge. "
                "Consider increasing max_iter or tolerance."
            )
    
    return {
        "beta": beta,
        "overdispersion": theta,
        "iterations": iterations,
        "size_factors": sf,
        "offset_vector": offset_vector,
        "design_matrix": design_matrix,
        "gene_names": gene_names,
        "n_genes": n_genes,
        "n_samples": n_samples,
        "converged": converged,
        "count_matrix": count_matrix,
        "use_gpu": use_gpu,
        "gpu_batch_size": gpu_batch_size if use_gpu else None
    }


def _fit_beta_gpu(
    count_matrix: np.ndarray,
    design_matrix: np.ndarray,
    beta_init: np.ndarray,
    offset_vector: np.ndarray,
    dispersion_init: np.ndarray,
    max_iter: int,
    tolerance: float,
    batch_size: int,
    dtype: np.dtype,
    verbose: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit beta coefficients using GPU acceleration."""
    n_genes, n_samples = count_matrix.shape
    n_features = design_matrix.shape[1]
    
    # Choose GPU implementation
    free_mem, _ = get_gpu_memory_info()
    gpu_impl = select_gpu_implementation(batch_size, n_samples, n_features, free_mem)
    
    if verbose:
        print(f"Using {gpu_impl} GPU implementation")
    
    # Process in batches
    beta_results = []
    iterations_results = []
    converged_results = []
    
    with GPUMemoryManager():
        for start_idx in tqdm(range(0, n_genes, batch_size), 
                             desc="GPU beta fitting", disable=not verbose):
            end_idx = min(start_idx + batch_size, n_genes)
            
            batch_counts = count_matrix[start_idx:end_idx]
            batch_beta_init = beta_init[start_idx:end_idx]
            batch_dispersion = dispersion_init[start_idx:end_idx]
            
            try:
                if gpu_impl == 'vectorized':
                    batch_beta, batch_iter, batch_conv = fit_beta_coefficients_gpu_vectorized(
                        batch_counts, design_matrix, batch_beta_init,
                        offset_vector, batch_dispersion, max_iter, tolerance, dtype
                    )
                else:
                    batch_beta, batch_iter, batch_conv = fit_beta_coefficients_gpu_batch(
                        batch_counts, design_matrix, batch_beta_init,
                        offset_vector, batch_dispersion, max_iter, tolerance, dtype
                    )
                
                beta_results.append(batch_beta)
                iterations_results.append(batch_iter)
                converged_results.append(batch_conv)
                
            except Exception as e:
                if verbose:
                    print(f"GPU batch {start_idx}-{end_idx} failed: {e}")
                    print("Falling back to CPU for this batch")
                
                # CPU fallback for this batch
                batch_results = Parallel(n_jobs=-1)(
                    delayed(fit_beta_coefficients)(
                        batch_counts[i], design_matrix, batch_beta_init[i],
                        offset_vector, batch_dispersion[i], max_iter, tolerance
                    ) for i in range(len(batch_counts))
                )
                
                batch_beta = np.array([r[0] for r in batch_results])
                batch_iter = np.array([r[1] for r in batch_results])
                batch_conv = np.array([r[2] for r in batch_results])
                
                beta_results.append(batch_beta)
                iterations_results.append(batch_iter)
                converged_results.append(batch_conv)
    
    # Combine results
    beta = np.vstack(beta_results)
    iterations = np.concatenate(iterations_results)
    converged = np.concatenate(converged_results)
    
    return beta, iterations, converged


def _fit_beta_cpu(
    count_matrix: np.ndarray,
    design_matrix: np.ndarray,
    beta_init: np.ndarray,
    offset_vector: np.ndarray,
    dispersion_init: np.ndarray,
    max_iter: int,
    tolerance: float,
    n_jobs: Optional[int],
    verbose: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit beta coefficients using CPU parallelization."""
    n_genes = count_matrix.shape[0]
    
    if n_jobs is None:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
    
    def fit_gene(gene_idx):
        """Fit model for a single gene."""
        beta, n_iter, converged = fit_beta_coefficients(
            count_matrix[gene_idx, :],
            design_matrix,
            beta_init[gene_idx, :],
            offset_vector,
            dispersion_init[gene_idx],
            max_iter=max_iter,
            tolerance=tolerance
        )
        return beta, n_iter, converged
    
    # Run parallel fitting
    if verbose:
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_gene)(i) for i in tqdm(range(n_genes), desc="CPU beta fitting")
        )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_gene)(i) for i in range(n_genes)
        )
    
    # Extract results
    beta = np.array([r[0] for r in results])
    iterations = np.array([r[1] for r in results])
    converged = np.array([r[2] for r in results])
    
    return beta, iterations, converged


def _fit_overdispersion_gpu(
    beta: np.ndarray,
    design_matrix: np.ndarray,
    count_matrix: np.ndarray,
    offset_vector: np.ndarray,
    tolerance: float,
    max_iter: int,
    do_cox_reid_adjustment: bool,
    batch_size: int,
    dtype: np.dtype,
    verbose: bool
) -> np.ndarray:
    """Fit overdispersion parameters using GPU acceleration."""
    n_genes = count_matrix.shape[0]
    
    # Check if GPU implementation is feasible
    free_mem, _ = get_gpu_memory_info()
    gpu_impl = select_dispersion_implementation(n_genes, batch_size, free_mem)
    
    if gpu_impl == 'cpu':
        if verbose:
            print("Using CPU for overdispersion fitting due to memory constraints")
        return _fit_overdispersion_cpu(
            beta, design_matrix, count_matrix, offset_vector,
            tolerance, max_iter, do_cox_reid_adjustment, None, verbose
        )
    
    # Process in batches on GPU
    theta_results = []
    
    with GPUMemoryManager():
        for start_idx in tqdm(range(0, n_genes, batch_size),
                             desc="GPU overdispersion fitting", disable=not verbose):
            end_idx = min(start_idx + batch_size, n_genes)
            
            batch_beta = beta[start_idx:end_idx]
            batch_counts = count_matrix[start_idx:end_idx]
            
            try:
                batch_theta = fit_dispersion_gpu_batch(
                    batch_beta, design_matrix, batch_counts, offset_vector,
                    tolerance, max_iter, do_cox_reid_adjustment, dtype
                )
                theta_results.append(batch_theta)
                
            except Exception as e:
                if verbose:
                    print(f"GPU overdispersion batch {start_idx}-{end_idx} failed: {e}")
                    print("Falling back to CPU for this batch")
                
                # CPU fallback
                batch_results = Parallel(n_jobs=-1)(
                    delayed(fit_dispersion)(
                        batch_beta[i], design_matrix, batch_counts[i],
                        offset_vector, tolerance, max_iter, do_cox_reid_adjustment
                    ) for i in range(len(batch_beta))
                )
                
                theta_results.append(np.array(batch_results))
    
    return np.concatenate(theta_results)


def _fit_overdispersion_cpu(
    beta: np.ndarray,
    design_matrix: np.ndarray,
    count_matrix: np.ndarray,
    offset_vector: np.ndarray,
    tolerance: float,
    max_iter: int,
    do_cox_reid_adjustment: bool,
    n_jobs: Optional[int],
    verbose: bool
) -> np.ndarray:
    """Fit overdispersion parameters using CPU parallelization."""
    n_genes = count_matrix.shape[0]
    
    if n_jobs is None:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
    
    def fit_gene_dispersion(gene_idx):
        """Fit dispersion for a single gene."""
        return fit_dispersion(
            beta[gene_idx, :],
            design_matrix,
            count_matrix[gene_idx, :],
            offset_vector,
            tolerance=tolerance,
            max_iter=max_iter,
            do_cox_reid_adjustment=do_cox_reid_adjustment
        )
    
    if verbose:
        theta = Parallel(n_jobs=n_jobs)(
            delayed(fit_gene_dispersion)(i) 
            for i in tqdm(range(n_genes), desc="CPU overdispersion fitting")
        )
    else:
        theta = Parallel(n_jobs=n_jobs)(
            delayed(fit_gene_dispersion)(i) for i in range(n_genes)
        )
    
    return np.array(theta)
