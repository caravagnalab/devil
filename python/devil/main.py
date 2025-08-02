"""Main module for fitting the devil statistical model with exact beta fitting."""

from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad
from joblib import Parallel, delayed
import warnings
from tqdm import tqdm

# Import implementations (with "_exact" suffix removed as requested)
from .beta import init_beta, fit_beta_coefficients, init_beta_matrix
from .overdispersion import estimate_initial_dispersion, fit_dispersion
from .size_factors import calculate_size_factors, compute_offset_vector
from .utils import handle_input_data, validate_inputs
from .gpu import (
    is_gpu_available, check_gpu_requirements, estimate_batch_size,
    GPUMemoryManager, get_gpu_memory_info
)
from .beta_gpu import (
    init_beta_gpu, fit_beta_coefficients_gpu_batch,
    fit_beta_coefficients_gpu_batch
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
    Fit statistical model for count data with exact beta coefficient estimation.
    
    This function implements the exact mathematical algorithms from the R package,
    providing mathematically identical results with optional GPU acceleration.
    
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
        Dictionary containing fitted parameters and model information:
        - 'beta': Fitted regression coefficients (genes × features)
        - 'overdispersion': Estimated overdispersion parameters (genes,)
        - 'size_factors': Computed size factors (samples,)
        - 'offset_vector': Offset values used in fitting (samples,)
        - 'iterations': Number of iterations per gene (genes,)
        - 'converged': Convergence status per gene (genes,)
        - 'n_genes': Number of genes processed
        - 'n_samples': Number of samples
        - 'n_features': Number of features in design matrix
        - 'design_matrix': Design matrix used (samples × features)
        - 'count_matrix': Original count data (genes × samples)
        - 'feature_names': Names of design matrix features
        - 'gene_names': Names of genes (if available)
        - 'sample_names': Names of samples (if available)
    """
    # Input validation and preprocessing
    count_matrix, gene_names, sample_names, obs_df = handle_input_data(
        adata, layer=layer
    )
    
    # Handle design matrix/formula
    if design_matrix is None and design_formula is None:
        raise ValueError("Must provide either design_matrix or design_formula")
    
    if design_formula is not None:
        if not isinstance(adata, ad.AnnData):
            raise ValueError("design_formula requires AnnData input with .obs metadata")
        # Note: Design formula parsing would need to be implemented
        # For now, require explicit design_matrix
        raise NotImplementedError("design_formula support not yet implemented")
    
    # Validate inputs
    validate_inputs(count_matrix, design_matrix)
    
    n_genes, n_samples = count_matrix.shape
    _, n_features = design_matrix.shape
    
    # Generate feature names for design matrix
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    if verbose:
        print(f"Fitting model for {n_genes} genes, {n_samples} samples, {n_features} features")
    
    # Determine GPU usage
    if use_gpu is None:
        use_gpu = is_gpu_available()
        if verbose and use_gpu:
            print("GPU detected and will be used for acceleration")
        elif verbose:
            print("GPU not available, using CPU")
    elif use_gpu and not is_gpu_available():
        warnings.warn("GPU requested but not available, falling back to CPU")
        use_gpu = False
    
    # Compute size factors
    if size_factors:
        if verbose:
            print("Computing size factors...")
        sf = calculate_size_factors(count_matrix, verbose=verbose)
    else:
        sf = np.ones(n_samples)
    
    # Calculate offset vector
    offset_vector = compute_offset_vector(offset, n_samples, sf)
    
    # Initialize overdispersion parameters
    if init_overdispersion is not None:
        dispersion_init = np.full(n_genes, init_overdispersion)
    else:
        if verbose:
            print("Estimating initial overdispersion parameters...")
        if use_gpu:
            dispersion_init = estimate_initial_dispersion_gpu(
                count_matrix, offset_vector, dtype=np.dtype(gpu_dtype)
            )
        else:
            dispersion_init = estimate_initial_dispersion(count_matrix, offset_vector)
    
    # Initialize beta coefficients using exact algorithm
    if verbose:
        print("Initializing beta coefficients...")
    
    if use_gpu:
        beta_init = init_beta_gpu(
            count_matrix, design_matrix, dtype=np.dtype(gpu_dtype)
        )
    else:
        beta_init = init_beta_matrix(count_matrix, design_matrix)
    
    # Fit beta coefficients using exact algorithm
    if verbose:
        print("Fitting beta coefficients...")
    
    if use_gpu:
        # Determine batch size for GPU processing
        if gpu_batch_size is None:
            gpu_batch_size = estimate_batch_size(
                n_genes, n_samples, n_features, 
                dtype=np.dtype(gpu_dtype), memory_fraction=0.8
            )
            if verbose:
                print(f"Using GPU batch size: {gpu_batch_size}")
        
        # GPU fitting with exact algorithm
        beta_fitted, beta_iterations, beta_converged = fit_beta_coefficients_gpu_batch(
            count_matrix=count_matrix,
            design_matrix=design_matrix,
            beta_init=beta_init,
            offset_vector=offset_vector,
            dispersion_vector=dispersion_init,
            batch_size=gpu_batch_size,
            max_iter=max_iter,
            tolerance=tolerance,
            dtype=np.dtype(gpu_dtype),
            verbose=verbose
        )
    else:
        # CPU fitting
        beta_fitted, beta_iterations, beta_converged = _fit_beta_cpu(
            count_matrix=count_matrix,
            design_matrix=design_matrix,
            beta_init=beta_init,
            offset_vector=offset_vector,
            dispersion_init=dispersion_init,
            max_iter=max_iter,
            tolerance=tolerance,
            n_jobs=n_jobs,
            verbose=verbose
        )
    
    # Fit overdispersion parameters if requested
    if overdispersion:
        if verbose:
            print("Fitting overdispersion parameters...")
        
        overdispersion_fitted = _fit_overdispersion(
            beta=beta_fitted,
            design_matrix=design_matrix,
            count_matrix=count_matrix,
            offset_vector=offset_vector,
            tolerance=tolerance,
            max_iter=max_iter,
            do_cox_reid_adjustment=do_cox_reid_adjustment,
            use_gpu=use_gpu,
            batch_size=gpu_batch_size,
            dtype=gpu_dtype if use_gpu else None,
            n_jobs=n_jobs,
            verbose=verbose
        )
    else:
        overdispersion_fitted = np.zeros(n_genes)
    
    # Compile results
    results = {
        'beta': beta_fitted,
        'overdispersion': overdispersion_fitted,
        'size_factors': sf,
        'offset_vector': offset_vector,
        'iterations': beta_iterations,
        'converged': beta_converged,
        'n_genes': n_genes,
        'n_samples': n_samples,
        'n_features': n_features,
        'design_matrix': design_matrix,
        'count_matrix': count_matrix,
        'feature_names': feature_names,
        'gene_names': gene_names,
        'sample_names': sample_names
    }
    
    if verbose:
        convergence_rate = np.mean(beta_converged) * 100
        print(f"Beta fitting completed: {convergence_rate:.1f}% genes converged")
        if overdispersion:
            print("Overdispersion fitting completed")
        print("Model fitting complete!")
    
    return results


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
        """Fit beta coefficients for a single gene."""
        try:
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
        except Exception as e:
            warnings.warn(f"Gene {gene_idx} fitting failed: {e}")
            # Return initial values as fallback
            return beta_init[gene_idx, :], 0, False
    
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


def _fit_overdispersion(
    beta: np.ndarray,
    design_matrix: np.ndarray,
    count_matrix: np.ndarray,
    offset_vector: np.ndarray,
    tolerance: float,
    max_iter: int,
    do_cox_reid_adjustment: bool,
    use_gpu: bool,
    batch_size: Optional[int],
    dtype: Optional[str],
    n_jobs: Optional[int],
    verbose: bool
) -> np.ndarray:
    """Fit overdispersion parameters."""
    n_genes = count_matrix.shape[0]
    
    if use_gpu:
        # GPU overdispersion fitting
        try:
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
                            tolerance, max_iter, do_cox_reid_adjustment, np.dtype(dtype)
                        )
                        theta_results.append(batch_theta)
                    except Exception as e:
                        if verbose:
                            print(f"GPU overdispersion batch {start_idx}-{end_idx} failed: {e}")
                            print("Falling back to CPU for this batch")
                        
                        # CPU fallback for this batch
                        batch_results = Parallel(n_jobs=-1)(
                            delayed(fit_dispersion)(
                                batch_beta[i], design_matrix, batch_counts[i],
                                offset_vector, tolerance, max_iter, do_cox_reid_adjustment
                            ) for i in range(len(batch_beta))
                        )
                        theta_results.append(np.array(batch_results))
            
            return np.concatenate(theta_results)
            
        except Exception as e:
            if verbose:
                print(f"GPU overdispersion fitting failed: {e}")
                print("Falling back to CPU")
            use_gpu = False
    
    if not use_gpu:
        # CPU overdispersion fitting
        if n_jobs is None:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        
        def fit_gene_dispersion(gene_idx):
            """Fit dispersion for a single gene."""
            try:
                return fit_dispersion(
                    beta[gene_idx], design_matrix, count_matrix[gene_idx, :],
                    offset_vector, tolerance, max_iter, do_cox_reid_adjustment
                )
            except Exception as e:
                warnings.warn(f"Gene {gene_idx} overdispersion fitting failed: {e}")
                return 1.0  # Default dispersion value
        
        if verbose:
            results = Parallel(n_jobs=n_jobs)(
                delayed(fit_gene_dispersion)(i) 
                for i in tqdm(range(n_genes), desc="CPU overdispersion fitting")
            )
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(fit_gene_dispersion)(i) for i in range(n_genes)
            )
        
        return np.array(results)

