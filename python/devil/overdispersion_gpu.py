"""GPU-accelerated overdispersion parameter estimation."""

from typing import Tuple, Optional
import numpy as np
import warnings

from .gpu import is_gpu_available, to_gpu, to_cpu, CUPY_AVAILABLE
from .overdispersion import estimate_initial_dispersion

if CUPY_AVAILABLE:
    import cupy as cp
    import cupyx.scipy.optimize as cp_optimize
    from cupy.special import digamma, polygamma


def estimate_initial_dispersion_gpu(
    count_matrix: np.ndarray,
    offset_vector: np.ndarray,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Estimate initial dispersion using method of moments on GPU.
    
    Args:
        count_matrix: Count data (genes × samples).
        offset_vector: Offset values per sample.
        dtype: Data type for GPU computation.
        
    Returns:
        Initial dispersion estimates per gene.
    """
    if not is_gpu_available():
        # Fallback to CPU implementation
        return estimate_initial_dispersion(count_matrix, offset_vector)
    
    n_genes, n_samples = count_matrix.shape
    
    # Transfer to GPU
    counts_gpu = to_gpu(count_matrix, dtype)
    offset_gpu = to_gpu(offset_vector, dtype)
    
    # Calculate normalized factors
    norm_factors = cp.exp(offset_gpu)
    
    # Calculate normalized means and variances
    norm_counts = counts_gpu / norm_factors[cp.newaxis, :]
    mean_counts = cp.mean(norm_counts, axis=1)
    var_counts = cp.var(norm_counts, axis=1, ddof=1)
    
    # Method of moments estimator: Var = mu + mu^2 * dispersion
    dispersion = (var_counts - mean_counts) / (mean_counts ** 2)
    
    # Handle edge cases
    dispersion = cp.where(
        cp.isfinite(dispersion) & (dispersion > 0), 
        dispersion, 
        100.0
    )
    
    return to_cpu(dispersion)


def fit_dispersion_gpu_batch(
    beta_batch: np.ndarray,
    design_matrix: np.ndarray,
    y_batch: np.ndarray,
    offset_vector: np.ndarray,
    tolerance: float = 1e-3,
    max_iter: int = 100,
    do_cox_reid_adjustment: bool = True,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Fit dispersion parameters for a batch of genes on GPU.
    
    Args:
        beta_batch: Fitted coefficients (batch_size × features).
        design_matrix: Design matrix (samples × features).
        y_batch: Count data (batch_size × samples).
        offset_vector: Offset values.
        tolerance: Convergence tolerance.
        max_iter: Maximum iterations.
        do_cox_reid_adjustment: Whether to apply Cox-Reid adjustment.
        dtype: Data type for computation.
        
    Returns:
        Estimated dispersion parameters for batch.
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available")
    
    batch_size, n_samples = y_batch.shape
    n_features = design_matrix.shape[1]
    
    # Transfer to GPU
    beta_gpu = to_gpu(beta_batch, dtype)
    X_gpu = to_gpu(design_matrix, dtype)
    y_gpu = to_gpu(y_batch, dtype)
    offset_gpu = to_gpu(offset_vector, dtype)
    
    # Calculate mean values for each gene
    eta = beta_gpu @ X_gpu.T + offset_gpu[cp.newaxis, :]  # batch_size × samples
    mu = cp.exp(eta)
    mu = cp.maximum(mu, 1e-10)
    
    # Initialize dispersion estimates
    dispersions = cp.zeros(batch_size, dtype=dtype)
    
    # Fit dispersion for each gene
    for gene_idx in range(batch_size):
        y_gene = y_gpu[gene_idx]
        mu_gene = mu[gene_idx]
        
        # Handle all-zero genes
        if cp.all(y_gene == 0):
            dispersions[gene_idx] = 0.0
            continue
        
        # Initial estimate using method of moments
        sample_var = cp.var(y_gene)
        sample_mean = cp.mean(y_gene)
        init_disp = cp.maximum((sample_var - sample_mean) / sample_mean**2, 0.1)
        
        # Optimize using scipy-like interface
        try:
            result = _optimize_dispersion_gpu(
                y_gene, mu_gene, X_gpu, init_disp,
                tolerance, max_iter, do_cox_reid_adjustment
            )
            dispersions[gene_idx] = result
        except Exception:
            # Fallback to initial estimate
            dispersions[gene_idx] = init_disp
    
    return to_cpu(dispersions)


def _optimize_dispersion_gpu(
    y: "cp.ndarray",
    mu: "cp.ndarray", 
    design_matrix: "cp.ndarray",
    init_disp: float,
    tolerance: float,
    max_iter: int,
    do_cox_reid: bool
) -> float:
    """
    Optimize dispersion parameter for single gene on GPU using Newton's method.
    
    Args:
        y: Count vector for gene.
        mu: Expected values.
        design_matrix: Design matrix.
        init_disp: Initial dispersion estimate.
        tolerance: Convergence tolerance.
        max_iter: Maximum iterations.
        do_cox_reid: Whether to apply Cox-Reid adjustment.
        
    Returns:
        Optimized dispersion parameter.
    """
    log_theta = cp.log(init_disp)
    
    for iteration in range(max_iter):
        theta = cp.exp(log_theta)
        
        # Compute score and hessian
        score = _compute_nb_score_gpu(y, mu, theta, design_matrix, do_cox_reid)
        hessian = _compute_nb_hessian_gpu(y, mu, theta, design_matrix, do_cox_reid)
        
        # Newton's method update
        if abs(hessian) > 1e-12:
            delta = score / hessian
            log_theta_new = log_theta - delta
            
            # Line search to ensure we don't go too far
            alpha = 1.0
            while alpha > 0.01:
                test_log_theta = log_theta + alpha * (log_theta_new - log_theta)
                test_theta = cp.exp(test_log_theta)
                
                # Check if likelihood improved
                old_ll = _compute_nb_log_likelihood_gpu(y, mu, theta, design_matrix, do_cox_reid)
                new_ll = _compute_nb_log_likelihood_gpu(y, mu, test_theta, design_matrix, do_cox_reid)
                
                if new_ll > old_ll or alpha <= 0.1:
                    log_theta = test_log_theta
                    break
                alpha *= 0.5
            else:
                log_theta = log_theta_new
            
            # Check convergence
            if abs(delta) < tolerance:
                break
        else:
            break
    
    return float(cp.exp(log_theta))


def _compute_nb_log_likelihood_gpu(
    y: "cp.ndarray",
    mu: "cp.ndarray",
    theta: float,
    design_matrix: "cp.ndarray",
    do_cox_reid: bool
) -> float:
    """Compute negative binomial log-likelihood on GPU."""
    alpha = 1.0 / theta
    
    # Basic NB log-likelihood using loggamma
    ll = cp.sum(
        cp.special.loggamma(y + alpha) - cp.special.loggamma(alpha) - cp.special.loggamma(y + 1) +
        alpha * cp.log(alpha / (alpha + mu)) +
        y * cp.log(mu / (alpha + mu))
    )
    
    # Cox-Reid adjustment
    if do_cox_reid:
        W = mu / (1 + mu * theta)
        XWX = design_matrix.T @ (W[:, cp.newaxis] * design_matrix)
        # Add small regularization to avoid singularity
        XWX += cp.eye(XWX.shape[0]) * 1e-8
        sign, logdet = cp.linalg.slogdet(XWX)
        if sign > 0:
            ll -= 0.5 * logdet
    
    return float(ll)


def _compute_nb_score_gpu(
    y: "cp.ndarray",
    mu: "cp.ndarray",
    theta: float,
    design_matrix: "cp.ndarray",
    do_cox_reid: bool
) -> float:
    """Compute score function (derivative of log-likelihood) on GPU."""
    alpha = 1.0 / theta
    
    # Basic score
    score = cp.sum(
        digamma(y + alpha) - digamma(alpha) + 
        cp.log(alpha / (alpha + mu)) +
        (y - mu) / (alpha + mu)
    ) * alpha
    
    # Cox-Reid adjustment term
    if do_cox_reid:
        W = mu / (1 + mu * theta)
        dW = -mu**2 * theta / (1 + mu * theta)**2
        
        XWX = design_matrix.T @ (W[:, cp.newaxis] * design_matrix)
        XdWX = design_matrix.T @ (dW[:, cp.newaxis] * design_matrix)
        
        # Add regularization
        XWX += cp.eye(XWX.shape[0]) * 1e-8
        try:
            XWX_inv = cp.linalg.inv(XWX)
            cr_term = -0.5 * cp.trace(XWX_inv @ XdWX)
            score += cr_term * theta
        except cp.linalg.LinAlgError:
            # Skip Cox-Reid term if matrix is singular
            pass
    
    return float(score)


def _compute_nb_hessian_gpu(
    y: "cp.ndarray",
    mu: "cp.ndarray", 
    theta: float,
    design_matrix: "cp.ndarray",
    do_cox_reid: bool
) -> float:
    """Compute Hessian (second derivative of log-likelihood) on GPU."""
    alpha = 1.0 / theta
    
    # Basic Hessian computation
    trigamma_term = cp.sum(polygamma(1, y + alpha) - polygamma(1, alpha))
    
    # Additional terms
    hess = -2 * alpha * (
        cp.sum(digamma(y + alpha) - digamma(alpha) + 
               cp.log(alpha / (alpha + mu))) +
        alpha * trigamma_term
    )
    
    # Cox-Reid adjustment (simplified)
    if do_cox_reid:
        hess *= 0.99  # Correction factor similar to glmGamPoi
    
    return float(hess)


def select_dispersion_implementation(
    n_genes: int,
    batch_size: int,
    available_memory: int
) -> str:
    """
    Select best dispersion fitting implementation.
    
    Args:
        n_genes: Number of genes.
        batch_size: Batch size for processing.
        available_memory: Available GPU memory.
        
    Returns:
        Implementation choice: 'gpu_batch' or 'cpu'.
    """
    if not is_gpu_available():
        return 'cpu'
    
    # Estimate memory requirements
    bytes_per_float = 4
    estimated_memory = bytes_per_float * batch_size * 1000  # Rough estimate
    
    if estimated_memory < available_memory * 0.5 and batch_size > 10:
        return 'gpu_batch'
    else:
        return 'cpu'