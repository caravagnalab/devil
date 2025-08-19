"""GPU-accelerated overdispersion parameter estimation with full vectorization."""

from typing import Tuple, Optional
import numpy as np
import warnings
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    # need to mock for cp.ndarray
    import numpy as cp
    CUPY_AVAILABLE = False
from .gpu import is_gpu_available, to_gpu, to_cpu, CUPY_AVAILABLE
from .overdispersion import estimate_initial_dispersion

if CUPY_AVAILABLE:
    import cupy as cp
    from scipy.special import digamma, polygamma


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
    Fit dispersion parameters for a batch of genes on GPU with full vectorization.
    This implementation uses vectorized Newton-Raphson optimization to fit
    dispersion parameters for all genes in the batch simultaneously.
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
    
    # Initialize dispersion estimates using method of moments
    dispersions = _initialize_dispersions_vectorized(y_gpu, mu)
    
    # Convert to log scale for optimization
    log_theta = cp.log(dispersions)
    
    # Pre-allocate arrays for optimization
    converged = cp.zeros(batch_size, dtype=bool)
    
    # Handle all-zero genes
    all_zero_mask = cp.all(y_gpu == 0, axis=1)
    converged[all_zero_mask] = True
    log_theta[all_zero_mask] = -cp.inf  # log(0)
    
    # Cox-Reid adjustment term (computed once if needed)
    cox_reid_term = None
    if do_cox_reid_adjustment:
        cox_reid_term = _compute_cox_reid_adjustment(X_gpu, n_features)
    
    # Newton-Raphson optimization
    for iter_num in range(max_iter):
        if cp.all(converged):
            break
        
        # Compute score and information for all genes
        scores, info = _compute_score_and_info_vectorized(
            y_gpu, mu, log_theta, converged, cox_reid_term
        )
        
        # Newton update
        updates = scores / cp.maximum(info, 1e-10)
        
        # Only update non-converged genes
        active_mask = ~converged
        log_theta[active_mask] += updates[active_mask]
        
        # Check convergence
        newly_converged = (cp.abs(updates) < tolerance) & active_mask
        converged |= newly_converged
    
    # Convert back from log scale
    dispersions = cp.exp(log_theta)
    dispersions[all_zero_mask] = 0.0
    
    # Ensure dispersions are in valid range
    dispersions = cp.maximum(dispersions, 0.0)
    dispersions = cp.minimum(dispersions, 1e6)  # Cap at reasonable maximum
    
    return to_cpu(dispersions)


def _initialize_dispersions_vectorized(
    y: cp.ndarray,
    mu: cp.ndarray
) -> cp.ndarray:
    """
    Initialize dispersion parameters using method of moments for all genes.
    Args:
        y: Count data (batch_size × samples).
        mu: Expected values (batch_size × samples).
    Returns:
        Initial dispersion estimates.
    """
    # Compute sample statistics
    sample_var = cp.var(y, axis=1)
    sample_mean = cp.mean(y, axis=1)
    
    # Method of moments estimator
    init_disp = (sample_var - sample_mean) / cp.maximum(sample_mean**2, 1e-10)
    
    # Handle edge cases
    init_disp = cp.where(
        cp.isfinite(init_disp) & (init_disp > 0),
        init_disp,
        0.5  # Default starting value
    )
    
    # Ensure reasonable starting range
    init_disp = cp.maximum(init_disp, 0.01)
    init_disp = cp.minimum(init_disp, 100.0)
    
    return init_disp


def _compute_cox_reid_adjustment(
    X: cp.ndarray,
    n_features: int
) -> float:
    """
    Compute Cox-Reid adjustment term.
    For efficiency, we compute a simplified version that works well in practice.
    """
    # Simplified Cox-Reid adjustment
    # Full computation would require the hat matrix diagonal
    # Here we use an approximation based on the number of parameters
    return 0.5 * n_features / X.shape[0]


def _compute_score_and_info_vectorized(
    y: cp.ndarray,
    mu: cp.ndarray,
    log_theta: cp.ndarray,
    converged: cp.ndarray,
    cox_reid_term: Optional[float]
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Compute score function and expected information for all genes.
    
    This uses the negative binomial log-likelihood derivatives.
    """
    batch_size = y.shape[0]
    n_samples = y.shape[1]
    
    # Convert from log scale
    theta = cp.exp(log_theta)
    
    # Pre-allocate results
    scores = cp.zeros(batch_size, dtype=log_theta.dtype)
    info = cp.zeros(batch_size, dtype=log_theta.dtype)
    
    # Only compute for non-converged genes
    active_mask = ~converged
    if not cp.any(active_mask):
        return scores, info
    
    # Active genes
    active_theta = theta[active_mask]
    active_y = y[active_mask]
    active_mu = mu[active_mask]
    
    # Compute score function components
    # Score = sum over samples of score contribution
    
    # First term: sum(digamma(y + 1/theta))
    # Ensure proper broadcasting by explicitly reshaping theta
    theta_reshaped = active_theta.reshape(-1, 1)  # (batch_size, 1)
    y_plus_inv_theta = active_y + 1.0 / theta_reshaped
    
    # Note: CuPy doesn't have digamma/polygamma, so we need workarounds
    # For now, we'll use approximations or transfer to CPU for special functions
    try:
        # Try to use scipy special functions on GPU data
        import cupyx.scipy.special as cp_special
        term1 = cp.sum(cp_special.digamma(y_plus_inv_theta), axis=1)
        term2 = n_samples * cp_special.digamma(1.0 / active_theta)
    except (ImportError, AttributeError):
        # Fallback: Use approximations
        term1 = cp.sum(_digamma_approx(y_plus_inv_theta), axis=1)
        term2 = n_samples * _digamma_approx(1.0 / active_theta)
    
    # Third term: sum(log(1 + mu * theta))
    term3 = cp.sum(cp.log1p(active_mu * theta_reshaped), axis=1)
    
    # Fourth term: sum((y - mu) / (mu + 1/theta))
    term4 = cp.sum(
        (active_y - active_mu) / (active_mu + 1.0 / theta_reshaped),
        axis=1
    )
    
    # Combine terms for score
    active_scores = (term1 - term2 - term3 + term4) / active_theta
    
    # Cox-Reid adjustment
    if cox_reid_term is not None:
        active_scores -= cox_reid_term
    
    scores[active_mask] = active_scores
    
    # Compute expected information (second derivative)
    # This is more complex, so we use a simplified version based on the expected Fisher information
    
    # Approximate information using second-order terms
    try:
        info_term1 = cp.sum(cp_special.polygamma(1, y_plus_inv_theta), axis=1)
        info_term2 = n_samples * cp_special.polygamma(1, 1.0 / active_theta)
    except (ImportError, AttributeError):
        # Use approximation
        info_term1 = cp.sum(1.0 / y_plus_inv_theta, axis=1)  # Simplified
        info_term2 = n_samples * active_theta**2  # Simplified
    
    # Additional terms
    info_term3 = cp.sum(
        active_mu**2 * theta_reshaped / (1 + active_mu * theta_reshaped)**2,
        axis=1
    )
    
    active_info = (info_term1 - info_term2 + info_term3) / active_theta**2
    
    # Ensure positive definiteness
    active_info = cp.maximum(active_info, 1e-6)
    
    info[active_mask] = active_info
    
    return scores, info


def _digamma_approx(x: cp.ndarray) -> cp.ndarray:
    """
    Approximation of digamma function for GPU computation.
    
    Uses asymptotic expansion for large x and series for small x.
    """
    result = cp.zeros_like(x)
    
    # For large x (> 10), use asymptotic expansion
    large_mask = x > 10
    if cp.any(large_mask):
        x_large = x[large_mask]
        result[large_mask] = cp.log(x_large) - 0.5/x_large - 1.0/(12*x_large**2)
    
    # For medium x (1 < x <= 10), use recurrence relation and series
    medium_mask = (x > 1) & (x <= 10)
    if cp.any(medium_mask):
        x_medium = x[medium_mask]
        # Shift to large x region
        shifts = cp.ceil(11 - x_medium).astype(int)
        x_shifted = x_medium + shifts
        result_shifted = cp.log(x_shifted) - 0.5/x_shifted - 1.0/(12*x_shifted**2)
        
        # Apply recurrence relation backwards
        for i in range(cp.max(shifts).item()):
            mask_i = shifts > i
            if cp.any(mask_i):
                x_shifted[mask_i] -= 1
                result_shifted[mask_i] -= 1.0 / x_shifted[mask_i]
        
        result[medium_mask] = result_shifted
    
    # For small x (0 < x <= 1), use series expansion
    small_mask = (x > 0) & (x <= 1)
    if cp.any(small_mask):
        x_small = x[small_mask]
        # Shift to x > 1 and use recurrence
        result[small_mask] = -1.0 / x_small + _digamma_approx(x_small + 1)
    
    # Handle special values
    result[x <= 0] = cp.nan
    
    return result


def _optimize_dispersion_gpu(
    y: cp.ndarray,
    mu: cp.ndarray,
    design_matrix: cp.ndarray,
    init_disp: float,
    tolerance: float,
    max_iter: int,
    do_cox_reid: bool
) -> float:
    """
    Optimize dispersion parameter for single gene on GPU using Newton's method.
    
    This is kept for compatibility but is deprecated in favor of the
    vectorized batch implementation.
    
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
    # Convert to batch format and use vectorized implementation
    y_batch = y.reshape(1, -1)
    mu_batch = mu.reshape(1, -1)
    
    # Dummy beta for single gene
    beta_batch = cp.zeros((1, design_matrix.shape[1]), dtype=y.dtype)
    
    # Use the batch function
    result = fit_dispersion_gpu_batch(
        beta_batch,
        design_matrix,
        y_batch,
        cp.zeros(y.shape[0], dtype=y.dtype),  # dummy offset
        tolerance,
        max_iter,
        do_cox_reid,
        y.dtype
    )
    
    return float(result[0])


def select_dispersion_implementation(
    batch_size: int,
    available_memory: int
) -> str:
    """
    Select implementation based on problem size and memory.
    
    Args:
        batch_size: Proposed batch size.
        available_memory: Available GPU memory in bytes.
    Returns:
        'gpu' or 'cpu' recommendation.
    """
    # Estimate memory requirements
    # Dispersion fitting is less memory intensive than beta fitting
    bytes_per_float = 4  # float32
    
    # Memory per gene (approximate)
    mem_per_gene = 1000 * bytes_per_float  # Conservative estimate
    
    total_memory = batch_size * mem_per_gene
    
    # Check if we have enough memory (with safety margin)
    if total_memory > available_memory * 0.7:
        return 'cpu'
    
    # Also check if it's worth using GPU (too small batches aren't efficient)
    if batch_size < 10:
        return 'cpu'
    
    return 'gpu'
