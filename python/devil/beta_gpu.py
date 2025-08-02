"""GPU-accelerated beta coefficient estimation with full vectorization."""

from typing import Tuple, Optional
import numpy as np
import warnings

from .gpu import is_gpu_available, to_gpu, to_cpu, CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp
    try:
        import cupyx.scipy.linalg as cp_linalg
    except ImportError:
        cp_linalg = None


def init_beta_gpu(
    count_matrix: np.ndarray,
    design_matrix: np.ndarray,
    offset_vector: np.ndarray,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Initialize beta coefficients using QR decomposition on GPU.
    
    Args:
        count_matrix: Count data (genes × samples).
        design_matrix: Design matrix (samples × features).
        offset_vector: Offset values per sample.
        dtype: Data type for GPU computation.
        
    Returns:
        Initial beta estimates (genes × features).
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available")
    
    n_genes, n_samples = count_matrix.shape
    n_features = design_matrix.shape[1]
    
    # Transfer to GPU
    X_gpu = to_gpu(design_matrix, dtype)
    offset_gpu = to_gpu(offset_vector, dtype)
    
    # QR decomposition with fallback
    try:
        Q, R = cp.linalg.qr(X_gpu)
    except (AttributeError, ImportError):
        # Fallback to CPU QR decomposition
        import numpy.linalg as np_linalg
        Q, R = np_linalg.qr(design_matrix)
        Q = to_gpu(Q, dtype)
        R = to_gpu(R, dtype)
    
    # Process data in batches to manage memory
    batch_size = min(n_genes, 1000)  # Adjust based on memory
    beta_init = np.zeros((n_genes, n_features), dtype=dtype)
    
    for start_idx in range(0, n_genes, batch_size):
        end_idx = min(start_idx + batch_size, n_genes)
        batch_counts = to_gpu(count_matrix[start_idx:end_idx], dtype)
        
        # Normalize counts
        norm_log_counts = cp.log1p(
            batch_counts.T / cp.exp(offset_gpu)[:, cp.newaxis]
        )
        
        # Solve for initial beta with fallback
        try:
            batch_beta = cp_linalg.solve_triangular(
                R, Q.T @ norm_log_counts, lower=False
            ).T
        except (AttributeError, ImportError):
            # Fallback to CPU solve
            import numpy.linalg as np_linalg
            R_cpu = to_cpu(R)
            Q_cpu = to_cpu(Q)
            norm_log_counts_cpu = to_cpu(norm_log_counts)
            batch_beta_cpu = np_linalg.solve_triangular(
                R_cpu, Q_cpu.T @ norm_log_counts_cpu, lower=False
            ).T
            batch_beta = to_gpu(batch_beta_cpu, dtype)
        
        beta_init[start_idx:end_idx] = to_cpu(batch_beta)
    
    return beta_init


def fit_beta_coefficients_gpu_batch(
    count_batch: np.ndarray,
    design_matrix: np.ndarray,
    beta_init_batch: np.ndarray,
    offset_vector: np.ndarray,
    dispersion_batch: np.ndarray,
    max_iter: int = 100,
    tolerance: float = 1e-3,
    dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit beta coefficients for a batch of genes on GPU with full vectorization.
    
    This is the primary optimized implementation that processes all genes
    in the batch simultaneously.
    
    Args:
        count_batch: Count data for batch (batch_size × samples).
        design_matrix: Design matrix (samples × features).
        beta_init_batch: Initial beta values (batch_size × features).
        offset_vector: Offset vector.
        dispersion_batch: Dispersion parameters for batch.
        max_iter: Maximum iterations.
        tolerance: Convergence tolerance.
        dtype: Data type for computation.
        
    Returns:
        Tuple of (fitted_beta, n_iterations, converged) for batch.
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available")
    
    batch_size, n_samples = count_batch.shape
    n_features = design_matrix.shape[1]
    
    # Transfer all data to GPU at once
    y_gpu = cp.asarray(count_batch, dtype=dtype)
    X_gpu = cp.asarray(design_matrix, dtype=dtype)
    beta_gpu = cp.asarray(beta_init_batch, dtype=dtype)
    offset_gpu = cp.asarray(offset_vector, dtype=dtype)
    k_gpu = cp.asarray(1.0 / dispersion_batch, dtype=dtype)
    
    # Pre-compute reusable matrices
    XT = X_gpu.T  # (n_features, n_samples) - cache transpose
    
    # Initialize tracking arrays
    converged = cp.zeros(batch_size, dtype=bool)
    iterations = cp.zeros(batch_size, dtype=cp.int32)
    
    # Pre-allocate work arrays for efficiency
    eta = cp.empty((batch_size, n_samples), dtype=dtype)
    mu = cp.empty_like(eta)
    variance = cp.empty_like(eta)
    weights = cp.empty_like(eta)
    residuals = cp.empty_like(eta)
    weighted_residuals = cp.empty_like(eta)
    scores = cp.empty((batch_size, n_features), dtype=dtype)
    
    for iter_num in range(max_iter):
        # Vectorized computation for all genes at once
        # eta = beta @ X.T + offset
        cp.matmul(beta_gpu, XT, out=eta)
        eta += offset_gpu
        
        # mu = exp(eta) with numerical stability
        cp.exp(eta, out=mu)
        cp.maximum(mu, 1e-10, out=mu)
        
        # Compute variance for negative binomial
        # variance = mu + mu^2 / k
        k_expanded = k_gpu[:, cp.newaxis]
        cp.square(mu, out=variance)
        variance /= k_expanded
        variance += mu
        
        # weights = mu / variance
        cp.divide(mu, variance, out=weights)
        
        # residuals = (y - mu) / variance
        cp.subtract(y_gpu, mu, out=residuals)
        residuals /= variance
        
        # Compute scores efficiently
        # scores = (weights * residuals) @ X
        cp.multiply(weights, residuals, out=weighted_residuals)
        cp.matmul(weighted_residuals, X_gpu, out=scores)
        
        # Batch solve using optimized method
        deltas = _batch_solve_weighted_least_squares(
            X_gpu, XT, weights, scores, n_features, batch_size, 
            dtype, converged
        )
        
        # Update beta for non-converged genes only
        mask = ~converged
        if cp.any(mask):
            beta_gpu[mask] += deltas[mask]
        
        # Check convergence
        max_delta = cp.max(cp.abs(deltas), axis=1)
        newly_converged = (max_delta < tolerance) & mask
        converged |= newly_converged
        iterations[newly_converged] = iter_num + 1
        
        # Early termination if all converged
        if cp.all(converged):
            break
    
    # Set iterations for non-converged genes
    iterations[~converged] = max_iter
    
    # Transfer results back to CPU
    return beta_gpu.get(), iterations.get(), converged.get()


def fit_beta_coefficients_gpu_vectorized(
    count_batch: np.ndarray,
    design_matrix: np.ndarray,
    beta_init_batch: np.ndarray,
    offset_vector: np.ndarray,
    dispersion_batch: np.ndarray,
    max_iter: int = 100,
    tolerance: float = 1e-3,
    dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized GPU implementation for batch beta fitting.
    
    This is an alias for fit_beta_coefficients_gpu_batch as both are now
    fully vectorized. Kept for backward compatibility.
    
    Args:
        count_batch: Count data for batch (batch_size × samples).
        design_matrix: Design matrix (samples × features).
        beta_init_batch: Initial beta values (batch_size × features).
        offset_vector: Offset vector.
        dispersion_batch: Dispersion parameters for batch.
        max_iter: Maximum iterations.
        tolerance: Convergence tolerance.
        dtype: Data type for computation.
        
    Returns:
        Tuple of (fitted_beta, n_iterations, converged) for batch.
    """
    return fit_beta_coefficients_gpu_batch(
        count_batch, design_matrix, beta_init_batch, offset_vector,
        dispersion_batch, max_iter, tolerance, dtype
    )


def _batch_solve_weighted_least_squares(
    X: cp.ndarray,
    XT: cp.ndarray,
    weights: cp.ndarray,
    scores: cp.ndarray,
    n_features: int,
    batch_size: int,
    dtype: cp.dtype,
    converged: cp.ndarray
) -> cp.ndarray:
    """
    Efficiently solve weighted least squares for multiple genes.
    
    Solves (X^T W X) delta = score for each gene in the batch using
    fully vectorized operations.
    """
    # For small feature sizes, use einsum for maximum efficiency
    if n_features <= 30:
        # Compute sqrt(W) for numerical stability
        sqrt_weights = cp.sqrt(weights)  # (batch_size, n_samples)
        
        # Mask out converged genes to save computation
        active_mask = ~converged
        if not cp.any(active_mask):
            return cp.zeros((batch_size, n_features), dtype=dtype)
        
        # Create weighted X: sqrt(W)[:, :, None] * X[None, :, :]
        # Only compute for active genes
        active_sqrt_weights = sqrt_weights[active_mask]
        WX = active_sqrt_weights[:, :, cp.newaxis] * X[cp.newaxis, :, :]
        
        # Batch matrix multiplication: WX^T @ WX
        # Using optimal einsum path
        info_matrices = cp.einsum('bsi,bsj->bij', WX, WX, optimize=True)
        
        # Add regularization for numerical stability
        reg_eye = cp.eye(n_features, dtype=dtype) * 1e-6
        info_matrices += reg_eye
        
        # Initialize result
        deltas = cp.zeros((batch_size, n_features), dtype=dtype)
        active_scores = scores[active_mask]
        
        try:
            # Batch Cholesky decomposition
            L = cp.linalg.cholesky(info_matrices)
            
            # Batch solve using triangular systems
            # First solve L @ y = scores
            y = _batch_forward_substitution(L, active_scores)
            
            # Then solve L^T @ delta = y
            active_deltas = _batch_backward_substitution(L, y)
            deltas[active_mask] = active_deltas
            
        except cp.linalg.LinAlgError:
            # Fallback to batch LU decomposition
            try:
                active_deltas = cp.linalg.solve(
                    info_matrices, 
                    active_scores[:, :, cp.newaxis]
                ).squeeze(-1)
                deltas[active_mask] = active_deltas
            except:
                # Ultimate fallback: process problematic genes individually
                deltas[active_mask] = _fallback_solve(
                    X, weights[active_mask], active_scores, n_features
                )
    
    else:
        # For larger feature sizes, use blocked algorithm
        deltas = _blocked_weighted_least_squares(
            X, XT, weights, scores, n_features, batch_size, 
            dtype, converged
        )
    
    return deltas


def _batch_forward_substitution(L: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
    """
    Batch forward substitution for lower triangular systems.
    
    Solves L @ x = b for multiple systems simultaneously.
    """
    batch_size, n = L.shape[0], L.shape[1]
    x = cp.zeros_like(b)
    
    for i in range(n):
        if i == 0:
            x[:, i] = b[:, i] / L[:, i, i]
        else:
            x[:, i] = (b[:, i] - cp.sum(L[:, i, :i] * x[:, :i], axis=1)) / L[:, i, i]
    
    return x


def _batch_backward_substitution(L: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
    """
    Batch backward substitution for upper triangular systems.
    
    Solves L^T @ x = b for multiple systems simultaneously.
    """
    batch_size, n = L.shape[0], L.shape[1]
    x = cp.zeros_like(b)
    
    for i in range(n-1, -1, -1):
        if i == n-1:
            x[:, i] = b[:, i] / L[:, i, i]
        else:
            x[:, i] = (b[:, i] - cp.sum(L[:, i+1:, i] * x[:, i+1:], axis=1)) / L[:, i, i]
    
    return x


def _blocked_weighted_least_squares(
    X: cp.ndarray,
    XT: cp.ndarray,
    weights: cp.ndarray,
    scores: cp.ndarray,
    n_features: int,
    batch_size: int,
    dtype: cp.dtype,
    converged: cp.ndarray
) -> cp.ndarray:
    """
    Blocked algorithm for large feature sizes to manage memory.
    """
    deltas = cp.zeros((batch_size, n_features), dtype=dtype)
    
    # Process in sub-batches for memory efficiency
    sub_batch_size = min(batch_size, 50)
    
    for start in range(0, batch_size, sub_batch_size):
        end = min(start + sub_batch_size, batch_size)
        sub_mask = ~converged[start:end]
        
        if not cp.any(sub_mask):
            continue
        
        active_indices = cp.arange(start, end)[sub_mask]
        sub_weights = weights[active_indices]
        sub_scores = scores[active_indices]
        
        # Compute information matrices for sub-batch
        sub_batch_actual = len(active_indices)
        sub_info = cp.empty((sub_batch_actual, n_features, n_features), dtype=dtype)
        
        # Vectorized computation of X^T @ diag(W) @ X
        for i in range(sub_batch_actual):
            W_diag = sub_weights[i]
            # Use optimized BLAS operations
            WX = X * W_diag[:, cp.newaxis]
            cp.matmul(XT, WX, out=sub_info[i])
        
        # Add regularization
        reg_eye = cp.eye(n_features, dtype=dtype) * 1e-6
        sub_info += reg_eye
        
        # Batch solve
        try:
            sub_deltas = cp.linalg.solve(
                sub_info, 
                sub_scores[:, :, cp.newaxis]
            ).squeeze(-1)
            deltas[active_indices] = sub_deltas
        except:
            # Fallback for problematic matrices
            deltas[active_indices] = _fallback_solve(
                X, sub_weights, sub_scores, n_features
            )
    
    return deltas


def _fallback_solve(
    X: cp.ndarray,
    weights: cp.ndarray,
    scores: cp.ndarray,
    n_features: int
) -> cp.ndarray:
    """
    Fallback solver for problematic cases using pseudo-inverse.
    """
    batch_size = weights.shape[0]
    deltas = cp.zeros((batch_size, n_features), dtype=weights.dtype)
    
    for i in range(batch_size):
        try:
            W_diag = weights[i]
            WX = X * W_diag[:, cp.newaxis]
            info = X.T @ WX + cp.eye(n_features) * 1e-4
            deltas[i] = cp.linalg.solve(info, scores[i])
        except:
            # Use pseudo-inverse as last resort
            try:
                deltas[i] = cp.linalg.pinv(info) @ scores[i]
            except:
                # Skip this gene if all else fails
                deltas[i] = 0
    
    return deltas


def select_gpu_implementation(
    batch_size: int,
    n_samples: int,
    n_features: int,
    available_memory: int
) -> str:
    """
    Select the best GPU implementation based on problem size and memory.
    
    Args:
        batch_size: Number of genes in batch.
        n_samples: Number of samples.
        n_features: Number of features.
        available_memory: Available GPU memory in bytes.
        
    Returns:
        Implementation name: 'vectorized' or 'loop'.
    """
    # Estimate memory requirements
    bytes_per_float = 4  # float32
    
    # Memory for main arrays: y, X, beta, weights, etc.
    main_memory = (
        batch_size * n_samples +  # y
        n_samples * n_features +  # X
        batch_size * n_features +  # beta
        batch_size * n_samples * 3  # mu, variance, weights
    ) * bytes_per_float
    
    # Memory for batch operations (einsum needs temporary space)
    if n_features <= 30:
        # Einsum approach needs WX array
        batch_op_memory = batch_size * n_samples * n_features * bytes_per_float
    else:
        # Blocked approach processes sub-batches
        sub_batch_size = min(batch_size, 50)
        batch_op_memory = sub_batch_size * n_features * n_features * bytes_per_float
    
    total_memory = main_memory + batch_op_memory
    
    # Check if we have enough memory (with safety margin)
    if total_memory > available_memory * 0.8:
        warnings.warn(
            f"GPU memory may be insufficient. Required: {total_memory/1e9:.1f}GB, "
            f"Available: {available_memory/1e9:.1f}GB. Consider reducing batch_size."
        )
    
    return 'vectorized'