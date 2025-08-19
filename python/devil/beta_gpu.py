"""
GPU-accelerated beta coefficient estimation using exact R package algorithm.

This implementation maintains mathematical exactness with the R package while
providing GPU acceleration for batch processing of multiple genes.
"""

from typing import Tuple, Optional
import numpy as np
import warnings

# GPU imports with fallback handling
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

from .gpu import is_gpu_available, to_gpu, to_cpu, GPUMemoryManager


def init_beta_gpu(
    count_matrix: np.ndarray,
    design_matrix: np.ndarray,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Initialize beta coefficients for multiple genes on GPU using exact algorithm.
    
    Args:
        count_matrix: Count data (genes × samples).
        design_matrix: Design matrix (samples × features).
        dtype: Data type for GPU computation.
        
    Returns:
        Initial beta estimates (genes × features).
    """
    if not is_gpu_available():
        # Fallback to CPU implementation
        from .beta import init_beta_matrix
        return init_beta_matrix(count_matrix, design_matrix)
    
    n_genes, n_samples = count_matrix.shape
    n_features = design_matrix.shape[1]
    
    # Transfer to GPU
    counts_gpu = to_gpu(count_matrix, dtype)
    X_gpu = to_gpu(design_matrix, dtype)
    
    # Apply log1p transformation (vectorized across all genes)
    # This matches: VectorXd norm_log_count_mat = y.array().log1p();
    norm_log_counts = cp.log1p(counts_gpu)  # Shape: (genes, samples)
    
    # Solve X @ beta = norm_log_counts for each gene
    # Using batch linear solver for efficiency
    try:
        # Transpose to match expected shapes for batch solve
        # norm_log_counts.T shape: (samples, genes)
        # X_gpu shape: (samples, features)
        # Result shape: (features, genes)
        beta_init_T = cp.linalg.lstsq(X_gpu, norm_log_counts.T, rcond=None)[0]
        beta_init = beta_init_T.T  # Shape: (genes, features)
    except Exception:
        # Fallback: solve gene by gene
        beta_init = cp.zeros((n_genes, n_features), dtype=dtype)
        for gene_idx in range(n_genes):
            try:
                beta_init[gene_idx, :] = cp.linalg.lstsq(
                    X_gpu, norm_log_counts[gene_idx, :], rcond=None
                )[0]
            except Exception:
                # Final fallback: use zeros
                beta_init[gene_idx, :] = 0.0
    
    return to_cpu(beta_init)


def beta_fit_gpu_batch(
    y_batch: np.ndarray,
    design_matrix: np.ndarray,
    mu_beta_batch: np.ndarray,
    offset_vector: np.ndarray,
    dispersion_batch: np.ndarray,
    max_iter: int = 100,
    tolerance: float = 1e-3,
    dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit beta coefficients for a batch of genes using exact GPU algorithm.
    
    This implements the exact mathematical algorithm from the R package's beta_fit
    function, but vectorized across multiple genes on GPU.
    
    Args:
        y_batch: Count data (batch_size × samples).
        design_matrix: Design matrix (samples × features).
        mu_beta_batch: Initial beta coefficients (batch_size × features).
        offset_vector: Offset values (samples,).
        dispersion_batch: Dispersion parameters (batch_size,).
        max_iter: Maximum iterations.
        tolerance: Convergence tolerance.
        dtype: Data type for computation.
        
    Returns:
        Tuple of (fitted_beta, n_iterations, converged).
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available")
    
    batch_size, n_samples = y_batch.shape
    n_features = design_matrix.shape[1]
    
    # Transfer to GPU
    y_gpu = to_gpu(y_batch, dtype)
    X_gpu = to_gpu(design_matrix, dtype)
    mu_beta_gpu = to_gpu(mu_beta_batch, dtype)
    offset_gpu = to_gpu(offset_vector, dtype)
    dispersion_gpu = to_gpu(dispersion_batch, dtype)
    
    # Convert dispersion to k (inverse dispersion) as in C++ code
    k_gpu = cp.where(dispersion_gpu > 0, 1.0 / dispersion_gpu, 1e6)
    k_gpu = k_gpu[:, cp.newaxis]  # Shape: (batch_size, 1)
    
    # Initialize convergence tracking
    converged = cp.zeros(batch_size, dtype=bool)
    iterations = cp.zeros(batch_size, dtype=cp.int32)
    
    # Regularization constant (as in C++ code)
    inv_sigma_beta_const = 0.01
    
    for iter_count in range(max_iter):
        if cp.all(converged):
            break
        
        # Only process non-converged genes
        active_mask = ~converged
        if not cp.any(active_mask):
            break
        
        # Calculate w_q: exp(-X @ mu_beta - offset) for active genes
        # This matches: w_q = (-X * mu_beta - off).array().exp();
        linear_pred = -(mu_beta_gpu @ X_gpu.T) - offset_gpu[cp.newaxis, :]  # (batch_size, samples)
        linear_pred = cp.clip(linear_pred, -50, 50)  # Numerical stability
        w_q = cp.exp(linear_pred)
        
        # Calculate mu_g: (k + y) / (1 + k * w_q) for active genes
        # This matches: mu_g = (k + y.array()) / (1 + k * w_q.array());
        mu_g = (k_gpu + y_gpu) / (1.0 + k_gpu * w_q)
        
        # Calculate weights: mu_g * w_q
        weights = mu_g * w_q  # Shape: (batch_size, samples)
        
        # Calculate Zigma (covariance matrix) for each gene in batch
        # This matches: Zigma = (k * X.transpose() * (mu_g.array() * w_q.array()).matrix().asDiagonal() * X).inverse();
        
        # Vectorized computation of X.T @ diag(weights) @ X for all genes
        XTwX_batch = cp.zeros((batch_size, n_features, n_features), dtype=dtype)
        
        for i in range(n_features):
            for j in range(n_features):
                # Use proper broadcasting: X[None, :, i] has shape (1, n_samples)
                # which broadcasts correctly with weights (batch_size, n_samples)
                XTwX_batch[:, i, j] = k_gpu.flatten() * cp.sum(
                    X_gpu[None, :, i] * weights * X_gpu[None, :, j], axis=1
                )
        
        # Add regularization
        reg_eye = cp.eye(n_features, dtype=dtype) * inv_sigma_beta_const
        XTwX_batch += reg_eye[cp.newaxis, :, :]  # Broadcast regularization
        
        # Compute inverse (Zigma) for each gene
        try:
            Zigma_batch = cp.linalg.inv(XTwX_batch)
        except cp.linalg.LinAlgError:
            # Fallback to pseudo-inverse for singular matrices
            Zigma_batch = cp.linalg.pinv(XTwX_batch)
        
        # Calculate delta (parameter update) for each gene
        # This matches: delta = Zigma * (k * X.transpose() * (mu_g.array() * w_q.array() - 1).matrix());
        residual = weights - 1.0  # Shape: (batch_size, samples)
        
        # Vectorized computation of X.T @ residual for all genes
        XT_residual = cp.zeros((batch_size, n_features), dtype=dtype)
        for i in range(n_features):
            # Use proper broadcasting: X[None, :, i] has shape (1, n_samples)
            # which broadcasts correctly with residual (batch_size, n_samples)
            XT_residual[:, i] = k_gpu.flatten() * cp.sum(X_gpu[None, :, i] * residual, axis=1)
        
        # Compute delta using batch matrix multiplication
        # Zigma_batch: (batch_size, n_features, n_features)
        # XT_residual: (batch_size, n_features)
        delta = cp.zeros((batch_size, n_features), dtype=dtype)
        for gene_idx in range(batch_size):
            if active_mask[gene_idx]:
                delta[gene_idx, :] = Zigma_batch[gene_idx, :, :] @ XT_residual[gene_idx, :]
        
        # Update mu_beta for active genes only
        mu_beta_gpu[active_mask] += delta[active_mask]
        
        # Check for NaN (as in C++ code)
        nan_mask = ~cp.all(cp.isfinite(delta), axis=1)
        converged[nan_mask] = True
        
        # Check convergence for active genes
        max_delta = cp.max(cp.abs(delta), axis=1)
        newly_converged = (max_delta < tolerance) & active_mask
        converged[newly_converged] = True
        
        # Update iteration count for active genes
        iterations[active_mask] += 1
    
    # Set final iteration count
    iterations[~converged] = max_iter
    
    return to_cpu(mu_beta_gpu), to_cpu(iterations), to_cpu(converged)


def beta_fit_group_gpu_batch(
    y_batch: np.ndarray,
    mu_beta_batch: np.ndarray,
    offset_vector: np.ndarray,
    dispersion_batch: np.ndarray,
    max_iter: int = 100,
    tolerance: float = 1e-3,
    dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit single beta coefficient (intercept-only) for batch of genes on GPU.
    This implements the exact beta_fit_group algorithm for GPU batch processing.
    Args:
        y_batch: Count data (batch_size × samples).
        mu_beta_batch: Initial beta coefficients (batch_size,).
        offset_vector: Offset values (samples,).
        dispersion_batch: Dispersion parameters (batch_size,).
        max_iter: Maximum iterations.
        tolerance: Convergence tolerance.
        dtype: Data type for computation.
    Returns:
        Tuple of (fitted_beta, n_iterations, converged).
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available")
    
    batch_size, n_samples = y_batch.shape
    
    # Transfer to GPU
    y_gpu = to_gpu(y_batch, dtype)
    mu_beta_gpu = to_gpu(mu_beta_batch, dtype)
    offset_gpu = to_gpu(offset_vector, dtype)
    dispersion_gpu = to_gpu(dispersion_batch, dtype)
    
    # Convert dispersion to k (inverse dispersion)
    k_gpu = cp.where(dispersion_gpu > 0, 1.0 / dispersion_gpu, 1e6)
    
    # Initialize convergence tracking
    converged = cp.zeros(batch_size, dtype=bool)
    iterations = cp.zeros(batch_size, dtype=cp.int32)
    
    for iter_count in range(max_iter):
        if cp.all(converged):
            break
        
        # Only process non-converged genes
        active_mask = ~converged
        if not cp.any(active_mask):
            break
        
        # Calculate w_q: exp(-mu_beta - offset) for all genes
        # This matches: w_q = (-mu_beta * ones - off).array().exp();
        linear_pred = -mu_beta_gpu[:, cp.newaxis] - offset_gpu[cp.newaxis, :]
        linear_pred = cp.clip(linear_pred, -50, 50)  # Numerical stability
        w_q = cp.exp(linear_pred)  # Shape: (batch_size, samples)
        
        # Calculate mu_g: (k + y) / (1 + k * w_q)
        # This matches: mu_g = (k + y.array()) / (1 + k * w_q.array());
        k_expanded = k_gpu[:, cp.newaxis]  # Shape: (batch_size, 1)
        mu_g = (k_expanded + y_gpu) / (1.0 + k_expanded * w_q)
        
        # Calculate Zigma (scalar covariance): 1 / (k * sum(mu_g * w_q))
        # This matches: Zigma = 1.0 / (k * (mu_g.array() * w_q.array()).sum());
        weights_sum = cp.sum(mu_g * w_q, axis=1)  # Shape: (batch_size,)
        Zigma = 1.0 / (k_gpu * weights_sum)
        
        # Calculate delta (parameter update)
        # This matches: delta = Zigma * (k * (mu_g.array() * w_q.array() - 1).sum());
        residual_sum = cp.sum(mu_g * w_q - 1.0, axis=1)  # Shape: (batch_size,)
        delta = Zigma * k_gpu * residual_sum
        
        # Update mu_beta for active genes only
        mu_beta_gpu[active_mask] += delta[active_mask]
        
        # Check for NaN
        nan_mask = ~cp.isfinite(delta)
        converged[nan_mask] = True
        
        # Check convergence for active genes
        newly_converged = (cp.abs(delta) < tolerance) & active_mask
        converged[newly_converged] = True
        
        # Update iteration count for active genes
        iterations[active_mask] += 1
    
    # Set final iteration count
    iterations[~converged] = max_iter
    
    return to_cpu(mu_beta_gpu), to_cpu(iterations), to_cpu(converged)


def fit_beta_coefficients_gpu(
    y_batch: np.ndarray,
    design_matrix: np.ndarray,
    beta_init_batch: Optional[np.ndarray] = None,
    offset_vector: Optional[np.ndarray] = None,
    dispersion_batch: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tolerance: float = 1e-3,
    dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    High-level interface for exact GPU beta coefficient fitting.
    Args:
        y_batch: Count data (batch_size × samples).
        design_matrix: Design matrix (samples × features).
        beta_init_batch: Initial beta coefficients. If None, will be computed.
        offset_vector: Offset vector. If None, defaults to zeros.
        dispersion_batch: Dispersion parameters. If None, defaults to ones.
        max_iter: Maximum iterations.
        tolerance: Convergence tolerance.
        dtype: Data type for computation.
    Returns:
        Tuple of (fitted_beta, n_iterations, converged).
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available")
    
    batch_size, n_samples = y_batch.shape
    n_features = design_matrix.shape[1]
    
    # Handle defaults
    if offset_vector is None:
        offset_vector = np.zeros(n_samples)
    
    if dispersion_batch is None:
        dispersion_batch = np.ones(batch_size)
    
    if beta_init_batch is None:
        beta_init_batch = init_beta_gpu(y_batch, design_matrix, dtype)
    
    # Handle all-zero genes
    all_zero_mask = np.all(y_batch == 0, axis=1)
    if np.any(all_zero_mask):
        beta_init_batch[all_zero_mask, :] = 0.0
    
    # Check for intercept-only model (optimization)
    if n_features == 1 and np.allclose(design_matrix[:, 0], 1.0):
        # Use the optimized group fitting function
        fitted_beta_flat, iterations, converged = beta_fit_group_gpu_batch(
            y_batch, beta_init_batch[:, 0], offset_vector, dispersion_batch,
            max_iter, tolerance, dtype
        )
        fitted_beta = fitted_beta_flat[:, np.newaxis]  # Reshape to (batch_size, 1)
    else:
        # Use the general fitting function
        fitted_beta, iterations, converged = beta_fit_gpu_batch(
            y_batch, design_matrix, beta_init_batch, offset_vector,
            dispersion_batch, max_iter, tolerance, dtype
        )
    
    # Set converged flag for all-zero genes
    converged[all_zero_mask] = True
    iterations[all_zero_mask] = 1
    
    return fitted_beta, iterations, converged


# Memory-efficient batch processing function
def fit_beta_coefficients_gpu_batch(
    count_matrix: np.ndarray,
    design_matrix: np.ndarray,
    beta_init: Optional[np.ndarray] = None,
    offset_vector: Optional[np.ndarray] = None,
    dispersion_vector: Optional[np.ndarray] = None,
    batch_size: int = 1024,
    max_iter: int = 100,
    tolerance: float = 1e-3,
    dtype: np.dtype = np.float32,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process large count matrices in batches using GPU acceleration.
    Args:
        count_matrix: Count data (genes × samples).
        design_matrix: Design matrix (samples × features).
        beta_init: Initial beta coefficients. If None, will be computed.
        offset_vector: Offset vector. If None, defaults to zeros.
        dispersion_vector: Dispersion parameters. If None, defaults to ones.
        batch_size: Number of genes to process per batch.
        max_iter: Maximum iterations.
        tolerance: Convergence tolerance.
        dtype: Data type for computation.
        verbose: Whether to show progress.
    Returns:
        Tuple of (fitted_beta, n_iterations, converged) for all genes.
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available")
    
    n_genes, n_samples = count_matrix.shape
    n_features = design_matrix.shape[1]
    
    # Handle defaults
    if offset_vector is None:
        offset_vector = np.zeros(n_samples)
    if dispersion_vector is None:
        dispersion_vector = np.ones(n_genes)
    if beta_init is None:
        if verbose:
            print("Computing initial beta coefficients...")
        beta_init = init_beta_gpu(count_matrix, design_matrix, dtype)
    
    # Initialize result arrays
    fitted_beta = np.zeros((n_genes, n_features), dtype=np.float64)
    iterations = np.zeros(n_genes, dtype=np.int32)
    converged = np.zeros(n_genes, dtype=bool)
    
    # Process in batches
    if verbose:
        from tqdm import tqdm
        batch_iterator = tqdm(range(0, n_genes, batch_size), desc="GPU beta fitting")
    else:
        batch_iterator = range(0, n_genes, batch_size)
    
    with GPUMemoryManager():
        for start_idx in batch_iterator:
            end_idx = min(start_idx + batch_size, n_genes)
            
            try:
                batch_beta, batch_iter, batch_conv = fit_beta_coefficients_gpu(
                    count_matrix[start_idx:end_idx, :],
                    design_matrix,
                    beta_init[start_idx:end_idx, :],
                    offset_vector,
                    dispersion_vector[start_idx:end_idx],
                    max_iter, tolerance, dtype
                )
                
                fitted_beta[start_idx:end_idx, :] = batch_beta
                iterations[start_idx:end_idx] = batch_iter
                converged[start_idx:end_idx] = batch_conv
                
            except Exception as e:
                if verbose:
                    print(f"GPU batch {start_idx}-{end_idx} failed: {e}")
                    print("Falling back to CPU for this batch")
                
                # CPU fallback using exact algorithm
                from .beta import fit_beta_coefficients
                for gene_idx in range(start_idx, end_idx):
                    try:
                        beta, n_iter, conv = fit_beta_coefficients(
                            count_matrix[gene_idx, :],
                            design_matrix,
                            beta_init[gene_idx, :],
                            offset_vector,
                            dispersion_vector[gene_idx],
                            max_iter, tolerance
                        )
                        fitted_beta[gene_idx, :] = beta
                        iterations[gene_idx] = n_iter
                        converged[gene_idx] = conv
                    except Exception:
                        # Final fallback: use initial values
                        fitted_beta[gene_idx, :] = beta_init[gene_idx, :]
                        iterations[gene_idx] = 0
                        converged[gene_idx] = False
    
    return fitted_beta, iterations, converged

