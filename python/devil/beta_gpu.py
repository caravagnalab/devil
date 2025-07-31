"""GPU-accelerated beta coefficient estimation."""

from typing import Tuple, Optional
import numpy as np
import warnings

from .gpu import is_gpu_available, to_gpu, to_cpu, CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp
    import cupyx.scipy.linalg as cp_linalg


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
    
    # QR decomposition
    Q, R = cp_linalg.qr(X_gpu)
    
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
        
        # Solve for initial beta
        batch_beta = cp_linalg.solve_triangular(
            R, Q.T @ norm_log_counts, lower=False
        ).T
        
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
    Fit beta coefficients for a batch of genes on GPU.
    
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
    
    # Transfer data to GPU
    y_gpu = to_gpu(count_batch, dtype)
    X_gpu = to_gpu(design_matrix, dtype)
    beta_gpu = to_gpu(beta_init_batch, dtype)
    offset_gpu = to_gpu(offset_vector, dtype)
    k_gpu = to_gpu(1.0 / dispersion_batch, dtype)  # Convert to precision
    
    # Initialize tracking arrays
    iterations = cp.zeros(batch_size, dtype=cp.int32)
    converged = cp.zeros(batch_size, dtype=bool)
    
    # Iterative fitting for each gene in batch
    for iter_num in range(max_iter):
        # Calculate linear predictor and mean
        eta = cp.einsum('ij,kj->ik', beta_gpu, X_gpu.T)  # batch_size × samples
        eta += offset_gpu[cp.newaxis, :]
        mu = cp.exp(eta)
        mu = cp.maximum(mu, 1e-10)  # Numerical stability
        
        # Negative binomial variance and weights
        k_expanded = k_gpu[:, cp.newaxis]  # batch_size × 1
        variance = mu + mu**2 / k_expanded
        weights = mu / variance
        
        # Calculate residuals
        residuals = (y_gpu - mu) / variance
        
        # Score and information matrix for each gene
        # This is the computationally intensive part
        deltas = cp.zeros_like(beta_gpu)
        
        for gene_idx in range(batch_size):
            if converged[gene_idx]:
                continue
                
            # Score vector
            w_r = weights[gene_idx] * residuals[gene_idx]
            score = X_gpu.T @ w_r
            
            # Information matrix (Fisher information)
            W_diag = weights[gene_idx]
            info = X_gpu.T @ (W_diag[:, cp.newaxis] * X_gpu)
            
            # Solve for update (with regularization for stability)
            try:
                reg_info = info + cp.eye(n_features) * 1e-6
                delta = cp_linalg.solve(reg_info, score)
                deltas[gene_idx] = delta
                
                # Check convergence for this gene
                if cp.max(cp.abs(delta)) < tolerance:
                    converged[gene_idx] = True
                    iterations[gene_idx] = iter_num + 1
                    
            except cp.linalg.LinAlgError:
                # Singular matrix - mark as converged with current values
                converged[gene_idx] = True
                iterations[gene_idx] = iter_num + 1
        
        # Update beta coefficients
        beta_gpu += deltas
        
        # Check if all genes converged
        if cp.all(converged):
            break
    
    # Set iterations for non-converged genes
    iterations = cp.where(converged, iterations, max_iter)
    
    return to_cpu(beta_gpu), to_cpu(iterations), to_cpu(converged)


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
    
    More memory-intensive but potentially faster for large batches.
    
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
    
    # Transfer data to GPU
    y_gpu = to_gpu(count_batch, dtype)  # batch_size × samples
    X_gpu = to_gpu(design_matrix, dtype)  # samples × features
    beta_gpu = to_gpu(beta_init_batch, dtype)  # batch_size × features
    offset_gpu = to_gpu(offset_vector, dtype)  # samples
    k_gpu = to_gpu(1.0 / dispersion_batch, dtype)  # batch_size
    
    # Tracking arrays
    iterations = cp.zeros(batch_size, dtype=cp.int32)
    converged = cp.zeros(batch_size, dtype=bool)
    
    # Pre-compute X.T for efficiency
    XT = X_gpu.T  # features × samples
    
    for iter_num in range(max_iter):
        # Vectorized computation across all genes in batch
        
        # Linear predictor: batch_size × samples
        eta = beta_gpu @ XT + offset_gpu[cp.newaxis, :]
        mu = cp.exp(eta)
        mu = cp.maximum(mu, 1e-10)
        
        # Variance and weights: batch_size × samples
        k_expanded = k_gpu[:, cp.newaxis]
        variance = mu + mu**2 / k_expanded
        weights = mu / variance
        
        # Residuals: batch_size × samples
        residuals = (y_gpu - mu) / variance
        weighted_residuals = weights * residuals
        
        # Score vectors: batch_size × features
        # This is matrix multiplication: (batch_size × samples) @ (samples × features)
        scores = weighted_residuals @ X_gpu
        
        # Information matrices are more complex to vectorize efficiently
        # We'll compute them in a loop but with vectorized operations
        deltas = cp.zeros_like(beta_gpu)
        
        # Batch process information matrices
        for gene_idx in range(batch_size):
            if converged[gene_idx]:
                continue
                
            W_diag = weights[gene_idx]  # samples
            # Information matrix: X.T @ diag(W) @ X
            WX = W_diag[:, cp.newaxis] * X_gpu  # samples × features
            info = XT @ WX  # features × features
            
            # Regularize for numerical stability
            info += cp.eye(n_features) * 1e-6
            
            try:
                delta = cp_linalg.solve(info, scores[gene_idx])
                deltas[gene_idx] = delta
                
                # Check convergence
                if cp.max(cp.abs(delta)) < tolerance:
                    converged[gene_idx] = True
                    iterations[gene_idx] = iter_num + 1
                    
            except cp.linalg.LinAlgError:
                converged[gene_idx] = True
                iterations[gene_idx] = iter_num + 1
        
        # Update beta
        beta_gpu += deltas
        
        # Early termination if all converged
        if cp.all(converged):
            break
    
    # Set final iteration counts
    iterations = cp.where(converged, iterations, max_iter)
    
    return to_cpu(beta_gpu), to_cpu(iterations), to_cpu(converged)


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
    # Estimate memory requirements for vectorized approach
    bytes_per_float = 4  # float32
    
    # Main arrays:
    # - y_gpu: batch_size × samples
    # - beta_gpu: batch_size × features  
    # - Working arrays: ~3 × batch_size × samples
    vectorized_memory = bytes_per_float * (
        batch_size * samples +
        batch_size * n_features +
        3 * batch_size * samples
    )
    
    # Use vectorized if we have enough memory and batch is large enough
    if vectorized_memory < available_memory * 0.7 and batch_size > 10:
        return 'vectorized'
    else:
        return 'loop'