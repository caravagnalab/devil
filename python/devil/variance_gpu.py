"""GPU-accelerated variance estimation functions."""

from typing import Tuple
import numpy as np

from .gpu import is_gpu_available, to_gpu, to_cpu, CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp
    try:
        import cupyx.scipy.linalg as cp_linalg
    except ImportError:
        cp_linalg = None


def compute_hessian_gpu_batch(
    beta_batch: np.ndarray,
    precision_batch: np.ndarray,
    y_batch: np.ndarray,
    design_matrix: np.ndarray,
    size_factors: np.ndarray,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Compute inverse Hessian matrices for a batch of genes on GPU.
    Args:
        beta_batch: Regression coefficients (batch_size × features).
        precision_batch: Inverse overdispersion parameters (batch_size,).
        y_batch: Response values (batch_size × samples).
        design_matrix: Predictor variables (samples × features).
        size_factors: Normalization factors (samples,).
        dtype: Data type for computation.
    Returns:
        Inverse Hessian matrices (batch_size × features × features).
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available")
    
    batch_size, n_samples = y_batch.shape
    n_features = design_matrix.shape[1]
    
    # Transfer to GPU
    beta_gpu = to_gpu(beta_batch, dtype)
    precision_gpu = to_gpu(precision_batch, dtype)
    y_gpu = to_gpu(y_batch, dtype)
    X_gpu = to_gpu(design_matrix, dtype)
    sf_gpu = to_gpu(size_factors, dtype)
    
    # Compute linear predictors and means
    eta = beta_gpu @ X_gpu.T  # batch_size × samples
    mu = sf_gpu[cp.newaxis, :] * cp.exp(eta)  # batch_size × samples
    
    # Compute weights for each gene
    precision_expanded = precision_gpu[:, cp.newaxis]  # batch_size × 1
    weights = mu / (1 + mu / precision_expanded)  # batch_size × samples
    
    # Compute Hessian matrices
    hessians = cp.zeros((batch_size, n_features, n_features), dtype=dtype)
    
    for gene_idx in range(batch_size):
        # Weight matrix for this gene
        W_diag = weights[gene_idx]  # samples
        
        # Hessian = -X.T @ diag(W) @ X
        WX = W_diag[:, cp.newaxis] * X_gpu  # samples × features
        H = -X_gpu.T @ WX  # features × features
        
        # Compute inverse with regularization
        try:
            H_reg = H - cp.eye(n_features) * 1e-8  # Small regularization
            hessians[gene_idx] = cp_linalg.inv(-H_reg)
        except cp.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            hessians[gene_idx] = cp_linalg.pinv(-H)
    
    return to_cpu(hessians)


def compute_scores_gpu_batch(
    design_matrix: np.ndarray,
    y_batch: np.ndarray,
    beta_batch: np.ndarray,
    overdispersion_batch: np.ndarray,
    size_factors: np.ndarray,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Compute score residuals for a batch of genes on GPU.
    
    Args:
        design_matrix: Predictor variables (samples × features).
        y_batch: Response values (batch_size × samples).
        beta_batch: Regression coefficients (batch_size × features).
        overdispersion_batch: Dispersion parameters (batch_size,).
        size_factors: Normalization factors (samples,).
        dtype: Data type for computation.
    Returns:
        Score residuals (batch_size × samples × features).
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available")
    
    batch_size, n_samples = y_batch.shape
    n_features = design_matrix.shape[1]
    
    # Transfer to GPU
    X_gpu = to_gpu(design_matrix, dtype)
    y_gpu = to_gpu(y_batch, dtype)
    beta_gpu = to_gpu(beta_batch, dtype)
    disp_gpu = to_gpu(overdispersion_batch, dtype)
    sf_gpu = to_gpu(size_factors, dtype)
    
    # Compute means and variances
    eta = beta_gpu @ X_gpu.T  # batch_size × samples
    mu = sf_gpu[cp.newaxis, :] * cp.exp(eta)  # batch_size × samples
    
    # Negative binomial variance
    alpha = 1.0 / disp_gpu[:, cp.newaxis]  # batch_size × 1
    variance = mu + alpha * mu**2  # batch_size × samples
    
    # Pearson residuals
    residuals = (y_gpu - mu) / cp.sqrt(variance)  # batch_size × samples
    
    # Weights
    weights = cp.sqrt(mu**2 / variance)  # batch_size × samples
    
    # Score contributions: batch_size × samples × features
    weighted_residuals = residuals * weights  # batch_size × samples
    
    # Broadcast multiplication to get scores
    scores = weighted_residuals[:, :, cp.newaxis] * X_gpu[cp.newaxis, :, :]
    
    return to_cpu(scores)


def compute_sandwich_estimator_gpu_batch(
    design_matrix: np.ndarray,
    y_batch: np.ndarray,
    beta_batch: np.ndarray,
    overdispersion_batch: np.ndarray,
    size_factors: np.ndarray,
    clusters: np.ndarray,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Compute clustered sandwich variance estimators for a batch of genes on GPU.
    Args:
        design_matrix: Predictor variables (samples × features).
        y_batch: Response values (batch_size × samples).
        beta_batch: Regression coefficients (batch_size × features).
        overdispersion_batch: Dispersion parameters (batch_size,).
        size_factors: Normalization factors (samples,).
        clusters: Cluster assignments (samples,).
        dtype: Data type for computation.
    Returns:
        Sandwich variance matrices (batch_size × features × features).
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available")
    
    batch_size, n_samples = y_batch.shape
    n_features = design_matrix.shape[1]
    
    # Transfer to GPU
    clusters_gpu = to_gpu(clusters.astype(np.int32))
    unique_clusters = cp.unique(clusters_gpu)
    n_clusters = len(unique_clusters)
    
    # Compute bread (inverse Hessian) matrices
    precision_batch = 1.0 / overdispersion_batch
    precision_batch = np.where(overdispersion_batch > 0, precision_batch, 1e6)
    
    bread_matrices = compute_hessian_gpu_batch(
        beta_batch, precision_batch, y_batch,
        design_matrix, size_factors, dtype
    )
    bread_gpu = to_gpu(bread_matrices, dtype)
    
    # Compute score residuals
    scores_batch = compute_scores_gpu_batch(
        design_matrix, y_batch, beta_batch,
        overdispersion_batch, size_factors, dtype
    )
    scores_gpu = to_gpu(scores_batch, dtype)  # batch_size × samples × features
    
    # Small sample adjustment
    adjustment = n_clusters / (n_clusters - 1) if n_clusters > 1 else 1
    
    # Compute meat matrices
    meat_matrices = cp.zeros((batch_size, n_features, n_features), dtype=dtype)
    
    for cluster_id in unique_clusters:
        # Find samples in this cluster
        cluster_mask = clusters_gpu == cluster_id
        
        # Sum scores within cluster for each gene
        cluster_scores = cp.sum(scores_gpu[:, cluster_mask, :], axis=1)  # batch_size × features
        
        # Add outer products to meat matrix
        for gene_idx in range(batch_size):
            cs = cluster_scores[gene_idx]  # features
            meat_matrices[gene_idx] += cp.outer(cs, cs)
    
    # Apply adjustment and normalize
    meat_matrices *= adjustment / n_samples
    
    # Compute sandwich estimator: bread @ meat @ bread
    sandwich_matrices = cp.zeros_like(meat_matrices)
    
    for gene_idx in range(batch_size):
        bread = bread_gpu[gene_idx]
        meat = meat_matrices[gene_idx]
        sandwich_matrices[gene_idx] = bread @ meat @ bread * n_samples
    
    return to_cpu(sandwich_matrices)


def compute_variance_batch_gpu(
    devil_fit: dict,
    gene_indices: np.ndarray,
    contrast: np.ndarray,
    clusters: np.ndarray = None,
    dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute variance estimates for a batch of genes on GPU.
    Args:
        devil_fit: Fitted model dictionary.
        gene_indices: Indices of genes to process.
        contrast: Contrast vector.
        clusters: Optional cluster assignments.
        dtype: Data type for computation.
    Returns:
        Tuple of (variances, standard_errors) for the batch.
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available")
    
    # Extract data for selected genes
    beta_batch = devil_fit["beta"][gene_indices]
    overdispersion_batch = devil_fit["overdispersion"][gene_indices]
    count_batch = devil_fit["count_matrix"][gene_indices]
    
    batch_size = len(gene_indices)
    n_features = len(contrast)
    
    if clusters is not None:
        # Use sandwich estimator
        variance_matrices = compute_sandwich_estimator_gpu_batch(
            devil_fit["design_matrix"],
            count_batch,
            beta_batch,
            overdispersion_batch,
            devil_fit["size_factors"],
            clusters,
            dtype
        )
    else:
        # Use standard Hessian
        precision_batch = 1.0 / overdispersion_batch
        precision_batch = np.where(overdispersion_batch > 0, precision_batch, 1e6)
        
        variance_matrices = compute_hessian_gpu_batch(
            beta_batch,
            precision_batch,
            count_batch,
            devil_fit["design_matrix"],
            devil_fit["size_factors"],
            dtype
        )
    
    # Compute contrast variances
    contrast_gpu = to_gpu(contrast, dtype)
    variances = cp.zeros(batch_size, dtype=dtype)
    
    variance_matrices_gpu = to_gpu(variance_matrices, dtype)
    
    for gene_idx in range(batch_size):
        var_matrix = variance_matrices_gpu[gene_idx]
        variances[gene_idx] = contrast_gpu.T @ var_matrix @ contrast_gpu
    
    variances_cpu = to_cpu(variances)
    standard_errors = np.sqrt(np.maximum(variances_cpu, 1e-12))
    
    return variances_cpu, standard_errors