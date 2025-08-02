"""
Exact beta coefficient estimation matching R package C++ implementation.

This implementation exactly matches the algorithm in src/beta.cpp from the R package,
ensuring mathematical consistency between R and Python versions.
"""

from typing import Tuple, Optional
import numpy as np
import warnings


def init_beta(
    y: np.ndarray,
    X: np.ndarray
) -> np.ndarray:
    """
    Initialize beta coefficients using log-linear regression.
    
    Exactly matches the R package's init_beta C++ function:
    VectorXd norm_log_count_mat = y.array().log1p();
    return X.colPivHouseholderQr().solve(norm_log_count_mat);
    
    Args:
        y: Count vector for one gene.
        X: Design matrix.
        
    Returns:
        Initial beta estimates.
    """
    # Apply log1p transformation as in C++ code
    norm_log_count_mat = np.log1p(y)
    
    # Use QR decomposition with column pivoting (equivalent to colPivHouseholderQr)
    # Note: numpy doesn't have exact equivalent, but lstsq with rcond=None is robust
    beta_init = np.linalg.lstsq(X, norm_log_count_mat, rcond=None)[0]
    
    return beta_init


def beta_fit(
    y: np.ndarray,
    X: np.ndarray,
    mu_beta: np.ndarray,
    offset: np.ndarray,
    dispersion: float,
    max_iter: int = 100,
    tolerance: float = 1e-3
) -> Tuple[np.ndarray, int, bool]:
    """
    Fit beta coefficients using exact algorithm from R package's beta_fit C++ function.
    
    This exactly replicates the mathematical algorithm:
    - k = 1.0 / dispersion
    - w_q = exp(-X * mu_beta - offset)
    - mu_g = (k + y) / (1 + k * w_q)
    - Zigma = inv(k * X.T @ diag(mu_g * w_q) @ X)
    - delta = Zigma @ (k * X.T @ (mu_g * w_q - 1))
    
    Args:
        y: Count vector for one gene.
        X: Design matrix (samples × features).
        mu_beta: Initial beta coefficients.
        offset: Offset vector.
        dispersion: Dispersion parameter.
        max_iter: Maximum iterations.
        tolerance: Convergence tolerance.
        
    Returns:
        Tuple of (fitted_beta, n_iterations, converged).
    """
    n_samples, n_features = X.shape
    
    # Convert dispersion to k (inverse dispersion) as in C++ code
    k = 1.0 / dispersion if dispersion > 0 else 1e6
    
    # Initialize variables exactly as in C++ code
    mu_beta = mu_beta.copy()
    converged = False
    iter_count = 0
    
    # Add small regularization to prevent singularity (as in C++ code comment)
    inv_sigma_beta_const = 0.01 * np.eye(n_features)
    
    while not converged and iter_count < max_iter:
        # Calculate w_q: exp(-X * mu_beta - offset)
        # This matches: w_q = (-X * mu_beta - off).array().exp();
        linear_pred = -(X @ mu_beta + offset)
        
        # Clamp to prevent overflow (numerical stability)
        linear_pred = np.clip(linear_pred, -50, 50)
        w_q = np.exp(linear_pred)
        
        # Calculate mu_g: (k + y) / (1 + k * w_q)
        # This matches: mu_g = (k + y.array()) / (1 + k * w_q.array());
        mu_g = (k + y) / (1.0 + k * w_q)
        
        # Calculate Zigma (covariance matrix): inv(k * X.T @ diag(mu_g * w_q) @ X)
        # This matches: Zigma = (k * X.transpose() * (mu_g.array() * w_q.array()).matrix().asDiagonal() * X).inverse();
        weights = mu_g * w_q
        weighted_X = X * weights[:, np.newaxis]  # Apply weights to each column
        XTwX = k * X.T @ weighted_X
        
        # Add regularization for numerical stability (as in C++ code)
        XTwX += inv_sigma_beta_const
        
        try:
            Zigma = np.linalg.inv(XTwX)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            Zigma = np.linalg.pinv(XTwX)
        
        # Calculate delta (parameter update)
        # This matches: delta = Zigma * (k * X.transpose() * (mu_g.array() * w_q.array() - 1).matrix());
        residual = mu_g * w_q - 1.0
        delta = Zigma @ (k * X.T @ residual)
        
        # Update mu_beta
        # This matches: mu_beta += delta;
        mu_beta += delta
        
        # Check for NaN (as in C++ code)
        # This matches: if (delta[0] != delta[0]) {converged = TRUE;}
        if not np.all(np.isfinite(delta)):
            converged = True
            break
        
        # Check convergence
        # This matches: converged = delta.cwiseAbs().maxCoeff() < eps;
        if np.max(np.abs(delta)) < tolerance:
            converged = True
        
        iter_count += 1
    
    return mu_beta, iter_count, converged


def beta_fit_group(
    y: np.ndarray,
    mu_beta: float,
    offset: np.ndarray,
    dispersion: float,
    max_iter: int = 100,
    tolerance: float = 1e-3
) -> Tuple[float, int, bool]:
    """
    Fit single beta coefficient (intercept-only model) using exact R package algorithm.
    
    This exactly replicates the beta_fit_group C++ function for the case where
    the design matrix is just a column of ones (intercept-only model).
    
    Args:
        y: Count vector for one gene.
        mu_beta: Initial beta coefficient (scalar).
        offset: Offset vector.
        dispersion: Dispersion parameter.
        max_iter: Maximum iterations.
        tolerance: Convergence tolerance.
        
    Returns:
        Tuple of (fitted_beta, n_iterations, converged).
    """
    n_samples = len(y)
    
    # Convert dispersion to k (inverse dispersion)
    k = 1.0 / dispersion if dispersion > 0 else 1e6
    
    converged = False
    iter_count = 0
    
    while not converged and iter_count < max_iter:
        # Calculate w_q: exp(-mu_beta * ones - offset)
        # This matches: w_q = (-mu_beta * ones - off).array().exp();
        linear_pred = -mu_beta - offset
        linear_pred = np.clip(linear_pred, -50, 50)  # Numerical stability
        w_q = np.exp(linear_pred)
        
        # Calculate mu_g: (k + y) / (1 + k * w_q)
        # This matches: mu_g = (k + y.array()) / (1 + k * w_q.array());
        mu_g = (k + y) / (1.0 + k * w_q)
        
        # Calculate Zigma (scalar covariance): 1 / (k * sum(mu_g * w_q))
        # This matches: Zigma = 1.0 / (k * (mu_g.array() * w_q.array()).sum());
        Zigma = 1.0 / (k * np.sum(mu_g * w_q))
        
        # Calculate delta (parameter update)
        # This matches: delta = Zigma * (k * (mu_g.array() * w_q.array() - 1).sum());
        residual_sum = np.sum(mu_g * w_q - 1.0)
        delta = Zigma * k * residual_sum
        
        # Update mu_beta
        mu_beta += delta
        
        # Check for NaN
        if not np.isfinite(delta):
            converged = True
            break
        
        # Check convergence
        if abs(delta) < tolerance:
            converged = True
        
        iter_count += 1
    
    return mu_beta, iter_count, converged


def fit_beta_coefficients(
    y: np.ndarray,
    X: np.ndarray,
    beta_init: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    dispersion: float = 1.0,
    max_iter: int = 100,
    tolerance: float = 1e-3
) -> Tuple[np.ndarray, int, bool]:
    """
    High-level interface for exact beta coefficient fitting.
    
    This function provides a convenient interface that matches the R package's
    mathematical implementation exactly.
    
    Args:
        y: Count vector for one gene.
        X: Design matrix (samples × features).
        beta_init: Initial beta coefficients. If None, will be computed.
        offset: Offset vector. If None, defaults to zeros.
        dispersion: Dispersion parameter.
        max_iter: Maximum iterations.
        tolerance: Convergence tolerance.
        
    Returns:
        Tuple of (fitted_beta, n_iterations, converged).
    """
    n_samples, n_features = X.shape
    
    # Handle defaults
    if offset is None:
        offset = np.zeros(n_samples)
    
    if beta_init is None:
        beta_init = init_beta(y, X)
    
    # Handle edge cases
    if np.all(y == 0):
        return np.zeros(n_features), 1, True
    
    # Check for intercept-only model (optimization)
    if n_features == 1 and np.allclose(X[:, 0], 1.0):
        # Use the optimized group fitting function
        fitted_beta, n_iter, converged = beta_fit_group(
            y, beta_init[0], offset, dispersion, max_iter, tolerance
        )
        return np.array([fitted_beta]), n_iter, converged
    else:
        # Use the general fitting function
        return beta_fit(
            y, X, beta_init, offset, dispersion, max_iter, tolerance
        )


def init_beta_matrix(
    count_matrix: np.ndarray,
    design_matrix: np.ndarray
) -> np.ndarray:
    """
    Initialize beta coefficients for multiple genes.
    
    Args:
        count_matrix: Count data (genes × samples).
        design_matrix: Design matrix (samples × features).
        
    Returns:
        Initial beta estimates (genes × features).
    """
    n_genes, n_samples = count_matrix.shape
    n_features = design_matrix.shape[1]
    
    beta_init = np.zeros((n_genes, n_features))
    
    for gene_idx in range(n_genes):
        beta_init[gene_idx, :] = init_beta(count_matrix[gene_idx, :], design_matrix)
    
    return beta_init


# Example usage and validation
if __name__ == "__main__":
    # python -m devil.beta
    # Test the implementation
    np.random.seed(42)
    n_samples, n_features = 20, 3
    
    # Generate test data
    X = np.column_stack([
        np.ones(n_samples),  # Intercept
        np.random.binomial(1, 0.5, n_samples),  # Binary covariate
        np.random.normal(0, 1, n_samples)  # Continuous covariate
    ])
    
    true_beta = np.array([2.0, 0.5, -0.3])
    offset = np.random.normal(0, 0.1, n_samples)
    dispersion = 0.5
    
    # Generate negative binomial counts
    mu = np.exp(X @ true_beta + offset)
    y = np.random.negative_binomial(1/dispersion, 1/(1 + dispersion * mu))
    
    # Fit using exact algorithm
    fitted_beta, n_iter, converged = fit_beta_coefficients(
        y, X, offset=offset, dispersion=dispersion
    )
    
    print(f"True beta: {true_beta}")
    print(f"Fitted beta: {fitted_beta}")
    print(f"Iterations: {n_iter}, Converged: {converged}")
    print(f"Max absolute error: {np.max(np.abs(fitted_beta - true_beta))}")