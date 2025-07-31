"""Beta coefficient estimation functions."""

from typing import Tuple
import numpy as np
from scipy import optimize
from scipy.special import loggamma


def init_beta(
    count_matrix: np.ndarray,
    design_matrix: np.ndarray, 
    offset_vector: np.ndarray
) -> np.ndarray:
    """
    Initialize beta coefficients using QR decomposition.
    
    Args:
        count_matrix: Count data (genes × samples).
        design_matrix: Design matrix (samples × features).
        offset_vector: Offset values per sample.
        
    Returns:
        Initial beta estimates (genes × features).
    """
    n_genes, n_samples = count_matrix.shape
    n_features = design_matrix.shape[1]
    
    # QR decomposition of design matrix
    Q, R = np.linalg.qr(design_matrix)
    
    # Normalize counts
    norm_log_counts = np.log1p(count_matrix.T / np.exp(offset_vector)[:, np.newaxis])
    
    # Solve for initial beta
    beta_init = np.linalg.solve(R, Q.T @ norm_log_counts).T
    
    return beta_init


def fit_beta_coefficients(
    y: np.ndarray,
    X: np.ndarray,
    beta_init: np.ndarray,
    offset: np.ndarray,
    dispersion: float,
    max_iter: int = 100,
    tolerance: float = 1e-3
) -> Tuple[np.ndarray, int, bool]:
    """
    Fit beta coefficients for a single gene using iterative optimization.
    
    Args:
        y: Count vector for one gene.
        X: Design matrix.
        beta_init: Initial beta values.
        offset: Offset vector.
        dispersion: Dispersion parameter.
        max_iter: Maximum iterations.
        tolerance: Convergence tolerance.
        
    Returns:
        Tuple of (fitted_beta, n_iterations, converged).
    """
    n_samples = len(y)
    n_features = X.shape[1]
    
    # Handle edge cases
    if np.all(y == 0):
        return np.zeros(n_features), 1, True
    
    # Convert dispersion to precision
    k = 1.0 / dispersion if dispersion > 0 else 1e6
    
    beta = beta_init.copy()
    converged = False
    
    for iteration in range(max_iter):
        # Calculate working weights and residuals
        mu = np.exp(X @ beta + offset)
        mu = np.maximum(mu, 1e-10)  # Avoid numerical issues
        
        # Negative binomial variance function
        V = mu + mu**2 / k
        
        # Working weights
        W = mu / V
        
        # Score and information
        score = X.T @ ((y - mu) / V)
        info = X.T @ (W[:, np.newaxis] * X)
        
        # Check for singular matrix
        try:
            delta = np.linalg.solve(info, score)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            delta = np.linalg.pinv(info) @ score
        
        # Update beta
        beta += delta
        
        # Check convergence
        if np.max(np.abs(delta)) < tolerance:
            converged = True
            break
    
    return beta, iteration + 1, converged
