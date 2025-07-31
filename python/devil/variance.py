 """Variance estimation functions including sandwich estimator."""

import numpy as np
from typing import Optional


def compute_hessian(
    beta: np.ndarray,
    precision: float,
    y: np.ndarray,
    design_matrix: np.ndarray,
    size_factors: np.ndarray
) -> np.ndarray:
    """
    Compute inverse of negative Hessian matrix for GLM.
    
    Args:
        beta: Regression coefficients.
        precision: Inverse of overdispersion (1/phi).
        y: Response values.
        design_matrix: Predictor variables matrix.
        size_factors: Normalization factors.
        
    Returns:
        Inverse of negative Hessian matrix.
    """
    n_samples = len(y)
    n_features = len(beta)
    
    # Calculate linear predictor and mean
    eta = design_matrix @ beta
    mu = size_factors * np.exp(eta)
    
    # Weight matrix diagonal
    weights = mu / (1 + mu / precision)
    
    # Hessian = -X'WX
    H = -design_matrix.T @ (weights[:, np.newaxis] * design_matrix)
    
    # Return inverse
    try:
        return np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        return np.linalg.pinv(-H)


def compute_scores(
    design_matrix: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    overdispersion: float,
    size_factors: np.ndarray
) -> np.ndarray:
    """
    Compute score residuals for each observation.
    
    Args:
        design_matrix: Predictor variables.
        y: Response values.
        beta: Regression coefficients.
        overdispersion: Dispersion parameter.
        size_factors: Normalization factors.
        
    Returns:
        Matrix of score residuals (n_samples × n_features).
    """
    # Calculate mean and variance
    mu = size_factors * np.exp(design_matrix @ beta)
    alpha = 1.0 / overdispersion if overdispersion > 0 else 0
    
    # Pearson residuals
    variance = mu + alpha * mu**2
    residuals = (y - mu) / np.sqrt(variance)
    
    # Weight by sqrt(variance)
    weights = np.sqrt(mu**2 / variance)
    
    # Score contributions
    scores = design_matrix * (residuals * weights)[:, np.newaxis]
    
    return scores


def compute_sandwich_estimator(
    design_matrix: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    overdispersion: float,
    size_factors: np.ndarray,
    clusters: np.ndarray
) -> np.ndarray:
    """
    Compute clustered sandwich variance estimator.
    
    Calculates robust variance estimation accounting for correlation
    within clusters (e.g., multiple samples from same patient).
    
    Args:
        design_matrix: Predictor variables.
        y: Response values.
        beta: Regression coefficients. 
        overdispersion: Dispersion parameter.
        size_factors: Normalization factors.
        clusters: Cluster assignments (1-indexed).
        
    Returns:
        Sandwich variance-covariance matrix.
    """
    n_samples = len(y)
    n_features = design_matrix.shape[1]
    
    # Compute bread (inverse Hessian)
    precision = 1.0 / overdispersion if overdispersion > 0 else 1e6
    bread = compute_hessian(beta, precision, y, design_matrix, size_factors)
    
    # Compute scores
    scores = compute_scores(design_matrix, y, beta, overdispersion, size_factors)
    
    # Aggregate scores by cluster
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    
    # Small sample adjustment
    adjustment = n_clusters / (n_clusters - 1) if n_clusters > 1 else 1
    
    # Compute meat matrix
    meat = np.zeros((n_features, n_features))
    
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        cluster_scores = scores[cluster_mask, :].sum(axis=0)
        meat += np.outer(cluster_scores, cluster_scores)
    
    meat = adjustment * meat / n_samples
    
    # Sandwich estimator
    sandwich = bread @ meat @ bread * n_samples
    
    return sandwich