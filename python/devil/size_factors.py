"""Size factor calculation for normalization."""

import numpy as np
from typing import Optional


def calculate_size_factors(
    count_matrix: np.ndarray,
    method: str = "median_ratio",
    verbose: bool = False
) -> np.ndarray:
    """
    Calculate size factors for count data normalization.
    Computes normalization factors to account for differences in sequencing
    depth across samples.
    Args:
        count_matrix: Count data (genes × samples).
        method: Normalization method. Options:
            - 'median_ratio': DESeq2-style median ratio method
            - 'total_count': Simple library size normalization
        verbose: Whether to print messages.
    Returns:
        Size factors array (one per sample).
    Raises:
        ValueError: If any sample has all zeros.
    """
    n_genes, n_samples = count_matrix.shape
    
    # Handle single gene case - return all ones
    if n_genes == 1:
        return np.ones(n_samples)
    
    if method == "total_count":
        # Simple total count normalization
        size_factors = np.sum(count_matrix, axis=0).astype(float)
        zero_mask = size_factors == 0
        # Replace zeros with 1 to avoid log issues; these samples will
        # effectively get a size factor of 1 after normalization.
        if np.any(zero_mask):
            size_factors[zero_mask] = 1.0

    elif method == "median_ratio":
        # DESeq2-style median ratio normalization
        # Calculate geometric mean per gene
        # Add pseudocount to handle zeros
        log_counts = np.log(count_matrix + 1)
        log_geom_means = np.mean(log_counts, axis=1)
        
        # Calculate size factors
        size_factors = np.zeros(n_samples)
        for j in range(n_samples):
            # Ratio of each gene to its geometric mean
            ratios = log_counts[:, j] - log_geom_means
            # Use median of ratios (excluding -inf values)
            finite_ratios = ratios[np.isfinite(ratios)]
            if len(finite_ratios) > 0:
                size_factors[j] = np.exp(np.median(finite_ratios))
            else:
                size_factors[j] = 1.0       
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Check for all-zero samples using column sums
    zero_samples = np.sum(count_matrix, axis=0) == 0
    if np.any(zero_samples):
        raise ValueError(
            f"Samples {np.where(zero_samples)[0]} have all zeros. "
            "Please filter out empty samples."
        )
    
    # Normalize to geometric mean
    geom = np.exp(np.mean(np.log(size_factors))) if np.all(size_factors > 0) else 1.0
    size_factors = size_factors / geom
    
    if verbose:
        print(f"Size factor range: [{np.min(size_factors):.3f}, "
              f"{np.max(size_factors):.3f}]")
    
    return size_factors


def compute_offset_vector(
    base_offset: float,
    n_samples: int,
    size_factors: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute offset vector for model fitting.
    Args:
        base_offset: Base offset value.
        n_samples: Number of samples.
        size_factors: Optional size factors for normalization.
    Returns:
        Offset vector for each sample.
    """
    offset_vector = np.full(n_samples, base_offset)
    
    if size_factors is not None:
        offset_vector = offset_vector + np.log(size_factors)
    
    return offset_vector