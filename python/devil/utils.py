"""Utility functions for devil package."""

from typing import Union, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad


def handle_input_data(
    data: Union[ad.AnnData, np.ndarray, sparse.spmatrix],
    layer: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[pd.DataFrame]]:
    """
    Handle various input data formats and extract count matrix.
    
    Args:
        data: Input data as AnnData, numpy array, or sparse matrix.
        layer: For AnnData, which layer to use.
        
    Returns:
        Tuple of (count_matrix, gene_names, sample_names, obs_dataframe).
    """
    if isinstance(data, ad.AnnData):
        # Extract from AnnData
        if layer is not None:
            if layer not in data.layers:
                raise ValueError(f"Layer '{layer}' not found in AnnData object")
            count_matrix = data.layers[layer]
        else:
            count_matrix = data.X
        
        # Convert sparse to dense if needed
        if sparse.issparse(count_matrix):
            count_matrix = count_matrix.toarray()
        
        # Get metadata
        gene_names = data.var_names.values
        sample_names = data.obs_names.values
        obs_df = data.obs
        
    elif isinstance(data, (np.ndarray, sparse.spmatrix)):
        # Handle array input
        if sparse.issparse(data):
            count_matrix = data.toarray()
        else:
            count_matrix = data
        
        # Generate default names
        n_genes, n_samples = count_matrix.shape
        gene_names = np.array([f"Gene_{i}" for i in range(n_genes)])
        sample_names = np.array([f"Sample_{i}" for i in range(n_samples)])
        obs_df = None
        
    else:
        raise TypeError(
            f"Unsupported data type: {type(data)}. "
            "Expected AnnData, numpy array, or scipy sparse matrix."
        )
    
    # Ensure numeric type
    if not np.issubdtype(count_matrix.dtype, np.number):
        raise ValueError("Count matrix must contain numeric values")
    
    # Convert to float64 for numerical stability
    count_matrix = count_matrix.astype(np.float64)
    
    return count_matrix, gene_names, sample_names, obs_df


def validate_inputs(
    count_matrix: np.ndarray,
    design_matrix: np.ndarray
) -> None:
    """
    Validate input matrices for compatibility.
    
    Args:
        count_matrix: Gene expression matrix (genes × samples).
        design_matrix: Design matrix (samples × features).
        
    Raises:
        ValueError: If inputs are incompatible.
    """
    n_genes, n_samples = count_matrix.shape
    n_design_samples, n_features = design_matrix.shape
    
    if n_samples != n_design_samples:
        raise ValueError(
            f"Sample count mismatch: count_matrix has {n_samples} samples, "
            f"but design_matrix has {n_design_samples} samples"
        )
    
    if n_samples < n_features:
        raise ValueError(
            f"Insufficient samples: {n_samples} samples but {n_features} features. "
            "Model is not identifiable."
        )
    
    # Check for negative counts
    if np.any(count_matrix < 0):
        raise ValueError("Count matrix contains negative values")
    
    # Check for non-integer counts (warning only)
    if not np.allclose(count_matrix, count_matrix.astype(int)):
        import warnings
        warnings.warn(
            "Count matrix contains non-integer values. "
            "This may indicate normalized data rather than raw counts."
        )
    
    # Check design matrix rank
    rank = np.linalg.matrix_rank(design_matrix)
    if rank < n_features:
        raise ValueError(
            f"Design matrix is rank deficient: rank {rank} < {n_features} features. "
            "Consider removing collinear variables."
        )


def check_convergence(
    iterations: np.ndarray,
    max_iter: int,
    gene_names: np.ndarray
) -> None:
    """
    Check and report convergence issues.
    
    Args:
        iterations: Number of iterations per gene.
        max_iter: Maximum iterations allowed.
        gene_names: Gene identifiers.
    """
    non_converged = iterations >= max_iter
    n_non_converged = np.sum(non_converged)
    
    if n_non_converged > 0:
        import warnings
        warnings.warn(
            f"{n_non_converged} genes did not converge within {max_iter} iterations. "
            "Consider increasing max_iter or tolerance."
        )
        
        # Report worst genes
        if n_non_converged <= 10:
            worst_genes = gene_names[non_converged]
            warnings.warn(f"Non-converged genes: {', '.join(worst_genes)}")
