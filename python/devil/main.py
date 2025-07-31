"""Main module for fitting the devil statistical model to count data."""

from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad
from joblib import Parallel, delayed
import warnings
from tqdm import tqdm

from .beta import init_beta, fit_beta_coefficients
from .overdispersion import estimate_initial_dispersion, fit_dispersion
from .size_factors import calculate_size_factors, compute_offset_vector
from .utils import handle_input_data, validate_inputs


def fit_devil(
    adata: Union[ad.AnnData, np.ndarray, sparse.spmatrix],
    design_matrix: Optional[np.ndarray] = None,
    design_formula: Optional[str] = None,
    overdispersion: bool = True,
    init_overdispersion: Optional[float] = None,
    do_cox_reid_adjustment: bool = True,
    offset: float = 1e-6,
    size_factors: bool = True,
    verbose: bool = False,
    max_iter: int = 100,
    tolerance: float = 1e-3,
    n_jobs: Optional[int] = None,
    layer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fit statistical model for count data, designed for RNA sequencing data analysis.
    
    Fits a negative binomial regression model with support for overdispersion estimation,
    size factor normalization, and parallel processing.
    
    Args:
        adata: Input data as AnnData object, numpy array (genes × samples), or sparse matrix.
            If AnnData, uses .X matrix or specified layer.
        design_matrix: Design matrix of predictor variables (samples × predictors).
            If None, must provide design_formula.
        design_formula: Formula string for design matrix construction (e.g., "~ condition + batch").
            Only used if design_matrix is None. Requires adata to be AnnData.
        overdispersion: Whether to estimate overdispersion parameter. If False, uses Poisson model.
        init_overdispersion: Initial value for overdispersion. If None, estimates from data.
        do_cox_reid_adjustment: Whether to apply Cox-Reid adjustment in overdispersion estimation.
        offset: Value added to counts to avoid numerical issues with zeros.
        size_factors: Whether to compute normalization factors for sequencing depth.
        verbose: Whether to print progress messages.
        max_iter: Maximum iterations for optimization.
        tolerance: Convergence criterion for optimization.
        n_jobs: Number of parallel jobs. If None, uses all available cores.
        layer: If adata is AnnData, which layer to use. If None, uses .X.
        
    Returns:
        Dictionary containing:
            - beta: Fitted coefficients array (genes × predictors)
            - overdispersion: Fitted overdispersion parameters (one per gene)  
            - iterations: Iteration counts for convergence (one per gene)
            - size_factors: Computed size factors (one per sample)
            - offset_vector: Offset values used in the model
            - design_matrix: Design matrix used (as provided or constructed)
            - gene_names: Gene identifiers
            - n_genes: Number of genes
            - n_samples: Number of samples
            - converged: Boolean array indicating convergence per gene
            
    Raises:
        ValueError: If inputs are invalid or incompatible.
        
    Examples:
        >>> # Using AnnData object
        >>> result = fit_devil(adata, design_formula="~ condition + batch")
        
        >>> # Using numpy array with explicit design matrix  
        >>> X = np.random.negative_binomial(5, 0.3, size=(1000, 100))
        >>> design = np.column_stack([np.ones(100), np.random.binomial(1, 0.5, 100)])
        >>> result = fit_devil(X, design_matrix=design)
    """
    # Handle input data
    count_matrix, gene_names, sample_names, obs_df = handle_input_data(
        adata, layer=layer
    )
    
    # Construct design matrix if needed
    if design_matrix is None:
        if design_formula is None:
            raise ValueError("Must provide either design_matrix or design_formula")
        if obs_df is None:
            raise ValueError("design_formula requires AnnData input with .obs")
        
        import patsy
        design_matrix = patsy.dmatrix(design_formula, obs_df, return_type="numpy")
        if verbose:
            print(f"Created design matrix from formula: {design_formula}")
            print(f"Design matrix shape: {design_matrix.shape}")
    
    # Validate inputs
    validate_inputs(count_matrix, design_matrix)
    
    n_genes, n_samples = count_matrix.shape
    n_features = design_matrix.shape[1]
    
    if verbose:
        print(f"Fitting model for {n_genes} genes and {n_samples} samples")
        print(f"Design matrix has {n_features} features")
    
    # Compute size factors
    if size_factors:
        if verbose:
            print("Computing size factors...")
        sf = calculate_size_factors(count_matrix, verbose=verbose)
    else:
        sf = np.ones(n_samples)
    
    # Calculate offset vector
    offset_vector = compute_offset_vector(offset, n_samples, sf)
    
    # Initialize overdispersion
    if overdispersion:
        if init_overdispersion is None:
            if verbose:
                print("Estimating initial overdispersion...")
            dispersion_init = estimate_initial_dispersion(count_matrix, offset_vector)
        else:
            dispersion_init = np.full(n_genes, init_overdispersion)
    else:
        dispersion_init = np.zeros(n_genes)
    
    # Initialize beta coefficients
    if verbose:
        print("Initializing beta coefficients...")
    beta_init = init_beta(count_matrix, design_matrix, offset_vector)
    
    # Set up parallel processing
    if n_jobs is None:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
    
    # Fit beta coefficients
    if verbose:
        print(f"Fitting beta coefficients using {n_jobs} cores...")
    
    def fit_gene(gene_idx):
        """Fit model for a single gene."""
        beta, n_iter, converged = fit_beta_coefficients(
            count_matrix[gene_idx, :],
            design_matrix,
            beta_init[gene_idx, :],
            offset_vector,
            dispersion_init[gene_idx],
            max_iter=max_iter,
            tolerance=tolerance
        )
        return beta, n_iter, converged
    
    # Run parallel fitting
    if verbose:
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_gene)(i) for i in tqdm(range(n_genes), desc="Fitting genes")
        )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_gene)(i) for i in range(n_genes)
        )
    
    # Extract results
    beta = np.array([r[0] for r in results])
    iterations = np.array([r[1] for r in results])
    converged = np.array([r[2] for r in results])
    
    # Fit overdispersion if requested
    if overdispersion:
        if verbose:
            print("Fitting overdispersion parameters...")
        
        def fit_gene_dispersion(gene_idx):
            """Fit dispersion for a single gene."""
            return fit_dispersion(
                beta[gene_idx, :],
                design_matrix,
                count_matrix[gene_idx, :],
                offset_vector,
                tolerance=tolerance,
                max_iter=max_iter,
                do_cox_reid_adjustment=do_cox_reid_adjustment
            )
        
        if verbose:
            theta = Parallel(n_jobs=n_jobs)(
                delayed(fit_gene_dispersion)(i) 
                for i in tqdm(range(n_genes), desc="Fitting overdispersion")
            )
        else:
            theta = Parallel(n_jobs=n_jobs)(
                delayed(fit_gene_dispersion)(i) for i in range(n_genes)
            )
        
        theta = np.array(theta)
    else:
        theta = np.zeros(n_genes)
    
    # Report convergence
    if verbose:
        n_converged = np.sum(converged)
        print(f"Optimization converged for {n_converged}/{n_genes} genes")
        if n_converged < n_genes:
            warnings.warn(
                f"{n_genes - n_converged} genes did not converge. "
                "Consider increasing max_iter or tolerance."
            )
    
    return {
        "beta": beta,
        "overdispersion": theta,
        "iterations": iterations,
        "size_factors": sf,
        "offset_vector": offset_vector,
        "design_matrix": design_matrix,
        "gene_names": gene_names,
        "n_genes": n_genes,
        "n_samples": n_samples,
        "converged": converged,
        "count_matrix": count_matrix  # Store for later use in test_de
    }
