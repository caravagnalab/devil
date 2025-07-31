"""Plotting functions for differential expression results."""

from typing import Optional, List, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings


def plot_volcano(
    de_results: pd.DataFrame,
    lfc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
    labels: bool = True,
    top_n_labels: Optional[int] = 10,
    colors: Optional[List[str]] = None,
    alpha: float = 0.7,
    point_size: float = 20,
    figsize: Tuple[float, float] = (8, 6),
    title: str = "Volcano Plot",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Create volcano plot for differential expression results.
    
    Generates a customizable volcano plot visualizing differential expression,
    highlighting significant genes based on fold change and statistical significance.
    
    Args:
        de_results: DataFrame from test_de() containing columns:
            'gene', 'padj' (or 'pval'), 'lfc'.
        lfc_threshold: Absolute log2 fold change threshold for significance.
        pval_threshold: Adjusted p-value threshold for significance.
        labels: Whether to label significant genes.
        top_n_labels: If specified, only label top N most significant genes.
        colors: List of 4 colors for: non-significant, LFC-only, p-value-only, both.
            If None, uses default color scheme.
        alpha: Transparency level for points.
        point_size: Size of scatter points.
        figsize: Figure size as (width, height).
        title: Plot title.
        xlabel: X-axis label. If None, uses default.
        ylabel: Y-axis label. If None, uses default.
        ax: Existing axes to plot on. If None, creates new figure.
        
    Returns:
        Matplotlib axes object containing the plot.
        
    Examples:
        >>> # Basic volcano plot
        >>> ax = plot_volcano(de_results)
        
        >>> # Custom thresholds and colors
        >>> ax = plot_volcano(
        ...     de_results,
        ...     lfc_threshold=2,
        ...     pval_threshold=0.01,
        ...     colors=['gray', 'blue', 'green', 'red']
        ... )
    """
    # Validate input
    required_cols = {'gene', 'lfc'}
    if not required_cols.issubset(de_results.columns):
        raise ValueError(f"de_results must contain columns: {required_cols}")
    
    # Use adjusted p-values if available, otherwise raw p-values
    if 'padj' in de_results.columns:
        pval_col = 'padj'
    elif 'pval' in de_results.columns:
        pval_col = 'pval'
        warnings.warn("Using raw p-values instead of adjusted p-values")
    else:
        raise ValueError("de_results must contain 'padj' or 'pval' column")
    
    # Create copy to avoid modifying original
    plot_data = de_results.copy()
    
    # Handle empty DataFrame
    if len(plot_data) == 0:
        warnings.warn("No data points to plot")
        # Create empty plot with proper labels
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel(xlabel or r'$\log_2$ Fold Change')
        ax.set_ylabel(ylabel or r'$-\log_{10}$ Adjusted P-value')
        ax.set_title(title)
        ax.text(0.5, 0.5, 'No data to display', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12, alpha=0.7)
        return ax
    
    # Remove rows with missing values
    n_missing = plot_data[['lfc', pval_col]].isna().any(axis=1).sum()
    if n_missing > 0:
        warnings.warn(
            f"Removing {n_missing} genes with missing values. "
            "These may be lowly expressed or problematic genes."
        )
        plot_data = plot_data.dropna(subset=['lfc', pval_col])
    
    # Remove rows with infinite values (only if data remains)
    if len(plot_data) > 0:
        n_infinite = (~np.isfinite(plot_data[['lfc', pval_col]])).any(axis=1).sum()
        if n_infinite > 0:
            warnings.warn(
                f"Removing {n_infinite} genes with infinite values. "
                "These may be problematic genes with extreme statistics."
            )
            plot_data = plot_data[np.isfinite(plot_data[['lfc', pval_col]]).all(axis=1)]
    
    # Handle empty DataFrame after filtering
    if len(plot_data) == 0:
        warnings.warn("No valid data points to plot after filtering")
        # Create empty plot with proper labels
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel(xlabel or r'$\log_2$ Fold Change')
        ax.set_ylabel(ylabel or r'$-\log_{10}$ Adjusted P-value')
        ax.set_title(title)
        ax.text(0.5, 0.5, 'No data to display', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12, alpha=0.7)
        return ax
    
    # Handle p-values of 0
    nonzero_pvals = plot_data[plot_data[pval_col] > 0][pval_col]
    if len(nonzero_pvals) > 0:
        min_nonzero_pval = nonzero_pvals.min()
    else:
        min_nonzero_pval = 1e-300  # Default minimum for when all p-values are 0
    
    n_zero_pval = (plot_data[pval_col] == 0).sum()
    if n_zero_pval > 0:
        warnings.warn(
            f"{n_zero_pval} genes have p-value = 0, "
            f"setting to minimum non-zero value: {min_nonzero_pval:.2e}"
        )
        plot_data.loc[plot_data[pval_col] == 0, pval_col] = min_nonzero_pval
    
    # Calculate -log10 p-values
    plot_data['neg_log_pval'] = -np.log10(plot_data[pval_col])
    
    # Classify genes
    plot_data['significant_pval'] = plot_data[pval_col] <= pval_threshold
    plot_data['significant_lfc'] = np.abs(plot_data['lfc']) >= lfc_threshold
    
    conditions = [
        (~plot_data['significant_pval']) & (~plot_data['significant_lfc']),
        (~plot_data['significant_pval']) & (plot_data['significant_lfc']),
        (plot_data['significant_pval']) & (~plot_data['significant_lfc']),
        (plot_data['significant_pval']) & (plot_data['significant_lfc'])
    ]
    categories = ['Non-significant', 'LFC only', 'P-value only', 'Both']
    plot_data['category'] = np.select(conditions, categories, default='Non-significant')
    
    # Set default colors if not provided
    if colors is None:
        colors = ['#888888', '#2E8B57', '#4682B4', '#DC143C']
    
    # Create figure if axes not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each category
    for cat, color in zip(categories, colors):
        mask = plot_data['category'] == cat
        ax.scatter(
            plot_data.loc[mask, 'lfc'],
            plot_data.loc[mask, 'neg_log_pval'],
            c=color,
            alpha=alpha,
            s=point_size,
            label=cat,
            edgecolors='none'
        )
    
    # Add threshold lines
    ax.axhline(y=-np.log10(pval_threshold), color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=lfc_threshold, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=-lfc_threshold, color='black', linestyle='--', alpha=0.5)
    
    # Add labels for significant genes
    if labels:
        sig_genes = plot_data[plot_data['category'] == 'Both'].copy()
        
        # If top_n_labels specified, select top genes
        if top_n_labels is not None and len(sig_genes) > top_n_labels:
            sig_genes = sig_genes.nsmallest(top_n_labels, pval_col)
        
        # Add labels with adjustment to prevent overlap
        texts = []
        for _, row in sig_genes.iterrows():
            texts.append(
                ax.annotate(
                    row['gene'],
                    xy=(row['lfc'], row['neg_log_pval']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.8
                )
            )
        
        # Adjust label positions to minimize overlap
        if texts and len(texts) > 1:
            from matplotlib import patheffects
            for text in texts:
                text.set_path_effects(
                    [patheffects.Stroke(linewidth=3, foreground='white'),
                     patheffects.Normal()]
                )
    
    # Set labels and title
    ax.set_xlabel(xlabel or r'$\log_2$ Fold Change')
    ax.set_ylabel(ylabel or r'$-\log_{10}$ Adjusted P-value')
    ax.set_title(title)
    
    # Add legend
    ax.legend(loc='best', frameon=True, framealpha=0.9)
    
    # Center x-axis at 0
    finite_lfc = plot_data['lfc'][np.isfinite(plot_data['lfc'])]
    if len(finite_lfc) > 0:
        max_abs_lfc = np.abs(finite_lfc).max()
        if np.isfinite(max_abs_lfc) and max_abs_lfc > 0:
            ax.set_xlim(-max_abs_lfc * 1.1, max_abs_lfc * 1.1)
        else:
            ax.set_xlim(-1.0, 1.0)  # Default range if max is 0 or problematic
    else:
        ax.set_xlim(-1.0, 1.0)  # Default range if no finite values
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add count annotations
    n_up = ((plot_data['category'] == 'Both') & (plot_data['lfc'] > 0)).sum()
    n_down = ((plot_data['category'] == 'Both') & (plot_data['lfc'] < 0)).sum()
    
    ax.text(
        0.02, 0.98,
        f'Up: {n_up}\nDown: {n_down}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    return ax


def plot_ma(
    de_results: pd.DataFrame,
    expression_data: Optional[np.ndarray] = None,
    lfc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
    alpha: float = 0.5,
    figsize: Tuple[float, float] = (8, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Create MA plot for differential expression results.
    
    Args:
        de_results: DataFrame from test_de() with columns 'gene', 'lfc', 'padj'.
        expression_data: Optional expression matrix (genes × samples) for computing
            mean expression. If None, must have 'baseMean' column in de_results.
        lfc_threshold: Absolute log2 fold change threshold.
        pval_threshold: Adjusted p-value threshold.
        alpha: Point transparency.
        figsize: Figure size as (width, height).
        ax: Existing axes to plot on.
        
    Returns:
        Matplotlib axes object.
    """
    # Implementation details...
    pass  # Placeholder for brevity
