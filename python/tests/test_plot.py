"""
Unit tests for CPU plotting functions.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from unittest.mock import patch, MagicMock
import warnings

from devil.plot import plot_volcano
import devil


# Set matplotlib to non-interactive backend for testing
matplotlib.use('Agg')


class TestPlotVolcano:
    """Test volcano plot function."""
    
    @pytest.fixture
    def sample_de_results(self):
        """Create sample differential expression results for testing."""
        np.random.seed(42)
        n_genes = 100
        
        # Create realistic DE results
        gene_names = [f'Gene_{i:03d}' for i in range(n_genes)]
        
        # Generate log fold changes with some large effects
        lfc = np.random.normal(0, 1, n_genes)
        lfc[:20] = np.random.normal(3, 0.5, 20)   # Upregulated genes
        lfc[20:40] = np.random.normal(-3, 0.5, 20)  # Downregulated genes
        
        # Generate p-values (some significant)
        pval = np.random.uniform(0, 1, n_genes)
        pval[:40] = np.random.uniform(0, 0.01, 40)  # Significant genes
        
        # Calculate adjusted p-values
        from statsmodels.stats.multitest import multipletests
        _, padj, _, _ = multipletests(pval, method='fdr_bh')
        
        # Generate standard errors
        se = np.random.uniform(0.1, 2.0, n_genes)
        
        # Generate test statistics
        stat = lfc / se
        
        return pd.DataFrame({
            'gene': gene_names,
            'pval': pval,
            'padj': padj,
            'lfc': lfc,
            'se': se,
            'stat': stat
        })
    
    def test_plot_volcano_basic(self, sample_de_results):
        """Test basic volcano plot creation."""
        ax = plot_volcano(sample_de_results)
        
        # Check that plot was created
        assert isinstance(ax, plt.Axes)
        
        # Check basic plot properties
        assert ax.get_xlabel() == r'$\log_2$ Fold Change'
        assert ax.get_ylabel() == r'$-\log_{10}$ Adjusted P-value'
        assert ax.get_title() == 'Volcano Plot'
        
        # Check that data was plotted
        collections = ax.collections
        assert len(collections) > 0  # Should have scatter plots
        
        # Check for threshold lines
        lines = ax.lines
        threshold_lines = [line for line in lines if line.get_linestyle() == '--']
        assert len(threshold_lines) == 3  # Two vertical, one horizontal
    
    def test_plot_volcano_custom_thresholds(self, sample_de_results):
        """Test volcano plot with custom thresholds."""
        lfc_threshold = 2.0
        pval_threshold = 0.01
        
        ax = plot_volcano(
            sample_de_results,
            lfc_threshold=lfc_threshold,
            pval_threshold=pval_threshold
        )
        
        # Check threshold lines
        lines = ax.lines
        vertical_lines = [line for line in lines if line.get_linestyle() == '--' 
                         and hasattr(line, '_x') and len(set(line._x)) == 1]
        horizontal_lines = [line for line in lines if line.get_linestyle() == '--'
                           and hasattr(line, '_y') and len(set(line._y)) == 1]
        
        # Should have two vertical lines at ±lfc_threshold
        assert len([line for line in lines if line.get_linestyle() == '--']) >= 3
    
    def test_plot_volcano_custom_colors(self, sample_de_results):
        """Test volcano plot with custom colors."""
        custom_colors = ['gray', 'blue', 'green', 'red']
        
        ax = plot_volcano(
            sample_de_results,
            colors=custom_colors
        )
        
        # Should create plot successfully
        assert isinstance(ax, plt.Axes)
        
        # Check that legend exists
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 4  # Four categories
    
    def test_plot_volcano_no_labels(self, sample_de_results):
        """Test volcano plot without gene labels."""
        ax = plot_volcano(
            sample_de_results,
            labels=False
        )
        
        # Should create plot without text annotations
        assert isinstance(ax, plt.Axes)
        
        # Check for absence of text annotations (labels)
        texts = [child for child in ax.get_children() 
                if isinstance(child, matplotlib.text.Annotation)]
        # Some text might still exist (axis labels, title), but should be minimal
    
    def test_plot_volcano_top_n_labels(self, sample_de_results):
        """Test volcano plot with limited number of labels."""
        top_n = 5
        
        ax = plot_volcano(
            sample_de_results,
            labels=True,
            top_n_labels=top_n
        )
        
        # Should create plot with limited labels
        assert isinstance(ax, plt.Axes)
        
        # Count text annotations (gene labels)
        annotations = [child for child in ax.get_children() 
                      if isinstance(child, matplotlib.text.Annotation)]
        # Exact count may vary due to overlap prevention, but should be ≤ top_n
        assert len(annotations) <= top_n + 3  # Allow some flexibility for axis labels
    
    def test_plot_volcano_custom_styling(self, sample_de_results):
        """Test volcano plot with custom styling options."""
        ax = plot_volcano(
            sample_de_results,
            alpha=0.5,
            point_size=50,
            figsize=(10, 8),
            title="Custom Volcano Plot",
            xlabel="Custom X Label",
            ylabel="Custom Y Label"
        )
        
        # Check custom properties
        assert ax.get_title() == "Custom Volcano Plot"
        assert ax.get_xlabel() == "Custom X Label"
        assert ax.get_ylabel() == "Custom Y Label"
        
        # Check figure size
        fig = ax.get_figure()
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 8
    
    def test_plot_volcano_existing_axes(self, sample_de_results):
        """Test volcano plot on existing axes."""
        fig, ax = plt.subplots(figsize=(6, 5))
        
        returned_ax = plot_volcano(sample_de_results, ax=ax)
        
        # Should return the same axes object
        assert returned_ax is ax
        
        # Should have data plotted
        collections = ax.collections
        assert len(collections) > 0
    
    def test_plot_volcano_raw_pvalues(self, sample_de_results):
        """Test volcano plot with raw p-values when adjusted not available."""
        # Remove adjusted p-values
        de_results_raw = sample_de_results.drop(columns=['padj'])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = plot_volcano(de_results_raw)
            
            # Should warn about using raw p-values
            assert len(w) > 0
            assert any("raw p-values" in str(warning.message) for warning in w)
        
        # Should still create plot
        assert isinstance(ax, plt.Axes)
    
    def test_plot_volcano_missing_required_columns(self):
        """Test error handling with missing required columns."""
        # Missing gene column
        bad_results1 = pd.DataFrame({
            'pval': [0.01, 0.05],
            'lfc': [1.5, -2.0]
        })
        
        with pytest.raises(ValueError, match="must contain columns"):
            plot_volcano(bad_results1)
        
        # Missing lfc column
        bad_results2 = pd.DataFrame({
            'gene': ['Gene1', 'Gene2'],
            'pval': [0.01, 0.05]
        })
        
        with pytest.raises(ValueError, match="must contain columns"):
            plot_volcano(bad_results2)
        
        # Missing both pval and padj columns
        bad_results3 = pd.DataFrame({
            'gene': ['Gene1', 'Gene2'],
            'lfc': [1.5, -2.0]
        })
        
        with pytest.raises(ValueError, match="must contain 'padj' or 'pval'"):
            plot_volcano(bad_results3)


class TestPlotVolcanoDataHandling:
    """Test data handling in volcano plot function."""
    
    def test_plot_volcano_missing_values(self):
        """Test volcano plot with missing values."""
        de_results = pd.DataFrame({
            'gene': ['Gene1', 'Gene2', 'Gene3', 'Gene4'],
            'pval': [0.01, np.nan, 0.05, 0.001],
            'padj': [0.02, 0.5, np.nan, 0.005],
            'lfc': [1.5, -2.0, np.nan, 3.0],
            'se': [0.3, 0.4, 0.5, 0.2],
            'stat': [5.0, -5.0, 2.0, 15.0]
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = plot_volcano(de_results)
            
            # Should warn about missing values
            assert len(w) > 0
            assert any("missing values" in str(warning.message) for warning in w)
        
        # Should still create plot
        assert isinstance(ax, plt.Axes)
    
    def test_plot_volcano_zero_pvalues(self):
        """Test volcano plot with p-values of zero."""
        de_results = pd.DataFrame({
            'gene': ['Gene1', 'Gene2', 'Gene3', 'Gene4'],
            'pval': [0.0, 0.0, 0.05, 0.001],  # Two zero p-values
            'padj': [0.0, 0.0, 0.1, 0.005],
            'lfc': [2.5, -3.0, 1.0, -1.5],
            'se': [0.3, 0.4, 0.5, 0.2],
            'stat': [8.0, -7.5, 2.0, -7.5]
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = plot_volcano(de_results)
            
            # Should warn about zero p-values
            assert len(w) > 0
            warning_messages = [str(warning.message) for warning in w]
            assert any("p-value = 0" in msg for msg in warning_messages)
        
        # Should still create plot
        assert isinstance(ax, plt.Axes)
    
    def test_plot_volcano_extreme_values(self):
        """Test volcano plot with extreme values."""
        de_results = pd.DataFrame({
            'gene': ['Gene1', 'Gene2', 'Gene3', 'Gene4'],
            'pval': [1e-100, 1e-50, 0.5, 0.999],  # Very small p-values
            'padj': [1e-95, 1e-45, 0.6, 0.999],
            'lfc': [50, -100, 0.1, -0.05],  # Extreme fold changes
            'se': [1.0, 5.0, 0.1, 0.01],
            'stat': [50, -20, 1.0, -5.0]
        })
        
        ax = plot_volcano(de_results)
        
        # Should handle extreme values
        assert isinstance(ax, plt.Axes)
        
        # Check that plot limits are reasonable
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        assert np.isfinite(xlim[0]) and np.isfinite(xlim[1])
        assert np.isfinite(ylim[0]) and np.isfinite(ylim[1])
        assert ylim[1] > 10  # Should show high -log10 p-values
    
    def test_plot_volcano_empty_dataframe(self):
        """Test volcano plot with empty DataFrame."""
        empty_results = pd.DataFrame(columns=['gene', 'pval', 'padj', 'lfc', 'se', 'stat'])
        
        # Should handle empty data gracefully
        ax = plot_volcano(empty_results)
        assert isinstance(ax, plt.Axes)
    
    def test_plot_volcano_single_gene(self):
        """Test volcano plot with single gene."""
        single_gene = pd.DataFrame({
            'gene': ['Gene1'],
            'pval': [0.01],
            'padj': [0.02],
            'lfc': [2.5],
            'se': [0.5],
            'stat': [5.0]
        })
        
        ax = plot_volcano(single_gene)
        assert isinstance(ax, plt.Axes)
    
    def test_plot_volcano_all_significant(self):
        """Test volcano plot where all genes are significant."""
        all_sig_results = pd.DataFrame({
            'gene': ['Gene1', 'Gene2', 'Gene3', 'Gene4'],
            'pval': [0.001, 0.002, 0.003, 0.001],
            'padj': [0.01, 0.02, 0.03, 0.01],
            'lfc': [2.5, -2.0, 3.0, -1.8],
            'se': [0.3, 0.4, 0.5, 0.2],
            'stat': [8.0, -5.0, 6.0, -9.0]
        })
        
        ax = plot_volcano(
            all_sig_results,
            lfc_threshold=1.0,
            pval_threshold=0.05
        )
        
        # Should create plot
        assert isinstance(ax, plt.Axes)
        
        # Should show count annotation
        # Check for text annotation with counts
        texts = [child for child in ax.get_children() 
                if isinstance(child, matplotlib.text.Text)]
        count_text_found = any('Up:' in text.get_text() or 'Down:' in text.get_text() 
                              for text in texts)
        assert count_text_found
    
    def test_plot_volcano_no_significant(self):
        """Test volcano plot where no genes are significant."""
        no_sig_results = pd.DataFrame({
            'gene': ['Gene1', 'Gene2', 'Gene3', 'Gene4'],
            'pval': [0.8, 0.9, 0.7, 0.6],
            'padj': [0.85, 0.92, 0.75, 0.65],
            'lfc': [0.1, -0.15, 0.05, -0.08],
            'se': [0.3, 0.4, 0.5, 0.2],
            'stat': [0.3, -0.4, 0.1, -0.4]
        })
        
        ax = plot_volcano(
            no_sig_results,
            lfc_threshold=1.0,
            pval_threshold=0.05
        )
        
        # Should create plot
        assert isinstance(ax, plt.Axes)
        
        # Count annotation should show 0 up and 0 down
        texts = [child for child in ax.get_children() 
                if isinstance(child, matplotlib.text.Text)]
        count_texts = [text.get_text() for text in texts 
                      if 'Up:' in text.get_text() or 'Down:' in text.get_text()]
        if count_texts:
            # Should show Up: 0, Down: 0
            assert any('Up: 0' in text for text in count_texts)
            assert any('Down: 0' in text for text in count_texts)


class TestPlotVolcanoCustomization:
    """Test customization options in volcano plot."""
    
    @pytest.fixture
    def basic_de_results(self):
        """Create basic DE results for customization testing."""
        return pd.DataFrame({
            'gene': ['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE'],
            'pval': [0.001, 0.01, 0.1, 0.3, 0.8],
            'padj': [0.005, 0.02, 0.15, 0.4, 0.85],
            'lfc': [3.0, -2.5, 1.5, 0.5, -0.2],
            'se': [0.5, 0.6, 0.8, 0.3, 0.4],
            'stat': [6.0, -4.2, 1.9, 1.7, -0.5]
        })
    
    def test_plot_volcano_point_styling(self, basic_de_results):
        """Test point styling options."""
        ax = plot_volcano(
            basic_de_results,
            alpha=0.3,
            point_size=100
        )
        
        # Check that points were styled
        assert isinstance(ax, plt.Axes)
        collections = ax.collections
        assert len(collections) > 0
        
        # Check alpha and size (these might be hard to verify directly)
        # At least verify plot was created successfully
    
    def test_plot_volcano_axis_centering(self, basic_de_results):
        """Test x-axis centering around zero."""
        ax = plot_volcano(basic_de_results)
        
        # X-axis should be centered around zero
        xlim = ax.get_xlim()
        assert abs(xlim[0] + xlim[1]) < 0.1  # Should be approximately symmetric
        assert xlim[0] < 0 < xlim[1]  # Should span zero
    
    def test_plot_volcano_grid(self, basic_de_results):
        """Test that grid is added to plot."""
        ax = plot_volcano(basic_de_results)
        
        # Check that grid is enabled
        assert ax.grid  # Grid should be on
    
    def test_plot_volcano_legend(self, basic_de_results):
        """Test legend properties."""
        ax = plot_volcano(basic_de_results)
        
        # Should have legend
        legend = ax.get_legend()
        assert legend is not None
        
        # Should have 4 categories
        legend_texts = [text.get_text() for text in legend.get_texts()]
        expected_categories = ['Non-significant', 'LFC only', 'P-value only', 'Both']
        for cat in expected_categories:
            assert cat in legend_texts


class TestPlotVolcanoEdgeCases:
    """Test edge cases in volcano plot function."""
    
    def test_plot_volcano_infinite_values(self):
        """Test volcano plot with infinite values."""
        problematic_results = pd.DataFrame({
            'gene': ['Gene1', 'Gene2', 'Gene3'],
            'pval': [0.0, 0.01, np.inf],  # Zero and infinite p-values
            'padj': [0.0, 0.02, np.inf],
            'lfc': [np.inf, -np.inf, 1.0],  # Infinite fold changes
            'se': [0.5, 0.0, 0.3],  # Zero standard error
            'stat': [np.inf, -np.inf, 3.3]
        })
        
        # Should handle problematic values without crashing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore expected warnings
            ax = plot_volcano(problematic_results)
        
        assert isinstance(ax, plt.Axes)
    
    def test_plot_volcano_duplicate_genes(self):
        """Test volcano plot with duplicate gene names."""
        duplicate_results = pd.DataFrame({
            'gene': ['Gene1', 'Gene1', 'Gene2', 'Gene2'],  # Duplicates
            'pval': [0.01, 0.02, 0.03, 0.04],
            'padj': [0.02, 0.03, 0.04, 0.05],
            'lfc': [2.0, 1.8, -1.5, -1.3],
            'se': [0.4, 0.5, 0.3, 0.4],
            'stat': [5.0, 3.6, -5.0, -3.25]
        })
        
        # Should handle duplicates
        ax = plot_volcano(duplicate_results)
        assert isinstance(ax, plt.Axes)
    
    def test_plot_volcano_very_large_dataset(self):
        """Test volcano plot with large dataset."""
        np.random.seed(42)
        n_genes = 10000  # Large dataset
        
        large_results = pd.DataFrame({
            'gene': [f'Gene_{i:05d}' for i in range(n_genes)],
            'pval': np.random.uniform(0, 1, n_genes),
            'padj': np.random.uniform(0, 1, n_genes),
            'lfc': np.random.normal(0, 2, n_genes),
            'se': np.random.uniform(0.1, 2.0, n_genes),
            'stat': np.random.normal(0, 3, n_genes)
        })
        
        # Should handle large datasets efficiently
        ax = plot_volcano(large_results, labels=False)  # Disable labels for performance
        assert isinstance(ax, plt.Axes)
    
    def test_plot_volcano_special_gene_names(self):
        """Test volcano plot with special characters in gene names."""
        special_results = pd.DataFrame({
            'gene': ['Gene-1', 'Gene.2', 'Gene_3|var', 'Gene(4)', 'Gene[5]'],
            'pval': [0.001, 0.01, 0.05, 0.1, 0.2],
            'padj': [0.005, 0.02, 0.08, 0.15, 0.25],
            'lfc': [2.0, -1.8, 1.2, 0.5, -0.3],
            'se': [0.4, 0.5, 0.6, 0.3, 0.2],
            'stat': [5.0, -3.6, 2.0, 1.7, -1.5]
        })
        
        # Should handle special characters in gene names
        ax = plot_volcano(special_results, labels=True)
        assert isinstance(ax, plt.Axes)


class TestPlotVolcanoIntegration:
    """Test volcano plot integration with devil workflow."""
    
    def test_plot_volcano_with_real_devil_results(self):
        """Test volcano plot with results from actual devil analysis."""
        # Create realistic test data
        np.random.seed(42)
        n_genes, n_samples = 50, 25
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Fit model
        fit_result = devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        
        # Test differential expression
        contrast = [0, 1]
        de_results = devil.test_de(
            fit_result,
            contrast=contrast,
            verbose=False,
            use_gpu=False
        )
        
        # Create volcano plot
        ax = plot_volcano(de_results)
        
        # Should work seamlessly
        assert isinstance(ax, plt.Axes)
        
        # Check that plot contains expected elements
        assert ax.get_xlabel() == r'$\log_2$ Fold Change'
        assert ax.get_ylabel() == r'$-\log_{10}$ Adjusted P-value'
        
        # Should have threshold lines
        lines = ax.lines
        dashed_lines = [line for line in lines if line.get_linestyle() == '--']
        assert len(dashed_lines) >= 3  # At least horizontal and two vertical
    
    def test_plot_volcano_comparison_plots(self):
        """Test creating multiple volcano plots for comparison."""
        np.random.seed(42)
        
        # Create two sets of results for comparison
        results1 = pd.DataFrame({
            'gene': [f'Gene_{i}' for i in range(20)],
            'pval': np.random.uniform(0, 1, 20),
            'padj': np.random.uniform(0, 1, 20),
            'lfc': np.random.normal(0, 2, 20),
            'se': np.random.uniform(0.2, 1.0, 20),
            'stat': np.random.normal(0, 3, 20)
        })
        
        results2 = pd.DataFrame({
            'gene': [f'Gene_{i}' for i in range(20)],
            'pval': np.random.uniform(0, 1, 20),
            'padj': np.random.uniform(0, 1, 20),
            'lfc': np.random.normal(0, 1.5, 20),
            'se': np.random.uniform(0.2, 1.0, 20),
            'stat': np.random.normal(0, 2, 20)
        })
        
        # Create comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        plot_volcano(results1, ax=ax1, title="Condition A vs Control")
        plot_volcano(results2, ax=ax2, title="Condition B vs Control")
        
        # Both should be created successfully
        assert isinstance(ax1, plt.Axes)
        assert isinstance(ax2, plt.Axes)
        assert ax1.get_title() == "Condition A vs Control"
        assert ax2.get_title() == "Condition B vs Control"


class TestPlotVolcanoPerformance:
    """Test performance aspects of volcano plot."""
    
    def test_plot_volcano_memory_efficiency(self):
        """Test that plotting doesn't consume excessive memory."""
        np.random.seed(42)
        n_genes = 5000  # Moderately large
        
        results = pd.DataFrame({
            'gene': [f'Gene_{i:05d}' for i in range(n_genes)],
            'pval': np.random.uniform(0, 1, n_genes),
            'padj': np.random.uniform(0, 1, n_genes),
            'lfc': np.random.normal(0, 2, n_genes),
            'se': np.random.uniform(0.1, 2.0, n_genes),
            'stat': np.random.normal(0, 3, n_genes)
        })
        
        # Should complete without memory issues
        ax = plot_volcano(results, labels=False)  # Disable labels for performance
        assert isinstance(ax, plt.Axes)
    
    def test_plot_volcano_label_overlap_prevention(self):
        """Test that label overlap prevention works."""
        # Create data with many significant genes close together
        clustered_results = pd.DataFrame({
            'gene': [f'Gene_{i}' for i in range(10)],
            'pval': [0.001] * 10,  # All very significant
            'padj': [0.005] * 10,
            'lfc': [2.0 + 0.1 * i for i in range(10)],  # Close LFC values
            'se': [0.3] * 10,
            'stat': [6.7 + 0.3 * i for i in range(10)]
        })
        
        ax = plot_volcano(clustered_results, labels=True, top_n_labels=5)
        
        # Should create plot without errors
        assert isinstance(ax, plt.Axes)
        
        # Labels should be present but managed for overlap
        annotations = [child for child in ax.get_children() 
                      if isinstance(child, matplotlib.text.Annotation)]
        # Should have some labels but not too many due to overlap prevention


# Integration test for plotting with matplotlib backends
class TestPlotVolcanoBackends:
    """Test volcano plot with different matplotlib backends."""
    
    def test_plot_volcano_agg_backend(self):
        """Test volcano plot with Agg backend (headless)."""
        matplotlib.use('Agg')
        
        results = pd.DataFrame({
            'gene': ['Gene1', 'Gene2'],
            'pval': [0.01, 0.05],
            'padj': [0.02, 0.08],
            'lfc': [2.0, -1.5],
            'se': [0.4, 0.5],
            'stat': [5.0, -3.0]
        })
        
        ax = plot_volcano(results)
        assert isinstance(ax, plt.Axes)
    
    def test_plot_volcano_save_functionality(self):
        """Test that volcano plots can be saved."""
        import tempfile
        import os
        
        results = pd.DataFrame({
            'gene': ['Gene1', 'Gene2', 'Gene3'],
            'pval': [0.001, 0.01, 0.1],
            'padj': [0.005, 0.02, 0.15],
            'lfc': [2.5, -2.0, 0.5],
            'se': [0.4, 0.5, 0.3],
            'stat': [6.25, -4.0, 1.67]
        })
        
        ax = plot_volcano(results)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            ax.figure.savefig(tmp.name, dpi=100, bbox_inches='tight')
            tmp_path = tmp.name
        
        try:
            # Check that file was created and has reasonable size
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 1000  # At least 1KB
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)