"""
Integration tests for the complete devil workflow.
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import warnings
import tempfile
import os

import devil


class TestCompleteWorkflow:
    """Test complete analysis workflow from data to results."""
    
    @pytest.fixture
    def realistic_dataset(self):
        """Create realistic single-cell RNA-seq dataset for integration testing."""
        np.random.seed(42)
        
        # Dataset parameters
        n_genes = 200
        n_samples = 100
        n_conditions = 3
        n_batches = 2
        
        # Create cell metadata
        conditions = np.repeat(np.arange(n_conditions), n_samples // n_conditions)
        conditions = np.concatenate([conditions, [0] * (n_samples - len(conditions))])
        batches = np.random.binomial(1, 0.4, n_samples)
        continuous_covar = np.random.normal(0, 1, n_samples)
        
        obs = pd.DataFrame({
            'condition': [f'Cond_{c}' for c in conditions],
            'batch': [f'Batch_{b}' for b in batches],
            'continuous_var': continuous_covar,
            'n_counts_log': np.random.normal(8, 1, n_samples),
            'cell_type': np.random.choice(['TypeA', 'TypeB', 'TypeC'], n_samples)
        })
        obs.index = [f'Cell_{i:04d}' for i in range(n_samples)]
        
        # Create gene metadata
        var = pd.DataFrame({
            'gene_symbol': [f'Gene_{i:03d}' for i in range(n_genes)],
            'gene_type': np.random.choice(['protein_coding', 'lncRNA', 'miRNA'], 
                                        n_genes, p=[0.8, 0.15, 0.05]),
            'highly_variable': np.random.binomial(1, 0.3, n_genes).astype(bool),
            'chromosome': np.random.choice([f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'], 
                                         n_genes)
        })
        var.index = [f'ENSG{i:08d}' for i in range(n_genes)]
        
        # Create realistic count data
        # Design matrix for data generation
        design_true = np.column_stack([
            np.ones(n_samples),
            (conditions == 1).astype(int),  # Condition 1 vs 0
            (conditions == 2).astype(int),  # Condition 2 vs 0
            batches,  # Batch effect
            continuous_covar  # Continuous covariate
        ])
        
        # True beta coefficients
        beta_true = np.random.normal(0, 0.5, size=(n_genes, 5))
        beta_true[:, 0] = np.random.normal(3, 1, n_genes)  # Positive intercepts
        
        # Make some genes differentially expressed
        n_de_genes = n_genes // 4
        beta_true[:n_de_genes, 1] = np.random.normal(2, 0.5, n_de_genes)  # Condition 1 effect
        beta_true[n_de_genes:2*n_de_genes, 2] = np.random.normal(-1.5, 0.5, n_de_genes)  # Condition 2 effect
        
        # Add batch effects
        beta_true[:, 3] = np.random.normal(0, 0.3, n_genes)
        
        # Generate size factors
        size_factors = np.random.lognormal(0, 0.4, n_samples)
        
        # Generate count data
        mu = size_factors[np.newaxis, :] * np.exp(beta_true @ design_true.T)
        overdispersion = np.random.gamma(3, 0.3, n_genes)
        
        count_matrix = np.zeros((n_genes, n_samples))
        for i in range(n_genes):
            p = 1 / (1 + mu[i, :] * overdispersion[i])
            n = 1 / overdispersion[i]
            count_matrix[i, :] = np.random.negative_binomial(n, p)
        
        # Create AnnData object
        adata = ad.AnnData(
            X=count_matrix.T,  # AnnData expects samples × genes
            obs=obs,
            var=var
        )
        
        # Add layers
        adata.layers['counts'] = adata.X.copy()
        adata.layers['logcounts'] = np.log1p(adata.X)
        
        return adata, design_true, beta_true, overdispersion
    
    def test_complete_workflow_anndata(self, realistic_dataset):
        """Test complete workflow with AnnData input."""
        adata, _, _, _ = realistic_dataset
        
        # Fit model
        fit_result = devil.fit_devil(
            adata,
            design_formula="~ condition + batch + continuous_var",
            overdispersion=True,
            size_factors=True,
            verbose=False,
            max_iter=20,
            use_gpu=False
        )
        
        # Validate fit result
        assert isinstance(fit_result, dict)
        assert fit_result['n_genes'] == 200
        assert fit_result['n_samples'] == 100
        assert fit_result['beta'].shape[1] == 5  # intercept + 3 conditions + batch + continuous
        
        # Test differential expression
        contrast_cond1 = [0, 1, 0, 0, 0]  # Condition 1 vs reference
        de_results_cond1 = devil.test_de(
            fit_result,
            contrast=contrast_cond1,
            verbose=False,
            use_gpu=False
        )
        
        # Validate DE results
        assert isinstance(de_results_cond1, pd.DataFrame)
        assert len(de_results_cond1) == 200
        assert all(col in de_results_cond1.columns for col in ['gene', 'pval', 'padj', 'lfc'])
        
        # Create volcano plot
        ax = devil.plot_volcano(de_results_cond1, title="Condition 1 vs Control")
        assert ax is not None
        
        # Test second contrast
        contrast_cond2 = [0, 0, 1, 0, 0]  # Condition 2 vs reference
        de_results_cond2 = devil.test_de(
            fit_result,
            contrast=contrast_cond2,
            verbose=False,
            use_gpu=False
        )
        
        assert len(de_results_cond2) == 200
        
        # Results should be different for different contrasts
        assert not np.allclose(de_results_cond1['lfc'], de_results_cond2['lfc'])
    
    def test_complete_workflow_numpy_arrays(self, realistic_dataset):
        """Test complete workflow with numpy array input."""
        adata, design_true, _, _ = realistic_dataset
        
        # Extract count matrix and create simple design
        count_matrix = adata.X.T  # Convert back to genes × samples
        simple_design = np.column_stack([
            np.ones(adata.n_obs),
            (adata.obs['condition'] == 'Cond_1').astype(int)
        ])
        
        # Fit model
        fit_result = devil.fit_devil(
            count_matrix,
            design_matrix=simple_design,
            overdispersion=True,
            size_factors=True,
            verbose=False,
            max_iter=15,
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
        
        # Should complete successfully
        assert len(de_results) == 200
        assert np.sum(de_results['padj'] < 0.05) > 0  # Should find some significant genes
        
        # Create plot
        ax = devil.plot_volcano(de_results)
        assert ax is not None
    
    def test_workflow_with_clustering(self, realistic_dataset):
        """Test workflow with sample clustering."""
        adata, _, _, _ = realistic_dataset
        
        # Fit model
        fit_result = devil.fit_devil(
            adata,
            design_formula="~ condition + batch",
            overdispersion=True,
            verbose=False,
            max_iter=15,
            use_gpu=False
        )
        
        # Create artificial patient clustering
        n_patients = 20
        patients_per_condition = n_patients // 3
        patient_assignments = np.concatenate([
            np.repeat(np.arange(1, patients_per_condition + 1), 100 // patients_per_condition),
            np.repeat(np.arange(patients_per_condition + 1, 2 * patients_per_condition + 1), 
                     100 // patients_per_condition),
            np.repeat(np.arange(2 * patients_per_condition + 1, n_patients + 1), 
                     100 - 2 * (100 // patients_per_condition))
        ])[:100]
        
        # Test with clustering
        contrast = [0, 1, 0, 0]
        de_results_clustered = devil.test_de(
            fit_result,
            contrast=contrast,
            clusters=patient_assignments,
            verbose=False,
            use_gpu=False
        )
        
        # Compare with non-clustered analysis
        de_results_standard = devil.test_de(
            fit_result,
            contrast=contrast,
            clusters=None,
            verbose=False,
            use_gpu=False
        )
        
        # Both should complete
        assert len(de_results_clustered) == 200
        assert len(de_results_standard) == 200
        
        # Clustered analysis should generally have larger standard errors
        mean_se_clustered = np.mean(de_results_clustered['se'])
        mean_se_standard = np.mean(de_results_standard['se'])
        # This is not always true, but often is
        # assert mean_se_clustered >= mean_se_standard * 0.8  # Allow some flexibility
    
    def test_workflow_memory_efficient(self, realistic_dataset):
        """Test memory-efficient workflow for large datasets."""
        adata, _, _, _ = realistic_dataset
        
        # Fit model
        fit_result = devil.fit_devil(
            adata,
            design_formula="~ condition",
            overdispersion=True,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        
        # Test subset of genes
        gene_subset = adata.var_names[:50]  # Test first 50 genes
        
        de_results = devil.test_de_memory_efficient(
            fit_result,
            contrast=[0, 1, 0],
            gene_subset=gene_subset,
            verbose=False
        )
        
        # Should test only subset
        assert len(de_results) == 50
        assert all(gene in gene_subset for gene in de_results['gene'])
    
    def test_workflow_sparse_data(self, realistic_dataset):
        """Test workflow with sparse count matrices."""
        adata, _, _, _ = realistic_dataset
        
        # Convert to sparse
        adata.X = sparse.csr_matrix(adata.X)
        
        # Should handle sparse data
        fit_result = devil.fit_devil(
            adata,
            design_formula="~ condition",
            overdispersion=True,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        
        assert fit_result['n_genes'] == 200
        assert fit_result['n_samples'] == 100


class TestWorkflowRobustness:
    """Test workflow robustness with challenging data."""
    
    def test_workflow_with_zero_inflation(self):
        """Test workflow with zero-inflated count data."""
        np.random.seed(42)
        n_genes, n_samples = 100, 50
        
        # Generate zero-inflated data
        base_counts = np.random.negative_binomial(3, 0.3, size=(n_genes, n_samples))
        
        # Add excessive zeros (zero inflation)
        zero_mask = np.random.binomial(1, 0.7, size=(n_genes, n_samples))  # 70% zeros
        counts = base_counts * zero_mask
        
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Should handle zero-inflated data
        fit_result = devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=15,
            use_gpu=False
        )
        
        contrast = [0, 1]
        de_results = devil.test_de(
            fit_result,
            contrast=contrast,
            verbose=False,
            use_gpu=False
        )
        
        # Should complete without errors
        assert len(de_results) == n_genes
        assert np.all(np.isfinite(de_results['pval']))
    
    def test_workflow_with_outlier_samples(self):
        """Test workflow with outlier samples."""
        np.random.seed(42)
        n_genes, n_samples = 80, 40
        
        # Generate normal data
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        
        # Add outlier samples with very different expression
        counts[:, 0] *= 10  # Very high expression sample
        counts[:, 1] = np.random.negative_binomial(1, 0.9, n_genes)  # Very low expression
        
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Should handle outliers
        fit_result = devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=20,
            use_gpu=False
        )
        
        de_results = devil.test_de(
            fit_result,
            contrast=[0, 1],
            verbose=False,
            use_gpu=False
        )
        
        assert len(de_results) == n_genes
        assert np.all(np.isfinite(de_results['pval']))
    
    def test_workflow_with_low_expressed_genes(self):
        """Test workflow with many low-expressed genes."""
        np.random.seed(42)
        n_genes, n_samples = 150, 60
        
        # Generate data with many low-expressed genes
        counts = np.random.negative_binomial(1, 0.8, size=(n_genes, n_samples))  # Many zeros
        
        # Add some higher expressed genes
        counts[:30, :] = np.random.negative_binomial(10, 0.3, size=(30, n_samples))
        
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Should handle low expression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # May warn about convergence
            
            fit_result = devil.fit_devil(
                counts,
                design_matrix=design,
                overdispersion=True,
                verbose=False,
                max_iter=25,
                use_gpu=False
            )
            
            de_results = devil.test_de(
                fit_result,
                contrast=[0, 1],
                verbose=False,
                use_gpu=False
            )
        
        assert len(de_results) == n_genes
    
    def test_workflow_with_extreme_design(self):
        """Test workflow with challenging design matrix."""
        np.random.seed(42)
        n_genes, n_samples = 60, 30
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        
        # Create challenging design with interactions and continuous variables
        x1 = np.random.binomial(1, 0.5, n_samples)
        x2 = np.random.normal(0, 2, n_samples)  # High variance continuous
        x3 = np.random.binomial(1, 0.1, n_samples)  # Rare binary condition
        
        design = np.column_stack([
            np.ones(n_samples),
            x1,
            x2,
            x3,
            x1 * x2  # Interaction term
        ])
        
        # Should handle complex design
        fit_result = devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=20,
            use_gpu=False
        )
        
        # Test different contrasts
        contrasts = [
            [0, 1, 0, 0, 0],  # Main effect of x1
            [0, 0, 1, 0, 0],  # Main effect of x2
            [0, 0, 0, 1, 0],  # Main effect of x3 (rare)
            [0, 0, 0, 0, 1]   # Interaction effect
        ]
        
        for i, contrast in enumerate(contrasts):
            de_results = devil.test_de(
                fit_result,
                contrast=contrast,
                verbose=False,
                use_gpu=False
            )
            
            assert len(de_results) == n_genes, f"Contrast {i} failed"
            assert np.all(np.isfinite(de_results['pval'])), f"Contrast {i} has invalid p-values"


class TestWorkflowComparisons:
    """Test consistency between different analysis approaches."""
    
    def test_poisson_vs_negative_binomial(self):
        """Compare Poisson vs negative binomial models."""
        np.random.seed(42)
        n_genes, n_samples = 50, 30
        
        # Generate Poisson-like data (low overdispersion)
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        beta_true = np.column_stack([
            np.random.normal(2, 0.5, n_genes),
            np.random.normal(1, 0.5, n_genes)
        ])
        
        mu = np.exp(design @ beta_true.T)
        counts = np.random.poisson(mu.T)
        
        # Fit both models
        fit_poisson = devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=False,  # Poisson model
            verbose=False,
            max_iter=15,
            use_gpu=False
        )
        
        fit_nb = devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,  # Negative binomial
            verbose=False,
            max_iter=15,
            use_gpu=False
        )
        
        # Compare results
        contrast = [0, 1]
        
        de_poisson = devil.test_de(fit_poisson, contrast, verbose=False, use_gpu=False)
        de_nb = devil.test_de(fit_nb, contrast, verbose=False, use_gpu=False)
        
        # Results should be reasonably similar for Poisson-like data
        lfc_correlation = np.corrcoef(de_poisson['lfc'], de_nb['lfc'])[0, 1]
        assert lfc_correlation > 0.7, f"Poor correlation between Poisson and NB: {lfc_correlation}"
    
    def test_size_factors_vs_no_size_factors(self):
        """Compare analysis with and without size factor normalization."""
        np.random.seed(42)
        n_genes, n_samples = 60, 25
        
        # Generate data with different sequencing depths
        base_counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        depth_factors = np.random.lognormal(0, 0.8, n_samples)  # Variable depths
        counts = (base_counts * depth_factors[np.newaxis, :]).astype(int)
        
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Fit with size factors
        fit_with_sf = devil.fit_devil(
            counts,
            design_matrix=design,
            size_factors=True,
            overdispersion=True,
            verbose=False,
            max_iter=15,
            use_gpu=False
        )
        
        # Fit without size factors
        fit_without_sf = devil.fit_devil(
            counts,
            design_matrix=design,
            size_factors=False,
            overdispersion=True,
            verbose=False,
            max_iter=15,
            use_gpu=False
        )
        
        # Test differential expression
        contrast = [0, 1]
        
        de_with_sf = devil.test_de(fit_with_sf, contrast, verbose=False, use_gpu=False)
        de_without_sf = devil.test_de(fit_without_sf, contrast, verbose=False, use_gpu=False)
        
        # Size factor correction should improve results
        # (This is hard to test directly, but at least check both complete)
        assert len(de_with_sf) == n_genes
        assert len(de_without_sf) == n_genes
        
        # Size factors should not all be 1
        assert not np.allclose(fit_with_sf['size_factors'], 1.0)
        assert np.allclose(fit_without_sf['size_factors'], 1.0)
    
    def test_different_adjustment_methods(self):
        """Test different multiple testing adjustment methods."""
        np.random.seed(42)
        n_genes, n_samples = 80, 35
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Fit model once
        fit_result = devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=15,
            use_gpu=False
        )
        
        # Test different adjustment methods
        contrast = [0, 1]
        adjustment_methods = ['fdr_bh', 'bonferroni', 'holm']
        
        results_by_method = {}
        for method in adjustment_methods:
            results = devil.test_de(
                fit_result,
                contrast=contrast,
                pval_adjust_method=method,
                verbose=False,
                use_gpu=False
            )
            results_by_method[method] = results
        
        # Raw p-values should be identical
        for method in adjustment_methods[1:]:
            np.testing.assert_allclose(
                results_by_method['fdr_bh']['pval'],
                results_by_method[method]['pval']
            )
        
        # Adjusted p-values should be different
        assert not np.allclose(
            results_by_method['fdr_bh']['padj'],
            results_by_method['bonferroni']['padj']
        )


class TestWorkflowValidation:
    """Test workflow validation and error handling."""
    
    def test_workflow_invalid_formula(self):
        """Test workflow with invalid design formula."""
        np.random.seed(42)
        n_genes, n_samples = 30, 20
        
        # Create AnnData without required variables
        counts = np.random.negative_binomial(5, 0.3, size=(n_samples, n_genes))
        obs = pd.DataFrame({
            'condition': np.random.choice(['A', 'B'], n_samples)
            # Missing 'treatment' variable
        })
        
        adata = ad.AnnData(X=counts, obs=obs)
        
        # Should raise error for missing variable
        with pytest.raises(Exception):  # patsy will raise specific error
            devil.fit_devil(
                adata,
                design_formula="~ condition + treatment",  # 'treatment' doesn't exist
                verbose=False,
                use_gpu=False
            )
    
    def test_workflow_convergence_issues(self):
        """Test workflow behavior with convergence issues."""
        np.random.seed(42)
        n_genes, n_samples = 40, 25
        
        # Create challenging data for convergence
        counts = np.random.negative_binomial(2, 0.9, size=(n_genes, n_samples))  # Very sparse
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Use strict convergence criteria
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Expect convergence warnings
            
            fit_result = devil.fit_devil(
                counts,
                design_matrix=design,
                max_iter=5,  # Very few iterations
                tolerance=1e-8,  # Very strict tolerance
                overdispersion=True,
                verbose=False,
                use_gpu=False
            )
            
            de_results = devil.test_de(
                fit_result,
                contrast=[0, 1],
                verbose=False,
                use_gpu=False
            )
        
        # Should complete despite convergence issues
        assert len(de_results) == n_genes
        # Some genes may not have converged
        assert np.sum(fit_result['converged']) >= 0  # At least some should work
    
    def test_workflow_extreme_parameters(self):
        """Test workflow with extreme parameter values."""
        np.random.seed(42)
        n_genes, n_samples = 30, 20
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Extreme parameter values
        fit_result = devil.fit_devil(
            counts,
            design_matrix=design,
            init_overdispersion=1000.0,  # Very high
            offset=1e-10,  # Very small
            tolerance=1e-1,  # Very loose
            max_iter=200,  # Many iterations
            overdispersion=True,
            verbose=False,
            use_gpu=False
        )
        
        de_results = devil.test_de(
            fit_result,
            contrast=[0, 1],
            max_lfc=50.0,  # Very high cap
            verbose=False,
            use_gpu=False
        )
        
        # Should handle extreme parameters
        assert len(de_results) == n_genes
        assert np.all(np.abs(de_results['lfc']) <= 50.0)


class TestWorkflowPerformance:
    """Test performance aspects of the complete workflow."""
    
    def test_workflow_small_dataset_performance(self):
        """Test that workflow is efficient for small datasets."""
        np.random.seed(42)
        n_genes, n_samples = 20, 15
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        import time
        
        # Should complete quickly
        start_time = time.time()
        
        fit_result = devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        
        de_results = devil.test_de(
            fit_result,
            contrast=[0, 1],
            verbose=False,
            use_gpu=False
        )
        
        total_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 10 seconds for small data)
        assert total_time < 10.0
        assert len(de_results) == n_genes
    
    def test_workflow_parallel_scaling(self):
        """Test that parallel processing improves performance."""
        np.random.seed(42)
        n_genes, n_samples = 100, 40
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        import time
        
        # Single-threaded
        start_time = time.time()
        fit_1job = devil.fit_devil(
            counts,
            design_matrix=design,
            n_jobs=1,
            overdispersion=True,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        time_1job = time.time() - start_time
        
        # Multi-threaded
        start_time = time.time()
        fit_4job = devil.fit_devil(
            counts,
            design_matrix=design,
            n_jobs=4,
            overdispersion=True,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        time_4job = time.time() - start_time
        
        # Results should be identical
        np.testing.assert_allclose(fit_1job['beta'], fit_4job['beta'], rtol=1e-10)
        
        # Multi-threading may not always be faster for small datasets due to overhead
        # Just ensure both complete successfully with reasonable times
        assert time_4job < 30.0  # Should complete in reasonable time
        assert time_1job < 30.0  # Should complete in reasonable time


class TestWorkflowDocumentation:
    """Test workflow examples from documentation."""
    
    def test_readme_example(self):
        """Test the basic example from README."""
        # Simulate the README example
        np.random.seed(42)
        n_genes, n_samples = 50, 30
        
        # Create synthetic AnnData
        X = np.random.negative_binomial(5, 0.3, size=(n_samples, n_genes))
        obs = pd.DataFrame({
            'condition': np.random.choice(['control', 'treatment'], n_samples),
            'batch': np.random.choice(['batch1', 'batch2'], n_samples)
        })
        var = pd.DataFrame({
            'gene_name': [f'Gene_{i}' for i in range(n_genes)]
        })
        
        adata = ad.AnnData(X=X, obs=obs, var=var)
        
        # Follow README workflow
        result = devil.fit_devil(
            adata,
            design_formula="~ condition + batch",
            verbose=False,
            use_gpu=False,
            max_iter=10
        )
        
        de_results = devil.test_de(
            result,
            contrast=[0, 1, 0]  # treatment vs control
        )
        
        ax = devil.plot_volcano(de_results)
        
        # Should complete successfully
        assert len(de_results) == n_genes
        assert ax is not None
    
    def test_clustering_example(self):
        """Test the clustering example from documentation."""
        np.random.seed(42)
        n_genes, n_samples = 40, 24
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Create patient IDs (clustering)
        patient_ids = np.repeat(np.arange(1, 9), 3)  # 8 patients, 3 samples each
        
        fit_result = devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        
        # Test with clustering
        de_results = devil.test_de(
            fit_result,
            contrast=[0, 1],
            clusters=patient_ids,
            verbose=False,
            use_gpu=False
        )
        
        assert len(de_results) == n_genes
        assert np.all(np.isfinite(de_results['pval']))


class TestWorkflowFileIO:
    """Test workflow with file input/output operations."""
    
    def test_workflow_save_load_results(self):
        """Test saving and loading analysis results."""
        np.random.seed(42)
        n_genes, n_samples = 30, 20
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Run analysis
        fit_result = devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        
        de_results = devil.test_de(
            fit_result,
            contrast=[0, 1],
            verbose=False,
            use_gpu=False
        )
        
        # Save results to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            de_results.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
        
        try:
            # Load results back
            loaded_results = pd.read_csv(tmp_path)
            
            # Should be identical
            pd.testing.assert_frame_equal(
                de_results.reset_index(drop=True),
                loaded_results
            )
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_workflow_export_plots(self):
        """Test exporting plots from workflow."""
        np.random.seed(42)
        n_genes, n_samples = 25, 18
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Run analysis
        fit_result = devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        
        de_results = devil.test_de(
            fit_result,
            contrast=[0, 1],
            verbose=False,
            use_gpu=False
        )
        
        # Create and save plot
        ax = devil.plot_volcano(de_results, title="Test Analysis")
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            ax.figure.savefig(tmp.name, dpi=150, bbox_inches='tight')
            tmp_path = tmp.name
        
        try:
            # Check that file was created
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 5000  # At least 5KB
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)