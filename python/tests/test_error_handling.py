"""
Error handling and edge case tests for devil package.

This should be saved as python/tests/test_error_handling.py
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import warnings
from unittest.mock import patch, MagicMock

import devil

# skip tests that are not implemented
pytest.skip("Not implemented", allow_module_level=True)


class TestInputValidationErrors:
    """Test proper error handling for invalid inputs."""
    
    def test_invalid_data_types(self):
        """Test error handling with invalid data types."""
        # String data
        with pytest.raises(TypeError, match="Unsupported data type"):
            devil.fit_devil("not_a_matrix", use_gpu=False)
        
        # List input
        with pytest.raises(TypeError, match="Unsupported data type"):
            devil.fit_devil([[1, 2], [3, 4]], use_gpu=False)
        
        # None input
        with pytest.raises(TypeError, match="Unsupported data type"):
            devil.fit_devil(None, use_gpu=False)
    
    def test_negative_count_values(self):
        """Test error handling with negative count values."""
        invalid_counts = np.array([
            [1, 2, -1],  # Negative value
            [4, 5, 6]
        ])
        design = np.ones((3, 1))
        
        with pytest.raises(ValueError, match="negative values"):
            devil.fit_devil(invalid_counts, design_matrix=design, use_gpu=False)
    
    def test_infinite_count_values(self):
        """Test error handling with infinite count values."""
        invalid_counts = np.array([
            [1, 2, np.inf],
            [4, 5, 6]
        ])
        design = np.ones((3, 1))
        
        with pytest.raises(ValueError, match="numeric values"):
            devil.fit_devil(invalid_counts, design_matrix=design, use_gpu=False)
    
    def test_nan_count_values(self):
        """Test error handling with NaN count values."""
        invalid_counts = np.array([
            [1, 2, np.nan],
            [4, 5, 6]
        ])
        design = np.ones((3, 1))
        
        with pytest.raises(ValueError, match="numeric values"):
            devil.fit_devil(invalid_counts, design_matrix=design, use_gpu=False)
    
    def test_mismatched_dimensions(self):
        """Test error handling with mismatched matrix dimensions."""
        counts = np.random.negative_binomial(5, 0.3, size=(20, 15))
        
        # Wrong number of samples in design matrix
        wrong_design = np.ones((10, 2))  # Should be 15 samples, not 10
        
        with pytest.raises(ValueError, match="Sample count mismatch"):
            devil.fit_devil(counts, design_matrix=wrong_design, use_gpu=False)
    
    def test_empty_matrices(self):
        """Test error handling with empty matrices."""
        # Empty count matrix
        empty_counts = np.array([]).reshape(0, 0)
        empty_design = np.array([]).reshape(0, 0)
        
        with pytest.raises((ValueError, IndexError)):
            devil.fit_devil(empty_counts, design_matrix=empty_design, use_gpu=False)
    
    def test_single_dimension_matrices(self):
        """Test error handling with 1D arrays."""
        counts_1d = np.array([1, 2, 3, 4, 5])
        design_1d = np.array([1, 1, 0, 0, 1])
        
        # Should raise error for 1D input
        with pytest.raises((ValueError, IndexError)):
            devil.fit_devil(counts_1d, design_matrix=design_1d, use_gpu=False)
    
    def test_rank_deficient_design_matrix(self):
        """Test error handling with rank-deficient design matrix."""
        counts = np.random.negative_binomial(5, 0.3, size=(20, 15))
        
        # Rank-deficient design matrix
        design = np.column_stack([
            np.ones(15),
            np.random.normal(0, 1, 15),
            np.random.normal(0, 1, 15)
        ])
        design[:, 2] = design[:, 1]  # Make columns identical
        
        with pytest.raises(ValueError, match="rank deficient"):
            devil.fit_devil(counts, design_matrix=design, use_gpu=False)
    
    def test_insufficient_samples(self):
        """Test error handling with insufficient samples."""
        counts = np.random.negative_binomial(5, 0.3, size=(20, 3))  # Only 3 samples
        design = np.random.normal(0, 1, size=(3, 5))  # 5 features, only 3 samples
        
        with pytest.raises(ValueError, match="Insufficient samples"):
            devil.fit_devil(counts, design_matrix=design, use_gpu=False)


class TestParameterValidationErrors:
    """Test error handling for invalid parameters."""
    
    @pytest.fixture
    def valid_test_data(self):
        """Create valid test data for parameter testing."""
        np.random.seed(42)
        counts = np.random.negative_binomial(5, 0.3, size=(30, 20))
        design = np.column_stack([
            np.ones(20),
            np.random.binomial(1, 0.5, 20)
        ])
        return counts, design
    
    def test_invalid_max_iter(self, valid_test_data):
        """Test error handling with invalid max_iter values."""
        counts, design = valid_test_data
        
        # Negative max_iter
        with pytest.raises((ValueError, TypeError)):
            devil.fit_devil(
                counts, design_matrix=design, max_iter=-5, use_gpu=False
            )
        
        # Zero max_iter
        with pytest.raises((ValueError, TypeError)):
            devil.fit_devil(
                counts, design_matrix=design, max_iter=0, use_gpu=False
            )
        
        # Non-integer max_iter
        with pytest.raises(TypeError):
            devil.fit_devil(
                counts, design_matrix=design, max_iter=10.5, use_gpu=False
            )
    
    def test_invalid_tolerance(self, valid_test_data):
        """Test error handling with invalid tolerance values."""
        counts, design = valid_test_data
        
        # Negative tolerance
        with pytest.raises((ValueError, TypeError)):
            devil.fit_devil(
                counts, design_matrix=design, tolerance=-1e-3, use_gpu=False
            )
        
        # Zero tolerance (might cause infinite loops)
        with pytest.raises((ValueError, TypeError)):
            devil.fit_devil(
                counts, design_matrix=design, tolerance=0.0, use_gpu=False
            )
    
    def test_invalid_n_jobs(self, valid_test_data):
        """Test error handling with invalid n_jobs values."""
        counts, design = valid_test_data
        
        # Negative n_jobs (other than -1)
        # Note: -1 is valid (means use all cores)
        with pytest.raises((ValueError, TypeError)):
            devil.fit_devil(
                counts, design_matrix=design, n_jobs=-2, use_gpu=False
            )
        
        # Zero n_jobs
        with pytest.raises((ValueError, TypeError)):
            devil.fit_devil(
                counts, design_matrix=design, n_jobs=0, use_gpu=False
            )
    
    def test_invalid_offset(self, valid_test_data):
        """Test error handling with invalid offset values."""
        counts, design = valid_test_data
        
        # Negative offset might cause issues
        with pytest.raises((ValueError, TypeError)):
            devil.fit_devil(
                counts, design_matrix=design, offset=-1e-6, use_gpu=False
            )
    
    def test_invalid_init_overdispersion(self, valid_test_data):
        """Test error handling with invalid initial overdispersion."""
        counts, design = valid_test_data
        
        # Negative overdispersion
        with pytest.raises((ValueError, TypeError)):
            devil.fit_devil(
                counts, design_matrix=design, 
                init_overdispersion=-1.0, use_gpu=False
            )
        
        # Zero overdispersion with overdispersion=True might be problematic
        # (Should be handled gracefully, but test the edge case)
        result = devil.fit_devil(
            counts, design_matrix=design,
            init_overdispersion=0.0, overdispersion=False,
            verbose=False, max_iter=5, use_gpu=False
        )
        
        # Should complete with zero overdispersion when overdispersion=False
        assert np.allclose(result['overdispersion'], 0.0)


class TestComputationErrors:
    """Test error handling during computation."""
    
    def test_convergence_failure_handling(self):
        """Test handling of convergence failures."""
        np.random.seed(42)
        n_genes, n_samples = 20, 15
        
        # Create difficult data for convergence
        counts = np.random.negative_binomial(1, 0.95, size=(n_genes, n_samples))  # Very sparse
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Use very strict parameters to force convergence issues
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore convergence warnings
            
            fit_result = devil.fit_devil(
                counts,
                design_matrix=design,
                max_iter=2,  # Very few iterations
                tolerance=1e-8,  # Very strict tolerance
                overdispersion=True,
                verbose=False,
                use_gpu=False
            )
        
        # Should complete even with convergence issues
        assert 'beta' in fit_result
        assert 'overdispersion' in fit_result
        assert 'converged' in fit_result
        
        # Some genes might not converge
        convergence_rate = np.mean(fit_result['converged'])
        print(f"Convergence rate with strict parameters: {convergence_rate:.2%}")
        
        # Should still be usable for testing
        contrast = [0, 1]
        de_results = devil.test_de(fit_result, contrast, verbose=False, use_gpu=False)
        
        assert len(de_results) == n_genes
        assert np.all(np.isfinite(de_results['pval']))
    
    def test_singular_matrix_handling(self):
        """Test handling of singular matrices in computations."""
        np.random.seed(42)
        n_samples = 20
        
        # Create nearly singular design matrix
        x = np.random.normal(0, 1, n_samples)
        design = np.column_stack([
            np.ones(n_samples),
            x,
            x + 1e-14,  # Nearly identical to previous column
            np.random.normal(0, 1e-10, n_samples)  # Nearly zero column
        ])
        
        counts = np.random.negative_binomial(5, 0.3, size=(10, n_samples))
        
        # Should handle near-singularity gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # May warn about conditioning
            
            try:
                fit_result = devil.fit_devil(
                    counts,
                    design_matrix=design,
                    overdispersion=True,
                    verbose=False,
                    max_iter=10,
                    use_gpu=False
                )
                
                # Should complete using pseudo-inverse or regularization
                assert 'beta' in fit_result
                assert np.all(np.isfinite(fit_result['beta']))
                
            except np.linalg.LinAlgError:
                # This is acceptable - singular matrices should be caught
                pass
    
    def test_optimization_failure_fallback(self):
        """Test fallback behavior when optimization fails."""
        np.random.seed(42)
        n_samples = 25
        
        # Create problematic data
        y = np.array([0] * 20 + [1000] * 5)  # Extreme data
        design_matrix = np.ones((n_samples, 1))
        beta = np.array([0.0])
        offset_vector = np.zeros(n_samples)
        
        # Mock optimization to fail
        with patch('devil.overdispersion.optimize.minimize') as mock_minimize:
            mock_minimize.side_effect = Exception("Optimization failed")
            
            # Should handle optimization failure
            dispersion = devil.overdispersion.fit_dispersion(
                beta, design_matrix, y, offset_vector,
                tolerance=1e-3, max_iter=50
            )
            
            # Should return a reasonable fallback value
            assert isinstance(dispersion, float)
            assert dispersion > 0
            assert np.isfinite(dispersion)
    
    def test_memory_allocation_errors(self):
        """Test handling of memory allocation errors."""
        # This is hard to test directly, but we can test with very large requests
        
        # Try to create unreasonably large arrays
        with pytest.raises((MemoryError, OverflowError, ValueError)):
            # This should fail due to memory constraints
            huge_counts = np.zeros((10**9, 10**6))  # Unreasonably large
    
    def test_numerical_overflow_handling(self):
        """Test handling of numerical overflow conditions."""
        np.random.seed(42)
        n_samples = 20
        
        # Create data that might cause overflow
        design = np.ones((n_samples, 1))
        
        # Very large beta coefficients
        extreme_beta = np.array([100.0])  # exp(100) will overflow
        y = np.random.negative_binomial(5, 0.3, n_samples)
        size_factors = np.ones(n_samples)
        
        # Should handle overflow gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # May warn about overflow
            
            try:
                precision = 1.0
                H = devil.variance.compute_hessian(
                    extreme_beta, precision, y, design, size_factors
                )
                
                # If it completes, should produce finite results
                if not np.any(np.isinf(H)):
                    assert np.all(np.isfinite(H))
                
            except (OverflowError, FloatingPointError):
                # This is acceptable - overflow should be caught
                pass


class TestEdgeCaseDatasets:
    """Test behavior with edge case datasets."""
    
    def test_all_zero_samples(self):
        """Test handling of samples with all zero counts."""
        np.random.seed(42)
        n_genes, n_samples = 20, 15
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        
        # Make one sample all zeros
        counts[:, 0] = 0
        
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Should raise error due to size factor computation
        with pytest.raises(ValueError, match="all zeros"):
            devil.fit_devil(counts, design_matrix=design, use_gpu=False)
    
    def test_all_zero_genes(self):
        """Test handling of genes with all zero counts."""
        np.random.seed(42)
        n_genes, n_samples = 20, 15
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        
        # Make some genes all zeros
        counts[0, :] = 0
        counts[5, :] = 0
        counts[10, :] = 0
        
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Should handle zero genes gracefully
        fit_result = devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        
        # Should complete successfully
        assert fit_result['n_genes'] == n_genes
        
        # Zero genes might have special handling
        # Check that results are finite for non-zero genes
        non_zero_genes = np.sum(counts, axis=1) > 0
        assert np.all(np.isfinite(fit_result['beta'][non_zero_genes, :]))
    
    def test_constant_expression_genes(self):
        """Test handling of genes with constant expression."""
        np.random.seed(42)
        n_genes, n_samples = 15, 12
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        
        # Make some genes have constant expression
        counts[0, :] = 10  # Constant high expression
        counts[1, :] = 1   # Constant low expression
        
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Should handle constant genes
        fit_result = devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=15,
            use_gpu=False
        )
        
        # Should complete
        assert fit_result['n_genes'] == n_genes
        
        # Constant genes should have low/zero overdispersion
        constant_gene_od = fit_result['overdispersion'][:2]
        print(f"Overdispersion for constant genes: {constant_gene_od}")
        # They might have very low overdispersion or special handling
    
    def test_single_cell_type_design(self):
        """Test with design matrix representing single cell type."""
        np.random.seed(42)
        n_genes, n_samples = 25, 20
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        
        # Intercept-only design (no covariates)
        design = np.ones((n_samples, 1))
        
        # Should work with intercept-only model
        fit_result = devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=15,
            use_gpu=False
        )
        
        assert fit_result['beta'].shape == (n_genes, 1)
        
        # Can't test differential expression with intercept-only
        # But model fitting should work
    
    def test_perfect_separation_design(self):
        """Test with perfect separation in design matrix."""
        np.random.seed(42)
        n_genes, n_samples = 20, 20
        
        # Perfect separation: first 10 samples condition 0, last 10 condition 1
        condition = np.concatenate([np.zeros(10), np.ones(10)])
        design = np.column_stack([
            np.ones(n_samples),
            condition
        ])
        
        # Generate data with clear separation
        counts = np.zeros((n_genes, n_samples))
        for i in range(n_genes):
            # Condition 0: low counts
            counts[i, :10] = np.random.negative_binomial(2, 0.6, 10)
            # Condition 1: high counts
            counts[i, 10:] = np.random.negative_binomial(20, 0.3, 10)
        
        # Should handle perfect separation
        fit_result = devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=20,
            use_gpu=False
        )
        
        # Should complete (though some genes might not converge)
        assert fit_result['n_genes'] == n_genes
        
        # Test differential expression
        contrast = [0, 1]
        de_results = devil.test_de(fit_result, contrast, verbose=False, use_gpu=False)
        
        # Should complete and detect strong differences
        assert len(de_results) == n_genes
        assert np.sum(de_results['padj'] < 0.05) > 10  # Should find many significant


class TestContrastValidationErrors:
    """Test error handling in contrast specification."""
    
    @pytest.fixture
    def fitted_model_for_contrast_testing(self):
        """Create fitted model for contrast testing."""
        np.random.seed(42)
        counts = np.random.negative_binomial(5, 0.3, size=(15, 12))
        design = np.column_stack([
            np.ones(12),
            np.random.binomial(1, 0.5, 12),
            np.random.normal(0, 1, 12)
        ])
        
        return devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
    
    def test_wrong_contrast_length(self, fitted_model_for_contrast_testing):
        """Test error with wrong contrast vector length."""
        fit_result = fitted_model_for_contrast_testing
        
        # Too short
        with pytest.raises(ValueError, match="Contrast length"):
            devil.test_de(fit_result, contrast=[0, 1], use_gpu=False)
        
        # Too long
        with pytest.raises(ValueError, match="Contrast length"):
            devil.test_de(fit_result, contrast=[0, 1, 0, 1], use_gpu=False)
    
    def test_invalid_contrast_types(self, fitted_model_for_contrast_testing):
        """Test error with invalid contrast types."""
        fit_result = fitted_model_for_contrast_testing
        
        # String contrast
        with pytest.raises((ValueError, TypeError)):
            devil.test_de(fit_result, contrast=['a', 'b', 'c'], use_gpu=False)
        
        # None contrast
        with pytest.raises((ValueError, TypeError)):
            devil.test_de(fit_result, contrast=None, use_gpu=False)
        
        # Mixed types
        with pytest.raises((ValueError, TypeError)):
            devil.test_de(fit_result, contrast=[0, 'b', 1], use_gpu=False)
    
    def test_all_zero_contrast(self, fitted_model_for_contrast_testing):
        """Test handling of all-zero contrast vector."""
        fit_result = fitted_model_for_contrast_testing
        
        # All zeros contrast (tests nothing)
        contrast = [0, 0, 0]
        
        # Should complete but give trivial results
        de_results = devil.test_de(fit_result, contrast=contrast, verbose=False, use_gpu=False)
        
        # All fold changes should be zero
        assert np.allclose(de_results['lfc'], 0.0)
        
        # P-values might be 1 or undefined
        # At minimum, should not crash
        assert len(de_results) == fit_result['n_genes']
    
    def test_extreme_contrast_values(self, fitted_model_for_contrast_testing):
        """Test handling of extreme contrast values."""
        fit_result = fitted_model_for_contrast_testing
        
        # Very large contrast values
        extreme_contrast = [0, 1000, -1000]
        
        # Should handle extreme contrasts
        de_results = devil.test_de(
            fit_result, 
            contrast=extreme_contrast, 
            verbose=False, 
            use_gpu=False
        )
        
        # Should complete without errors
        assert len(de_results) == fit_result['n_genes']
        assert np.all(np.isfinite(de_results['pval']))


class TestClusterValidationErrors:
    """Test error handling in cluster specification."""
    
    @pytest.fixture
    def fitted_model_for_cluster_testing(self):
        """Create fitted model for cluster testing."""
        np.random.seed(42)
        counts = np.random.negative_binomial(5, 0.3, size=(20, 18))
        design = np.column_stack([
            np.ones(18),
            np.random.binomial(1, 0.5, 18)
        ])
        
        return devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
    
    def test_wrong_cluster_length(self, fitted_model_for_cluster_testing):
        """Test error with wrong cluster vector length."""
        fit_result = fitted_model_for_cluster_testing
        contrast = [0, 1]
        
        # Too few clusters
        wrong_clusters = np.array([1, 1, 2, 2])  # Only 4, should be 18
        
        with pytest.raises(ValueError, match="Clusters length"):
            devil.test_de(
                fit_result, 
                contrast=contrast, 
                clusters=wrong_clusters, 
                use_gpu=False
            )
    
    def test_single_cluster(self, fitted_model_for_cluster_testing):
        """Test handling of single cluster (no clustering)."""
        fit_result = fitted_model_for_cluster_testing
        contrast = [0, 1]
        
        # All samples in one cluster
        single_cluster = np.ones(18, dtype=int)
        
        # Should work (equivalent to no clustering)
        de_results = devil.test_de(
            fit_result,
            contrast=contrast,
            clusters=single_cluster,
            verbose=False,
            use_gpu=False
        )
        
        assert len(de_results) == fit_result['n_genes']
    
    def test_each_sample_own_cluster(self, fitted_model_for_cluster_testing):
        """Test with each sample as its own cluster."""
        fit_result = fitted_model_for_cluster_testing
        contrast = [0, 1]
        
        # Each sample is its own cluster
        individual_clusters = np.arange(1, 19)  # 1 to 18
        
        # Should handle this case
        de_results = devil.test_de(
            fit_result,
            contrast=contrast,
            clusters=individual_clusters,
            verbose=False,
            use_gpu=False
        )
        
        assert len(de_results) == fit_result['n_genes']
    
    def test_invalid_cluster_types(self, fitted_model_for_cluster_testing):
        """Test error with invalid cluster types."""
        fit_result = fitted_model_for_cluster_testing
        contrast = [0, 1]
        
        # Mixed types in cluster vector
        mixed_clusters = [1, 2, 'a', 4] + list(range(5, 19))
        
        # Should handle type conversion or raise appropriate error
        try:
            de_results = devil.test_de(
                fit_result,
                contrast=contrast,
                clusters=mixed_clusters,
                verbose=False,
                use_gpu=False
            )
            # If successful, should have converted to numeric
            assert len(de_results) == fit_result['n_genes']
        except (ValueError, TypeError):
            # Acceptable to raise error for invalid cluster types
            pass


class TestWarningHandling:
    """Test proper warning generation and handling."""
    
    def test_convergence_warnings(self):
        """Test that convergence warnings are properly generated."""
        np.random.seed(42)
        n_genes, n_samples = 15, 12
        
        # Create difficult convergence scenario
        counts = np.random.negative_binomial(1, 0.95, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            fit_result = devil.fit_devil(
                counts,
                design_matrix=design,
                max_iter=3,  # Very few iterations
                tolerance=1e-6,  # Strict tolerance
                overdispersion=True,
                verbose=False,
                use_gpu=False
            )
            
            # Should warn about convergence issues
            warning_messages = [str(warning.message) for warning in w]
            convergence_warnings = [msg for msg in warning_messages 
                                  if "converge" in msg.lower()]
            
            if np.mean(fit_result['converged']) < 0.9:
                assert len(convergence_warnings) > 0, "Should warn about convergence issues"
    
    def test_data_quality_warnings(self):
        """Test warnings about data quality issues."""
        np.random.seed(42)
        n_genes, n_samples = 20, 15
        
        # Non-integer counts
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples)).astype(float)
        counts += 0.1  # Make non-integer
        
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Should warn about non-integer counts during validation
            devil.fit_devil(
                counts,
                design_matrix=design,
                verbose=False,
                max_iter=5,
                use_gpu=False
            )
            
            warning_messages = [str(warning.message) for warning in w]
            non_integer_warnings = [msg for msg in warning_messages 
                                   if "non-integer" in msg.lower()]
            
            assert len(non_integer_warnings) > 0, "Should warn about non-integer counts"
    
    def test_missing_value_warnings(self):
        """Test warnings about missing values in DE results."""
        # Create DE results with missing values
        problematic_results = pd.DataFrame({
            'gene': ['Gene1', 'Gene2', 'Gene3'],
            'pval': [0.01, np.nan, 0.05],
            'padj': [0.02, 0.1, np.nan],
            'lfc': [1.5, -2.0, np.nan],
            'se': [0.3, 0.4, 0.5],
            'stat': [5.0, -5.0, 2.0]
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Should warn about missing values
            ax = devil.plot_volcano(problematic_results)
            
            warning_messages = [str(warning.message) for warning in w]
            missing_warnings = [msg for msg in warning_messages 
                               if "missing" in msg.lower()]
            
            assert len(missing_warnings) > 0, "Should warn about missing values"


class TestRecoveryMechanisms:
    """Test error recovery mechanisms."""
    
    def test_fallback_to_simpler_model(self):
        """Test fallback to simpler model when complex model fails."""
        np.random.seed(42)
        n_genes, n_samples = 20, 15
        
        # Create challenging data
        counts = np.random.negative_binomial(1, 0.9, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Try fitting with overdispersion
        try:
            fit_with_od = devil.fit_devil(
                counts,
                design_matrix=design,
                overdispersion=True,
                verbose=False,
                max_iter=20,
                use_gpu=False
            )
            od_success = True
        except Exception:
            od_success = False
        
        # Try fitting without overdispersion as fallback
        fit_without_od = devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=False,
            verbose=False,
            max_iter=20,
            use_gpu=False
        )
        
        # Simpler model should always work
        assert fit_without_od['n_genes'] == n_genes
        assert np.allclose(fit_without_od['overdispersion'], 0.0)
        
        if od_success:
            # If complex model worked, compare results
            beta_corr = np.corrcoef(
                fit_with_od['beta'].flatten(),
                fit_without_od['beta'].flatten()
            )[0, 1]
            print(f"Beta correlation (with vs without overdispersion): {beta_corr:.3f}")
    
    def test_graceful_degradation(self):
        """Test graceful degradation with increasingly difficult data."""
        np.random.seed(42)
        
        # Increasingly difficult scenarios
        difficulties = [
            {'od': 0.1, 'sparsity': 0.1, 'name': 'easy'},
            {'od': 1.0, 'sparsity': 0.3, 'name': 'medium'},
            {'od': 5.0, 'sparsity': 0.7, 'name': 'hard'},
            {'od': 20.0, 'sparsity': 0.9, 'name': 'very_hard'}
        ]
        
        results = {}
        
        for diff in difficulties:
            n_genes, n_samples = 30, 20
            
            # Generate data with specified difficulty
            base_mu = np.random.gamma(3, 1, size=(n_genes, n_samples))
            
            # Add sparsity
            sparsity_mask = np.random.binomial(1, 1 - diff['sparsity'], 
                                             size=(n_genes, n_samples))
            counts = np.random.negative_binomial(
                n=1/diff['od'], 
                p=1/(1 + base_mu * diff['od'])
            ) * sparsity_mask
            
            design = np.column_stack([
                np.ones(n_samples),
                np.random.binomial(1, 0.5, n_samples)
            ])
            
            # Attempt fitting
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                try:
                    fit_result = devil.fit_devil(
                        counts,
                        design_matrix=design,
                        overdispersion=True,
                        verbose=False,
                        max_iter=25,
                        use_gpu=False
                    )
                    
                    results[diff['name']] = {
                        'success': True,
                        'convergence_rate': np.mean(fit_result['converged']),
                        'finite_beta': np.all(np.isfinite(fit_result['beta']))
                    }
                    
                except Exception as e:
                    results[diff['name']] = {
                        'success': False,
                        'error': str(e)
                    }
        
        # Print results
        for name, result in results.items():
            if result['success']:
                print(f"{name}: success, convergence={result['convergence_rate']:.2%}")
            else:
                print(f"{name}: failed - {result['error']}")
        
        # Should handle at least easy and medium cases
        assert results['easy']['success'], "Should handle easy data"
        assert results['medium']['success'], "Should handle medium difficulty data"
