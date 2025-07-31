"""
Unit tests for main CPU fitting functions.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad
from unittest.mock import patch, MagicMock
import warnings

import devil
from devil.main import fit_devil, _fit_beta_cpu, _fit_overdispersion_cpu


class TestFitDevil:
    """Test the main fit_devil function."""
    
    @pytest.fixture
    def synthetic_count_matrix(self):
        """Create synthetic count matrix for testing."""
        np.random.seed(42)
        n_genes, n_samples = 50, 30
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        return counts.astype(np.float64)
    
    @pytest.fixture
    def design_matrix(self):
        """Create design matrix for testing."""
        np.random.seed(42)
        n_samples = 30
        design = np.column_stack([
            np.ones(n_samples),  # intercept
            np.random.binomial(1, 0.5, n_samples),  # condition
            np.random.normal(0, 1, n_samples)  # continuous covariate
        ])
        return design
    
    @pytest.fixture
    def anndata_object(self, synthetic_count_matrix):
        """Create AnnData object for testing."""
        n_genes, n_samples = synthetic_count_matrix.shape
        
        obs = pd.DataFrame({
            'condition': np.random.choice(['A', 'B'], n_samples),
            'batch': np.random.choice(['batch1', 'batch2'], n_samples),
            'continuous_var': np.random.normal(0, 1, n_samples)
        })
        obs.index = [f'Cell_{i:03d}' for i in range(n_samples)]
        
        var = pd.DataFrame({
            'gene_name': [f'Gene_{i:03d}' for i in range(n_genes)],
            'highly_variable': np.random.binomial(1, 0.3, n_genes).astype(bool)
        })
        var.index = [f'ENSG{i:08d}' for i in range(n_genes)]
        
        adata = ad.AnnData(
            X=synthetic_count_matrix.T,  # AnnData expects samples × genes
            obs=obs,
            var=var
        )
        
        return adata
    
    def test_fit_devil_basic_numpy(self, synthetic_count_matrix, design_matrix):
        """Test basic fit_devil with numpy arrays."""
        result = fit_devil(
            synthetic_count_matrix,
            design_matrix=design_matrix,
            overdispersion=True,
            size_factors=True,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        
        # Check basic structure
        assert isinstance(result, dict)
        required_keys = [
            'beta', 'overdispersion', 'iterations', 'size_factors', 
            'offset_vector', 'design_matrix', 'gene_names', 'n_genes', 
            'n_samples', 'converged', 'count_matrix'
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        # Check dimensions
        assert result['beta'].shape == (50, 3)
        assert len(result['overdispersion']) == 50
        assert len(result['iterations']) == 50
        assert len(result['size_factors']) == 30
        assert len(result['offset_vector']) == 30
        assert result['n_genes'] == 50
        assert result['n_samples'] == 30
        
        # Check data types
        assert isinstance(result['converged'], np.ndarray)
        assert result['converged'].dtype == bool
        
    def test_fit_devil_anndata(self, anndata_object):
        """Test fit_devil with AnnData object."""
        result = fit_devil(
            anndata_object,
            design_formula="~ condition + batch",
            overdispersion=True,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        
        assert result['n_genes'] == 50
        assert result['n_samples'] == 30
        assert result['beta'].shape[1] == 3  # intercept + condition + batch
        
    def test_fit_devil_sparse_matrix(self, synthetic_count_matrix, design_matrix):
        """Test fit_devil with sparse matrix input."""
        sparse_counts = sparse.csr_matrix(synthetic_count_matrix)
        
        result = fit_devil(
            sparse_counts,
            design_matrix=design_matrix,
            overdispersion=True,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        
        assert result['n_genes'] == 50
        assert result['n_samples'] == 30
        
    def test_fit_devil_no_overdispersion(self, synthetic_count_matrix, design_matrix):
        """Test fit_devil without overdispersion estimation (Poisson model)."""
        result = fit_devil(
            synthetic_count_matrix,
            design_matrix=design_matrix,
            overdispersion=False,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        
        assert np.allclose(result['overdispersion'], 0)
        
    def test_fit_devil_no_size_factors(self, synthetic_count_matrix, design_matrix):
        """Test fit_devil without size factor computation."""
        result = fit_devil(
            synthetic_count_matrix,
            design_matrix=design_matrix,
            size_factors=False,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        
        assert np.allclose(result['size_factors'], 1.0)
        
    def test_fit_devil_init_overdispersion(self, synthetic_count_matrix, design_matrix):
        """Test fit_devil with initial overdispersion value."""
        init_disp = 50.0
        result = fit_devil(
            synthetic_count_matrix,
            design_matrix=design_matrix,
            init_overdispersion=init_disp,
            overdispersion=False,  # Don't fit, just use initial
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        
        # When overdispersion=False, should be all zeros regardless of init
        assert np.allclose(result['overdispersion'], 0)
        
    def test_fit_devil_custom_parameters(self, synthetic_count_matrix, design_matrix):
        """Test fit_devil with custom optimization parameters."""
        result = fit_devil(
            synthetic_count_matrix,
            design_matrix=design_matrix,
            max_iter=5,
            tolerance=1e-2,
            offset=1e-5,
            verbose=False,
            use_gpu=False
        )
        
        assert result['n_genes'] == 50
        # Some genes may not converge with strict parameters
        assert np.any(result['iterations'] <= 5)
        
    def test_fit_devil_parallel_jobs(self, synthetic_count_matrix, design_matrix):
        """Test fit_devil with specified number of parallel jobs."""
        result = fit_devil(
            synthetic_count_matrix,
            design_matrix=design_matrix,
            n_jobs=2,
            verbose=False,
            max_iter=5,
            use_gpu=False
        )
        
        assert result['n_genes'] == 50


class TestInputValidation:
    """Test input validation in fit_devil."""
    
    def test_missing_design_matrix_and_formula(self):
        """Test error when both design_matrix and design_formula are missing."""
        counts = np.random.negative_binomial(3, 0.4, size=(10, 10))
        
        with pytest.raises(ValueError, match="Must provide either"):
            fit_devil(counts, use_gpu=False)
    
    def test_design_formula_without_anndata(self):
        """Test error when using design_formula with non-AnnData input."""
        counts = np.random.negative_binomial(3, 0.4, size=(10, 10))
        
        with pytest.raises(ValueError, match="design_formula requires AnnData"):
            fit_devil(counts, design_formula="~ condition", use_gpu=False)
    
    def test_incompatible_dimensions(self):
        """Test error with incompatible matrix dimensions."""
        counts = np.random.negative_binomial(3, 0.4, size=(10, 15))
        design = np.random.normal(0, 1, size=(10, 3))  # Wrong number of samples
        
        with pytest.raises(ValueError, match="Sample count mismatch"):
            fit_devil(counts, design_matrix=design, use_gpu=False)
    
    def test_negative_counts(self):
        """Test error with negative count values."""
        counts = np.random.normal(0, 1, size=(10, 10))  # Can have negative values
        design = np.random.normal(0, 1, size=(10, 3))
        
        with pytest.raises(ValueError, match="negative values"):
            fit_devil(counts, design_matrix=design, use_gpu=False)
    
    def test_rank_deficient_design(self):
        """Test error with rank-deficient design matrix."""
        counts = np.random.negative_binomial(3, 0.4, size=(10, 10))
        design = np.column_stack([
            np.ones(10),
            np.ones(10)  # Linearly dependent column
        ])
        
        with pytest.raises(ValueError, match="rank deficient"):
            fit_devil(counts, design_matrix=design, use_gpu=False)
    
    def test_insufficient_samples(self):
        """Test error with insufficient samples relative to features."""
        counts = np.random.negative_binomial(3, 0.4, size=(10, 3))
        design = np.random.normal(0, 1, size=(3, 5))  # More features than samples
        
        with pytest.raises(ValueError, match="Insufficient samples"):
            fit_devil(counts, design_matrix=design, use_gpu=False)


class TestHelperFunctions:
    """Test helper functions used in main module."""
    
    @pytest.fixture
    def mock_fit_data(self):
        """Create mock fitted data for testing helper functions."""
        np.random.seed(42)
        n_genes, n_samples, n_features = 20, 15, 3
        
        return {
            'count_matrix': np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples)),
            'design_matrix': np.column_stack([
                np.ones(n_samples),
                np.random.binomial(1, 0.5, n_samples),
                np.random.normal(0, 1, n_samples)
            ]),
            'beta_init': np.random.normal(0, 1, size=(n_genes, n_features)),
            'offset_vector': np.random.normal(0, 0.1, n_samples),
            'dispersion_init': np.random.gamma(1, 1, n_genes)
        }
    
    def test_fit_beta_cpu(self, mock_fit_data):
        """Test CPU beta coefficient fitting."""
        beta, iterations, converged = _fit_beta_cpu(
            mock_fit_data['count_matrix'],
            mock_fit_data['design_matrix'],
            mock_fit_data['beta_init'],
            mock_fit_data['offset_vector'],
            mock_fit_data['dispersion_init'],
            max_iter=10,
            tolerance=1e-3,
            n_jobs=2,
            verbose=False
        )
        
        assert beta.shape == (20, 3)
        assert len(iterations) == 20
        assert len(converged) == 20
        assert isinstance(converged, np.ndarray)
        assert converged.dtype == bool
        assert np.all(iterations >= 1)
        assert np.all(iterations <= 10)
    
    def test_fit_overdispersion_cpu(self, mock_fit_data):
        """Test CPU overdispersion fitting."""
        # Create fitted beta values
        beta = np.random.normal(0, 1, size=(20, 3))
        
        theta = _fit_overdispersion_cpu(
            beta,
            mock_fit_data['design_matrix'],
            mock_fit_data['count_matrix'],
            mock_fit_data['offset_vector'],
            tolerance=1e-3,
            max_iter=10,
            do_cox_reid_adjustment=True,
            n_jobs=2,
            verbose=False
        )
        
        assert len(theta) == 20
        assert np.all(theta >= 0)  # Overdispersion should be non-negative
        assert np.all(np.isfinite(theta))  # Should be finite values


class TestEdgeCases:
    """Test edge cases and special conditions."""
    
    def test_all_zero_genes(self):
        """Test handling of genes with all zero counts."""
        counts = np.zeros((5, 10))  # All zero counts
        design = np.column_stack([
            np.ones(10),
            np.random.binomial(1, 0.5, 10)
        ])
        
        # Should handle gracefully but raise warning about size factors
        with pytest.raises(Exception):  # Should fail due to size factor computation
            fit_devil(counts, design_matrix=design, use_gpu=False, verbose=False)
    
    def test_single_gene(self):
        """Test fitting with single gene."""
        counts = np.random.negative_binomial(5, 0.3, size=(1, 20))
        design = np.column_stack([
            np.ones(20),
            np.random.binomial(1, 0.5, 20)
        ])
        
        result = fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        
        assert result['n_genes'] == 1
        assert result['beta'].shape == (1, 2)
        
    def test_single_sample_per_condition(self):
        """Test fitting with minimal samples per condition."""
        counts = np.random.negative_binomial(5, 0.3, size=(10, 4))
        design = np.column_stack([
            np.ones(4),
            [0, 0, 1, 1]  # Two samples per condition
        ])
        
        result = fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        
        assert result['n_genes'] == 10
        assert result['n_samples'] == 4
        
    def test_high_overdispersion_genes(self):
        """Test handling of genes with very high overdispersion."""
        np.random.seed(42)
        # Create data with very high variance (overdispersed)
        base_counts = np.random.negative_binomial(2, 0.8, size=(10, 20))  # Very overdispersed
        # Ensure no samples have all zeros by adding 1 to any zero columns
        col_sums = np.sum(base_counts, axis=0)
        zero_cols = col_sums == 0
        if np.any(zero_cols):
            base_counts[0, zero_cols] += 1  # Add 1 to first gene in zero columns
        
        design = np.column_stack([
            np.ones(20),
            np.random.binomial(1, 0.5, 20)
        ])
        
        result = fit_devil(
            base_counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        
        # Should complete without errors
        assert result['n_genes'] == 10
        # Some genes might have high overdispersion
        assert np.any(result['overdispersion'] > 0.5)
        
    def test_non_integer_counts_warning(self):
        """Test warning for non-integer count data."""
        counts = np.random.negative_binomial(5, 0.3, size=(10, 15)).astype(float)
        counts += 0.1  # Make non-integer
        design = np.column_stack([
            np.ones(15),
            np.random.binomial(1, 0.5, 15)
        ])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fit_devil(
                counts,
                design_matrix=design,
                verbose=False,
                max_iter=5,
                use_gpu=False
            )
            
            # Should warn about non-integer counts
            assert len(w) > 0
            assert any("non-integer" in str(warning.message) for warning in w)
        
        # But should still complete
        assert result['n_genes'] == 10


class TestConvergenceAndOptimization:
    """Test convergence behavior and optimization parameters."""
    
    def test_convergence_tracking(self):
        """Test that convergence is properly tracked."""
        counts = np.random.negative_binomial(5, 0.3, size=(20, 15))
        design = np.column_stack([
            np.ones(15),
            np.random.binomial(1, 0.5, 15)
        ])
        
        result = fit_devil(
            counts,
            design_matrix=design,
            max_iter=20,
            tolerance=1e-4,
            verbose=False,
            use_gpu=False
        )
        
        # Check convergence tracking
        assert 'converged' in result
        assert 'iterations' in result
        assert len(result['converged']) == 20
        assert len(result['iterations']) == 20
        
        # Most genes should converge with reasonable parameters
        convergence_rate = np.mean(result['converged'])
        assert convergence_rate > 0.5  # At least 50% should converge
        
    def test_strict_tolerance(self):
        """Test behavior with very strict tolerance."""
        counts = np.random.negative_binomial(5, 0.3, size=(10, 15))
        design = np.column_stack([
            np.ones(15),
            np.random.binomial(1, 0.5, 15)
        ])
        
        result = fit_devil(
            counts,
            design_matrix=design,
            max_iter=100,
            tolerance=1e-8,  # Very strict
            verbose=False,
            use_gpu=False
        )
        
        # Should still complete
        assert result['n_genes'] == 10
        # With strict tolerance, genes should take multiple iterations
        assert np.any(result['iterations'] >= 4)
        
    def test_loose_tolerance(self):
        """Test behavior with very loose tolerance."""
        counts = np.random.negative_binomial(5, 0.3, size=(10, 15))
        design = np.column_stack([
            np.ones(15),
            np.random.binomial(1, 0.5, 15)
        ])
        
        result = fit_devil(
            counts,
            design_matrix=design,
            max_iter=100,
            tolerance=1e-1,  # Very loose
            verbose=False,
            use_gpu=False
        )
        
        # Should converge quickly
        assert result['n_genes'] == 10
        # Most genes should converge in few iterations
        assert np.mean(result['iterations']) < 10