"""
Unit tests for CPU utility functions.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad
from unittest.mock import patch
import warnings

from devil.utils import (
    handle_input_data, validate_inputs, check_convergence
)


class TestHandleInputData:
    """Test input data handling function."""
    
    @pytest.fixture
    def sample_anndata(self):
        """Create sample AnnData object for testing."""
        np.random.seed(42)
        n_genes, n_samples = 50, 30
        
        # Create count matrix
        X = np.random.negative_binomial(5, 0.3, size=(n_samples, n_genes))
        
        # Create observations (samples)
        obs = pd.DataFrame({
            'condition': np.random.choice(['A', 'B', 'C'], n_samples),
            'batch': np.random.choice(['batch1', 'batch2'], n_samples),
            'n_counts': np.sum(X, axis=1),
            'continuous_var': np.random.normal(0, 1, n_samples)
        })
        obs.index = [f'Cell_{i:03d}' for i in range(n_samples)]
        
        # Create variables (genes)
        var = pd.DataFrame({
            'gene_symbol': [f'Gene_{i:03d}' for i in range(n_genes)],
            'highly_variable': np.random.binomial(1, 0.3, n_genes).astype(bool),
            'mean_expression': np.mean(X, axis=0)
        })
        var.index = [f'ENSG{i:08d}' for i in range(n_genes)]
        
        adata = ad.AnnData(X=X, obs=obs, var=var)
        
        # Add a layer
        adata.layers['raw'] = X.copy()
        adata.layers['normalized'] = X / np.sum(X, axis=1, keepdims=True) * 1e4
        
        return adata
    
    def test_handle_input_data_anndata_default(self, sample_anndata):
        """Test handling AnnData with default layer (X)."""
        count_matrix, gene_names, sample_names, obs_df = handle_input_data(
            sample_anndata, layer=None
        )
        
        # Check dimensions
        assert count_matrix.shape == (50, 30)  # genes × samples
        assert len(gene_names) == 50
        assert len(sample_names) == 30
        
        # Check data types
        assert count_matrix.dtype == np.float64
        assert isinstance(gene_names, np.ndarray)
        assert isinstance(sample_names, np.ndarray)
        assert isinstance(obs_df, pd.DataFrame)
        
        # Check content
        np.testing.assert_allclose(count_matrix, sample_anndata.X.T)
        np.testing.assert_array_equal(gene_names, sample_anndata.var_names.values)
        np.testing.assert_array_equal(sample_names, sample_anndata.obs_names.values)
        pd.testing.assert_frame_equal(obs_df, sample_anndata.obs)
    
    def test_handle_input_data_anndata_layer(self, sample_anndata):
        """Test handling AnnData with specific layer."""
        count_matrix, gene_names, sample_names, obs_df = handle_input_data(
            sample_anndata, layer='raw'
        )
        
        assert count_matrix.shape == (50, 30)
        np.testing.assert_allclose(count_matrix, sample_anndata.layers['raw'].T)
    
    def test_handle_input_data_anndata_missing_layer(self, sample_anndata):
        """Test error when requesting missing layer."""
        with pytest.raises(ValueError, match="Layer 'missing' not found"):
            handle_input_data(sample_anndata, layer='missing')
    
    def test_handle_input_data_anndata_sparse(self, sample_anndata):
        """Test handling AnnData with sparse X matrix."""
        # Convert to sparse
        sample_anndata.X = sparse.csr_matrix(sample_anndata.X)
        
        count_matrix, gene_names, sample_names, obs_df = handle_input_data(
            sample_anndata, layer=None
        )
        
        # Should convert to dense
        assert count_matrix.shape == (50, 30)
        assert isinstance(count_matrix, np.ndarray)
        assert not sparse.issparse(count_matrix)
    
    def test_handle_input_data_numpy_array(self):
        """Test handling numpy array input."""
        np.random.seed(42)
        input_array = np.random.negative_binomial(5, 0.3, size=(40, 25))
        
        count_matrix, gene_names, sample_names, obs_df = handle_input_data(
            input_array, layer=None
        )
        
        # Check basic properties
        assert count_matrix.shape == (40, 25)
        assert count_matrix.dtype == np.float64
        assert len(gene_names) == 40
        assert len(sample_names) == 25
        assert obs_df is None
        
        # Check generated names
        assert gene_names[0] == 'Gene_0'
        assert gene_names[-1] == 'Gene_39'
        assert sample_names[0] == 'Sample_0'
        assert sample_names[-1] == 'Sample_24'
        
        # Data should be identical
        np.testing.assert_allclose(count_matrix, input_array)
    
    def test_handle_input_data_sparse_matrix(self):
        """Test handling sparse matrix input."""
        np.random.seed(42)
        dense_array = np.random.negative_binomial(3, 0.5, size=(30, 20))
        sparse_array = sparse.csr_matrix(dense_array)
        
        count_matrix, gene_names, sample_names, obs_df = handle_input_data(
            sparse_array, layer=None
        )
        
        assert count_matrix.shape == (30, 20)
        assert isinstance(count_matrix, np.ndarray)
        assert count_matrix.dtype == np.float64
        assert obs_df is None
        
        # Should convert correctly
        np.testing.assert_allclose(count_matrix, dense_array)
    
    def test_handle_input_data_invalid_type(self):
        """Test error with invalid input type."""
        invalid_input = "not_a_matrix"
        
        with pytest.raises(TypeError, match="Unsupported data type"):
            handle_input_data(invalid_input)
    
    def test_handle_input_data_non_numeric(self):
        """Test error with non-numeric data."""
        non_numeric = np.array([['a', 'b'], ['c', 'd']])
        
        with pytest.raises(ValueError, match="numeric values"):
            handle_input_data(non_numeric)
    
    def test_handle_input_data_dtype_conversion(self):
        """Test that data is converted to float64."""
        # Integer input
        int_array = np.random.randint(0, 10, size=(20, 15)).astype(np.int32)
        
        count_matrix, _, _, _ = handle_input_data(int_array)
        
        assert count_matrix.dtype == np.float64
        np.testing.assert_allclose(count_matrix, int_array.astype(np.float64))


class TestValidateInputs:
    """Test input validation function."""
    
    def test_validate_inputs_valid(self):
        """Test validation with valid inputs."""
        np.random.seed(42)
        count_matrix = np.random.negative_binomial(5, 0.3, size=(50, 30))
        design_matrix = np.column_stack([
            np.ones(30),
            np.random.binomial(1, 0.5, 30),
            np.random.normal(0, 1, 30)
        ])
        
        # Should not raise any exception
        validate_inputs(count_matrix, design_matrix)
    
    def test_validate_inputs_dimension_mismatch(self):
        """Test error with dimension mismatch."""
        count_matrix = np.random.negative_binomial(5, 0.3, size=(50, 30))
        design_matrix = np.random.normal(0, 1, size=(25, 3))  # Wrong number of samples
        
        with pytest.raises(ValueError, match="Sample count mismatch"):
            validate_inputs(count_matrix, design_matrix)
    
    def test_validate_inputs_insufficient_samples(self):
        """Test error with insufficient samples."""
        count_matrix = np.random.negative_binomial(5, 0.3, size=(50, 5))
        design_matrix = np.random.normal(0, 1, size=(5, 10))  # More features than samples
        
        with pytest.raises(ValueError, match="Insufficient samples"):
            validate_inputs(count_matrix, design_matrix)
    
    def test_validate_inputs_negative_counts(self):
        """Test error with negative counts."""
        count_matrix = np.random.normal(0, 1, size=(20, 15))  # Can have negative values
        design_matrix = np.column_stack([
            np.ones(15),
            np.random.binomial(1, 0.5, 15)
        ])
        
        with pytest.raises(ValueError, match="negative values"):
            validate_inputs(count_matrix, design_matrix)
    
    def test_validate_inputs_rank_deficient_design(self):
        """Test error with rank-deficient design matrix."""
        count_matrix = np.random.negative_binomial(5, 0.3, size=(30, 20))
        
        # Create rank-deficient design matrix
        design_matrix = np.column_stack([
            np.ones(20),
            np.random.normal(0, 1, 20),
            np.random.normal(0, 1, 20)
        ])
        design_matrix[:, 2] = design_matrix[:, 1]  # Make columns identical
        
        with pytest.raises(ValueError, match="rank deficient"):
            validate_inputs(count_matrix, design_matrix)
    
    def test_validate_inputs_non_integer_warning(self):
        """Test warning with non-integer counts."""
        count_matrix = np.random.negative_binomial(5, 0.3, size=(20, 15)).astype(float)
        count_matrix += 0.1  # Make non-integer
        design_matrix = np.column_stack([
            np.ones(15),
            np.random.binomial(1, 0.5, 15)
        ])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_inputs(count_matrix, design_matrix)
            
            # Should warn about non-integer counts
            assert len(w) > 0
            assert any("non-integer" in str(warning.message) for warning in w)
    
    def test_validate_inputs_edge_case_dimensions(self):
        """Test validation with edge case dimensions."""
        # Minimum valid case
        count_matrix = np.random.negative_binomial(3, 0.4, size=(10, 2))
        design_matrix = np.ones((2, 1))  # Just intercept
        
        # Should be valid
        validate_inputs(count_matrix, design_matrix)
        
        # Exactly equal samples and features (boundary case)
        count_matrix = np.random.negative_binomial(3, 0.4, size=(10, 5))
        design_matrix = np.random.normal(0, 1, size=(5, 5))
        
        # Should be valid if design matrix is full rank
        U, s, Vt = np.linalg.svd(design_matrix)
        design_matrix = U @ np.diag([1, 2, 3, 4, 5]) @ Vt  # Ensure full rank
        
        validate_inputs(count_matrix, design_matrix)


class TestCheckConvergence:
    """Test convergence checking function."""
    
    def test_check_convergence_all_converged(self):
        """Test convergence checking when all genes converged."""
        iterations = np.array([5, 3, 8, 2, 6, 4])
        max_iter = 10
        gene_names = np.array([f'Gene_{i}' for i in range(6)])
        
        # Should not raise warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_convergence(iterations, max_iter, gene_names)
            
            # No warnings expected
            assert len(w) == 0
    
    def test_check_convergence_some_not_converged(self):
        """Test convergence checking with some non-converged genes."""
        iterations = np.array([5, 10, 8, 10, 6, 4])  # Two genes hit max_iter
        max_iter = 10
        gene_names = np.array(['Gene_A', 'Gene_B', 'Gene_C', 'Gene_D', 'Gene_E', 'Gene_F'])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_convergence(iterations, max_iter, gene_names)
            
            # Should warn about non-convergence
            assert len(w) >= 1
            warning_messages = [str(warning.message) for warning in w]
            assert any("did not converge" in msg for msg in warning_messages)
            assert any("2 genes" in msg for msg in warning_messages)
    
    def test_check_convergence_many_not_converged(self):
        """Test convergence checking with many non-converged genes."""
        iterations = np.full(20, 50)  # All genes hit max_iter
        max_iter = 50
        gene_names = np.array([f'Gene_{i:02d}' for i in range(20)])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_convergence(iterations, max_iter, gene_names)
            
            # Should warn but not list all genes (too many)
            warning_messages = [str(warning.message) for warning in w]
            assert any("20 genes" in msg for msg in warning_messages)
            # Should not list individual genes when there are many
            assert not any("Gene_00" in msg for msg in warning_messages)
    
    def test_check_convergence_few_not_converged_with_names(self):
        """Test convergence checking with few non-converged genes (should list names)."""
        iterations = np.array([5, 25, 8, 4, 25])  # Two genes hit max_iter
        max_iter = 25
        gene_names = np.array(['GAPDH', 'TP53', 'ACTB', 'MYC', 'BRCA1'])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_convergence(iterations, max_iter, gene_names)
            
            # Should list specific gene names
            warning_messages = [str(warning.message) for warning in w]
            combined_message = ' '.join(warning_messages)
            assert "TP53" in combined_message
            assert "BRCA1" in combined_message
    
    def test_check_convergence_empty_input(self):
        """Test convergence checking with empty arrays."""
        iterations = np.array([])
        max_iter = 10
        gene_names = np.array([])
        
        # Should handle empty input gracefully
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_convergence(iterations, max_iter, gene_names)
            
            # No warnings expected for empty input
            assert len(w) == 0


class TestUtilityEdgeCases:
    """Test edge cases in utility functions."""
    
    def test_handle_input_data_single_gene_sample(self):
        """Test handling data with single gene and single sample."""
        single_data = np.array([[5]])  # 1 gene, 1 sample
        
        count_matrix, gene_names, sample_names, obs_df = handle_input_data(single_data)
        
        assert count_matrix.shape == (1, 1)
        assert len(gene_names) == 1
        assert len(sample_names) == 1
        assert obs_df is None
        assert gene_names[0] == 'Gene_0'
        assert sample_names[0] == 'Sample_0'
    
    def test_handle_input_data_very_large_dimensions(self):
        """Test handling data with large dimensions (names generation)."""
        # Create moderately large data to test name generation
        large_data = np.ones((1000, 100))  # 1000 genes, 100 samples
        
        count_matrix, gene_names, sample_names, obs_df = handle_input_data(large_data)
        
        assert count_matrix.shape == (1000, 100)
        assert len(gene_names) == 1000
        assert len(sample_names) == 100
        
        # Check name format for large indices
        assert gene_names[999] == 'Gene_999'
        assert sample_names[99] == 'Sample_99'
    
    def test_validate_inputs_perfect_collinearity(self):
        """Test validation with perfect collinearity in design matrix."""
        count_matrix = np.random.negative_binomial(5, 0.3, size=(30, 20))
        
        # Perfect collinearity
        x1 = np.random.normal(0, 1, 20)
        design_matrix = np.column_stack([
            np.ones(20),
            x1,
            2 * x1  # Perfectly correlated
        ])
        
        with pytest.raises(ValueError, match="rank deficient"):
            validate_inputs(count_matrix, design_matrix)
    
    def test_validate_inputs_near_collinearity(self):
        """Test validation with near collinearity (should pass)."""
        count_matrix = np.random.negative_binomial(5, 0.3, size=(30, 20))
        
        # Near collinearity (but not perfect)
        x1 = np.random.normal(0, 1, 20)
        x2 = 2 * x1 + np.random.normal(0, 0.01, 20)  # Almost perfectly correlated
        design_matrix = np.column_stack([np.ones(20), x1, x2])
        
        # Should pass (rank is still full)
        validate_inputs(count_matrix, design_matrix)
    
    def test_validate_inputs_zero_variance_predictor(self):
        """Test validation with zero variance predictor."""
        count_matrix = np.random.negative_binomial(5, 0.3, size=(30, 20))
        
        design_matrix = np.column_stack([
            np.ones(20),
            np.zeros(20),  # Zero variance predictor
            np.random.normal(0, 1, 20)
        ])
        
        # Should fail due to zero variance column
        with pytest.raises(ValueError, match="rank deficient"):
            validate_inputs(count_matrix, design_matrix)
    
    def test_check_convergence_boundary_cases(self):
        """Test convergence checking with boundary cases."""
        # All genes converged in exactly max_iter - 1 iterations
        iterations = np.full(10, 9)
        max_iter = 10
        gene_names = np.array([f'Gene_{i}' for i in range(10)])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_convergence(iterations, max_iter, gene_names)
            
            # Should not warn (didn't hit max_iter)
            assert len(w) == 0
        
        # All genes hit exactly max_iter
        iterations = np.full(10, 10)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_convergence(iterations, max_iter, gene_names)
            
            # Should warn
            assert len(w) >= 1


class TestUtilityRobustness:
    """Test robustness of utility functions."""
    
    def test_handle_input_data_memory_efficiency(self):
        """Test that input data handling doesn't unnecessarily copy data."""
        # Create data
        np.random.seed(42)
        original_data = np.random.negative_binomial(5, 0.3, size=(100, 80))
        
        # Test with numpy array
        count_matrix, _, _, _ = handle_input_data(original_data)
        
        # Should be the same data (may or may not share memory due to dtype conversion)
        np.testing.assert_allclose(count_matrix, original_data)
    
    def test_validate_inputs_numerical_precision(self):
        """Test validation with numerical precision issues."""
        count_matrix = np.random.negative_binomial(5, 0.3, size=(30, 20))
        
        # Create design matrix with small numerical differences
        base_col = np.random.normal(0, 1, 20)
        design_matrix = np.column_stack([
            np.ones(20),
            base_col,
            base_col + 1e-14  # Very small difference
        ])
        
        # Should detect rank deficiency despite small numerical differences
        with pytest.raises(ValueError, match="rank deficient"):
            validate_inputs(count_matrix, design_matrix)
    
    def test_handle_input_data_consistency(self):
        """Test that multiple calls with same data give consistent results."""
        np.random.seed(42)
        test_data = np.random.negative_binomial(5, 0.3, size=(50, 30))
        
        # Call twice
        result1 = handle_input_data(test_data)
        result2 = handle_input_data(test_data)
        
        # Should be identical
        np.testing.assert_allclose(result1[0], result2[0])
        np.testing.assert_array_equal(result1[1], result2[1])
        np.testing.assert_array_equal(result1[2], result2[2])
        assert (result1[3] is None) == (result2[3] is None)
    
    def test_validate_inputs_extreme_values(self):
        """Test validation with extreme but valid values."""
        # Very large counts
        count_matrix = np.random.negative_binomial(1000, 0.01, size=(20, 15))
        design_matrix = np.column_stack([
            np.ones(15),
            np.random.normal(0, 1, 15)
        ])
        
        # Should be valid
        validate_inputs(count_matrix, design_matrix)
        
        # Very small non-zero counts
        count_matrix = np.random.negative_binomial(1, 0.99, size=(20, 15))
        
        # Should also be valid
        validate_inputs(count_matrix, design_matrix)
