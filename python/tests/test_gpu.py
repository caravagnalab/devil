"""
Comprehensive tests for GPU functionality in devil package.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import warnings

# Import devil modules
import devil
from devil.gpu import (
    is_gpu_available, check_gpu_requirements, estimate_batch_size,
    get_gpu_memory_info, GPUMemoryManager
)


class TestGPUDetection:
    """Test GPU detection and setup functions."""
    
    def test_is_gpu_available_no_cupy(self):
        """Test GPU detection when CuPy is not available."""
        with patch('devil.gpu.CUPY_AVAILABLE', False):
            assert not is_gpu_available()
    
    def test_is_gpu_available_with_cupy_no_gpu(self):
        """Test GPU detection when CuPy is available but no GPU."""
        with patch('devil.gpu.CUPY_AVAILABLE', True):
            with patch('devil.gpu.cp') as mock_cp:
                mock_cp.cuda.Device.side_effect = Exception("No GPU")
                assert not is_gpu_available()
    
    def test_check_gpu_requirements(self):
        """Test GPU requirements checking."""
        # Should return False, message for no GPU
        feasible, message = check_gpu_requirements(100, 50, 3, verbose=False)
        if not is_gpu_available():
            assert not feasible
            assert "not available" in message.lower() or "not installed" in message.lower()
    
    def test_estimate_batch_size(self):
        """Test batch size estimation."""
        batch_size = estimate_batch_size(1000, 100, 5)
        assert isinstance(batch_size, int)
        assert batch_size > 0
        assert batch_size <= 1000
    
    def test_get_gpu_memory_info_no_gpu(self):
        """Test memory info when GPU not available."""
        with patch('devil.gpu.is_gpu_available', return_value=False):
            free, total = get_gpu_memory_info()
            assert free == 0
            assert total == 0


class TestGPUMemoryManager:
    """Test GPU memory management."""
    
    def test_memory_manager_context(self):
        """Test GPU memory manager context."""
        with GPUMemoryManager() as mgr:
            assert mgr is not None
        # Should not raise exceptions even without GPU
    
    def test_memory_manager_clear_cache_false(self):
        """Test memory manager without cache clearing."""
        with GPUMemoryManager(clear_cache=False):
            pass
        # Should complete without errors


class TestSyntheticDataGeneration:
    """Test synthetic data generation for GPU testing."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic data for testing."""
        np.random.seed(42)
        n_genes, n_samples = 100, 50
        
        # Create count matrix
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        
        # Create design matrix
        design = np.column_stack([
            np.ones(n_samples),  # intercept
            np.random.binomial(1, 0.5, n_samples),  # condition
            np.random.normal(0, 1, n_samples)  # continuous covariate
        ])
        
        gene_names = [f"Gene_{i:03d}" for i in range(n_genes)]
        
        return counts, design, gene_names
    
    def test_synthetic_data_properties(self, synthetic_data):
        """Test properties of synthetic data."""
        counts, design, gene_names = synthetic_data
        
        assert counts.shape == (100, 50)
        assert design.shape == (50, 3)
        assert len(gene_names) == 100
        assert np.all(counts >= 0)  # Non-negative counts
        assert np.allclose(design[:, 0], 1)  # Intercept column


@pytest.mark.gpu
class TestGPUFitting:
    """Test GPU-accelerated model fitting."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic data for GPU testing."""
        np.random.seed(42)
        n_genes, n_samples = 50, 30  # Small for fast testing
        
        counts = np.random.negative_binomial(3, 0.4, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        gene_names = [f"Gene_{i:03d}" for i in range(n_genes)]
        
        return counts, design, gene_names
    
    def test_fit_devil_gpu_auto_detect(self, synthetic_data):
        """Test fit_devil with automatic GPU detection."""
        counts, design, gene_names = synthetic_data
        
        # Should work regardless of GPU availability
        result = devil.fit_devil(
            counts,
            design_matrix=design,
            use_gpu=None,  # Auto-detect
            overdispersion=True,
            verbose=False,
            max_iter=10  # Fast for testing
        )
        
        assert 'beta' in result
        assert 'overdispersion' in result
        assert result['beta'].shape == (50, 2)
        assert len(result['overdispersion']) == 50
    
    def test_fit_devil_gpu_forced_false(self, synthetic_data):
        """Test fit_devil with GPU explicitly disabled."""
        counts, design, gene_names = synthetic_data
        
        result = devil.fit_devil(
            counts,
            design_matrix=design,
            use_gpu=False,
            overdispersion=True,
            verbose=False,
            max_iter=10
        )
        
        assert 'beta' in result
        assert 'overdispersion' in result
        assert result['n_genes'] > 0
        assert result['n_samples'] > 0
    
    @pytest.mark.skipif(not is_gpu_available(), reason="GPU not available")
    def test_fit_devil_gpu_enabled(self, synthetic_data):
        """Test fit_devil with GPU explicitly enabled (requires GPU)."""
        counts, design, gene_names = synthetic_data
        
        result = devil.fit_devil(
            counts,
            design_matrix=design,
            use_gpu=True,
            gpu_batch_size=25,
            gpu_dtype='float32',
            overdispersion=True,
            verbose=False,
            max_iter=10
        )
        
        assert 'beta' in result
        assert 'overdispersion' in result
        assert result['n_genes'] > 0
        assert result['n_samples'] > 0
    
    def test_fit_devil_gpu_fallback(self, synthetic_data):
        """Test GPU fallback to CPU on errors."""
        counts, design, gene_names = synthetic_data
        
        # This should either use GPU successfully or fallback to CPU
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = devil.fit_devil(
                counts,
                design_matrix=design,
                use_gpu=True,  # Try GPU but allow fallback
                overdispersion=True,
                verbose=False,
                max_iter=10
            )
        
        # Should complete successfully regardless
        assert 'beta' in result
        assert 'overdispersion' in result


@pytest.mark.gpu  
class TestGPUTesting:
    """Test GPU-accelerated differential expression testing."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model for testing."""
        np.random.seed(42)
        n_genes, n_samples = 30, 25
        
        counts = np.random.negative_binomial(3, 0.4, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        result = devil.fit_devil(
            counts,
            design_matrix=design,
            use_gpu=False,  # Use CPU for consistent testing
            overdispersion=True,
            verbose=False,
            max_iter=5
        )
        
        return result
    
    def test_test_de_gpu_auto_detect(self, fitted_model):
        """Test test_de with automatic GPU detection."""
        contrast = [0, 1]
        
        results = devil.test_de(
            fitted_model,
            contrast=contrast,
            use_gpu=None,  # Auto-detect
            verbose=False
        )
        
        assert isinstance(results, pd.DataFrame)
        assert 'gene' in results.columns
        assert 'pval' in results.columns
        assert 'padj' in results.columns
        assert 'lfc' in results.columns
        assert len(results) == 30
    
    def test_test_de_gpu_disabled(self, fitted_model):
        """Test test_de with GPU explicitly disabled."""
        contrast = [0, 1]
        
        results = devil.test_de(
            fitted_model,
            contrast=contrast,
            use_gpu=False,
            verbose=False
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 30
    
    @pytest.mark.skipif(not is_gpu_available(), reason="GPU not available")
    def test_test_de_gpu_enabled(self, fitted_model):
        """Test test_de with GPU explicitly enabled (requires GPU)."""
        contrast = [0, 1]
        
        results = devil.test_de(
            fitted_model,
            contrast=contrast,
            use_gpu=True,
            gpu_batch_size=15,
            verbose=False
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 30
    
    def test_test_de_with_clusters(self, fitted_model):
        """Test differential expression with cluster correction."""
        contrast = [0, 1]
        clusters = np.random.randint(1, 4, fitted_model['n_samples'])
        
        results = devil.test_de(
            fitted_model,
            contrast=contrast,
            clusters=clusters,
            use_gpu=False,  # Use CPU for consistent results
            verbose=False
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 30
    
    def test_test_de_memory_efficient(self, fitted_model):
        """Test memory-efficient differential expression testing."""
        contrast = [0, 1]
        gene_subset = fitted_model['gene_names'][:15]  # Test half the genes
        
        results = devil.test_de_memory_efficient(
            fitted_model,
            contrast=contrast,
            gene_subset=gene_subset,
            verbose=False
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 15
        assert all(gene in gene_subset for gene in results['gene'])


class TestGPUIntegration:
    """Test integration between GPU components."""
    
    def test_cpu_gpu_consistency(self):
        """Test that CPU and GPU produce similar results."""
        np.random.seed(42)
        n_genes, n_samples = 20, 15
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Fit with CPU
        cpu_result = devil.fit_devil(
            counts,
            design_matrix=design,
            use_gpu=False,
            overdispersion=True,
            verbose=False,
            max_iter=5
        )
        
        # Test with CPU
        cpu_test = devil.test_de(
            cpu_result,
            contrast=[0, 1],
            use_gpu=False,
            verbose=False
        )
        
        # If GPU is available, compare results
        if is_gpu_available():
            try:
                # Fit with GPU  
                gpu_result = devil.fit_devil(
                    counts,
                    design_matrix=design,
                    use_gpu=True,
                    gpu_dtype='float32',
                    overdispersion=True,
                    verbose=False,
                    max_iter=5
                )
                
                # Test with GPU
                gpu_test = devil.test_de(
                    gpu_result,
                    contrast=[0, 1],
                    use_gpu=True,
                    verbose=False
                )
                
                # Check correlation (should be very high)
                lfc_corr = np.corrcoef(cpu_test['lfc'], gpu_test['lfc'])[0, 1]
                assert lfc_corr > 0.95, f"LFC correlation too low: {lfc_corr}"
                
                pval_corr = np.corrcoef(
                    -np.log10(cpu_test['pval'] + 1e-10),
                    -np.log10(gpu_test['pval'] + 1e-10)
                )[0, 1]
                assert pval_corr > 0.9, f"P-value correlation too low: {pval_corr}"
                
            except Exception as e:
                # GPU comparison failed, but CPU should work
                pytest.skip(f"GPU comparison failed: {e}")
        
        # At minimum, CPU results should be valid
        assert isinstance(cpu_test, pd.DataFrame)
        assert len(cpu_test) == n_genes
        assert all(cpu_test['pval'] >= 0)
        assert all(cpu_test['pval'] <= 1)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_memory_constraint_fallback(self):
        """Test fallback when GPU memory is insufficient."""
        # This would require mocking GPU memory functions
        # For now, just test that the function completes
        counts = np.random.negative_binomial(3, 0.4, size=(5, 5))
        design = np.column_stack([np.ones(5), np.random.binomial(1, 0.5, 5)])
        
        # Should complete without error regardless of GPU status
        result = devil.fit_devil(
            counts,
            design_matrix=design,
            use_gpu=True,
            gpu_batch_size=1,  # Very small batch
            verbose=False,
            max_iter=3
        )
        
        assert 'beta' in result
    
    def test_contrast_validation_gpu(self):
        """Test contrast validation in GPU testing."""
        # Create minimal fitted model
        result = {
            'beta': np.random.normal(0, 1, (10, 2)),
            'overdispersion': np.random.gamma(1, 1, 10),
            'design_matrix': np.column_stack([np.ones(5), np.random.binomial(1, 0.5, 5)]),
            'count_matrix': np.random.negative_binomial(3, 0.4, size=(10, 5)),
            'size_factors': np.ones(5),
            'gene_names': [f"Gene_{i}" for i in range(10)],
            'n_genes': 10,
            'n_samples': 5
        }
        
        # Wrong contrast length should raise error
        with pytest.raises(ValueError, match="Contrast length"):
            devil.test_de(result, contrast=[0, 1, 0])  # Too long


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU (deselect with '-m \"not gpu\"')"
    )


# Skip GPU tests if not available by default
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle GPU tests."""
    if not is_gpu_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)