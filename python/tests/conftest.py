"""
Pytest configuration and shared fixtures for devil package tests.
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import warnings
import matplotlib
import os
import sys

# Set matplotlib to non-interactive backend for testing
matplotlib.use('Agg')

# Add parent directory to path to import devil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import devil


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle special markers."""
    # Skip GPU tests if GPU not available
    if not devil.is_gpu_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    
    # Mark slow tests
    for item in items:
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def numpy_random_state():
    """Set numpy random state for reproducible tests."""
    np.random.seed(42)
    return np.random.get_state()


@pytest.fixture
def suppress_warnings():
    """Suppress warnings during testing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# Data generation fixtures
@pytest.fixture
def small_count_matrix():
    """Create small count matrix for fast testing."""
    np.random.seed(42)
    return np.random.negative_binomial(5, 0.3, size=(20, 15)).astype(np.float64)


@pytest.fixture
def medium_count_matrix():
    """Create medium-sized count matrix for standard testing."""
    np.random.seed(42)
    return np.random.negative_binomial(5, 0.3, size=(100, 50)).astype(np.float64)


@pytest.fixture
def large_count_matrix():
    """Create large count matrix for performance testing."""
    np.random.seed(42)
    return np.random.negative_binomial(5, 0.3, size=(2000, 200)).astype(np.float64)


@pytest.fixture
def simple_design_matrix():
    """Create simple design matrix (intercept + condition)."""
    np.random.seed(42)
    n_samples = 15
    return np.column_stack([
        np.ones(n_samples),
        np.random.binomial(1, 0.5, n_samples)
    ])


@pytest.fixture
def complex_design_matrix():
    """Create complex design matrix with multiple factors and interactions."""
    np.random.seed(42)
    n_samples = 50
    
    condition = np.random.binomial(1, 0.5, n_samples)
    batch = np.random.binomial(1, 0.3, n_samples)
    continuous = np.random.normal(0, 1, n_samples)
    
    return np.column_stack([
        np.ones(n_samples),  # intercept
        condition,           # main condition
        batch,              # batch effect
        continuous,         # continuous covariate
        condition * batch   # interaction
    ])


@pytest.fixture
def realistic_anndata():
    """Create realistic AnnData object for testing."""
    np.random.seed(42)
    n_genes, n_samples = 150, 80
    
    # Generate realistic count data
    base_expression = np.random.gamma(2, 2, n_genes)  # Gene-specific expression levels
    size_factors = np.random.lognormal(0, 0.5, n_samples)  # Sample-specific depths
    
    # Create conditions and batches
    conditions = np.random.choice(['Control', 'Treatment_A', 'Treatment_B'], n_samples)
    batches = np.random.choice(['Batch_1', 'Batch_2', 'Batch_3'], n_samples)
    patients = np.random.choice([f'Patient_{i}' for i in range(1, 21)], n_samples)
    
    # Generate counts with realistic structure
    condition_effects = np.random.normal(0, 1, n_genes)
    batch_effects = np.random.normal(0, 0.5, n_genes)
    
    counts = np.zeros((n_samples, n_genes))
    for i, (cond, batch) in enumerate(zip(conditions, batches)):
        mu = base_expression * size_factors[i]
        
        # Add condition effects
        if cond == 'Treatment_A':
            mu *= np.exp(condition_effects * 0.5)
        elif cond == 'Treatment_B':
            mu *= np.exp(condition_effects * -0.3)
        
        # Add batch effects
        if batch == 'Batch_2':
            mu *= np.exp(batch_effects * 0.2)
        elif batch == 'Batch_3':
            mu *= np.exp(batch_effects * -0.1)
        
        # Generate negative binomial counts
        overdispersion = np.random.gamma(3, 0.3, n_genes)
        for j in range(n_genes):
            p = 1 / (1 + mu[j] * overdispersion[j])
            n = 1 / overdispersion[j]
            counts[i, j] = np.random.negative_binomial(n, p)
    
    # Create observation metadata
    obs = pd.DataFrame({
        'condition': conditions,
        'batch': batches,
        'patient': patients,
        'n_genes': np.sum(counts > 0, axis=1),
        'total_counts': np.sum(counts, axis=1),
        'log_total_counts': np.log1p(np.sum(counts, axis=1)),
        'continuous_covar': np.random.normal(0, 1, n_samples),
        'cell_cycle_score': np.random.uniform(0, 1, n_samples)
    })
    obs.index = [f'Cell_{i:04d}' for i in range(n_samples)]
    
    # Create variable metadata
    var = pd.DataFrame({
        'gene_symbol': [f'Gene_{i:03d}' for i in range(n_genes)],
        'gene_type': np.random.choice(['protein_coding', 'lncRNA', 'pseudogene'], 
                                    n_genes, p=[0.85, 0.10, 0.05]),
        'chromosome': np.random.choice([f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'], 
                                     n_genes),
        'mean_expression': np.mean(counts, axis=0),
        'pct_dropout': np.mean(counts == 0, axis=0) * 100,
        'highly_variable': np.random.binomial(1, 0.4, n_genes).astype(bool)
    })
    var.index = [f'ENSG{i:08d}' for i in range(n_genes)]
    
    # Create AnnData object
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    
    # Add layers
    adata.layers['counts'] = adata.X.copy()
    adata.layers['logcounts'] = np.log1p(adata.X)
    adata.layers['normalized'] = adata.X / np.sum(adata.X, axis=1, keepdims=True) * 1e4
    
    return adata


@pytest.fixture
def fitted_model_small(small_count_matrix, simple_design_matrix):
    """Create small fitted model for testing."""
    return devil.fit_devil(
        small_count_matrix,
        design_matrix=simple_design_matrix,
        overdispersion=True,
        verbose=False,
        max_iter=10,
        use_gpu=False
    )


@pytest.fixture
def fitted_model_medium(medium_count_matrix):
    """Create medium fitted model for testing."""
    np.random.seed(42)
    n_samples = medium_count_matrix.shape[1]
    design = np.column_stack([
        np.ones(n_samples),
        np.random.binomial(1, 0.5, n_samples),
        np.random.normal(0, 1, n_samples)
    ])
    
    return devil.fit_devil(
        medium_count_matrix,
        design_matrix=design,
        overdispersion=True,
        verbose=False,
        max_iter=15,
        use_gpu=False
    )


@pytest.fixture
def de_results_small(fitted_model_small):
    """Create DE results from small fitted model."""
    contrast = [0, 1]
    return devil.test_de(
        fitted_model_small,
        contrast=contrast,
        verbose=False,
        use_gpu=False
    )


@pytest.fixture
def de_results_medium(fitted_model_medium):
    """Create DE results from medium fitted model."""
    contrast = [0, 1, 0]
    return devil.test_de(
        fitted_model_medium,
        contrast=contrast,
        verbose=False,
        use_gpu=False
    )


# Special test data fixtures
@pytest.fixture
def zero_inflated_data():
    """Create zero-inflated count data for testing."""
    np.random.seed(42)
    n_genes, n_samples = 50, 30
    
    # Generate base counts
    base_counts = np.random.negative_binomial(3, 0.4, size=(n_genes, n_samples))
    
    # Add zero inflation
    zero_mask = np.random.binomial(1, 0.6, size=(n_genes, n_samples))  # 60% dropout
    counts = base_counts * zero_mask
    
    design = np.column_stack([
        np.ones(n_samples),
        np.random.binomial(1, 0.5, n_samples)
    ])
    
    return counts, design


@pytest.fixture
def sparse_data():
    """Create very sparse count data for testing."""
    np.random.seed(42)
    n_genes, n_samples = 40, 25
    
    # Very sparse data (many zeros)
    counts = np.random.negative_binomial(1, 0.9, size=(n_genes, n_samples))
    
    design = np.column_stack([
        np.ones(n_samples),
        np.random.binomial(1, 0.5, n_samples)
    ])
    
    return counts, design


@pytest.fixture
def outlier_data():
    """Create count data with outliers for testing."""
    np.random.seed(42)
    n_genes, n_samples = 30, 20
    
    counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
    
    # Add outliers
    counts[0, 0] = 1000  # Extreme outlier
    counts[1, :] = 0     # All-zero gene
    counts[:, 1] = np.random.negative_binomial(50, 0.1, n_genes)  # High-expression sample
    
    design = np.column_stack([
        np.ones(n_samples),
        np.random.binomial(1, 0.5, n_samples)
    ])
    
    return counts, design


@pytest.fixture
def perfect_separation_data():
    """Create data with perfect separation for testing edge cases."""
    np.random.seed(42)
    n_genes, n_samples = 25, 20
    
    # Perfect separation between conditions
    conditions = np.concatenate([np.zeros(10), np.ones(10)])
    design = np.column_stack([
        np.ones(n_samples),
        conditions
    ])
    
    # Generate data with clear separation
    counts = np.zeros((n_genes, n_samples))
    for i in range(n_genes):
        # Condition 0: low expression
        counts[i, :10] = np.random.negative_binomial(2, 0.6, 10)
        # Condition 1: high expression
        counts[i, 10:] = np.random.negative_binomial(20, 0.3, 10)
    
    return counts, design


# Clustering fixtures
@pytest.fixture
def patient_clusters():
    """Create patient cluster assignments for testing."""
    np.random.seed(42)
    n_samples = 50
    n_patients = 10
    
    # Assign samples to patients (multiple samples per patient)
    samples_per_patient = n_samples // n_patients
    patient_ids = np.repeat(np.arange(1, n_patients + 1), samples_per_patient)
    
    # Handle remainder
    remainder = n_samples - len(patient_ids)
    if remainder > 0:
        patient_ids = np.concatenate([patient_ids, np.arange(1, remainder + 1)])
    
    return patient_ids[:n_samples]


@pytest.fixture
def unbalanced_clusters():
    """Create unbalanced cluster assignments for testing."""
    # Very unbalanced clustering
    return np.concatenate([
        np.ones(20),  # Large cluster
        np.arange(2, 7),  # Small clusters of size 1
        np.array([7, 7, 8, 8, 8])  # Medium clusters
    ])


# Performance testing fixtures
@pytest.fixture
def benchmark_data():
    """Create data for performance benchmarking."""
    np.random.seed(42)
    
    sizes = {
        'small': (50, 30),
        'medium': (500, 100),
        'large': (2000, 200)
    }
    
    datasets = {}
    for size_name, (n_genes, n_samples) in sizes.items():
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples),
            np.random.normal(0, 1, n_samples)
        ])
        datasets[size_name] = (counts, design)
    
    return datasets


# Error simulation fixtures
@pytest.fixture
def problematic_data():
    """Create data that might cause numerical issues."""
    np.random.seed(42)
    n_genes, n_samples = 30, 20
    
    # Various problematic scenarios
    problems = {
        'extreme_counts': np.random.negative_binomial(100, 0.01, size=(n_genes, n_samples)),
        'all_zeros': np.zeros((n_genes, n_samples)),
        'single_nonzero': np.zeros((n_genes, n_samples)),
        'extreme_variation': np.random.negative_binomial(1, 0.99, size=(n_genes, n_samples))
    }
    
    # Modify single_nonzero
    problems['single_nonzero'][0, 0] = 1000
    
    design = np.column_stack([
        np.ones(n_samples),
        np.random.binomial(1, 0.5, n_samples)
    ])
    
    return problems, design


# Validation fixtures
@pytest.fixture
def known_result_data():
    """Create data with known expected results for validation."""
    np.random.seed(42)
    n_genes, n_samples = 40, 30
    
    # Generate data with known parameters
    true_beta = np.column_stack([
        np.random.normal(2, 0.5, n_genes),  # Intercept
        np.concatenate([
            np.random.normal(2, 0.3, 10),   # Strong positive effect
            np.random.normal(-2, 0.3, 10),  # Strong negative effect
            np.random.normal(0, 0.1, 20)    # No effect
        ])
    ])
    
    true_overdispersion = np.random.gamma(2, 0.5, n_genes)
    
    design = np.column_stack([
        np.ones(n_samples),
        np.random.binomial(1, 0.5, n_samples)
    ])
    
    # Generate counts based on true model
    size_factors = np.random.lognormal(0, 0.3, n_samples)
    mu = size_factors[np.newaxis, :] * np.exp(design @ true_beta.T)
    
    counts = np.zeros((n_genes, n_samples))
    for i in range(n_genes):
        p = 1 / (1 + mu[i, :] * true_overdispersion[i])
        n = 1 / true_overdispersion[i]
        counts[i, :] = np.random.negative_binomial(n, p)
    
    return {
        'counts': counts,
        'design': design,
        'true_beta': true_beta,
        'true_overdispersion': true_overdispersion,
        'true_size_factors': size_factors
    }


# Comparison fixtures
@pytest.fixture
def comparison_datasets():
    """Create multiple datasets for method comparison."""
    np.random.seed(42)
    n_genes, n_samples = 60, 40
    
    # Different overdispersion scenarios
    datasets = {}
    
    # Low overdispersion (Poisson-like)
    counts_low_od = np.random.poisson(
        np.random.gamma(5, 1, size=(n_genes, n_samples))
    )
    
    # Medium overdispersion
    overdispersion_med = 0.5
    mu_med = np.random.gamma(5, 1, size=(n_genes, n_samples))
    counts_med_od = np.random.negative_binomial(
        n=1/overdispersion_med,
        p=1/(1 + mu_med * overdispersion_med)
    )
    
    # High overdispersion
    overdispersion_high = 2.0
    mu_high = np.random.gamma(5, 1, size=(n_genes, n_samples))
    counts_high_od = np.random.negative_binomial(
        n=1/overdispersion_high,
        p=1/(1 + mu_high * overdispersion_high)
    )
    
    design = np.column_stack([
        np.ones(n_samples),
        np.random.binomial(1, 0.5, n_samples)
    ])
    
    datasets['low_overdispersion'] = (counts_low_od, design)
    datasets['medium_overdispersion'] = (counts_med_od, design)
    datasets['high_overdispersion'] = (counts_high_od, design)
    
    return datasets


# GPU testing fixtures
@pytest.fixture
def gpu_test_data():
    """Create data specifically for GPU testing."""
    if not devil.is_gpu_available():
        pytest.skip("GPU not available")
    
    np.random.seed(42)
    n_genes, n_samples = 100, 60
    
    counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
    design = np.column_stack([
        np.ones(n_samples),
        np.random.binomial(1, 0.5, n_samples),
        np.random.normal(0, 1, n_samples)
    ])
    
    return counts, design


# Utility fixtures
@pytest.fixture
def temp_dir():
    """Create temporary directory for testing file operations."""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_cupy():
    """Mock CuPy for testing GPU code paths without actual GPU."""
    with patch('devil.gpu.CUPY_AVAILABLE', True):
        with patch('devil.gpu.cp') as mock_cp:
            # Set up basic mock behavior
            mock_cp.array = lambda x: np.array(x)
            mock_cp.asarray = lambda x: np.array(x)
            mock_cp.asnumpy = lambda x: np.array(x)
            mock_cp.zeros = np.zeros
            mock_cp.ones = np.ones
            mock_cp.eye = np.eye
            mock_cp.newaxis = np.newaxis
            
            # Mock CUDA device
            mock_device = MagicMock()
            mock_cp.cuda.Device.return_value = mock_device
            
            # Mock memory pool
            mock_mempool = MagicMock()
            mock_mempool.free_bytes.return_value = 1e9  # 1GB free
            mock_mempool.total_bytes.return_value = 4e9  # 4GB total
            mock_cp.get_default_memory_pool.return_value = mock_mempool
            
            yield mock_cp


# Test data validation helpers
def validate_fitted_model(fit_result, expected_n_genes, expected_n_samples, expected_n_features):
    """Validate that a fitted model has expected structure."""
    required_keys = [
        'beta', 'overdispersion', 'iterations', 'size_factors',
        'offset_vector', 'design_matrix', 'gene_names', 'n_genes',
        'n_samples', 'converged', 'count_matrix'
    ]
    
    for key in required_keys:
        assert key in fit_result, f"Missing key in fit result: {key}"
    
    assert fit_result['beta'].shape == (expected_n_genes, expected_n_features)
    assert len(fit_result['overdispersion']) == expected_n_genes
    assert len(fit_result['size_factors']) == expected_n_samples
    assert fit_result['n_genes'] == expected_n_genes
    assert fit_result['n_samples'] == expected_n_samples
    
    # Check data types and finite values
    assert np.all(np.isfinite(fit_result['beta']))
    assert np.all(fit_result['overdispersion'] >= 0)
    assert np.all(fit_result['size_factors'] > 0)
    assert np.all(np.isfinite(fit_result['size_factors']))


def validate_de_results(de_results, expected_n_genes):
    """Validate that DE results have expected structure."""
    required_columns = ['gene', 'pval', 'padj', 'lfc', 'se', 'stat']
    
    for col in required_columns:
        assert col in de_results.columns, f"Missing column in DE results: {col}"
    
    assert len(de_results) == expected_n_genes
    
    # Check data properties
    assert np.all(de_results['pval'] >= 0)
    assert np.all(de_results['pval'] <= 1)
    assert np.all(de_results['padj'] >= 0)
    assert np.all(de_results['padj'] <= 1)
    assert np.all(de_results['se'] > 0)
    assert np.all(np.isfinite(de_results['lfc']))
    assert np.all(np.isfinite(de_results['stat']))
    
    # Should be sorted by adjusted p-value
    assert list(de_results['padj']) == sorted(de_results['padj'])


# Custom pytest markers for conditional testing
def pytest_runtest_setup(item):
    """Setup function run before each test."""
    # Skip tests marked as slow unless explicitly requested
    if "slow" in item.keywords and not item.config.getoption("--run-slow", default=False):
        pytest.skip("Slow test skipped (use --run-slow to run)")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--run-gpu",
        action="store_true", 
        default=False,
        help="Run GPU tests even if GPU not available (will likely fail)"
    )


# Session-scoped fixtures for expensive operations
@pytest.fixture(scope="session")
def session_fitted_model():
    """Create fitted model once per test session for speed."""
    np.random.seed(42)
    counts = np.random.negative_binomial(5, 0.3, size=(80, 45))
    design = np.column_stack([
        np.ones(45),
        np.random.binomial(1, 0.5, 45),
        np.random.normal(0, 1, 45)
    ])
    
    return devil.fit_devil(
        counts,
        design_matrix=design,
        overdispersion=True,
        verbose=False,
        max_iter=20,
        use_gpu=False
    )


# Helper functions for tests
class TestDataGenerator:
    """Helper class for generating test data with specific properties."""
    
    @staticmethod
    def create_differential_expression_data(
        n_genes: int,
        n_samples: int, 
        n_de_genes: int,
        effect_size: float = 2.0,
        seed: int = 42
    ):
        """Create data with known differential expression."""
        np.random.seed(seed)
        
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Create beta coefficients
        beta = np.column_stack([
            np.random.normal(2, 0.5, n_genes),  # Intercept
            np.concatenate([
                np.random.normal(effect_size, 0.2, n_de_genes),  # DE genes
                np.random.normal(0, 0.1, n_genes - n_de_genes)   # Non-DE genes
            ])
        ])
        
        # Generate counts
        size_factors = np.random.lognormal(0, 0.3, n_samples)
        overdispersion = np.random.gamma(2, 0.3, n_genes)
        
        mu = size_factors[np.newaxis, :] * np.exp(design @ beta.T)
        counts = np.zeros((n_genes, n_samples))
        
        for i in range(n_genes):
            p = 1 / (1 + mu[i, :] * overdispersion[i])
            n = 1 / overdispersion[i]
            counts[i, :] = np.random.negative_binomial(n, p)
        
        return counts, design, beta, overdispersion
    
    @staticmethod
    def create_batch_effect_data(
        n_genes: int,
        n_samples: int,
        batch_strength: float = 0.5,
        seed: int = 42
    ):
        """Create data with batch effects."""
        np.random.seed(seed)
        
        conditions = np.random.binomial(1, 0.5, n_samples)
        batches = np.random.binomial(1, 0.4, n_samples)
        
        design = np.column_stack([
            np.ones(n_samples),
            conditions,
            batches
        ])
        
        # Beta with batch effects
        beta = np.column_stack([
            np.random.normal(2, 0.5, n_genes),  # Intercept
            np.random.normal(1, 0.5, n_genes),  # Condition effect
            np.random.normal(0, batch_strength, n_genes)  # Batch effect
        ])
        
        # Generate counts
        size_factors = np.ones(n_samples)
        overdispersion = np.random.gamma(2, 0.3, n_genes)
        
        mu = np.exp(design @ beta.T)
        counts = np.zeros((n_genes, n_samples))
        
        for i in range(n_genes):
            p = 1 / (1 + mu[i, :] * overdispersion[i])
            n = 1 / overdispersion[i]
            counts[i, :] = np.random.negative_binomial(n, p)
        
        return counts, design, conditions, batches
