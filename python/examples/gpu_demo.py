"""
Example demonstrating GPU-accelerated differential expression analysis with devil.

This example shows how to:
1. Check GPU availability and requirements
2. Fit models with GPU acceleration
3. Perform differential expression testing on GPU
4. Compare performance between CPU and GPU
5. Handle memory management and error recovery
"""

import numpy as np
import pandas as pd
import scanpy as sc
import devil
import time
from typing import Optional


def create_synthetic_data(
    n_genes: int = 2000,
    n_samples: int = 500,
    n_conditions: int = 2,
    seed: int = 42
) -> tuple:
    """Create synthetic single-cell RNA-seq data for testing."""
    np.random.seed(seed)
    
    # Create design matrix
    conditions = np.repeat(np.arange(n_conditions), n_samples // n_conditions)
    batch = np.random.binomial(1, 0.3, n_samples)  # 30% batch effect
    
    design_matrix = np.column_stack([
        np.ones(n_samples),  # intercept
        conditions,          # condition effect
        batch               # batch effect
    ])
    
    # Simulate count data
    # True beta coefficients
    true_beta = np.random.normal(0, 1, (n_genes, 3))
    true_beta[:, 0] = np.random.normal(5, 1, n_genes)  # Higher intercept for counts
    
    # Make some genes differentially expressed
    n_de_genes = n_genes // 10
    true_beta[:n_de_genes, 1] = np.random.normal(2, 0.5, n_de_genes)  # Strong effect
    
    # Simulate size factors
    size_factors = np.random.lognormal(0, 0.3, n_samples)
    
    # Generate counts
    log_mu = design_matrix @ true_beta.T + np.log(size_factors)[:, np.newaxis]
    mu = np.exp(log_mu)
    
    # Add overdispersion
    dispersion = np.random.gamma(2, 0.1, n_genes)
    counts = np.random.negative_binomial(
        n=1/dispersion[np.newaxis, :], 
        p=1/(1 + mu * dispersion[np.newaxis, :])
    ).T
    
    # Create gene and sample names
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    sample_names = [f"Cell_{i:04d}" for i in range(n_samples)]
    
    # Create metadata
    obs_df = pd.DataFrame({
        'condition': [f'Cond_{c}' for c in conditions],
        'batch': [f'Batch_{b}' for b in batch],
        'sample_id': sample_names
    })
    
    return counts, design_matrix, gene_names, sample_names, obs_df


def check_gpu_setup(verbose: bool = True) -> dict:
    """Check GPU availability and setup."""
    gpu_info = {
        'available': devil.is_gpu_available(),
        'cupy_installed': False,
        'memory_info': (0, 0),
        'feasible': False,
        'message': ""
    }
    
    if gpu_info['available']:
        try:
            import cupy as cp
            gpu_info['cupy_installed'] = True
            
            # Get memory info
            free_mem, total_mem = devil.gpu.get_gpu_memory_info()
            gpu_info['memory_info'] = (free_mem, total_mem)
            
            # Check feasibility for a typical dataset
            feasible, message = devil.check_gpu_requirements(
                n_genes=2000, n_samples=500, n_features=3, verbose=verbose
            )
            gpu_info['feasible'] = feasible
            gpu_info['message'] = message
            
        except ImportError:
            gpu_info['message'] = "CuPy not installed"
    else:
        gpu_info['message'] = "CUDA not available"
    
    if verbose:
        print("GPU Setup Check:")
        print(f"  GPU Available: {gpu_info['available']}")
        print(f"  CuPy Installed: {gpu_info['cupy_installed']}")
        if gpu_info['memory_info'][1] > 0:
            free_gb = gpu_info['memory_info'][0] / 1e9
            total_gb = gpu_info['memory_info'][1] / 1e9
            print(f"  GPU Memory: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
        print(f"  Feasible: {gpu_info['feasible']}")
        print(f"  Message: {gpu_info['message']}")
    
    return gpu_info


def benchmark_gpu_vs_cpu(
    counts: np.ndarray,
    design_matrix: np.ndarray,
    gene_names: list,
    obs_df: pd.DataFrame,
    gpu_available: bool = True
) -> dict:
    """Benchmark GPU vs CPU performance."""
    results = {
        'cpu_time': None,
        'gpu_time': None,
        'speedup': None,
        'cpu_fit': None,
        'gpu_fit': None
    }
    
    print("Benchmarking CPU vs GPU performance...")
    
    # CPU fitting
    print("Fitting model on CPU...")
    start_time = time.time()
    cpu_fit = devil.fit_devil(
        counts,
        design_matrix=design_matrix,
        overdispersion=True,
        verbose=False,
        use_gpu=False,
        n_jobs=4  # Use limited cores for fair comparison
    )
    cpu_time = time.time() - start_time
    results['cpu_time'] = cpu_time
    results['cpu_fit'] = cpu_fit
    
    print(f"CPU fitting completed in {cpu_time:.2f} seconds")
    
    # GPU fitting (if available)
    if gpu_available:
        print("Fitting model on GPU...")
        start_time = time.time()
        try:
            gpu_fit = devil.fit_devil(
                counts,
                design_matrix=design_matrix,
                overdispersion=True,
                verbose=False,
                use_gpu=True,
                gpu_batch_size=512,  # Smaller batch for memory efficiency
                gpu_dtype='float32'
            )
            gpu_time = time.time() - start_time
            results['gpu_time'] = gpu_time
            results['gpu_fit'] = gpu_fit
            results['speedup'] = cpu_time / gpu_time
            
            print(f"GPU fitting completed in {gpu_time:.2f} seconds")
            print(f"GPU speedup: {results['speedup']:.2f}x")
            
        except Exception as e:
            print(f"GPU fitting failed: {e}")
            results['gpu_time'] = None
    else:
        print("GPU not available, skipping GPU benchmark")
    
    return results


def demonstrate_gpu_testing(
    fit_result: dict,
    contrast: list = [0, 1, 0],
    gpu_available: bool = True
) -> dict:
    """Demonstrate GPU-accelerated differential expression testing."""
    results = {
        'cpu_results': None,
        'gpu_results': None,
        'cpu_time': None,
        'gpu_time': None,
        'correlation': None
    }
    
    print("\nDemonstrating differential expression testing...")
    
    # CPU testing
    print("Running DE test on CPU...")
    start_time = time.time()
    cpu_results = devil.test_de(
        fit_result,
        contrast=contrast,
        verbose=False,
        use_gpu=False,
        n_jobs=4
    )
    cpu_time = time.time() - start_time
    results['cpu_results'] = cpu_results
    results['cpu_time'] = cpu_time
    
    print(f"CPU testing completed in {cpu_time:.2f} seconds")
    print(f"Found {(cpu_results['padj'] < 0.05).sum()} significant genes")
    
    # GPU testing (if available)
    if gpu_available:
        print("Running DE test on GPU...")
        start_time = time.time()
        try:
            gpu_results = devil.test_de(
                fit_result,
                contrast=contrast,
                verbose=False,
                use_gpu=True,
                gpu_batch_size=512
            )
            gpu_time = time.time() - start_time
            results['gpu_results'] = gpu_results
            results['gpu_time'] = gpu_time
            
            print(f"GPU testing completed in {gpu_time:.2f} seconds")
            print(f"Found {(gpu_results['padj'] < 0.05).sum()} significant genes")
            print(f"GPU speedup: {cpu_time / gpu_time:.2f}x")
            
            # Check correlation between results
            correlation = np.corrcoef(
                cpu_results['lfc'].values,
                gpu_results['lfc'].values
            )[0, 1]
            results['correlation'] = correlation
            print(f"Correlation between CPU and GPU results: {correlation:.6f}")
            
        except Exception as e:
            print(f"GPU testing failed: {e}")
    else:
        print("GPU not available, skipping GPU testing")
    
    return results


def demonstrate_memory_efficient_testing(
    fit_result: dict,
    contrast: list = [0, 1, 0]
) -> pd.DataFrame:
    """Demonstrate memory-efficient testing for large datasets."""
    print("\nDemonstrating memory-efficient testing...")
    
    # Test only a subset of genes
    n_genes = fit_result['n_genes']
    gene_subset = fit_result['gene_names'][:min(500, n_genes)]  # Test first 500 genes
    
    results = devil.test_de_memory_efficient(
        fit_result,
        contrast=contrast,
        gene_subset=gene_subset,
        verbose=True,
        use_gpu=True if devil.is_gpu_available() else False
    )
    
    print(f"Tested {len(results)} genes from subset")
    print(f"Found {(results['padj'] < 0.05).sum()} significant genes")
    
    return results


def demonstrate_clustered_testing(
    fit_result: dict,
    contrast: list = [0, 1, 0],
    n_clusters: int = 3
) -> pd.DataFrame:
    """Demonstrate testing with clustered variance estimation."""
    print(f"\nDemonstrating clustered testing with {n_clusters} clusters...")
    
    # Create artificial clusters (e.g., patients)
    n_samples = fit_result['n_samples']
    clusters = np.random.randint(1, n_clusters + 1, n_samples)
    
    results = devil.test_de(
        fit_result,
        contrast=contrast,
        clusters=clusters,
        verbose=True,
        use_gpu=True if devil.is_gpu_available() else False
    )
    
    print(f"Found {(results['padj'] < 0.05).sum()} significant genes with cluster correction")
    
    return results


def main():
    """Main demonstration function."""
    print("Devil GPU Acceleration Demo")
    print("=" * 50)
    
    # Check GPU setup
    gpu_info = check_gpu_setup(verbose=True)
    print()
    
    # Create synthetic data
    print("Creating synthetic dataset...")
    counts, design_matrix, gene_names, sample_names, obs_df = create_synthetic_data(
        n_genes=2000, n_samples=500
    )
    print(f"Created dataset: {counts.shape[0]} genes × {counts.shape[1]} samples")
    print()
    
    # Benchmark CPU vs GPU
    benchmark_results = benchmark_gpu_vs_cpu(
        counts, design_matrix, gene_names, obs_df,
        gpu_available=gpu_info['feasible']
    )
    
    # Use the best available fit result
    fit_result = (benchmark_results['gpu_fit'] 
                 if benchmark_results['gpu_fit'] is not None 
                 else benchmark_results['cpu_fit'])
    
    # Demonstrate differential expression testing
    contrast = [0, 1, 0]  # Compare condition 1 vs condition 0
    testing_results = demonstrate_gpu_testing(
        fit_result, contrast, gpu_available=gpu_info['feasible']
    )
    
    # Memory-efficient testing
    memory_results = demonstrate_memory_efficient_testing(fit_result, contrast)
    
    # Clustered testing
    cluster_results = demonstrate_clustered_testing(fit_result, contrast)
    
    # Create visualization
    if testing_results['cpu_results'] is not None:
        print("\nCreating volcano plot...")
        ax = devil.plot_volcano(
            testing_results['cpu_results'],
            lfc_threshold=1.0,
            pval_threshold=0.05,
            title="GPU-Accelerated Differential Expression"
        )
        print("Volcano plot created (use plt.show() to display)")
    
    print("\nDemo completed successfully!")
    
    return {
        'gpu_info': gpu_info,
        'benchmark_results': benchmark_results,
        'testing_results': testing_results,
        'memory_results': memory_results,
        'cluster_results': cluster_results
    }


if __name__ == "__main__":
    demo_results = main()