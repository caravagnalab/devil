"""
Performance and benchmarking tests for devil CPU functions.

This should be saved as python/tests/test_performance.py
"""

import pytest
import numpy as np
import pandas as pd
import time
from unittest.mock import patch
import psutil
import os

import devil


class TestPerformanceBenchmarks:
    """Test performance benchmarks for different dataset sizes."""
    
    @pytest.mark.slow
    @pytest.mark.performance
    def test_fit_devil_scaling(self, benchmark_data):
        """Test how fit_devil performance scales with dataset size."""
        results = {}
        
        for size_name, (counts, design) in benchmark_data.items():
            n_genes, n_samples = counts.shape
            
            # Measure time and memory
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss
            
            start_time = time.time()
            
            fit_result = devil.fit_devil(
                counts,
                design_matrix=design,
                overdispersion=True,
                size_factors=True,
                verbose=False,
                max_iter=10,  # Limited for speed
                use_gpu=False
            )
            
            end_time = time.time()
            memory_after = process.memory_info().rss
            
            results[size_name] = {
                'n_genes': n_genes,
                'n_samples': n_samples,
                'time': end_time - start_time,
                'memory_mb': (memory_after - memory_before) / 1024 / 1024,
                'converged_fraction': np.mean(fit_result['converged'])
            }
            
            print(f"{size_name}: {n_genes}×{n_samples}, "
                  f"Time: {results[size_name]['time']:.2f}s, "
                  f"Memory: {results[size_name]['memory_mb']:.1f}MB, "
                  f"Converged: {results[size_name]['converged_fraction']:.2%}")
        
        # Validate scaling behavior
        small_time = results['small']['time']
        medium_time = results['medium']['time']
        
        # Medium dataset should take more time than small
        assert medium_time >= small_time
        
        # Time scaling should be reasonable (not exponential)
        genes_ratio = results['medium']['n_genes'] / results['small']['n_genes']
        time_ratio = medium_time / small_time
        
        # Expect roughly linear scaling with some overhead
        assert time_ratio < genes_ratio * 3  # Should not be worse than 3x linear scaling
    
    @pytest.mark.slow
    @pytest.mark.performance
    def test_test_de_scaling(self, benchmark_data):
        """Test how test_de performance scales with dataset size."""
        results = {}
        
        for size_name, (counts, design) in benchmark_data.items():
            if size_name == 'large':  # Skip large for DE testing to save time
                continue
                
            # Fit model first
            fit_result = devil.fit_devil(
                counts,
                design_matrix=design,
                overdispersion=True,
                verbose=False,
                max_iter=5,  # Fast fitting
                use_gpu=False
            )
            
            # Measure DE testing time
            start_time = time.time()
            
            de_results = devil.test_de(
                fit_result,
                contrast=[0, 1, 0],
                verbose=False,
                use_gpu=False
            )
            
            end_time = time.time()
            
            results[size_name] = {
                'n_genes': counts.shape[0],
                'time': end_time - start_time,
                'n_significant': np.sum(de_results['padj'] < 0.05)
            }
            
            print(f"DE {size_name}: {results[size_name]['n_genes']} genes, "
                  f"Time: {results[size_name]['time']:.2f}s, "
                  f"Significant: {results[size_name]['n_significant']}")
        
        # DE testing should scale approximately linearly
        if 'small' in results and 'medium' in results:
            genes_ratio = results['medium']['n_genes'] / results['small']['n_genes']
            time_ratio = results['medium']['time'] / results['small']['time']
            
            # Should be roughly linear scaling
            assert time_ratio < genes_ratio * 2
    
    def test_parallel_performance(self):
        """Test parallel processing performance improvements."""
        np.random.seed(42)
        n_genes, n_samples = 200, 60
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Test different numbers of jobs
        job_configs = [1, 2, 4]
        results = {}
        
        for n_jobs in job_configs:
            start_time = time.time()
            
            fit_result = devil.fit_devil(
                counts,
                design_matrix=design,
                n_jobs=n_jobs,
                overdispersion=True,
                verbose=False,
                max_iter=10,
                use_gpu=False
            )
            
            end_time = time.time()
            
            results[n_jobs] = {
                'time': end_time - start_time,
                'beta': fit_result['beta'].copy()
            }
            
            print(f"{n_jobs} jobs: {results[n_jobs]['time']:.2f}s")
        
        # Results should be numerically identical
        np.testing.assert_allclose(results[1]['beta'], results[2]['beta'], rtol=1e-10)
        np.testing.assert_allclose(results[1]['beta'], results[4]['beta'], rtol=1e-10)
        
        # Parallel should be faster (or at least not much slower)
        assert results[4]['time'] <= results[1]['time'] * 1.5  # Allow some overhead
    
    def test_memory_usage_optimization(self):
        """Test memory usage optimization strategies."""
        np.random.seed(42)
        n_genes, n_samples = 500, 100
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        process = psutil.Process(os.getpid())
        
        # Measure memory usage
        memory_before = process.memory_info().rss
        
        fit_result = devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=10,
            use_gpu=False
        )
        
        memory_after = process.memory_info().rss
        memory_used_mb = (memory_after - memory_before) / 1024 / 1024
        
        print(f"Memory used: {memory_used_mb:.1f}MB for {n_genes}×{n_samples} dataset")
        
        # Memory usage should be reasonable (less than 1GB for this size)
        assert memory_used_mb < 1000
        
        # Clean up
        del fit_result
    
    @pytest.mark.performance
    def test_convergence_speed(self):
        """Test convergence speed with different tolerances."""
        np.random.seed(42)
        n_genes, n_samples = 100, 50
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        tolerances = [1e-1, 1e-2, 1e-3, 1e-4]
        results = {}
        
        for tol in tolerances:
            start_time = time.time()
            
            fit_result = devil.fit_devil(
                counts,
                design_matrix=design,
                tolerance=tol,
                max_iter=100,
                overdispersion=True,
                verbose=False,
                use_gpu=False
            )
            
            end_time = time.time()
            
            results[tol] = {
                'time': end_time - start_time,
                'mean_iterations': np.mean(fit_result['iterations']),
                'converged_fraction': np.mean(fit_result['converged'])
            }
            
            print(f"Tolerance {tol}: {results[tol]['time']:.2f}s, "
                  f"Avg iterations: {results[tol]['mean_iterations']:.1f}, "
                  f"Converged: {results[tol]['converged_fraction']:.2%}")
        
        # Stricter tolerance should take more time and iterations
        assert results[1e-4]['time'] >= results[1e-1]['time']
        assert results[1e-4]['mean_iterations'] >= results[1e-1]['mean_iterations']


class TestPerformanceOptimizations:
    """Test specific performance optimizations."""
    
    def test_size_factor_computation_speed(self):
        """Test size factor computation performance."""
        sizes = [(100, 50), (500, 100), (1000, 200)]
        
        for n_genes, n_samples in sizes:
            np.random.seed(42)
            counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
            
            start_time = time.time()
            size_factors = devil.size_factors.calculate_size_factors(counts, verbose=False)
            end_time = time.time()
            
            print(f"Size factors {n_genes}×{n_samples}: {end_time - start_time:.3f}s")
            
            # Should complete in reasonable time
            assert end_time - start_time < 10.0  # Less than 10 seconds
            assert len(size_factors) == n_samples
            assert np.all(np.isfinite(size_factors))
    
    def test_overdispersion_fitting_speed(self):
        """Test overdispersion fitting performance."""
        np.random.seed(42)
        n_genes_list = [50, 100, 200]
        n_samples = 40
        
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        offset_vector = np.zeros(n_samples)
        
        for n_genes in n_genes_list:
            counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
            beta = np.random.normal(0, 1, size=(n_genes, 2))
            
            start_time = time.time()
            
            # Test single-threaded overdispersion fitting
            theta_results = []
            for i in range(min(n_genes, 20)):  # Test subset for speed
                theta = devil.overdispersion.fit_dispersion(
                    beta[i, :], design, counts[i, :], offset_vector,
                    tolerance=1e-3, max_iter=50
                )
                theta_results.append(theta)
            
            end_time = time.time()
            
            avg_time_per_gene = (end_time - start_time) / min(n_genes, 20)
            print(f"Overdispersion fitting: {avg_time_per_gene:.4f}s per gene")
            
            # Should be reasonably fast per gene
            assert avg_time_per_gene < 1.0  # Less than 1 second per gene
            assert all(theta > 0 for theta in theta_results)
    
    def test_variance_computation_speed(self):
        """Test variance computation performance."""
        np.random.seed(42)
        
        test_configs = [
            (50, 30, 3),   # Small
            (200, 60, 4),  # Medium
            (500, 100, 5)  # Large
        ]
        
        for n_genes, n_samples, n_features in test_configs:
            # Generate test data
            design_matrix = np.random.normal(0, 1, size=(n_samples, n_features))
            design_matrix[:, 0] = 1  # Intercept
            
            beta = np.random.normal(0, 1, size=(n_genes, n_features))
            overdispersion = np.random.gamma(2, 0.5, n_genes)
            size_factors = np.ones(n_samples)
            
            start_time = time.time()
            
            # Test Hessian computation for subset of genes
            for i in range(min(n_genes, 20)):
                y = np.random.negative_binomial(5, 0.3, n_samples)
                precision = 1.0 / overdispersion[i]
                
                H = devil.variance.compute_hessian(
                    beta[i, :], precision, y, design_matrix, size_factors
                )
                
                assert H.shape == (n_features, n_features)
                assert np.all(np.isfinite(H))
            
            end_time = time.time()
            
            avg_time = (end_time - start_time) / min(n_genes, 20)
            print(f"Hessian computation {n_samples}×{n_features}: {avg_time:.4f}s per gene")
            
            # Should be fast for variance computation
            assert avg_time < 0.1  # Less than 0.1 second per gene
    
    @pytest.mark.slow
    def test_full_workflow_benchmark(self):
        """Benchmark complete workflow for different scenarios."""
        scenarios = {
            'small_simple': (50, 30, 2),    # Small dataset, simple design
            'medium_simple': (200, 80, 2),  # Medium dataset, simple design
            'small_complex': (50, 30, 5),   # Small dataset, complex design
            'medium_complex': (200, 80, 5)  # Medium dataset, complex design
        }
        
        results = {}
        
        for scenario_name, (n_genes, n_samples, n_features) in scenarios.items():
            np.random.seed(42)
            
            # Generate data
            counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
            design = np.random.normal(0, 1, size=(n_samples, n_features))
            design[:, 0] = 1  # Intercept
            
            # Benchmark complete workflow
            start_time = time.time()
            
            # Fit model
            fit_result = devil.fit_devil(
                counts,
                design_matrix=design,
                overdispersion=True,
                size_factors=True,
                verbose=False,
                max_iter=15,
                use_gpu=False
            )
            
            # Test differential expression
            contrast = [0, 1] + [0] * (n_features - 2)
            de_results = devil.test_de(
                fit_result,
                contrast=contrast,
                verbose=False,
                use_gpu=False
            )
            
            # Create plot
            ax = devil.plot_volcano(de_results)
            
            end_time = time.time()
            
            results[scenario_name] = {
                'total_time': end_time - start_time,
                'n_significant': np.sum(de_results['padj'] < 0.05),
                'convergence_rate': np.mean(fit_result['converged'])
            }
            
            print(f"{scenario_name}: {results[scenario_name]['total_time']:.2f}s total, "
                  f"{results[scenario_name]['n_significant']} significant genes, "
                  f"{results[scenario_name]['convergence_rate']:.2%} converged")
        
        # Validate reasonable performance
        for scenario, result in results.items():
            assert result['total_time'] < 300  # Less than 5 minutes
            assert result['convergence_rate'] > 0.5  # At least 50% convergence
    
    def test_memory_efficient_processing(self):
        """Test memory-efficient processing strategies."""
        np.random.seed(42)
        n_genes, n_samples = 1000, 150
        
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
        
        # Test memory-efficient DE testing
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Test subset of genes
        gene_subset = fit_result['gene_names'][:200]  # Test first 200 genes
        
        start_time = time.time()
        de_results = devil.test_de_memory_efficient(
            fit_result,
            contrast=[0, 1],
            gene_subset=gene_subset,
            verbose=False
        )
        end_time = time.time()
        
        memory_after = process.memory_info().rss
        memory_used_mb = (memory_after - memory_before) / 1024 / 1024
        
        print(f"Memory-efficient DE: {end_time - start_time:.2f}s, "
              f"{memory_used_mb:.1f}MB for {len(gene_subset)} genes")
        
        # Should complete efficiently
        assert len(de_results) == 200
        assert end_time - start_time < 60  # Less than 1 minute
        assert memory_used_mb < 500  # Less than 500MB


class TestPerformanceRegression:
    """Test for performance regressions between versions."""
    
    def test_baseline_performance_small(self):
        """Test baseline performance for small datasets."""
        np.random.seed(42)
        n_genes, n_samples = 100, 40
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Baseline performance test
        start_time = time.time()
        
        fit_result = devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=15,
            use_gpu=False
        )
        
        de_results = devil.test_de(
            fit_result,
            contrast=[0, 1],
            verbose=False,
            use_gpu=False
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Baseline performance (100×40): {total_time:.2f}s")
        
        # Should complete in reasonable time
        assert total_time < 60  # Less than 1 minute for small dataset
        assert len(de_results) == n_genes
        assert np.mean(fit_result['converged']) > 0.8  # Good convergence
    
    def test_performance_consistency(self):
        """Test that performance is consistent across runs."""
        np.random.seed(42)
        n_genes, n_samples = 80, 35
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Run multiple times
        times = []
        for run in range(3):
            start_time = time.time()
            
            fit_result = devil.fit_devil(
                counts,
                design_matrix=design,
                overdispersion=True,
                verbose=False,
                max_iter=10,
                use_gpu=False
            )
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Times should be consistent (coefficient of variation < 50%)
        mean_time = np.mean(times)
        std_time = np.std(times)
        cv = std_time / mean_time
        
        print(f"Performance consistency: {mean_time:.2f}±{std_time:.2f}s (CV: {cv:.2%})")
        
        assert cv < 0.5  # Coefficient of variation should be reasonable


class TestComputationalComplexity:
    """Test computational complexity characteristics."""
    
    def test_gene_scaling_complexity(self):
        """Test how computation scales with number of genes."""
        n_samples = 50
        gene_counts = [50, 100, 200, 400]
        
        times = []
        
        for n_genes in gene_counts:
            np.random.seed(42)
            counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
            design = np.column_stack([
                np.ones(n_samples),
                np.random.binomial(1, 0.5, n_samples)
            ])
            
            start_time = time.time()
            
            fit_result = devil.fit_devil(
                counts,
                design_matrix=design,
                overdispersion=True,
                verbose=False,
                max_iter=10,
                use_gpu=False
            )
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            print(f"{n_genes} genes: {times[-1]:.2f}s")
        
        # Should scale approximately linearly with genes
        # (Each gene is processed independently)
        for i in range(1, len(times)):
            gene_ratio = gene_counts[i] / gene_counts[i-1]
            time_ratio = times[i] / times[i-1]
            
            # Allow for some overhead but should be roughly linear
            assert time_ratio < gene_ratio * 2
    
    def test_sample_scaling_complexity(self):
        """Test how computation scales with number of samples."""
        n_genes = 100
        sample_counts = [25, 50, 100, 200]
        
        times = []
        
        for n_samples in sample_counts:
            np.random.seed(42)
            counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
            design = np.column_stack([
                np.ones(n_samples),
                np.random.binomial(1, 0.5, n_samples)
            ])
            
            start_time = time.time()
            
            fit_result = devil.fit_devil(
                counts,
                design_matrix=design,
                overdispersion=True,
                verbose=False,
                max_iter=10,
                use_gpu=False
            )
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            print(f"{n_samples} samples: {times[-1]:.2f}s")
        
        # Scaling with samples should be better than quadratic
        # (Some operations are O(n_samples^2) but overall should be manageable)
        for i in range(1, len(times)):
            sample_ratio = sample_counts[i] / sample_counts[i-1]
            time_ratio = times[i] / times[i-1]
            
            # Should not be worse than quadratic scaling
            assert time_ratio < sample_ratio ** 2.5
    
    def test_feature_scaling_complexity(self):
        """Test how computation scales with number of features."""
        n_genes, n_samples = 80, 60
        feature_counts = [2, 3, 5, 8]
        
        times = []
        
        for n_features in feature_counts:
            np.random.seed(42)
            counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
            design = np.random.normal(0, 1, size=(n_samples, n_features))
            design[:, 0] = 1  # Intercept
            
            start_time = time.time()
            
            fit_result = devil.fit_devil(
                counts,
                design_matrix=design,
                overdispersion=True,
                verbose=False,
                max_iter=10,
                use_gpu=False
            )
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            print(f"{n_features} features: {times[-1]:.2f}s")
        
        # Should scale reasonably with features (matrix operations are O(p^3))
        for i in range(1, len(times)):
            feature_ratio = feature_counts[i] / feature_counts[i-1]
            time_ratio = times[i] / times[i-1]
            
            # Should not be worse than cubic scaling
            assert time_ratio < feature_ratio ** 3.5


class TestResourceUtilization:
    """Test resource utilization characteristics."""
    
    def test_cpu_utilization(self):
        """Test CPU utilization during computation."""
        np.random.seed(42)
        n_genes, n_samples = 300, 80
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples),
            np.random.normal(0, 1, n_samples)
        ])
        
        # Monitor CPU usage during computation
        import threading
        cpu_percentages = []
        monitoring = True
        
        def monitor_cpu():
            while monitoring:
                cpu_percentages.append(psutil.cpu_percent(interval=0.1))
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        try:
            # Run computation
            fit_result = devil.fit_devil(
                counts,
                design_matrix=design,
                n_jobs=4,  # Use multiple cores
                overdispersion=True,
                verbose=False,
                max_iter=15,
                use_gpu=False
            )
            
        finally:
            monitoring = False
            monitor_thread.join()
        
        if cpu_percentages:
            max_cpu = np.max(cpu_percentages)
            mean_cpu = np.mean(cpu_percentages)
            
            print(f"CPU utilization: {mean_cpu:.1f}% average, {max_cpu:.1f}% peak")
            
            # Should utilize CPU resources efficiently
            assert mean_cpu > 10  # Should use some CPU
            assert max_cpu < 100  # Should not max out completely
    
    def test_memory_scaling(self):
        """Test memory usage scaling."""
        base_size = (100, 50)
        scale_factors = [1, 2, 4]
        
        memory_usage = []
        
        for scale in scale_factors:
            np.random.seed(42)
            n_genes = base_size[0] * scale
            n_samples = base_size[1] * scale
            
            counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
            design = np.column_stack([
                np.ones(n_samples),
                np.random.binomial(1, 0.5, n_samples)
            ])
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss
            
            fit_result = devil.fit_devil(
                counts,
                design_matrix=design,
                overdispersion=True,
                verbose=False,
                max_iter=8,
                use_gpu=False
            )
            
            memory_after = process.memory_info().rss
            memory_used = (memory_after - memory_before) / 1024 / 1024  # MB
            
            memory_usage.append(memory_used)
            print(f"Scale {scale}x ({n_genes}×{n_samples}): {memory_used:.1f}MB")
            
            # Clean up
            del fit_result
        
        # Memory should scale reasonably (not exponentially)
        for i in range(1, len(memory_usage)):
            scale_ratio = scale_factors[i] / scale_factors[i-1]
            memory_ratio = memory_usage[i] / memory_usage[i-1]
            
            # Should not be worse than quadratic scaling
            assert memory_ratio < scale_ratio ** 2.5


class TestPerformanceOptimizationStrategies:
    """Test different optimization strategies."""
    
    def test_convergence_early_stopping(self):
        """Test early stopping optimization for convergence."""
        np.random.seed(42)
        n_genes, n_samples = 150, 60
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Test different max_iter values
        max_iters = [5, 10, 20, 50]
        results = {}
        
        for max_iter in max_iters:
            start_time = time.time()
            
            fit_result = devil.fit_devil(
                counts,
                design_matrix=design,
                max_iter=max_iter,
                tolerance=1e-3,
                overdispersion=True,
                verbose=False,
                use_gpu=False
            )
            
            end_time = time.time()
            
            results[max_iter] = {
                'time': end_time - start_time,
                'mean_iterations': np.mean(fit_result['iterations']),
                'convergence_rate': np.mean(fit_result['converged'])
            }
            
            print(f"max_iter={max_iter}: {results[max_iter]['time']:.2f}s, "
                  f"avg_iter={results[max_iter]['mean_iterations']:.1f}, "
                  f"converged={results[max_iter]['convergence_rate']:.2%}")
        
        # Early stopping should save time when convergence is achieved
        assert results[5]['time'] <= results[50]['time']
        
        # More iterations should allow better convergence (up to a point)
        assert results[50]['convergence_rate'] >= results[5]['convergence_rate']
    
    def test_tolerance_optimization(self):
        """Test optimization with different tolerance levels."""
        np.random.seed(42)
        n_genes, n_samples = 100, 50
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        tolerances = [1e-1, 1e-2, 1e-3, 1e-4]
        results = {}
        
        for tol in tolerances:
            start_time = time.time()
            
            fit_result = devil.fit_devil(
                counts,
                design_matrix=design,
                tolerance=tol,
                max_iter=50,
                overdispersion=True,
                verbose=False,
                use_gpu=False
            )
            
            end_time = time.time()
            
            results[tol] = {
                'time': end_time - start_time,
                'mean_iterations': np.mean(fit_result['iterations']),
                'beta_precision': np.std(fit_result['beta'])
            }
            
            print(f"tolerance={tol}: {results[tol]['time']:.2f}s, "
                  f"iterations={results[tol]['mean_iterations']:.1f}")
        
        # Tighter tolerance should take more time but give more precise results
        assert results[1e-4]['time'] >= results[1e-1]['time']
        assert results[1e-4]['mean_iterations'] >= results[1e-1]['mean_iterations']
    
    def test_batch_processing_optimization(self):
        """Test batch processing for large datasets."""
        np.random.seed(42)
        n_genes, n_samples = 1000, 100
        
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
            max_iter=8,
            use_gpu=False
        )
        
        # Test batch processing for DE testing
        batch_sizes = [100, 250, 500]
        times = []
        
        for batch_size in batch_sizes:
            # Simulate batch processing by testing subsets
            start_time = time.time()
            
            all_results = []
            for start_idx in range(0, n_genes, batch_size):
                end_idx = min(start_idx + batch_size, n_genes)
                gene_subset = fit_result['gene_names'][start_idx:end_idx]
                
                batch_results = devil.test_de_memory_efficient(
                    fit_result,
                    contrast=[0, 1],
                    gene_subset=gene_subset,
                    verbose=False
                )
                all_results.append(batch_results)
            
            # Combine results
            combined_results = pd.concat(all_results, ignore_index=True)
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            print(f"Batch size {batch_size}: {times[-1]:.2f}s")
            
            assert len(combined_results) == n_genes
        
        # Different batch sizes should give similar total times
        # (Might vary due to overhead, but should be in same ballpark)
        time_range = max(times) - min(times)
        assert time_range < max(times) * 0.5  # Variation should be < 50%


class TestNumericalPerformance:
    """Test numerical computation performance."""
    
    def test_matrix_operation_efficiency(self):
        """Test efficiency of matrix operations."""
        sizes = [(100, 50), (200, 100), (500, 200)]
        
        for n_genes, n_samples in sizes:
            np.random.seed(42)
            
            # Test Hessian computation speed
            design_matrix = np.column_stack([
                np.ones(n_samples),
                np.random.binomial(1, 0.5, n_samples),
                np.random.normal(0, 1, n_samples)
            ])
            
            beta = np.random.normal(0, 1, 3)
            y = np.random.negative_binomial(5, 0.3, n_samples)
            precision = 1.0
            size_factors = np.ones(n_samples)
            
            start_time = time.time()
            
            # Multiple Hessian computations
            for _ in range(50):
                H = devil.variance.compute_hessian(
                    beta, precision, y, design_matrix, size_factors
                )
            
            end_time = time.time()
            
            time_per_computation = (end_time - start_time) / 50
            print(f"Hessian ({n_samples}×3): {time_per_computation:.4f}s per computation")
            
            # Should be fast for matrix operations
            assert time_per_computation < 0.01  # Less than 10ms per Hessian
    
    def test_optimization_algorithm_efficiency(self):
        """Test efficiency of optimization algorithms."""
        np.random.seed(42)
        n_samples = 100
        
        # Test beta fitting efficiency
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples),
            np.random.normal(0, 1, n_samples)
        ])
        
        y = np.random.negative_binomial(5, 0.3, n_samples)
        beta_init = np.array([1.0, 0.5, -0.2])
        offset = np.zeros(n_samples)
        dispersion = 1.0
        
        start_time = time.time()
        
        # Multiple beta fittings
        for _ in range(100):
            fitted_beta, n_iter, converged = devil.beta.fit_beta_coefficients(
                y, design, beta_init, offset, dispersion,
                max_iter=20, tolerance=1e-4
            )
        
        end_time = time.time()
        
        time_per_fit = (end_time - start_time) / 100
        print(f"Beta fitting: {time_per_fit:.4f}s per gene")
        
        # Should be efficient for single gene fitting
        assert time_per_fit < 0.05  # Less than 50ms per gene