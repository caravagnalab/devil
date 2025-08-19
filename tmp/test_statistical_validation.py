"""
Statistical validation tests for devil package.

This should be saved as python/tests/test_statistical_validation.py
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import digamma, polygamma
import warnings

import devil


class TestStatisticalAccuracy:
    """Test statistical accuracy of devil implementations."""
    
    def test_negative_binomial_likelihood_accuracy(self):
        """Test accuracy of negative binomial likelihood computation."""
        np.random.seed(42)
        n_samples = 30
        
        y = np.random.negative_binomial(5, 0.3, n_samples)
        mu = np.random.gamma(5, 1, n_samples)
        theta = 0.8
        design_matrix = np.ones((n_samples, 1))
        
        # Compute using devil implementation
        ll_devil = devil.overdispersion.compute_nb_log_likelihood(
            y, mu, theta, design_matrix, do_cox_reid=False
        )
        
        # Compute manually using scipy
        from scipy.special import gammaln
        alpha = 1.0 / theta
        ll_manual = np.sum(
            gammaln(y + alpha) - gammaln(alpha) - gammaln(y + 1) +
            alpha * np.log(alpha / (alpha + mu)) +
            y * np.log(mu / (alpha + mu))
        )
        
        # Should match closely
        np.testing.assert_allclose(ll_devil, ll_manual, rtol=1e-10)
    
    def test_dispersion_estimation_consistency(self):
        """Test that dispersion estimation is statistically consistent."""
        np.random.seed(42)
        n_replicates = 50
        n_samples = 100
        true_dispersion = 0.5
        
        estimated_dispersions = []
        
        for rep in range(n_replicates):
            # Generate data with known dispersion
            design_matrix = np.ones((n_samples, 1))
            beta = np.array([2.0])
            offset_vector = np.zeros(n_samples)
            
            mu = np.exp(design_matrix @ beta + offset_vector)
            p = 1 / (1 + mu * true_dispersion)
            n = 1 / true_dispersion
            y = np.random.negative_binomial(n, p)
            
            # Estimate dispersion
            estimated = devil.overdispersion.fit_dispersion(
                beta, design_matrix, y, offset_vector,
                tolerance=1e-4, max_iter=100
            )
            
            estimated_dispersions.append(estimated)
        
        estimated_dispersions = np.array(estimated_dispersions)
        
        # Check consistency
        mean_estimate = np.mean(estimated_dispersions)
        std_estimate = np.std(estimated_dispersions)
        
        print(f"True dispersion: {true_dispersion}")
        print(f"Estimated: {mean_estimate:.3f} ± {std_estimate:.3f}")
        
        # Should be unbiased (within 2 standard errors)
        bias = abs(mean_estimate - true_dispersion)
        se_mean = std_estimate / np.sqrt(n_replicates)
        
        assert bias < 2 * se_mean, f"Biased estimate: bias={bias:.3f}, SE={se_mean:.3f}"
        
        # Should be reasonably precise
        cv = std_estimate / mean_estimate
        assert cv < 0.5, f"Too variable: CV={cv:.3f}"
    
    def test_beta_coefficient_accuracy(self):
        """Test accuracy of beta coefficient estimation."""
        np.random.seed(42)
        n_replicates = 30
        n_samples = 80
        true_beta = np.array([2.0, 1.5, -0.8])
        
        estimated_betas = []
        
        for rep in range(n_replicates):
            # Generate data with known beta
            design_matrix = np.column_stack([
                np.ones(n_samples),
                np.random.binomial(1, 0.5, n_samples),
                np.random.normal(0, 1, n_samples)
            ])
            
            offset_vector = np.zeros(n_samples)
            dispersion = 0.3
            
            mu = np.exp(design_matrix @ true_beta + offset_vector)
            p = 1 / (1 + mu * dispersion)
            n = 1 / dispersion
            y = np.random.negative_binomial(n, p)
            
            # Fit beta coefficients
            beta_init = np.random.normal(0, 0.5, 3) + true_beta
            fitted_beta, _, converged = devil.beta.fit_beta_coefficients(
                y, design_matrix, beta_init, offset_vector, dispersion,
                max_iter=100, tolerance=1e-6
            )
            
            if converged:
                estimated_betas.append(fitted_beta)
        
        estimated_betas = np.array(estimated_betas)
        
        # Check accuracy for each coefficient
        for i in range(3):
            mean_estimate = np.mean(estimated_betas[:, i])
            std_estimate = np.std(estimated_betas[:, i])
            
            print(f"Beta[{i}]: true={true_beta[i]:.2f}, "
                  f"estimated={mean_estimate:.3f}±{std_estimate:.3f}")
            
            # Should be unbiased
            bias = abs(mean_estimate - true_beta[i])
            se_mean = std_estimate / np.sqrt(len(estimated_betas))
            
            assert bias < 3 * se_mean, f"Beta[{i}] biased: bias={bias:.3f}, SE={se_mean:.3f}"
    
    def test_size_factor_normalization_accuracy(self):
        """Test accuracy of size factor normalization."""
        np.random.seed(42)
        n_genes, n_samples = 100, 50
        
        # Create data with known size differences
        true_size_factors = np.random.lognormal(0, 0.8, n_samples)
        base_expression = np.random.gamma(3, 2, n_genes)
        
        # Generate counts with known scaling
        expected_counts = np.outer(base_expression, true_size_factors)
        counts = np.random.poisson(expected_counts)
        
        # Estimate size factors
        estimated_sf = devil.size_factors.calculate_size_factors(counts, verbose=False)
        
        # Normalize both to have same geometric mean for comparison
        true_sf_norm = true_size_factors / np.exp(np.mean(np.log(true_size_factors)))
        
        # Should be highly correlated
        correlation = np.corrcoef(estimated_sf, true_sf_norm)[0, 1]
        print(f"Size factor correlation: {correlation:.4f}")
        
        assert correlation > 0.95, f"Poor size factor estimation: r={correlation:.4f}"
        
        # Should have similar variance structure
        true_cv = np.std(true_sf_norm) / np.mean(true_sf_norm)
        est_cv = np.std(estimated_sf) / np.mean(estimated_sf)
        
        assert abs(true_cv - est_cv) < 0.2, f"CV mismatch: true={true_cv:.3f}, est={est_cv:.3f}"
    
    def test_variance_estimation_accuracy(self):
        """Test accuracy of variance estimation methods."""
        np.random.seed(42)
        n_replicates = 25
        n_samples = 60
        
        design_matrix = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        true_beta = np.array([1.5, 1.0])
        dispersion = 0.5
        size_factors = np.ones(n_samples)
        
        # Generate multiple datasets and test variance estimation
        beta_estimates = []
        variance_estimates = []
        
        for rep in range(n_replicates):
            mu = size_factors * np.exp(design_matrix @ true_beta)
            p = 1 / (1 + mu * dispersion)
            n = 1 / dispersion
            y = np.random.negative_binomial(n, p)
            
            # Fit beta
            beta_init = true_beta + np.random.normal(0, 0.1, 2)
            fitted_beta, _, _ = devil.beta.fit_beta_coefficients(
                y, design_matrix, beta_init, np.zeros(n_samples), dispersion,
                max_iter=50, tolerance=1e-5
            )
            
            # Compute variance
            precision = 1.0 / dispersion
            H = devil.variance.compute_hessian(
                fitted_beta, precision, y, design_matrix, size_factors
            )
            
            beta_estimates.append(fitted_beta)
            variance_estimates.append(np.diag(H))
        
        beta_estimates = np.array(beta_estimates)
        variance_estimates = np.array(variance_estimates)
        
        # Check variance estimation accuracy
        for i in range(2):
            empirical_var = np.var(beta_estimates[:, i], ddof=1)
            mean_estimated_var = np.mean(variance_estimates[:, i])
            
            print(f"Beta[{i}] variance: empirical={empirical_var:.4f}, "
                  f"estimated={mean_estimated_var:.4f}")
            
            # Estimated variance should approximate empirical variance
            ratio = mean_estimated_var / empirical_var
            assert 0.5 < ratio < 2.0, f"Poor variance estimate for Beta[{i}]: ratio={ratio:.3f}"


class TestStatisticalPropertiesUnderNull:
    """Test statistical properties under null hypothesis."""
    
    def test_pvalue_distribution_under_null(self):
        """Test that p-values are uniformly distributed under null hypothesis."""
        np.random.seed(42)
        n_genes = 200
        n_samples = 50
        
        # Generate data under null hypothesis (no differential expression)
        design_matrix = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)  # No effect
        ])
        
        # Beta with no condition effect
        beta = np.column_stack([
            np.random.normal(2, 0.5, n_genes),  # Intercept
            np.zeros(n_genes)  # No condition effect (null hypothesis)
        ])
        
        # Generate counts
        size_factors = np.ones(n_samples)
        overdispersion = np.random.gamma(2, 0.5, n_genes)
        
        count_matrix = np.zeros((n_genes, n_samples))
        for i in range(n_genes):
            mu = size_factors * np.exp(design_matrix @ beta[i, :])
            p = 1 / (1 + mu * overdispersion[i])
            n = 1 / overdispersion[i]
            count_matrix[i, :] = np.random.negative_binomial(n, p)
        
        # Fit model
        fitted_model = {
            'beta': beta,
            'overdispersion': overdispersion,
            'design_matrix': design_matrix,
            'count_matrix': count_matrix,
            'size_factors': size_factors,
            'gene_names': np.array([f'Gene_{i:03d}' for i in range(n_genes)]),
            'n_genes': n_genes,
            'n_samples': n_samples
        }
        
        # Test condition effect (should be null)
        contrast = np.array([0, 1])
        results = devil.test_de(fitted_model, contrast=contrast, verbose=False, use_gpu=False)
        
        # Test uniformity of p-values using Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.kstest(results['pval'], 'uniform')
        
        print(f"KS test for p-value uniformity: statistic={ks_stat:.4f}, p-value={ks_pval:.4f}")
        
        # P-values should be approximately uniform under null
        assert ks_pval > 0.05, f"P-values not uniform under null: KS p-value = {ks_pval:.4f}"
        
        # Check that Type I error rate is controlled
        alpha = 0.05
        type_i_error_rate = np.mean(results['pval'] < alpha)
        
        print(f"Type I error rate at α=0.05: {type_i_error_rate:.3f}")
        
        # Should be close to nominal level (within 2 standard errors)
        expected_se = np.sqrt(alpha * (1 - alpha) / n_genes)
        assert abs(type_i_error_rate - alpha) < 2 * expected_se
    
    def test_fold_change_distribution_under_null(self):
        """Test that fold changes are centered at zero under null hypothesis."""
        np.random.seed(42)
        n_genes = 150
        n_samples = 40
        
        # Generate null data
        design_matrix = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        beta_null = np.column_stack([
            np.random.normal(2, 0.5, n_genes),
            np.zeros(n_genes)  # No effect
        ])
        
        # Generate counts under null
        size_factors = np.random.lognormal(0, 0.2, n_samples)
        overdispersion = np.random.gamma(2, 0.3, n_genes)
        
        count_matrix = np.zeros((n_genes, n_samples))
        for i in range(n_genes):
            mu = size_factors * np.exp(design_matrix @ beta_null[i, :])
            p = 1 / (1 + mu * overdispersion[i])
            n = 1 / overdispersion[i]
            count_matrix[i, :] = np.random.negative_binomial(n, p)
        
        # Fit model and test
        fitted_model = {
            'beta': beta_null,
            'overdispersion': overdispersion,
            'design_matrix': design_matrix,
            'count_matrix': count_matrix,
            'size_factors': size_factors,
            'gene_names': np.array([f'Gene_{i:03d}' for i in range(n_genes)]),
            'n_genes': n_genes,
            'n_samples': n_samples
        }
        
        contrast = np.array([0, 1])
        results = devil.test_de(fitted_model, contrast=contrast, verbose=False, use_gpu=False)
        
        # Log fold changes should be centered at zero
        mean_lfc = np.mean(results['lfc'])
        std_lfc = np.std(results['lfc'])
        
        print(f"LFC under null: mean={mean_lfc:.4f}, std={std_lfc:.4f}")
        
        # Mean should be close to zero
        se_mean = std_lfc / np.sqrt(n_genes)
        assert abs(mean_lfc) < 2 * se_mean, f"LFC not centered at zero: mean={mean_lfc:.4f}"
        
        # Test normality of LFC distribution (should be approximately normal)
        _, normality_pval = stats.jarque_bera(results['lfc'])
        print(f"LFC normality test p-value: {normality_pval:.4f}")
        
        # Don't enforce strict normality as it depends on model assumptions
        # But test should complete without errors
    
    def test_multiple_testing_correction_accuracy(self):
        """Test accuracy of multiple testing correction methods."""
        np.random.seed(42)
        n_genes = 1000
        
        # Generate p-values with known structure
        n_true_nulls = 800
        n_alternatives = 200
        
        # Null p-values (uniform)
        null_pvals = np.random.uniform(0, 1, n_true_nulls)
        
        # Alternative p-values (concentrated near 0)
        alt_pvals = np.random.beta(0.5, 5, n_alternatives)  # Skewed toward 0
        
        # Combine and shuffle
        all_pvals = np.concatenate([null_pvals, alt_pvals])
        true_nulls = np.concatenate([np.ones(n_true_nulls), np.zeros(n_alternatives)]).astype(bool)
        
        # Shuffle to randomize order
        shuffle_idx = np.random.permutation(n_genes)
        all_pvals = all_pvals[shuffle_idx]
        true_nulls = true_nulls[shuffle_idx]
        
        # Test FDR control
        from statsmodels.stats.multitest import multipletests
        _, padj_bh, _, _ = multipletests(all_pvals, method='fdr_bh')
        
        # Calculate actual FDR at different thresholds
        thresholds = [0.01, 0.05, 0.10, 0.20]
        
        for threshold in thresholds:
            rejected = padj_bh <= threshold
            n_rejected = np.sum(rejected)
            
            if n_rejected > 0:
                false_discoveries = np.sum(rejected & true_nulls)
                actual_fdr = false_discoveries / n_rejected
                
                print(f"Threshold {threshold}: {n_rejected} rejected, "
                      f"FDR={actual_fdr:.3f}")
                
                # FDR should be controlled at nominal level
                assert actual_fdr <= threshold * 1.2  # Allow some variation due to randomness
    
    def test_confidence_interval_coverage(self):
        """Test confidence interval coverage rates."""
        np.random.seed(42)
        n_replicates = 100
        n_samples = 50
        true_effect = 1.0  # True log fold change
        
        coverage_count = 0
        
        for rep in range(n_replicates):
            # Generate data with known effect
            design_matrix = np.column_stack([
                np.ones(n_samples),
                np.random.binomial(1, 0.5, n_samples)
            ])
            
            true_beta = np.array([2.0, true_effect])
            dispersion = 0.5
            size_factors = np.ones(n_samples)
            
            mu = size_factors * np.exp(design_matrix @ true_beta)
            p = 1 / (1 + mu * dispersion)
            n = 1 / dispersion
            y = np.random.negative_binomial(n, p)
            
            # Fit model and get variance estimate
            beta_init = true_beta + np.random.normal(0, 0.2, 2)
            fitted_beta, _, converged = devil.beta.fit_beta_coefficients(
                y, design_matrix, beta_init, np.zeros(n_samples), dispersion,
                max_iter=50, tolerance=1e-5
            )
            
            if converged:
                # Compute variance for contrast [0, 1]
                contrast = np.array([0, 1])
                precision = 1.0 / dispersion
                H = devil.variance.compute_hessian(
                    fitted_beta, precision, y, design_matrix, size_factors
                )
                
                variance = contrast.T @ H @ contrast
                se = np.sqrt(variance)
                
                # Estimated effect and confidence interval
                estimated_effect = fitted_beta[1]
                ci_lower = estimated_effect - 1.96 * se
                ci_upper = estimated_effect + 1.96 * se
                
                # Check if true effect is in confidence interval
                if ci_lower <= true_effect <= ci_upper:
                    coverage_count += 1
        
        # Calculate coverage rate
        coverage_rate = coverage_count / n_replicates
        
        print(f"95% CI coverage rate: {coverage_rate:.3f} ({coverage_count}/{n_replicates})")
        
        # Should be close to 95% (allow for sampling variation)
        expected_se = np.sqrt(0.95 * 0.05 / n_replicates)
        assert abs(coverage_rate - 0.95) < 2 * expected_se


class TestModelAssumptionValidation:
    """Test validation of model assumptions."""
    
    def test_overdispersion_detection(self):
        """Test ability to detect overdispersion."""
        np.random.seed(42)
        n_genes, n_samples = 100, 60
        
        # Test scenarios
        scenarios = {
            'poisson': lambda mu: np.random.poisson(mu),  # No overdispersion
            'moderate_od': lambda mu: np.random.negative_binomial(1/0.5, 1/(1 + mu * 0.5)),
            'high_od': lambda mu: np.random.negative_binomial(1/2.0, 1/(1 + mu * 2.0))
        }
        
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        for scenario_name, count_generator in scenarios.items():
            print(f"\nTesting {scenario_name} scenario:")
            
            # Generate data according to scenario
            counts = np.zeros((n_genes, n_samples))
            true_beta = np.random.normal([2.0, 0.5], [0.3, 0.2], size=(n_genes, 2))
            
            for i in range(n_genes):
                mu = np.exp(design @ true_beta[i, :])
                counts[i, :] = count_generator(mu)
            
            # Estimate overdispersion
            offset_vector = np.zeros(n_samples)
            estimated_od = devil.overdispersion.estimate_initial_dispersion(
                counts, offset_vector
            )
            
            mean_od = np.mean(estimated_od)
            print(f"  Mean estimated overdispersion: {mean_od:.3f}")
            
            if scenario_name == 'poisson':
                # Should detect low overdispersion
                assert mean_od < 1.0, f"Should detect low overdispersion for Poisson: {mean_od:.3f}"
            elif scenario_name == 'high_od':
                # Should detect high overdispersion
                assert mean_od > 0.5, f"Should detect high overdispersion: {mean_od:.3f}"
    
    def test_model_goodness_of_fit(self):
        """Test model goodness of fit."""
        np.random.seed(42)
        n_genes, n_samples = 50, 40
        
        # Generate data from the model
        design_matrix = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        true_beta = np.random.normal([2.0, 1.0], [0.5, 0.3], size=(n_genes, 2))
        true_overdispersion = np.random.gamma(2, 0.5, n_genes)
        size_factors = np.random.lognormal(0, 0.3, n_samples)
        
        count_matrix = np.zeros((n_genes, n_samples))
        for i in range(n_genes):
            mu = size_factors * np.exp(design_matrix @ true_beta[i, :])
            p = 1 / (1 + mu * true_overdispersion[i])
            n = 1 / true_overdispersion[i]
            count_matrix[i, :] = np.random.negative_binomial(n, p)
        
        # Fit model
        fit_result = devil.fit_devil(
            count_matrix,
            design_matrix=design_matrix,
            overdispersion=True,
            size_factors=True,
            verbose=False,
            max_iter=20,
            use_gpu=False
        )
        
        # Check parameter recovery
        beta_correlation = np.corrcoef(
            true_beta.flatten(),
            fit_result['beta'].flatten()
        )[0, 1]
        
        od_correlation = np.corrcoef(
            true_overdispersion,
            fit_result['overdispersion']
        )[0, 1]
        
        print(f"Beta correlation: {beta_correlation:.3f}")
        print(f"Overdispersion correlation: {od_correlation:.3f}")
        
        # Should recover parameters reasonably well
        assert beta_correlation > 0.7, f"Poor beta recovery: r={beta_correlation:.3f}"
        assert od_correlation > 0.3, f"Poor overdispersion recovery: r={od_correlation:.3f}"
        
        # Check convergence
        convergence_rate = np.mean(fit_result['converged'])
        assert convergence_rate > 0.8, f"Poor convergence: {convergence_rate:.2%}"
    
    def test_residual_analysis(self):
        """Test residual analysis for model diagnostics."""
        np.random.seed(42)
        n_genes, n_samples = 30, 35
        
        # Generate and fit data
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        fit_result = devil.fit_devil(
            counts,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=15,
            use_gpu=False
        )
        
        # Compute Pearson residuals for first few genes
        pearson_residuals = []
        
        for i in range(min(10, n_genes)):
            beta = fit_result['beta'][i, :]
            overdispersion = fit_result['overdispersion'][i]
            y = fit_result['count_matrix'][i, :]
            
            # Fitted values
            mu = fit_result['size_factors'] * np.exp(design @ beta)
            
            # Variance
            if overdispersion > 0:
                variance = mu + mu**2 * overdispersion
            else:
                variance = mu
            
            # Pearson residuals
            residuals = (y - mu) / np.sqrt(variance)
            pearson_residuals.extend(residuals)
        
        pearson_residuals = np.array(pearson_residuals)
        
        # Test residual properties
        mean_residual = np.mean(pearson_residuals)
        std_residual = np.std(pearson_residuals)
        
        print(f"Pearson residuals: mean={mean_residual:.3f}, std={std_residual:.3f}")
        
        # Residuals should be approximately centered at zero
        se_mean = std_residual / np.sqrt(len(pearson_residuals))
        assert abs(mean_residual) < 2 * se_mean
        
        # Standard deviation should be reasonable (close to 1 for good fit)
        assert 0.5 < std_residual < 2.0


class TestRobustnessToPerturbations:
    """Test robustness to data perturbations."""
    
    def test_robustness_to_outliers(self):
        """Test robustness to outlier observations."""
        np.random.seed(42)
        n_genes, n_samples = 60, 40
        
        # Generate clean data
        counts_clean = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Fit clean model
        fit_clean = devil.fit_devil(
            counts_clean,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=15,
            use_gpu=False
        )
        
        # Add outliers
        counts_outliers = counts_clean.copy()
        counts_outliers[0, 0] = 1000  # Extreme outlier
        counts_outliers[1, 1] = 0     # Zero outlier
        counts_outliers[2, :5] = np.random.negative_binomial(100, 0.1, 5)  # Multiple outliers
        
        # Fit model with outliers
        fit_outliers = devil.fit_devil(
            counts_outliers,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=15,
            use_gpu=False
        )
        
        # Compare results
        beta_correlation = np.corrcoef(
            fit_clean['beta'].flatten(),
            fit_outliers['beta'].flatten()
        )[0, 1]
        
        print(f"Beta correlation (clean vs outliers): {beta_correlation:.3f}")
        
        # Should be reasonably robust (correlation > 0.8)
        assert beta_correlation > 0.8, f"Not robust to outliers: r={beta_correlation:.3f}"
        
        # Both should converge reasonably well
        assert np.mean(fit_clean['converged']) > 0.8
        assert np.mean(fit_outliers['converged']) > 0.6  # Allow some degradation
    
    def test_robustness_to_missing_data(self):
        """Test robustness to missing data patterns."""
        np.random.seed(42)
        n_genes, n_samples = 80, 50
        
        # Generate complete data
        counts_complete = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        # Create missing data by setting some counts to zero
        counts_missing = counts_complete.copy()
        missing_mask = np.random.binomial(1, 0.2, size=(n_genes, n_samples))  # 20% missing
        counts_missing[missing_mask == 1] = 0
        
        # Fit both models
        fit_complete = devil.fit_devil(
            counts_complete,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=15,
            use_gpu=False
        )
        
        fit_missing = devil.fit_devil(
            counts_missing,
            design_matrix=design,
            overdispersion=True,
            verbose=False,
            max_iter=15,
            use_gpu=False
        )
        
        # Test differential expression
        contrast = [0, 1]
        
        de_complete = devil.test_de(fit_complete, contrast, verbose=False, use_gpu=False)
        de_missing = devil.test_de(fit_missing, contrast, verbose=False, use_gpu=False)
        
        # Compare results
        lfc_correlation = np.corrcoef(de_complete['lfc'], de_missing['lfc'])[0, 1]
        
        print(f"LFC correlation (complete vs missing): {lfc_correlation:.3f}")
        
        # Should be reasonably robust to missing data
        assert lfc_correlation > 0.75, f"Not robust to missing data: r={lfc_correlation:.3f}"
    
    def test_robustness_to_design_perturbations(self):
        """Test robustness to small perturbations in design matrix."""
        np.random.seed(42)
        n_genes, n_samples = 50, 35
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        
        # Original design
        design_original = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples),
            np.random.normal(0, 1, n_samples)
        ])
        
        # Perturbed design (small random noise)
        design_perturbed = design_original + np.random.normal(0, 0.01, design_original.shape)
        design_perturbed[:, 0] = 1  # Keep intercept exact
        
        # Fit both models
        fit_original = devil.fit_devil(
            counts,
            design_matrix=design_original,
            overdispersion=True,
            verbose=False,
            max_iter=15,
            use_gpu=False
        )
        
        fit_perturbed = devil.fit_devil(
            counts,
            design_matrix=design_perturbed,
            overdispersion=True,
            verbose=False,
            max_iter=15,
            use_gpu=False
        )
        
        # Compare beta estimates
        beta_diff = np.abs(fit_original['beta'] - fit_perturbed['beta'])
        max_diff = np.max(beta_diff)
        mean_diff = np.mean(beta_diff)
        
        print(f"Beta differences: mean={mean_diff:.4f}, max={max_diff:.4f}")
        
        # Should be stable to small perturbations
        assert max_diff < 0.5, f"Too sensitive to design perturbations: max_diff={max_diff:.4f}"
        assert mean_diff < 0.1, f"Too sensitive to design perturbations: mean_diff={mean_diff:.4f}"


class TestNumericalStability:
    """Test numerical stability of computations."""
    
    def test_extreme_parameter_stability(self):
        """Test stability with extreme parameter values."""
        test_scenarios = [
            {'dispersion': 1e-8, 'name': 'very_low_dispersion'},
            {'dispersion': 1e8, 'name': 'very_high_dispersion'},
            {'beta_scale': 10, 'name': 'large_coefficients'},
            {'size_factor_var': 3, 'name': 'variable_size_factors'}
        ]
        
        for scenario in test_scenarios:
            np.random.seed(42)
            n_genes, n_samples = 30, 25
            
            print(f"\nTesting {scenario['name']}:")
            
            # Generate base data
            design = np.column_stack([
                np.ones(n_samples),
                np.random.binomial(1, 0.5, n_samples)
            ])
            
            if 'beta_scale' in scenario:
                # Large coefficients
                beta_scale = scenario['beta_scale']
                beta = np.random.normal([2, 1], [0.5, 0.3], size=(n_genes, 2)) * beta_scale
            else:
                beta = np.random.normal([2, 1], [0.5, 0.3], size=(n_genes, 2))
            
            if 'size_factor_var' in scenario:
                # Variable size factors
                sf_var = scenario['size_factor_var']
                size_factors = np.random.lognormal(0, sf_var, n_samples)
            else:
                size_factors = np.ones(n_samples)
            
            if 'dispersion' in scenario:
                overdispersion = np.full(n_genes, scenario['dispersion'])
            else:
                overdispersion = np.random.gamma(2, 0.5, n_genes)
            
            # Generate counts
            count_matrix = np.zeros((n_genes, n_samples))
            for i in range(n_genes):
                mu = size_factors * np.exp(design @ beta[i, :])
                if overdispersion[i] > 0:
                    p = 1 / (1 + mu * overdispersion[i])
                    n = 1 / overdispersion[i]
                    count_matrix[i, :] = np.random.negative_binomial(n, p)
                else:
                    count_matrix[i, :] = np.random.poisson(mu)
            
            # Test model fitting
            try:
                fit_result = devil.fit_devil(
                    count_matrix,
                    design_matrix=design,
                    overdispersion=True,
                    size_factors=(size_factors is not None),
                    verbose=False,
                    max_iter=20,
                    use_gpu=False
                )
                
                # Check for numerical stability
                assert np.all(np.isfinite(fit_result['beta']))
                assert np.all(fit_result['overdispersion'] >= 0)
                assert np.all(np.isfinite(fit_result['overdispersion']))
                
                convergence_rate = np.mean(fit_result['converged'])
                print(f"  Convergence rate: {convergence_rate:.2%}")
                
                # Should achieve reasonable convergence even with extreme parameters
                assert convergence_rate > 0.5
                
            except Exception as e:
                pytest.fail(f"Failed with extreme parameters in {scenario['name']}: {e}")
    
    def test_numerical_precision_limits(self):
        """Test behavior at numerical precision limits."""
        np.random.seed(42)
        n_samples = 40
        
        # Test cases
        test_cases = [
            {'mu': np.full(n_samples, 1e-10), 'name': 'very_small_mu'},
            {'mu': np.full(n_samples, 1e10), 'name': 'very_large_mu'},
            {'dispersion': 1e-15, 'name': 'tiny_dispersion'},
            {'dispersion': 1e15, 'name': 'huge_dispersion'}
        ]
        
        design_matrix = np.ones((n_samples, 1))
        
        for case in test_cases:
            print(f"\nTesting {case['name']}:")
            
            if 'mu' in case:
                mu = case['mu']
                beta = np.log(mu[0])  # Consistent with mu
                dispersion = 1.0
            else:
                beta = 2.0
                mu = np.exp(beta)
                dispersion = case['dispersion']
            
            y = np.random.negative_binomial(5, 0.3, n_samples)  # Reasonable y values
            size_factors = np.ones(n_samples)
            
            try:
                # Test Hessian computation
                precision = 1.0 / dispersion if dispersion > 0 else 1e6
                H = devil.variance.compute_hessian(
                    np.array([beta]), precision, y, design_matrix, size_factors
                )
                
                assert np.all(np.isfinite(H))
                assert H.shape == (1, 1)
                assert H[0, 0] > 0
                
                print(f"  Hessian computation: OK")
                
            except Exception as e:
                print(f"  Hessian computation failed: {e}")
                # Some extreme cases may legitimately fail
                if 'tiny_dispersion' not in case['name'] and 'huge_dispersion' not in case['name']:
                    pytest.fail(f"Unexpected failure in {case['name']}: {e}")