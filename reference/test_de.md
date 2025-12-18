# Test for Differential Expression

Performs statistical testing for differential expression using results
from a fitted devil model. Supports both standard and robust (clustered)
variance estimation, with multiple testing correction and customizable
fold change thresholds.

## Usage

``` r
test_de(
  devil.fit,
  contrast,
  pval_adjust_method = "BH",
  max_lfc = 10,
  clusters = NULL,
  parallel.cores = 1
)
```

## Arguments

- devil.fit:

  A fitted model object from fit_devil(). Must contain beta
  coefficients, design matrix, and overdispersion parameters.

- contrast:

  Numeric vector or matrix specifying the comparison of interest. Length
  must match number of coefficients in the model. For example, c(0, 1,
  -1) tests difference between second and third coefficient.

- pval_adjust_method:

  Character. Method for p-value adjustment. Passed to stats::p.adjust().
  Common choices:

  - "BH": Benjamini-Hochberg (default)

  - "bonferroni": Bonferroni correction

  - "holm": Holm's step-down method

- max_lfc:

  Numeric. Maximum absolute log2 fold change to report. Larger values
  are capped at ±max_lfc. Default: 10

- clusters:

  Numeric vector or factor. Sample cluster assignments for robust
  variance estimation. Length must match number of samples. Default:
  NULL

- parallel.cores:

  Integer or NULL. Number of CPU cores for parallel processing. If NULL,
  uses all available cores. Default: 1

## Value

A tibble with columns:

- name:

  Character. Gene identifiers from input data

- pval:

  Numeric. Raw p-values from statistical tests

- adj_pval:

  Numeric. Adjusted p-values after multiple testing correction

- lfc:

  Numeric. Log2 fold changes, capped at ±max_lfc

## Details

The function implements the following analysis pipeline:

1.  Calculates log fold changes using contrast vectors/matrices

2.  Computes test statistics using either standard or robust variance
    estimation

3.  Calculates p-values using t-distribution with appropriate degrees of
    freedom

4.  Adjusts p-values for multiple testing

5.  Applies fold change thresholding

The variance estimation can account for sample clustering (e.g.,
multiple samples from the same patient) using a sandwich estimator for
robust inference.

## Examples

``` r
## Example: test_de() on a simple two-group comparison
set.seed(1)

# Simulate counts (genes x cells)
counts <- matrix(rnbinom(1000, mu = 0.2, size = 1), nrow = 100, ncol = 10)
rownames(counts) <- paste0("gene", seq_len(nrow(counts)))

# Two-group design (no intercept)
group <- factor(rep(c("A", "B"), each = 5))
design <- model.matrix(~ 0 + group)
colnames(design) <- levels(group)

# Fit model
fit <- fit_devil(
    input_matrix  = counts,
    design_matrix = design,
    size_factors  = "normed_sum"
)

# Test A vs B (contrast = +1*A -1*B)
res <- test_de(
    fit,
    contrast = c(A = 1, B = -1)
)
head(res[order(res$adj_pval), ])
#> # A tibble: 6 × 4
#>   name        pval adj_pval    lfc
#>   <chr>      <dbl>    <dbl>  <dbl>
#> 1 gene4  0.0000194  0.00125  -8.97
#> 2 gene86 0.0000251  0.00125   8.74
#> 3 gene63 0.0000488  0.00139   7.20
#> 4 gene70 0.0000621  0.00139 -10   
#> 5 gene75 0.0000696  0.00139   8.46
#> 6 gene18 0.0000850  0.00142   8.53

## Example: clustered (patient-aware) variance (sandwich SE)

patient <- factor(rep(c("P1", "P2"), each = 5))

res_clustered <- test_de(
    fit,
    contrast = c(A = 1, B = -1),
    clusters = patient
)
#> Converting clusters to numeric factors

head(res_clustered[order(res_clustered$adj_pval), ])
#> # A tibble: 6 × 4
#>   name       pval adj_pval    lfc
#>   <chr>     <dbl>    <dbl>  <dbl>
#> 1 gene1  0.00179    0.0336   9.37
#> 2 gene4  0.00232    0.0336  -8.97
#> 3 gene18 0.00325    0.0336   8.53
#> 4 gene57 0.00161    0.0336  -9.54
#> 5 gene65 0.00155    0.0336   9.60
#> 6 gene70 0.000731   0.0336 -10   
```
