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
  clusters = NULL,
  pval_adjust_method = "BH",
  max_lfc = 10,
  BPPARAM = BiocParallel::SerialParam()
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

- clusters:

  Numeric vector or factor. Sample cluster assignments for robust
  variance estimation. Length must match number of samples. Default:
  NULL

- pval_adjust_method:

  Character. Method for p-value adjustment. Passed to stats::p.adjust().
  Common choices:

  - "BH": Benjamini-Hochberg (default)

  - "bonferroni": Bonferroni correction

  - "holm": Holm's step-down method

- max_lfc:

  Numeric. Maximum absolute log2 fold change to report. Larger values
  are capped at ±max_lfc. Default: 10

- BPPARAM:

  A
  [`BiocParallelParam`](https://rdrr.io/pkg/BiocParallel/man/BiocParallelParam-class.html)
  object controlling parallel evaluation. Default:
  [`BiocParallel::SerialParam()`](https://rdrr.io/pkg/BiocParallel/man/SerialParam-class.html).

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
    x             = counts,
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
#>   name   pval adj_pval    lfc
#>   <chr> <dbl>    <dbl>  <dbl>
#> 1 gene1 0.702    0.924  9.37 
#> 2 gene2 0.829    0.924  0.304
#> 3 gene4 0.582    0.924 -8.97 
#> 4 gene5 0.869    0.924  0.269
#> 5 gene6 0.564    0.924 -3.71 
#> 6 gene7 0.484    0.924  4.53 

## Example: clustered (patient-aware) variance (sandwich SE)

patient <- factor(rep(c("P1", "P2"), each = 5))

res_clustered <- test_de(
    fit,
    contrast = c(A = 1, B = -1),
    clusters = patient
)

head(res_clustered[order(res_clustered$adj_pval), ])
#> # A tibble: 6 × 4
#>   name   pval adj_pval    lfc
#>   <chr> <dbl>    <dbl>  <dbl>
#> 1 gene1 0.702    0.924  9.37 
#> 2 gene2 0.829    0.924  0.304
#> 3 gene4 0.582    0.924 -8.97 
#> 4 gene5 0.869    0.924  0.269
#> 5 gene6 0.564    0.924 -3.71 
#> 6 gene7 0.484    0.924  4.53 
```
