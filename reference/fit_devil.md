# Fit Statistical Model for Count Data

Fits a statistical model to count data, particularly designed for
RNA-seq data analysis. The function estimates regression coefficients
(beta), gene-wise overdispersion parameters, and normalizes data using
size factors. It supports both CPU and (optionally) GPU-based
computation with parallel processing capabilities.

## Usage

``` r
fit_devil(
  input_matrix,
  design_matrix,
  overdispersion = "MOM",
  init_overdispersion = NULL,
  init_beta_rough = FALSE,
  offset = 0,
  size_factors = NULL,
  verbose = FALSE,
  max_iter = 200,
  tolerance = 0.001,
  CUDA = FALSE,
  batch_size = 1024L,
  parallel.cores = 1
)
```

## Arguments

- input_matrix:

  A numeric matrix of count data (genes × samples). Rows represent
  genes/features, columns represent samples/cells.

- design_matrix:

  A numeric matrix of predictor variables (samples × predictors). Each
  row corresponds to a sample, each column to a predictor variable. Must
  have `nrow(design_matrix) == ncol(input_matrix)`.

- overdispersion:

  Character or logical. Strategy for estimating overdispersion: one of
  `"new"`, `"I"`, `"old"`, `"MLE"`, `"MOM"`, or `FALSE` to disable
  overdispersion fitting (Poisson model). Default: `"MOM"`.

- init_overdispersion:

  Numeric scalar or `NULL`. Initial value for the overdispersion
  parameter used as a starting point for the iterative procedures. If
  `NULL`, an initial value is estimated from the data via
  [`estimate_dispersion()`](https://caravagnalab.github.io/devil/reference/estimate_dispersion.md).
  Recommended value if specified: `100`. Default: `NULL`.

- init_beta_rough:

  Logial. Whether to initialize betas in a rough but extremely fast way.
  Default: `FALSE`.

- offset:

  Numeric scalar. Value used when computing the offset vector to avoid
  numerical issues with zero counts. Default: `0`.

- size_factors:

  Character string or `NULL`. Method for computing normalization factors
  to account for different sequencing depths. Options are:

  - `NULL` (default): No normalization (all size factors set to 1)

  - `"normed_sum"`: Geometric mean normalization

  - `"psinorm"`: Psi-normalization

  - `"edgeR"`: edgeR TMM method

- verbose:

  Logical. Whether to print progress messages during execution. Default:
  `FALSE`.

- max_iter:

  Integer. Maximum number of iterations for parameter optimization (both
  beta and overdispersion routines). Default: `200`.

- tolerance:

  Numeric. Convergence criterion for parameter optimization. Default:
  `1e-3`.

- CUDA:

  Logical. Whether to use GPU acceleration (requires CUDA support and a
  compiled `beta_fit_gpu()` implementation). Default: `FALSE`.

- batch_size:

  Integer. Number of genes to process per batch in GPU mode. Only
  relevant if `CUDA = TRUE`. Default: `1024`.

- parallel.cores:

  Integer or `NULL`. Number of CPU cores for parallel processing with
  [`parallel::mclapply`](https://rdrr.io/r/parallel/mclapply.html). If
  `NULL`, uses all available cores. Default: `1`.

## Value

A list containing:

- beta:

  Matrix of fitted coefficients (genes × predictors).

- overdispersion:

  Vector of fitted overdispersion parameters (one per gene).

- iterations:

  List with elements `beta_iters` and `theta_iters` giving the number of
  iterations used for each gene.

- size_factors:

  Vector of computed size factors (one per sample).

- offset_vector:

  Vector of offset values used in the model (length = number of
  samples).

- design_matrix:

  Input design matrix (as provided, possibly coerced to numeric matrix).

- input_matrix:

  Input count matrix (as provided, possibly coerced to numeric matrix).

- input_parameters:

  List of used parameter values (`max_iter`, `tolerance`,
  `parallel.cores`).

## Details

The function implements a negative binomial regression model with the
following steps:

1.  Computes size factors for data normalization (if requested)

2.  Initializes model parameters including beta coefficients and
    overdispersion

3.  Fits the regression coefficients using either CPU (parallel) or GPU
    computation

4.  Optionally fits/updates overdispersion parameters using one of
    several strategies

The model fitting process uses iterative optimization with configurable
convergence criteria and maximum iterations. For large datasets, the GPU
implementation processes genes in batches for improved memory
efficiency.

## Size Factor Methods

Three normalization methods are available when `size_factors` is a
character string:

- `"normed_sum"` (default): Geometric mean normalization based on
  library sizes. Fast and works well for most datasets.

- `"psinorm"`: Psi-normalization using Pareto distribution MLE. More
  robust to highly variable genes.

- `"edgeR"`: edgeR's TMM with singleton pairing method. Requires the
  edgeR package from Bioconductor.

If `size_factors = NULL`, no normalization is performed and all size
factors are set to 1.

## Overdispersion Strategies

The `overdispersion` argument controls how gene-wise overdispersion is
handled:

- `"old"` or `"MLE"`: Overdispersion is fit via the original (legacy) NB
  MLE-based procedure (with Cox–Reid adjustment inside
  [`fit_dispersion()`](https://caravagnalab.github.io/devil/reference/fit_dispersion.md)).

- `"new"` or `"I"`: Overdispersion is fit via the new iterative NB
  routine implemented in `fit_overdispersion_cppp()`, typically faster
  and more stable for large single-cell datasets.

- `"MOM"`: Overdispersion is estimated using a method-of-moments
  approach via `estimate_mom_dispersion_cpp()`, which is cheap and
  provides a rough dispersion estimate.

- `FALSE`: Disable overdispersion fitting and use a Poisson model
  (overdispersion fixed to 0).

## Examples

``` r
## Example: fit a simple two-group model
set.seed(1)

# Simulate a small counts matrix (genes x cells)
counts <- matrix(
    rnbinom(1000, mu = 0.2, size = 1),
    nrow = 100, ncol = 10
)
rownames(counts) <- paste0("gene", seq_len(nrow(counts)))

# Two-group design (no intercept)
group <- factor(rep(c("A", "B"), each = 5))
design <- model.matrix(~ 0 + group)
colnames(design) <- levels(group)

# Fit the model
fit <- fit_devil(
    input_matrix  = counts,
    design_matrix = design,
    size_factors  = "normed_sum",
    verbose       = TRUE
)
#> Compute size factors
#> Calculating size factors using method: normed_sum
#> Size factors calculated successfully.
#> Range: [0.53, 1.4311]
#> Initialize theta
#> Initialize beta
#> Fitting beta coefficients
#> Fit overdispersion (mode = MOM)
```
