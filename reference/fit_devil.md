# Fit Statistical Model for Count Data

Fits a statistical model to count data, particularly designed for
RNA-seq data analysis. The function estimates regression coefficients
(beta), gene-wise overdispersion parameters, and normalizes data using
size factors. It supports both CPU and (optionally) GPU-based
computation with parallel processing capabilities.

## Usage

``` r
fit_devil(
  x,
  design_matrix,
  assay.type = "counts",
  clusters = NULL,
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
  BPPARAM = BiocParallel::SerialParam()
)
```

## Arguments

- x:

  A numeric matrix of count data (genes x samples), a
  [`SummarizedExperiment`](https://rdrr.io/pkg/SummarizedExperiment/man/SummarizedExperiment-class.html),
  or a
  [`SingleCellExperiment`](https://rdrr.io/pkg/SingleCellExperiment/man/SingleCellExperiment.html)
  object. Rows represent genes/features, columns represent
  samples/cells. When `x` is an SE/SCE object, the assay named by
  `assay.type` is used.

- design_matrix:

  A numeric matrix of predictor variables (samples x predictors). Each
  row corresponds to a sample, each column to a predictor variable. Must
  have `nrow(design_matrix) == ncol(x)`.

- assay.type:

  Character. Name of the assay to extract when `x` is a
  `SummarizedExperiment` or `SingleCellExperiment`. Default: `"counts"`.

- clusters:

  Vector of cluster or patient identifiers (one element per
  cell/sample). Cells belonging to the same cluster must be contiguous —
  use
  [`group_data()`](https://caravagnalab.github.io/devil/reference/group_data.md)
  to sort them first. When provided, enables clustered sandwich
  covariance estimation. Default: `NULL`.

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

  Logical. Whether to initialize betas in a rough but extremely fast
  way. Default: `FALSE`.

- offset:

  Numeric scalar. Value used when computing the offset vector to avoid
  numerical issues with zero counts. Default: `0`.

- size_factors:

  Character string, numeric vector, or `NULL`. Controls normalization
  for sequencing depth. Options are:

  - `NULL` (default): No normalization (all size factors set to 1)

  - `"normed_sum"`: Geometric mean normalization

  - `"psinorm"`: Psi-normalization

  - `"edgeR"`: edgeR TMM method

  - A numeric vector of length `ncol(x)`: precomputed size factors used
    directly without further normalization

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

- BPPARAM:

  A
  [`BiocParallelParam`](https://rdrr.io/pkg/BiocParallel/man/BiocParallelParam-class.html)
  object controlling parallel evaluation. Default:
  [`BiocParallel::SerialParam()`](https://rdrr.io/pkg/BiocParallel/man/SerialParam-class.html).

## Value

A list containing:

- beta:

  Matrix of fitted coefficients (genes x predictors).

- beta_sandwiches_null:

  List of per-gene Hessian inverse matrices (\\H^{-1}\\).

- beta_sandwiches:

  List of per-gene clustered sandwich matrices (\\H^{-1} M H^{-1} \times
  n\\samples\\); `NULL` entries when no `clusters` are provided.

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

  List of used parameter values (`max_iter`, `tolerance`).

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
factors are set to 1. If `size_factors` is a numeric vector of
precomputed size factors (one per sample), they will be used directly.

## Overdispersion Strategies

The `overdispersion` argument controls how gene-wise overdispersion is
handled:

- `"old"` or `"MLE"`: Overdispersion is fit via the original (legacy) NB
  MLE-based procedure (with Cox-Reid adjustment inside
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

# Fit from a raw matrix
fit <- fit_devil(
    x             = counts,
    design_matrix = design,
    size_factors  = "normed_sum"
)

## Example: fit from a SingleCellExperiment object
library(SingleCellExperiment)
#> Loading required package: SummarizedExperiment
#> Loading required package: MatrixGenerics
#> Loading required package: matrixStats
#> 
#> Attaching package: ‘MatrixGenerics’
#> The following objects are masked from ‘package:matrixStats’:
#> 
#>     colAlls, colAnyNAs, colAnys, colAvgsPerRowSet, colCollapse,
#>     colCounts, colCummaxs, colCummins, colCumprods, colCumsums,
#>     colDiffs, colIQRDiffs, colIQRs, colLogSumExps, colMadDiffs,
#>     colMads, colMaxs, colMeans2, colMedians, colMins, colOrderStats,
#>     colProds, colQuantiles, colRanges, colRanks, colSdDiffs, colSds,
#>     colSums2, colTabulates, colVarDiffs, colVars, colWeightedMads,
#>     colWeightedMeans, colWeightedMedians, colWeightedSds,
#>     colWeightedVars, rowAlls, rowAnyNAs, rowAnys, rowAvgsPerColSet,
#>     rowCollapse, rowCounts, rowCummaxs, rowCummins, rowCumprods,
#>     rowCumsums, rowDiffs, rowIQRDiffs, rowIQRs, rowLogSumExps,
#>     rowMadDiffs, rowMads, rowMaxs, rowMeans2, rowMedians, rowMins,
#>     rowOrderStats, rowProds, rowQuantiles, rowRanges, rowRanks,
#>     rowSdDiffs, rowSds, rowSums2, rowTabulates, rowVarDiffs, rowVars,
#>     rowWeightedMads, rowWeightedMeans, rowWeightedMedians,
#>     rowWeightedSds, rowWeightedVars
#> Loading required package: GenomicRanges
#> Loading required package: stats4
#> Loading required package: BiocGenerics
#> Loading required package: generics
#> 
#> Attaching package: ‘generics’
#> The following objects are masked from ‘package:base’:
#> 
#>     as.difftime, as.factor, as.ordered, intersect, is.element, setdiff,
#>     setequal, union
#> 
#> Attaching package: ‘BiocGenerics’
#> The following objects are masked from ‘package:stats’:
#> 
#>     IQR, mad, sd, var, xtabs
#> The following objects are masked from ‘package:base’:
#> 
#>     Filter, Find, Map, Position, Reduce, anyDuplicated, aperm, append,
#>     as.data.frame, basename, cbind, colnames, dirname, do.call,
#>     duplicated, eval, evalq, get, grep, grepl, is.unsorted, lapply,
#>     mapply, match, mget, order, paste, pmax, pmax.int, pmin, pmin.int,
#>     rank, rbind, rownames, sapply, saveRDS, table, tapply, unique,
#>     unsplit, which.max, which.min
#> Loading required package: S4Vectors
#> 
#> Attaching package: ‘S4Vectors’
#> The following object is masked from ‘package:utils’:
#> 
#>     findMatches
#> The following objects are masked from ‘package:base’:
#> 
#>     I, expand.grid, unname
#> Loading required package: IRanges
#> Loading required package: Seqinfo
#> Loading required package: Biobase
#> Welcome to Bioconductor
#> 
#>     Vignettes contain introductory material; view with
#>     'browseVignettes()'. To cite Bioconductor, see
#>     'citation("Biobase")', and for packages 'citation("pkgname")'.
#> 
#> Attaching package: ‘Biobase’
#> The following object is masked from ‘package:MatrixGenerics’:
#> 
#>     rowMedians
#> The following objects are masked from ‘package:matrixStats’:
#> 
#>     anyMissing, rowMedians
sce <- SingleCellExperiment(assays = list(counts = counts))
fit_sce <- fit_devil(
    x             = sce,
    design_matrix = design,
    assay.type    = "counts"
)
```
