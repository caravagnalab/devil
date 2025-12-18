# Compute Sandwich Estimator for Statistical Model

Calculates the sandwich estimator for robust covariance estimation,
particularly useful in clustered or heteroskedastic data scenarios.

## Usage

``` r
compute_sandwich(
  design_matrix,
  y,
  beta,
  overdispersion,
  size_factors,
  clusters
)
```

## Arguments

- design_matrix:

  Matrix of predictor variables

- y:

  Vector of response variables

- beta:

  Vector of coefficient estimates

- overdispersion:

  Scalar overdispersion parameter

- size_factors:

  Vector of normalization factors for each sample

- clusters:

  Vector indicating cluster membership

## Value

Matrix containing the sandwich estimator
