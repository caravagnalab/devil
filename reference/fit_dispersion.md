# Fit Dispersion Parameter for Negative Binomial Model

Estimates the dispersion parameter in a negative binomial GLM using
maximum likelihood estimation. Implementation from the glmGamPoi
package.

## Usage

``` r
fit_dispersion(
  beta,
  model_matrix,
  y,
  offset_matrix,
  tolerance,
  max_iter,
  do_cox_reid_adjustment = TRUE
)
```

## Arguments

- beta:

  Vector of coefficient estimates

- model_matrix:

  Design matrix of predictor variables

- y:

  Vector of response variables (counts)

- offset_matrix:

  Matrix of offset values

- tolerance:

  Convergence tolerance for optimization

- max_iter:

  Maximum number of iterations for optimization

- do_cox_reid_adjustment:

  Logical indicating whether to apply Cox-Reid adjustment

## Value

Estimated dispersion parameter

## Details

This implementation comes from the glmGamPoi package:
https://github.com/const-ae/glmGamPoi

Original publication: Ahlmann-Eltze, C., Huber, W. (2020). glmGamPoi:
Fitting Gamma-Poisson Generalized Linear Models on Single Cell Count
Data. Bioinformatics. https://doi.org/10.1093/bioinformatics/btaa1009
