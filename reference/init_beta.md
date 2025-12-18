# Initialize Beta Coefficients Using Design Matrix

Initializes regression coefficients using QR decomposition of the design
matrix and normalized log counts.

## Usage

``` r
init_beta(y, design_matrix, offset_matrix)
```

## Arguments

- y:

  Count data matrix

- design_matrix:

  Matrix of predictor variables

- offset_matrix:

  Matrix of offset values

## Value

Matrix of initial beta coefficients
