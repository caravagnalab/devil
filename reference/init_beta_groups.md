# Initialize Beta Coefficients Using Groups

Initializes regression coefficients based on group-wise means of
normalized counts.

## Usage

``` r
init_beta_groups(y, groups, offset_matrix)
```

## Arguments

- y:

  Count data matrix

- groups:

  Vector indicating group membership

- offset_matrix:

  Matrix of offset values

## Value

Matrix of initial beta coefficients by group
