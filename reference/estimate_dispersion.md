# Estimate Dispersion Parameters for Count Matrix

Calculates per-gene dispersion estimates for a count matrix using a
method of moments approach. Handles edge cases by setting a default high
dispersion value.

## Usage

``` r
estimate_dispersion(count_matrix, offset_vector)
```

## Arguments

- count_matrix:

  Matrix of count data with genes in rows and samples in columns

- offset_vector:

  Vector of offset values for normalization

## Value

Vector of dispersion estimates, one per gene
