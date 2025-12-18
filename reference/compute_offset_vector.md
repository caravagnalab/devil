# Compute Offset Matrix for Statistical Model

Creates an offset matrix incorporating base offsets and optional size
factors for model fitting.

## Usage

``` r
compute_offset_vector(off, Y, size_factors)
```

## Arguments

- off:

  Base offset value

- Y:

  Count data matrix with genes in rows and samples in columns

- size_factors:

  Optional vector of size factors for normalization

## Value

Matrix of offset values for each gene-sample combination
