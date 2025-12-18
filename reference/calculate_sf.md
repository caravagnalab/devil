# Calculate Size Factors for Count Data Normalization

Computes normalization factors for count data using one of three
methods: geometric mean normalization (normed_sum), psi-normalization
(psinorm), or edgeR's TMM with singleton pairing (edgeR). Handles edge
cases like all-zero columns and matrices with too few rows.

## Usage

``` r
calculate_sf(Y, method = c("normed_sum", "psinorm", "edgeR"), verbose = FALSE)
```

## Arguments

- Y:

  Count data matrix with genes in rows and samples in columns

- method:

  Character string specifying the normalization method. One of:

  - `"normed_sum"` (default): Geometric mean normalization based on
    library sizes

  - `"psinorm"`: Psi-normalization using Pareto distribution MLE

  - `"edgeR"`: edgeR's TMM with singleton pairing method

- verbose:

  Logical indicating whether to print progress messages

## Value

Numeric vector of size factors, one per sample (column). Size factors
are scaled to have a geometric mean of 1.

## Details

Size factors are used to normalize count data for differences in
sequencing depth and RNA composition across samples. The function will
return a vector of 1s if the input matrix has only one row.

The `"normed_sum"` method computes size factors as the column sums
divided by their geometric mean.

The `"psinorm"` method uses maximum likelihood estimation of Pareto
distribution parameters to compute size factors robust to highly
variable genes.

The `"edgeR"` method requires the edgeR package to be installed and uses
the TMM (trimmed mean of M-values) method with singleton pairing.
