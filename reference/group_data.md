# Group and Reorder Data by Clusters

Rearranges the input count matrix and design matrix so that observations
belonging to the same cluster (e.g., patient) are contiguous. This is a
required preprocessing step for block-wise variance estimation to ensure
the C++ backend can iterate through clusters without jumping across
memory.

## Usage

``` r
group_data(input_matrix, design_matrix, clusters)
```

## Arguments

- input_matrix:

  A matrix-like object (e.g., sparse or dense count matrix) where
  columns represent individual cells/observations.

- design_matrix:

  A matrix of predictor variables where rows represent individual
  cells/observations.

- clusters:

  A vector of cluster or patient identifiers.

## Value

A named list containing three elements:

- `input_matrix`: The reordered count matrix.

- `design_matrix`: The reordered design matrix.

- `clusters`: The reordered cluster vector, converted to numeric indices
  based on the order of appearance.

## Details

The function converts the `clusters` vector into a factor based on its
unique levels in order of appearance, then sorts all inputs based on
these levels.

## Examples

``` r
set.seed(1)
counts <- matrix(rnbinom(500, mu = 3, size = 1), nrow = 50, ncol = 10)
rownames(counts) <- paste0("gene", seq_len(nrow(counts)))
design <- model.matrix(~ 0 + factor(rep(c("A", "B"), each = 5)))
patient <- rep(c("P1", "P2"), times = 5)
grouped <- group_data(counts, design, patient)
str(grouped)
#> List of 3
#>  $ input_matrix : num [1:50, 1:10] 1 0 1 0 1 2 1 2 1 0 ...
#>   ..- attr(*, "dimnames")=List of 2
#>   .. ..$ : chr [1:50] "gene1" "gene2" "gene3" "gene4" ...
#>   .. ..$ : NULL
#>  $ design_matrix: num [1:10, 1:2] 1 1 1 0 0 1 1 0 0 0 ...
#>   ..- attr(*, "dimnames")=List of 2
#>   .. ..$ : chr [1:10] "1" "3" "5" "7" ...
#>   .. ..$ : chr [1:2] "factor(rep(c(\"A\", \"B\"), each = 5))A" "factor(rep(c(\"A\", \"B\"), each = 5))B"
#>  $ clusters     : num [1:10] 1 1 1 1 1 2 2 2 2 2
```
