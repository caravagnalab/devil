# Create Volcano Plot for Differential Expression Results

Generates a customizable volcano plot visualizing differential
expression results, highlighting significant genes based on both fold
change and statistical significance. The plot supports various
customization options including color schemes, point sizes, and gene
labeling.

## Usage

``` r
plot_volcano(
  devil.res,
  lfc_cut = 1,
  pval_cut = 0.05,
  labels = TRUE,
  colors = c("gray", "forestgreen", "steelblue", "indianred"),
  color_alpha = 0.7,
  point_size = 1,
  center = TRUE,
  title = "Volcano plot"
)
```

## Arguments

- devil.res:

  A tibble from test_de() containing columns:

  - name: Gene identifiers

  - adj_pval: Adjusted p-values

  - lfc: Log2 fold changes

- lfc_cut:

  Numeric. Absolute log2 fold change threshold for significance.
  Default: 1

- pval_cut:

  Numeric. Adjusted p-value threshold for significance. Default: 0.05

- labels:

  Logical. Whether to label genes meeting both significance criteria.
  Default: TRUE

- colors:

  Character vector of length 4 specifying colors for:

  1.  Non-significant genes

  2.  Fold-change significant only

  3.  P-value significant only

  4.  Both significant Default: c("gray", "forestgreen", "steelblue",
      "indianred")

- color_alpha:

  Numeric between 0 and 1. Transparency level for points. Default: 0.7

- point_size:

  Numeric. Size of plotting points. Default: 1

- center:

  Logical. Whether to center the x-axis at zero. Default: TRUE

- title:

  Character. Plot title. Default: "Volcano plot"

## Value

A ggplot2 object containing the volcano plot.

## Details

The function creates a scatter plot with:

- X-axis: Log2 fold change

- Y-axis: -Log10 adjusted p-value Points are colored based on
  significance categories:

1.  Non-significant: Neither p-value nor fold change threshold met

2.  LFC significant: Only fold change threshold met

3.  P-value significant: Only p-value threshold met

4.  Both significant: Both thresholds met

The plot includes dashed lines indicating significance thresholds and
optionally labels genes meeting both significance criteria.

## Note

- Genes with adj_pval = 0 are assigned the smallest non-zero p-value in
  the dataset

- NA values are removed with a warning

- Gene labels are placed with overlap prevention

## Examples

``` r
set.seed(1)
y <- t(as.matrix(rnbinom(1000, 1, .1)))
fit <- devil::fit_devil(input_matrix = y, design_matrix = matrix(1, ncol = 1, nrow = 1000))
de_results <- devil::test_de(devil.fit = fit, contrast = c(1))

# Basic volcano plot
de_results$name = paste0("Fake gene")
plot_volcano(de_results)
#> Warning: no non-missing arguments to min; returning Inf
#> 1 genes have adjusted p-value equal to 0, will be set to Inf


# Custom thresholds and colors
plot_volcano(de_results,
    lfc_cut = 2,
    pval_cut = 0.01,
    colors = c("grey80", "blue", "green", "red")
)
#> Warning: no non-missing arguments to min; returning Inf
#> 1 genes have adjusted p-value equal to 0, will be set to Inf


# Without gene labels
de_results$name <- "fake gene"
plot_volcano(de_results, labels = FALSE)
#> Warning: no non-missing arguments to min; returning Inf
#> 1 genes have adjusted p-value equal to 0, will be set to Inf

```
