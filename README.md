
<!-- README.md is generated from README.Rmd. Please edit that file -->

<!-- Use devtools::build_readme() to update the files -->

# devil <a href="caravagnalab.github.io/rdevil"><img src="man/figures/logo.png" align="right" height="139" alt="devil website" /></a>

<!-- badges: start -->

<!-- badges: end -->

`devil` is an R package for differential expression analysis in
single-cell RNA sequencing (scRNA-seq) data. It supports both single-
and multi-patient experimental designs, implementing robust statistical
methods to identify differentially expressed genes while accounting for
technical and biological variability.

Key features are:

1.  Flexible experimental design support (single/multiple patients)
2.  Robust statistical testing framework
3.  Efficient implementation for large-scale datasets

## Installation

You can install the development version of `devil` from
[GitHub](https://github.com/) with:

``` r
devtools::install_github("caravagnalab/devil")
```

## Example

This is a basic example which shows you how to fit the expression for a
single gene observed in 1000 cells.

``` r
library(devil)
y <- t(as.matrix(rnbinom(1000, 1, .1)))
fit <- devil::fit_devil(input_matrix=y, design_matrix=matrix(1, ncol = 1, nrow = 1000), verbose=T, size_factors=T, overdispersion = T)
#> Compute size factors
#> Initialize beta estimate
#> Fit beta coefficients
#> Fit overdispersion
test <- devil::test_de(fit, c(1))
```

------------------------------------------------------------------------

#### Copyright and contacts

Giulio Caravagna, Giovanni Santacatterina. Cancer Data Science (CDS)
Laboratory.

[![](https://img.shields.io/badge/CDS%20Lab%20Github-caravagnalab-seagreen.svg)](https://github.com/caravagnalab/)
[![](https://img.shields.io/badge/CDS%20Lab%20webpage-https://www.caravagnalab.org/-red.svg)](https://www.caravagnalab.org/)
