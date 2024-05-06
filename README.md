
<!-- README.md is generated from README.Rmd. Please edit that file -->
<!-- Use devtools::build_readme() to update the files -->

# devil <a href="caravagnalab.github.io/rdevil"><img src="man/figures/logo.png" align="right" height="139" alt="devil website" /></a>

<!-- badges: start -->
<!-- badges: end -->

`devil` is a package to perform differential expression analysis for
scRNA-seq dataset considering either single- or multi-patients
experimental designs

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
#> Skipping size factor estimation! Only one gene is present!
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
