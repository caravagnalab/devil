# Quick setup (CPU & GPU)

## Quick Start

This guide provides streamlined installation instructions for the
`devil` package. Choose the installation method that best suits your
computational needs.

## Standard Installation

For most users, the standard CPU-based installation is recommended:

``` r

# Install from GitHub
devtools::install_github("caravagnalab/devil")
```

If you don’t have `devtools` installed:

``` r

install.packages("devtools")
devtools::install_github("caravagnalab/devil")
```

## GPU-Accelerated Installation

For users with NVIDIA GPUs who want to leverage GPU acceleration for
faster computations.

### Prerequisites

Before installing the GPU-accelerated version, ensure you have:

- **CUDA Toolkit** (version 12.0 or higher) - [Download
  here](https://developer.nvidia.com/cuda-downloads)
- **cuTENSOR Library** - [Download
  here](https://developer.nvidia.com/cutensor)
- **Environment Variables** properly configured:
  - `CUDA_HOME`: Path to your CUDA installation directory
  - `CUTENSOR_HOME`: Path to your cuTENSOR installation directory

### Verifying Your Environment

Check if your environment variables are set correctly:

``` r

# Check CUDA_HOME
Sys.getenv("CUDA_HOME")

# Check CUTENSOR_HOME
Sys.getenv("CUTENSOR_HOME")
```

Both commands should return valid directory paths. If they return empty
strings, you’ll need to set these variables before proceeding.

### Installation Command

Once prerequisites are met:

``` r

devtools::install_github("caravagnalab/devil", configure.args = "--with-cuda")
```

The package will automatically detect your CUDA installation and compile
with GPU support.

## Troubleshooting

If installation fails:

1.  Verify CUDA toolkit installation: `nvcc --version` in your terminal
2.  Confirm environment variables are set system-wide, not just in your
    R session
3.  Ensure your GPU drivers are up to date
4.  Check that your CUDA version is compatible with your GPU
    architecture

For additional help, please open an issue on the [GitHub
repository](https://github.com/caravagnalab/devil/issues).

## Next Steps

After successful installation, load the package:

``` r

library(devil)
```

Continue to the main vignettes to learn how to use `devil` for your
analysis.

## Session info

``` r

sessionInfo()
#> R version 4.6.1 (2026-06-24)
#> Platform: x86_64-pc-linux-gnu
#> Running under: Ubuntu 24.04.4 LTS
#> 
#> Matrix products: default
#> BLAS:   /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 
#> LAPACK: /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.26.so;  LAPACK version 3.12.0
#> 
#> locale:
#>  [1] LC_CTYPE=C.UTF-8       LC_NUMERIC=C           LC_TIME=C.UTF-8       
#>  [4] LC_COLLATE=C.UTF-8     LC_MONETARY=C.UTF-8    LC_MESSAGES=C.UTF-8   
#>  [7] LC_PAPER=C.UTF-8       LC_NAME=C              LC_ADDRESS=C          
#> [10] LC_TELEPHONE=C         LC_MEASUREMENT=C.UTF-8 LC_IDENTIFICATION=C   
#> 
#> time zone: UTC
#> tzcode source: system (glibc)
#> 
#> attached base packages:
#> [1] stats     graphics  grDevices utils     datasets  methods   base     
#> 
#> loaded via a namespace (and not attached):
#>  [1] digest_0.6.39     desc_1.4.3        R6_2.6.1          fastmap_1.2.0    
#>  [5] xfun_0.59         cachem_1.1.0      knitr_1.51        htmltools_0.5.9  
#>  [9] rmarkdown_2.31    lifecycle_1.0.5   cli_3.6.6         sass_0.4.10      
#> [13] pkgdown_2.2.0     textshaping_1.0.5 jquerylib_0.1.4   systemfonts_1.3.2
#> [17] compiler_4.6.1    tools_4.6.1       ragg_1.5.2        bslib_0.11.0     
#> [21] evaluate_1.0.5    yaml_2.3.12       otel_0.2.0        jsonlite_2.0.0   
#> [25] rlang_1.2.0       fs_2.1.0
```
