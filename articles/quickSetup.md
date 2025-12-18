# Quick Setup

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
