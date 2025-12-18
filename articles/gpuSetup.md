# GPU and CUDA Setup

## Overview

This guide provides instructions for installing `devil` on HPC clusters.
We cover both standard installation and GPU-accelerated installation
with CUDA support.

## Prerequisites

Before proceeding, ensure that:

- You have access to an HPC cluster with module system support (e.g.,
  `Lmod`)
- CUDA toolkit is installed (version 12.0+ requested)
- You have appropriate permissions to install software in your user
  space, or root access if installing system-wide (for system
  administrators). In principle, you should be able to install the
  package in your user space assuming working Nvidia drivers and CUDA
  installation.

**Note:** In shared HPC environments, you typically don’t have root
access, so the module system (`Lmod`) is necessary for managing software
dependencies.

## Standard Installation on HPC Cluster

The installation will follow the steps below: 1. Compile and install
OpenBLAS and its module (*optional*) 2. Compile and install R with
optimized BLAS support and its module (*optional*) 3. Install `devil`
from GitHub within R

In case of GPU acceleration, additional steps will be provided later.

### Step 1: OpenBLAS Setup

Since R can use internal BLAS as a fallback, we recommend compiling
OpenBLAS for optimal performance on your specific hardware architecture.

#### Compile OpenBLAS

Create a bash script to compile OpenBLAS:

``` bash
#!/usr/bin/bash -e

SOURCE_PATH="software/source"
INSTALL_PATH="software/programs/openBLAS"
MODULE_PATH="software/modules/openBLAS"

VERSION="0.3.30"
PLATFORM="DGX"  # Adjust for your architecture

mkdir -p "$INSTALL_PATH/$VERSION-$PLATFORM"

# Clone OpenBLAS
cd $SOURCE_PATH
rm -rf OpenBLAS/
git clone --depth 1 --branch v$VERSION git@github.com:OpenMathLib/OpenBLAS.git

# Compile
cd $SOURCE_PATH/OpenBLAS
mkdir -p ${INSTALL_PATH}/$VERSION-$PLATFORM/
make clean
make distclean
make USE_OPENMP=0 -j $(nproc --all)
make PREFIX=${INSTALL_PATH}/$VERSION-$PLATFORM/ install
```

**Note:** We use pthread-based libraries instead of OpenMP-based ones to
avoid conflicts, since devil uses OpenMP internally.

#### Create OpenBLAS Module File

Create a Lua module file for OpenBLAS:

``` lua
-- -*- lua -*-

local name      = "openBLAS"
local version   = "0.3.29-DGX"

whatis("Name         : " .. name)
whatis("Version      : " .. version)

family("BLAS")

local home    = "software/programs/openBLAS/0.3.29-DGX/"

prepend_path{"PATH", home .. "/bin",delim=":",priority="0"}
prepend_path{"LD_LIBRARY_PATH", home .. "/lib",delim=":",priority="0"}
prepend_path{"LIBRARY_PATH", home .. "/lib",delim=":",priority="0"}
prepend_path{"CPATH", home .. "/include",delim=":",priority="0"}
setenv("OPENBLAS_DIR", home)
setenv("OPENBLAS_ROOT", home)
setenv("OPENBLAS_LIB", home .. "/lib")
setenv("OPENBLAS_IN", home .. "/include")
```

**Note:** Adjust the `home` variable to the absolute path where OpenBLAS
is installed ! Example:
`/path/to/software/programs/openBLAS/0.3.29-DGX/`

### Step 2: R Setup

Compile R with optimized BLAS support for maximum performance.

#### Compile R

``` bash
#!/usr/bin/bash

SOURCE_PATH="software/source"
INSTALL_PATH="software/programs/R"
MODULE_PATH="software/modules"
VERSION="4.3.3"
PLATFORM="DGX"
mkdir -p "$INSTALL_PATH/$VERSION-$PLATFORM"

# Download R source
cd $SOURCE_PATH/
rm -rf R-4.3.3*
wget https://cloud.r-project.org/src/base/R-4/R-${VERSION}.tar.gz
tar -xzf R-${VERSION}.tar.gz

# Load modules
module use $MODULE_PATH
module load openBLAS/0.3.29-$PLATFORM

echo $OPENBLAS_LIB

# Configure and compile
cd $SOURCE_PATH/R-$VERSION
./configure --prefix=${INSTALL_PATH}/${VERSION}-${PLATFORM} --with-x=no \
    --with-blas="-L${OPENBLAS_LIB} -lopenblas" \
    --with-lapack=yes \
    --with-system-valgrind-headers \
    --enable-memory-profiling \
    --with-valgrind-instrumentation=2 \
    --with-jpeglib \
    --with-libpng \
    --with-tcltk \
    --with-readline \
    --with-cairo=yes \
    --enable-R-profiling \
    --with-libtiff \
    --enable-lto \
    --enable-R-shlib

make -j $(nproc --all)
make install
```

#### Create R Module File

``` lua
-- -*- lua -*-

local name      = "R"
local version   = "4.3.3-DGX"
whatis("Name         : " .. name)
whatis("Version      : " .. version)

family("R")
depends_on("openBLAS/0.3.30-DGX")

local home    = "software/programs/R/4.3.3-DGX"

prepend_path("PATH", home .. "/bin")
prepend_path("LD_LIBRARY_PATH", home .."/lib64")
prepend_path("MANPATH", home .."/share/man")
prepend_path("R_LIBS_USER","software/programs/r_packages_DGX")
```

**Important:** Specify a custom `R_LIBS_USER` to avoid installation
conflicts between different architectures.

### Step 3: Install devil

Load the required modules and install devil:

``` bash
# Add your software modules to the module path
module use software/modules/

# Load required modules
module load R/4.3.3-DGX openBLAS/0.3.30-DGX

# Install devil
R
```

``` r
# In R console
devtools::install_github("caravagnalab/devill")
```

**Note** You can specify a different branch with the following syntax:
`devtools::install_github("caravagnalab/devill@branch_name")`

## GPU-Accelerated Installation with CUDA Support

For GPU acceleration, additional setup is required to enable `CUDA` and
`cuTENSOR` support.

If `CUDA` and `cuTENSOR` are already installed and configured on your
HPC system, you can skip to **Step 3**.

### Step 1: cuTENSOR Setup

`cuTENSOR` is distributed by NVIDIA and is required for GPU-accelerated
tensor operations.

#### Install cuTENSOR

``` bash
#!/usr/bin/bash

SOURCE_PATH="software/source"
INSTALL_PATH="software/programs/cutensor"
MODULE_PATH="software/modules"
VERSION="2.2.0.0"

# Download cuTENSOR
cd $SOURCE_PATH/
rm -rf libcutensor*

wget https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/libcutensor-linux-x86_64-$VERSION-archive.tar.xz

tar -xf libcutensor-linux-x86_64-$VERSION-archive.tar.xz
cd libcutensor-linux-x86_64-$VERSION-archive/
mkdir -p $INSTALL_PATH/cutensor/$VERSION/
cp -r ./* $INSTALL_PATH/cutensor/$VERSION/
```

This will install several versions of `cuTENSOR` compatible with
multiple CUDA versions. We use version 12.0+.

#### Create cuTENSOR Module File

This will create a module for `cuTENSOR`, adjusting the `home` variable
to the absolute path where `cuTENSOR` is installed (e.g.,
\`/path/to/software/programs/cutensor/

``` lua
-- -*- lua -*-

local name      = "cutensor"
local version   = "2.2.0.0"
whatis("Name         : " .. name)
whatis("Version      : " .. version)

family("cutensor")
depends_on("cuda")

local home    = "software/programs/cutensor/2.2.0.0"

prepend_path("CPATH", home .. "/include")
prepend_path("INCLUDE", home .. "/include")
prepend_path("LD_LIBRARY_PATH", home .. "/lib/12/")
prepend_path("LIBRARY_PATH", home .. "/lib/12/")
setenv("CUTENSOR_HOME", home)
```

### Step 2: Load All Required Modules

``` bash
# Add your software modules to the module path
module use software/modules/

# Load all required modules for GPU support
module load R/4.3.3-h100 openBLAS/0.3.29-h100 cutensor/2.2.0.0
```

### Step 3: Verify Environment Variables

Before installing devil with GPU support, verify that the essential
environment variables are correctly set:

``` bash
# Check OpenBLAS
echo $OPENBLAS_LIB
# Expected output (example):
# /path/to/software/programs/openBLAS/0.3.29-h100/lib

# Check CUDA
echo $CUDA_HOME
# Expected output (example):
# /opt/programs/cuda/12.1

# Check cuTENSOR
echo $CUTENSOR_HOME
# Expected output (example):
# /path/to/software/programs/cutensor/2.2.0.0
```

**Important:** The environment variables `CUDA_HOME` and `CUTENSOR_HOME`
are essential for compiling devil with GPU support.

### Step 4: Install devil with GPU Support

Once all modules are loaded and environment variables are verified, just
install devil with the following command:

``` r
# In R console
devtools::install_github("caravagnalab/devil", configure.args = "--with-cuda")
```

The installation will automatically detect the CUDA and cuTENSOR
libraries and compile the GPU-accelerated code.

## Notes on Module Paths

**Important:** The module files provided use relative paths. You need to
adjust these to absolute paths on your system. Additionally, ensure you
have a working CUDA installation configured by your HPC system
administrators.
