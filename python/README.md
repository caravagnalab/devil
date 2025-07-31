# devil Python Package

A Python implementation of the devil (Differential Expression analysis) package for single-cell RNA sequencing data.

## Installation

```bash
cd python
uv venv --python=python3.10
source .venv/bin/activate
uv pip install .
```

GPU support

```bash
uv pip install ".[gpu]"
```

## For development:

```bash
cd python
uv venv --python=python3.10
source .venv/bin/activate
uv pip install -e ".[dev]"
```

GPU support

```bash
uv pip install -e ".[all]"
```

## Quick Start

```python
import devil
import scanpy as sc

# Load data
adata = sc.read_h5ad("data.h5ad")

# Fit model
result = devil.fit_devil(
    adata,
    design_formula="~ condition + batch",
    verbose=True
)

# Test differential expression
de_results = devil.test_de(
    result,
    contrast=[0, 1, -1]  # Compare conditions
)

# Visualize results
ax = devil.plot_volcano(de_results)
```

## Features

- Direct AnnData support: Works seamlessly with the single-cell ecosystem
- Fast parallel processing: Utilizes all available CPU cores
- Robust statistics: Includes clustered variance estimation for multi-patient designs
- Flexible design matrices: Support for complex experimental designs
- Publication-ready plots: Beautiful volcano and MA plots

## Documentation

See the full documentation at https://caravagnalab.github.io/devil

## Citation

If you use devil in your research, please cite:

> Caravagna G, Santacatterina G. devil: fast and scalable differential 
> expression analysis for single-cell RNA-seq data. 2024.