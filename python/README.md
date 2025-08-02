# devil Python Package

A Python implementation of the devil (Differential Expression analysis) package for single-cell RNA sequencing data.


## Installation

> Assuming that you are in the root directory of the devil repository.

```bash
uv venv --python=python3.10
source .venv/bin/activate
uv pip install python/
```

GPU support

```bash
uv pip install "python/.[gpu]"
```

## Development

```bash
uv pip install -e "python/.[dev]"
```

GPU support

```bash
uv pip install -e "python/.[all]"
```

## Quick Start

See the [notebooks](notebooks) for examples.

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