"""
devil: Differential Expression analysis for single-cell RNA sequencing data.

A Python package for fast and scalable differential expression analysis in scRNA-seq data,
supporting both single- and multi-patient experimental designs with optional GPU acceleration.
"""

from .main import fit_devil
from .test import test_de, test_de_memory_efficient
from .plot import plot_volcano
from .gpu import is_gpu_available, check_gpu_requirements

__version__ = "0.1.0"
__all__ = [
    "fit_devil", 
    "test_de", 
    "test_de_memory_efficient",
    "plot_volcano", 
    "is_gpu_available", 
    "check_gpu_requirements"
]