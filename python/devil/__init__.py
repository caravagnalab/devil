"""
devil: Differential Expression analysis for single-cell RNA sequencing data.

A Python package for fast and scalable differential expression analysis in scRNA-seq data,
supporting both single- and multi-patient experimental designs.
"""

from .main import fit_devil
from .test import test_de
from .plot import plot_volcano

__version__ = "0.1.0"
__all__ = ["fit_devil", "test_de", "plot_volcano"]