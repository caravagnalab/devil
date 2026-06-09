# devil: Differential Expression via Negative Binomial GLMs with Sandwich Covariance for Single-Cell RNA-Seq

devil is an R package for fast, scalable differential expression
analysis of single-cell and single-nucleus RNA-sequencing data. It is
designed to operate directly at the cell level while accounting for
complex experimental structure. It employs a statistical framework based
on generalized linear models with cluster-robust (sandwich) covariance
estimators, enabling valid inference in the presence of multi-patient,
multi-sample, or repeated-measurement designs. Input can be a raw count
matrix, a SingleCellExperiment, or a SummarizedExperiment object. To
support analyses at atlas scale, devil combines efficient variational
inference with optional GPU acceleration, delivering substantial gains
in runtime and memory efficiency while retaining accurate and
well-calibrated statistical tests.

## See also

Useful links:

- <https://caravagnalab.github.io/devil/>

- Report bugs at <https://github.com/caravagnalab/devil/issues>

## Author

**Maintainer**: Giulio Caravagna <gcaravagn@gmail.com>
([ORCID](https://orcid.org/0000-0003-4240-3265))

Authors:

- Giovanni Santacatterina <santacatterinagiovanni@gmail.com>
  ([ORCID](https://orcid.org/0009-0003-8507-6521))

- Niccolò Tosato (https://github.com/NiccoloTosato)
