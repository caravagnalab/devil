# devil 0.99.0

* Initial submission to Bioconductor.
* Implements negative binomial GLM differential expression with cluster-robust sandwich covariance estimators.
* Accepts raw count matrices, SingleCellExperiment, and SummarizedExperiment objects as input via `fit_devil()`.
* Supports CPU parallelism via BiocParallel and optional GPU acceleration (CUDA).
* Multiple overdispersion estimation strategies: MOM, iterative MLE, and Poisson (no overdispersion).
* Multiple size factor normalization methods: normed_sum, psinorm, and edgeR TMM.
