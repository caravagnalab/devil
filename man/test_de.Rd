% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/test.R
\name{test_de}
\alias{test_de}
\title{Test for Differential Expression}
\usage{
test_de(
  devil.fit,
  contrast,
  pval_adjust_method = "BH",
  max_lfc = 10,
  clusters = NULL
)
}
\arguments{
\item{devil.fit}{The fitted devil model.}

\item{contrast}{The contrast matrix for differential expression testing.}

\item{pval_adjust_method}{Method for adjusting p-values. Default is "BH" (Benjamini & Hochberg).}

\item{max_lfc}{Maximum absolute log fold change to consider for filtering results. Default is 10.}

\item{clusters}{.}
}
\value{
A tibble containing the results of the differential expression testing.
}
\description{
This function tests for differential expression between conditions using the provided contrast matrix.
}
\details{
This function computes log fold changes and p-values for each gene in parallel and filters the results based on the maximum absolute log fold change specified.
}
