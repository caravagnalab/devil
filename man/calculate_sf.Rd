% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/size_factors.R
\name{calculate_sf}
\alias{calculate_sf}
\title{Calculate Size Factors for Count Data Normalization}
\usage{
calculate_sf(Y, verbose = FALSE)
}
\arguments{
\item{Y}{Count data matrix with genes in rows and samples in columns}

\item{verbose}{Logical indicating whether to print progress messages}
}
\value{
Vector of size factors, one per sample
}
\description{
Computes normalization factors for count data, handling edge cases
like all-zero columns. Uses geometric mean normalization approach.
}
\keyword{internal}
