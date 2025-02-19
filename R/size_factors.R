
#' Calculate Size Factors for Count Data Normalization
#'
#' @description Computes normalization factors for count data, handling edge cases
#'   like all-zero columns. Uses geometric mean normalization approach.
#'
#' @param Y Count data matrix with genes in rows and samples in columns
#' @param verbose Logical indicating whether to print progress messages
#'
#' @return Vector of size factors, one per sample
#'
#' @keywords internal
calculate_sf <- function(Y, verbose=FALSE) {
  if (nrow(Y) <= 1) { return(rep(1, ncol(Y))) }

  sf <- DelayedMatrixStats::colSums2(Y, useNames=TRUE)

  # Check for all-zero columns
  all_zero_genes <- (sf == 0)

  if (any(all_zero_genes)) {
    stop("Error: At least one column (i.e. gene) contains all zeros, unable to compute size factor. Please filter out non-expressed genes from input matrix")
  }

  # Compute size factors
  sf <- sf / exp(mean(log(sf), na.rm=TRUE))

  return(sf)
}

compute_offset_matrix <- function (off, Y, size_factors) {
  n_samples <- ncol(Y)
  n_genes <- nrow(Y)

  offset_matrix <- matrix(off, nrow = n_genes, ncol = n_samples)

  if (!(is.null(size_factors))) {
    lsf <- DelayedArray::DelayedArray(DelayedArray::SparseArraySeed(c(n_samples, 1))) + log(size_factors)
    offset_matrix <- DelayedArray::sweep(offset_matrix, 2, lsf, "+")
  }

  return(offset_matrix)
}

#' Compute Offset Matrix for Statistical Model
#'
#' @description Creates an offset matrix incorporating base offsets and optional
#'   size factors for model fitting.
#'
#' @param off Base offset value
#' @param Y Count data matrix with genes in rows and samples in columns
#' @param size_factors Optional vector of size factors for normalization
#'
#' @return Matrix of offset values for each gene-sample combination
#'
#' @keywords internal
compute_offset_vector <- function(off, Y, size_factors) {
  n_samples <- ncol(Y)
  n_genes <- nrow(Y)

  # Create the offset vector
  offset_vec <- rep(off, n_samples)

  if (!is.null(size_factors)) {
    offset_vec <- offset_vec + log(size_factors)
  }

  offset_vec
}

