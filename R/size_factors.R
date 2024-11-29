
calculate_sf <- function(Y, verbose=FALSE) {
  if (nrow(Y) <= 1) {
    if (verbose) {
      cat("Skipping size factor estimation! Only one gene is present!\n")
    }
    return(rep(1, ncol(Y)))
  }

  sf <- DelayedMatrixStats::colSums2(Y, useNames=TRUE)

  # Check for all-zero columns
  all_zero_column <- is.nan(sf) | (sf <= 0)

  # Replace all-zero columns with NA
  sf[all_zero_column] <- 0

  if (any(all_zero_column)) {
    warning_message <- paste(sum(all_zero_column), "columns contain too many zeros to calculate a size factor. The size factor will be fixed to 0.001")
    cat(warning_message, "\n")

    # Apply the required transformations
    sf <- sf / exp(mean(log(sf[!all_zero_column]), na.rm=TRUE))
    sf[all_zero_column] <- 0.001
  } else {
    sf <- sf / exp(mean(log(sf), na.rm=TRUE))
  }

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

