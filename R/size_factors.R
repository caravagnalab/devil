
calculate_sf <- function(Y, method, verbose=FALSE) {
  if (dim(Y)[2] <= 1) {
    if (verbose) {
      cat("Skipping size factor estimation! Only one gene is present!\n")
    }
    return(rep(1, dim(Y)[1]))
  }

  sf <- DelayedMatrixStats::colSums2(Y)

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

compute_offset_matrix <- function(off, Y, size_factors) {
  n_samples <- dim(Y)[1]
  n_genes <- dim(Y)[2]

  make_offset_hdf5_mat <- methods::is(Y, "DelayedMatrix") && methods::is(DelayedArray::seed(Y), "HDF5ArraySeed")

  # Create the offset matrix
  if (make_offset_hdf5_mat) {
    offset_matrix <- DelayedArray::DelayedArray(DelayedArray::SparseArraySeed(c(n_samples, n_genes)))
    offset_matrix <- offset_matrix + off
  } else {
    offset_matrix <- matrix(off, nrow = n_samples, ncol = n_genes)
  }

  if (!(is.null(size_factors))) {
    # Update the offset_matrix with size_factors
    lsf <- DelayedArray::DelayedArray(DelayedArray::SparseArraySeed(c(n_genes, 1))) + log(size_factors)
    offset_matrix <- DelayedArray::sweep(offset_matrix, 2, lsf, "+")
  }


  if(make_offset_hdf5_mat){
    offset_matrix <- HDF5Array::writeHDF5Array(offset_matrix)
  }

  # Return the result
  return(offset_matrix)
}

