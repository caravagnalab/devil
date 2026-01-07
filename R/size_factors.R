#' Calculate Size Factors for Count Data Normalization
#'
#' @description Computes normalization factors for count data using one of three
#'   methods: geometric mean normalization (normed_sum), psi-normalization
#'   (psinorm), or edgeR's TMM with singleton pairing (edgeR). Handles edge cases
#'   like all-zero columns and matrices with too few rows.
#'
#' @param Y Count data matrix with genes in rows and samples in columns
#' @param method Character string specifying the normalization method. One of:
#'   \itemize{
#'     \item \code{"normed_sum"} (default): Geometric mean normalization based on
#'       library sizes
#'     \item \code{"psinorm"}: Psi-normalization using Pareto distribution MLE
#'     \item \code{"edgeR"}: edgeR's TMM with singleton pairing method
#'   }
#' @param verbose Logical indicating whether to print progress messages
#'
#' @return Numeric vector of size factors, one per sample (column). Size factors
#'   are scaled to have a geometric mean of 1.
#'
#' @details
#' Size factors are used to normalize count data for differences in sequencing
#' depth and RNA composition across samples. The function will return a vector
#' of 1s if the input matrix has only one row.
#'
#' The \code{"normed_sum"} method computes size factors as the column sums
#' divided by their geometric mean.
#'
#' The \code{"psinorm"} method uses maximum likelihood estimation of Pareto
#' distribution parameters to compute size factors robust to highly variable
#' genes.
#'
#' The \code{"edgeR"} method requires the edgeR package to be installed and
#' uses the TMM (trimmed mean of M-values) method with singleton pairing.
#'
#' @keywords internal
calculate_sf <- function(Y, method = c("normed_sum", "psinorm", "edgeR"), verbose = FALSE) {
    # Handle edge case: matrix with only one row
    if (nrow(Y) <= 1) {
        if (verbose) message("Matrix has only one row. Returning unit size factors.")
        return(rep(1, ncol(Y)))
    }

    # Match and validate method argument
    method <- match.arg(method)

    if (verbose) message("Calculating size factors using method: ", method)

    # Calculate size factors based on selected method
    sf <- switch(method,
        normed_sum = normed_sum_sf(Y),
        psinorm = psinorm_sf(Y),
        edgeR = edgeR_sf(Y),
        stop("Unknown method: ", method)
    )

    if (verbose) {
        message("Size factors calculated successfully.")
        message("Range: [", round(min(sf), 4), ", ", round(max(sf), 4), "]")
    }

    return(sf)
}

#' Geometric Mean Normalization Size Factors
#'
#' @description Computes size factors based on library sizes (column sums)
#'   normalized by their geometric mean.
#'
#' @param Y Count data matrix with genes in rows and samples in columns
#'
#' @return Numeric vector of size factors
#'
#' @keywords internal
normed_sum_sf <- function(Y) {
    sf <- DelayedMatrixStats::colSums2(Y, useNames = TRUE)
    all_zero_cols <- (sf == 0) # Check for all-zero columns

    if (any(all_zero_cols)) {
        stop("At least one column (sample) contains all zeros, unable to compute size factor. ",
            "Please filter out empty samples from input matrix.",
            call. = FALSE
        )
    }

    sf <- sf / exp(mean(log(sf), na.rm = TRUE)) # Compute size factors
    return(sf)
}

#' Psi-Normalization Size Factors
#'
#' @description Computes size factors using psi-normalization based on Pareto
#'   distribution maximum likelihood estimation.
#'
#' @param Y Count data matrix with genes in rows and samples in columns
#'
#' @return Numeric vector of size factors
#'
#' @keywords internal
psinorm_sf <- function(Y) {
    pareto.MLE <- function(Y) {
        n <- nrow(Y)
        m <- DelayedMatrixStats::colMins(Y)
        a <- n / DelayedMatrixStats::colSums2(t(t(log(Y)) - log(m)))
        return(a)
    }

    computePsiNormSF <- function(x) {
        1 / pareto.MLE(x + 1)
    }

    sf <- computePsiNormSF(Y)
    sf <- sf / exp(mean(log(sf), na.rm = TRUE))
    return(sf)
}

#' edgeR TMM Size Factors
#'
#' @description Computes size factors using edgeR's TMM (trimmed mean of M-values)
#'   method with singleton pairing.
#'
#' @param Y Count data matrix with genes in rows and samples in columns
#'
#' @return Numeric vector of size factors
#'
#' @keywords internal
edgeR_sf <- function(Y) {
    if (requireNamespace("edgeR", quietly = TRUE)) {
        edgeR <- asNamespace("edgeR")
        y <- edgeR$DGEList(counts = Y)
        y <- edgeR$calcNormFactors(y, method = "TMMwsp")
        sf <- DelayedMatrixStats::colSums2(Y) * y$samples$norm.factors
        sf <- sf / exp(mean(log(sf), na.rm = TRUE))
        return(sf)
    } else {
        stop("To use the \"edgeR\" method for size factor calculation, you need to install the ",
            "'edgeR' package from Bioconductor.",
            call. = FALSE
        )
    }
}

compute_offset_matrix <- function(off, Y, size_factors) {
    n_samples <- ncol(Y)
    n_genes <- nrow(Y)

    offset_matrix <- matrix(off, nrow = n_genes, ncol = n_samples)

    if (!(is.null(size_factors))) {
        # lsf <- DelayedArray::DelayedArray(DelayedArray::SparseArraySeed(c(n_samples, 1))) + log(size_factors)
        lsf <- DelayedArray::DelayedArray(matrix(log(size_factors), ncol = 1))
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
