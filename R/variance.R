#' Compute Sandwich Estimator for Statistical Model
#'
#' @description Calculates the sandwich estimator for robust covariance estimation,
#'   particularly useful in clustered or heteroskedastic data scenarios.
#'
#' @param design_matrix Matrix of predictor variables
#' @param y Vector of response variables
#' @param beta Vector of coefficient estimates
#' @param overdispersion Scalar overdispersion parameter
#' @param size_factors Vector of normalization factors for each sample
#' @param clusters Vector indicating cluster membership
#'
#' @return Matrix containing the sandwich estimator
#'
#' @keywords internal
compute_sandwich <- function(design_matrix, y, beta, overdispersion, size_factors, clusters) {
    b <- compute_hessian(beta, 1 / overdispersion, y, design_matrix, size_factors)

    if (!is.null(clusters)) {
        m <- compute_clustered_meat(design_matrix, y, beta, overdispersion, size_factors, clusters)
    } else {
        m <- compute_meat(design_matrix, y, beta, overdispersion, size_factors)
    }

    (b %*% m %*% b) * length(y)
}
