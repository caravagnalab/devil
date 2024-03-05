
compute_scores <- function(design_matrix, y, beta, overdispersion, size_factors) {
  xmat = design_matrix
  alpha <- 1 / overdispersion

  mu = size_factors * exp(xmat %*% beta)
  residuals <- (y - mu) / mu
  weights <- (mu**2) / (mu + mu**2 / alpha)
  wr <- as.vector(residuals * weights)

  xmat * wr
}

compute_clustered_meat <- function(design_matrix, y, beta, overdispersion, size_factors, clusters = NULL) {
  ef = compute_scores(design_matrix, y, beta, overdispersion, size_factors)
  k <- ncol(ef)
  n <- nrow(ef)

  rval <- matrix(0, nrow = k, ncol = k)

  cl <- 1
  sign <- 1
  g <- length(unique(clusters))

  for (i in seq_along(cl)) {
    efi <- ef  # Make a copy to avoid modifying the original

    adj <- ifelse(g[i] > 1, g[i] / (g[i] - 1), 1)

    if (g[i] < n) {
      efi <- matrix(0, nrow = g[i], ncol = k)

      for (j in seq_along(unique(clusters))) {
        group <- unique(clusters)[j]
        mask <- clusters == group
        efi[j,] <- colSums(ef[mask,,drop=FALSE])
      }
    }

    rval <- rval + sign * adj * t(efi) %*% efi / n
  }
  rval
}

compute_sandwich <- function(design_matrix, y, beta, overdispersion, size_factors, clusters) {
  b = devil:::compute_hessian(beta, overdispersion, y, design_matrix, size_factors)
  m = compute_clustered_meat(design_matrix, y, beta, overdispersion, size_factors, clusters)
  (b %*% m %*% b) * length(y)
}
