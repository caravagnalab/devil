
compute_sandwich <- function(design_matrix, y, beta, overdispersion, size_factors, clusters) {
  b = compute_hessian(beta, 1 / overdispersion, y, design_matrix, size_factors)
  m = compute_clustered_meat(design_matrix, y, beta, overdispersion, size_factors, clusters)
  (b %*% m %*% b) * length(y)
}
