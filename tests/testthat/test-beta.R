
test_that("init_beta returns matrix with correct dimensions", {
  set.seed(1)
  n_genes   <- 5
  n_samples <- 8
  p         <- 1  # intercept only

  y <- matrix(rpois(n_genes * n_samples, lambda = 10),
              nrow = n_genes, ncol = n_samples)
  design_matrix <- matrix(1, nrow = n_samples, ncol = p)
  offset_matrix <- matrix(0, nrow = n_samples, ncol = n_genes)  # samples x genes

  beta_init <- init_beta(y, design_matrix, offset_matrix)

  expect_true(is.matrix(beta_init))
  expect_equal(dim(beta_init), c(n_genes, p))
})

test_that("init_beta with intercept-only design matches log1p of mean normalized counts", {
  set.seed(2)
  n_genes   <- 4
  n_samples <- 6

  y <- matrix(rpois(n_genes * n_samples, lambda = 5),
              nrow = n_genes, ncol = n_samples)
  design_matrix <- matrix(1, nrow = n_samples, ncol = 1)
  offset_matrix <- matrix(0, nrow = n_samples, ncol = n_genes)

  beta_init <- init_beta(y, design_matrix, offset_matrix)

  # norm_log_count_mat = log1p( t(y) / exp(offset) ) = log1p(t(y))
  # Intercept-only LS fit: coefficient ~= rowMeans(log1p(norm_counts))
  norm_log <- log1p(t(y))  # samples x genes
  expected <- colMeans(norm_log)  # one value per gene

  # beta_init is genes x 1
  expect_equal(as.numeric(beta_init[, 1]), expected, tolerance = 1e-10)
})

test_that("init_beta matches explicit least squares solution per gene", {
  set.seed(3)
  n_genes   <- 3
  n_samples <- 10
  p         <- 3

  y <- matrix(rpois(n_genes * n_samples, lambda = 10),
              nrow = n_genes, ncol = n_samples)

  # Full rank design
  design_matrix <- cbind(
    1,
    rnorm(n_samples),
    rnorm(n_samples)
  )

  # Non-zero offsets
  offset_matrix <- matrix(runif(n_samples * n_genes, min = -0.5, max = 0.5),
                          nrow = n_samples, ncol = n_genes)

  beta_init <- init_beta(y, design_matrix, offset_matrix)
  X <- design_matrix
  XtX_inv <- solve(t(X) %*% X)

  # manually compute LS for each gene
  norm_log <- log1p(t(y) / exp(offset_matrix))  # samples x genes

  beta_manual <- sapply(seq_len(n_genes), function(g) {
    z_g <- norm_log[, g]
    as.numeric(XtX_inv %*% (t(X) %*% z_g))
  })
  beta_manual <- t(beta_manual)  # genes x p

  expect_equal(beta_init, beta_manual, tolerance = 1e-10)
})

test_that("init_beta behaves correctly under scaling of design columns", {
  set.seed(4)
  n_genes   <- 3
  n_samples <- 12
  p         <- 2

  y <- matrix(rpois(n_genes * n_samples, lambda = 7),
              nrow = n_genes, ncol = n_samples)
  X <- cbind(1, rnorm(n_samples))
  offset_matrix <- matrix(0, nrow = n_samples, ncol = n_genes)

  beta_orig <- init_beta(y, X, offset_matrix)

  c_scale <- 5
  X_scaled <- cbind(X[, 1], c_scale * X[, 2])
  beta_scaled <- init_beta(y, X_scaled, offset_matrix)

  # Intercept should be (almost) unchanged; slope scaled by 1/c_scale
  expect_equal(beta_orig[, 1], beta_scaled[, 1], tolerance = 1e-8)
  expect_equal(beta_orig[, 2], beta_scaled[, 2] * c_scale, tolerance = 1e-8)
})
