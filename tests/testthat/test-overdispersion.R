
test_that("estimate_dispersion matches manual method-of-moments formula without offsets", {
  set.seed(1)
  count_matrix <- matrix(rpois(50, lambda = 10), nrow = 5)  # 5 genes, 10 samples
  offset_vector <- rep(0, ncol(count_matrix))

  disp <- estimate_dispersion(count_matrix, offset_vector)

  # Manual calculation per gene
  var_manual  <- DelayedMatrixStats::rowVars(count_matrix)
  mean_manual <- DelayedMatrixStats::rowMeans2(count_matrix)
  mom <- (var_manual - mean_manual) / (mean_manual^2)
  mom[mom < 0 | is.na(mom)] <- 0.01

  expect_equal(disp, mom)
})

test_that("estimate_dispersion depends on offset_vector", {
  set.seed(2)
  count_matrix <- matrix(rpois(100, lambda = 20), nrow = 10)
  offset0 <- rep(0, ncol(count_matrix))
  offset1 <- rep(log(2), ncol(count_matrix))  # exp(offset) = 2

  disp0 <- estimate_dispersion(count_matrix, offset0)
  disp1 <- estimate_dispersion(count_matrix, offset1)

  # The two should not be identical
  expect_false(isTRUE(all.equal(disp0, disp1)))
})

test_that("estimate_dispersion truncates negative or NA estimates to 0.01", {
  # 1) Pure Poisson: var ≈ mean
  set.seed(3)
  count_matrix <- matrix(rpois(50, lambda = 5), nrow = 5)
  offset_vector <- rep(0, ncol(count_matrix))
  disp <- estimate_dispersion(count_matrix, offset_vector)

  expect_true(all(disp >= 0.01))
  expect_true(any(disp == 0.01))  # at least some genes truncated

  # 2) All-zero gene: mean = var = 0 → NA → 0.01
  cm2 <- rbind(0 * count_matrix[1, ], count_matrix)
  disp2 <- estimate_dispersion(cm2, offset_vector)

  expect_equal(disp2[1], 0.01)
})

test_that("fit_dispersion returns 0 when all counts are zero", {
  n <- 20
  y <- rep(0L, n)
  X <- matrix(1, nrow = n, ncol = 1)
  beta <- 0
  offset_matrix <- matrix(0, nrow = n, ncol = 1)

  disp <- fit_dispersion(
    beta = beta,
    model_matrix = X,
    y = y,
    offset_matrix = offset_matrix,
    tolerance = 1e-8,
    max_iter = 50,
    do_cox_reid_adjustment = TRUE
  )

  expect_identical(disp, 0)
})

test_that("fit_dispersion yields small dispersion for near-Poisson data", {
  set.seed(10)
  n <- 200
  mu_true <- 10
  y <- rpois(n, lambda = mu_true)

  X <- matrix(1, nrow = n, ncol = 1)        # intercept-only
  beta <- log(mu_true)                      # so that exp(X %*% beta) = mu_true
  offset_matrix <- matrix(0, nrow = n, ncol = 1)

  disp_hat <- fit_dispersion(
    beta = beta,
    model_matrix = X,
    y = y,
    offset_matrix = offset_matrix,
    tolerance = 1e-8,
    max_iter = 100,
    do_cox_reid_adjustment = TRUE
  )

  expect_true(is.finite(disp_hat))
  expect_true(disp_hat >= 0)
  # For Poisson-like data this should be quite small
  expect_lt(disp_hat, 0.2)
})

test_that("fit_dispersion gives larger dispersion for NB data than Poisson", {
  set.seed(11)
  n <- 400
  mu_true <- 10

  # Poisson data
  y_pois <- rpois(n, lambda = mu_true)

  # NB data with dispersion phi_true
  phi_true <- 0.5
  size_true <- 1 / phi_true
  y_nb <- rnbinom(n, mu = mu_true, size = size_true)

  X <- matrix(1, nrow = n, ncol = 1)
  beta <- log(mu_true)
  offset_matrix <- matrix(0, nrow = n, ncol = 1)

  disp_pois <- fit_dispersion(
    beta, X, y_pois, offset_matrix,
    tolerance = 1e-8, max_iter = 100, do_cox_reid_adjustment = TRUE
  )

  disp_nb <- fit_dispersion(
    beta, X, y_nb, offset_matrix,
    tolerance = 1e-8, max_iter = 100, do_cox_reid_adjustment = TRUE
  )

  expect_true(disp_nb > disp_pois)
  expect_true(is.finite(disp_nb))

  # Optional: check it is in the same ballpark as phi_true
  expect_equal(disp_nb, phi_true, tolerance = 0.3)
})

test_that("fit_dispersion depends on Cox-Reid adjustment flag", {
  set.seed(12)
  n <- 200
  mu_true <- 5
  phi_true <- 0.8
  size_true <- 1 / phi_true

  y <- rnbinom(n, mu = mu_true, size = size_true)
  X <- matrix(1, nrow = n, ncol = 1)
  beta <- log(mu_true)
  offset_matrix <- matrix(0, nrow = n, ncol = 1)

  disp_cr  <- fit_dispersion(
    beta, X, y, offset_matrix,
    tolerance = 1e-8, max_iter = 100, do_cox_reid_adjustment = TRUE
  )
  disp_ncr <- fit_dispersion(
    beta, X, y, offset_matrix,
    tolerance = 1e-8, max_iter = 100, do_cox_reid_adjustment = FALSE
  )

  expect_true(is.finite(disp_cr))
  expect_true(is.finite(disp_ncr))
  # They often differ; if in some unlucky random seed they don't, change seed.
  expect_false(isTRUE(all.equal(disp_cr, disp_ncr)))
})
