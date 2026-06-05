make_mock_fit <- function(ngenes, nsamples, p, lambda = 10, seed = 1,
                          sandwich_scale = 1) {
  set.seed(seed)
  beta <- matrix(rnorm(ngenes * p), nrow = ngenes, ncol = p)
  rownames(beta) <- paste0("g", seq_len(ngenes))

  # Minimal positive-definite sandwich matrices: scaled identity
  make_bread <- function() (sandwich_scale / nsamples) * diag(p)

  list(
    input_matrix        = matrix(rpois(ngenes * nsamples, lambda = lambda), nrow = ngenes),
    design_matrix       = matrix(rnorm(nsamples * p), nrow = nsamples, ncol = p),
    beta                = beta,
    overdispersion      = rep(0.5, ngenes),
    size_factors        = rep(1, nsamples),
    beta_sandwiches_null = lapply(seq_len(ngenes), function(i) make_bread()),
    beta_sandwiches      = vector("list", ngenes)  # NULLs = no clusters
  )
}

test_that("test_de returns tibble with correct dimensions and columns", {
  fit <- make_mock_fit(ngenes = 5, nsamples = 8, p = 3)
  contrast <- c(0, 1, -1)

  res <- test_de(fit, contrast = contrast, pval_adjust_method = "BH", max_lfc = 10)

  expect_s3_class(res, "tbl_df")
  expect_equal(nrow(res), 5L)
  expect_true(all(c("name", "pval", "adj_pval", "lfc") %in% names(res)))
})

test_that("test_de computes lfcs correctly and applies clipping", {
  ngenes   <- 4
  nsamples <- 6
  p        <- 2

  beta <- rbind(
    c(0,  0.5),
    c(0, -1),
    c(0,  4),   # huge positive
    c(0, -5)    # huge negative
  )
  rownames(beta) <- paste0("g", seq_len(ngenes))
  contrast <- c(0, 1)

  fit <- list(
    input_matrix        = matrix(1L, nrow = ngenes, ncol = nsamples),
    design_matrix       = cbind(1, rnorm(nsamples)),
    beta                = beta,
    overdispersion      = rep(0.5, ngenes),
    size_factors        = rep(1, nsamples),
    beta_sandwiches_null = lapply(seq_len(ngenes), function(i) (1 / nsamples) * diag(p)),
    beta_sandwiches      = vector("list", ngenes)
  )

  max_lfc <- 2
  res <- test_de(fit, contrast = contrast, max_lfc = max_lfc)

  expected_raw     <- as.numeric(beta %*% contrast) / log(2)
  expected_clipped <- pmin(pmax(expected_raw, -max_lfc), max_lfc)
  expect_equal(res$lfc, expected_clipped)
})

test_that("test_de uses requested p-value adjustment method", {
  fit      <- make_mock_fit(ngenes = 10, nsamples = 8, p = 2, seed = 2)
  contrast <- c(0, 1)

  res_BH <- test_de(fit, contrast, pval_adjust_method = "BH")
  res_BF <- test_de(fit, contrast, pval_adjust_method = "bonferroni")

  expect_equal(res_BH$pval, res_BF$pval)
  expect_true(all(res_BF$adj_pval >= res_BH$adj_pval))
})

test_that("test_de errors on old fit object missing beta_sandwiches_null", {
  old_fit <- list(
    input_matrix  = matrix(1L, nrow = 3, ncol = 4),
    design_matrix = cbind(1, rnorm(4)),
    beta          = matrix(rnorm(6), nrow = 3),
    overdispersion = rep(0.5, 3),
    size_factors   = rep(1, 4)
  )
  rownames(old_fit$beta) <- paste0("g", seq_len(3))

  expect_error(test_de(old_fit, contrast = c(0, 1)), "re-fit")
})

test_that("test_de returns more conservative p-values when clustered sandwiches have larger variance", {
  ngenes   <- 6
  nsamples <- 12
  p        <- 2
  set.seed(4)

  beta <- matrix(rnorm(ngenes * p), nrow = ngenes, ncol = p)
  rownames(beta) <- paste0("g", seq_len(ngenes))

  make_bread <- function(scale) (scale / nsamples) * diag(p)

  fit_nocluster <- list(
    input_matrix        = matrix(rpois(ngenes * nsamples, lambda = 8), nrow = ngenes),
    design_matrix       = cbind(1, rnorm(nsamples)),
    beta                = beta,
    overdispersion      = rep(0.5, ngenes),
    size_factors        = rep(1, nsamples),
    beta_sandwiches_null = lapply(seq_len(ngenes), function(i) make_bread(1)),
    beta_sandwiches      = vector("list", ngenes)  # NULLs → no clustered SE
  )

  fit_cluster <- fit_nocluster
  # Clustered sandwich: 5× larger variance → lower test stat → larger p-values
  fit_cluster$beta_sandwiches <- lapply(seq_len(ngenes), function(i) make_bread(5))

  contrast <- c(0, 1)
  res_nocluster <- test_de(fit_nocluster, contrast)
  res_cluster   <- test_de(fit_cluster,   contrast)

  expect_true(all(res_cluster$pval >= res_nocluster$pval))
})
