test_that("fit_devil (CPU, no size_factors) returns well-formed result", {
    set.seed(123)
    ngenes <- 5
    nsamples <- 8
    p <- 2

    input_matrix <- matrix(rpois(ngenes * nsamples, lambda = 10),
        nrow = ngenes, ncol = nsamples
    )
    rownames(input_matrix) <- paste0("g", seq_len(ngenes))

    design_matrix <- cbind(
        intercept = 1,
        x         = rnorm(nsamples)
    )

    res <- fit_devil(
        input_matrix   = input_matrix,
        design_matrix  = design_matrix,
        overdispersion = FALSE, # Poisson model, simplest
        size_factors   = NULL,
        CUDA           = FALSE,
        parallel.cores = 1,
        verbose        = FALSE
    )

    # Structure and names
    expect_type(res, "list")
    expect_true(all(c(
        "beta", "overdispersion", "iterations",
        "size_factors", "offset_vector",
        "design_matrix", "input_matrix", "input_parameters"
    ) %in% names(res)))

    # Dimensions
    expect_equal(dim(res$beta), c(ngenes, p))
    expect_equal(length(res$overdispersion), ngenes)
    expect_equal(length(res$iterations$beta_iters), ngenes)
    expect_equal(length(res$iterations$theta_iters), 1L) # scalar 0L in Poisson case

    # Size factors and offset
    expect_equal(res$size_factors, rep(1, nsamples))
    expect_length(res$offset_vector, nsamples)

    # Poisson overdispersion = FALSE ⇒ theta all zeros
    expect_true(all(res$overdispersion == 0))

    # Names preserved
    expect_equal(rownames(res$beta), rownames(input_matrix))

    # Inputs passed through
    expect_identical(res$design_matrix, design_matrix)
    expect_identical(res$input_matrix, input_matrix)

    # Input parameters stored
    expect_equal(res$input_parameters$parallel.cores, 1)
})


test_that("fit_devil uses calculate_sf when size_factors is a method string", {
    set.seed(1)
    ngenes <- 4
    nsamples <- 6

    input_matrix <- matrix(rpois(ngenes * nsamples, lambda = 10),
        nrow = ngenes, ncol = nsamples
    )
    rownames(input_matrix) <- paste0("g", seq_len(ngenes))
    design_matrix <- cbind(
        intercept = 1,
        x         = rnorm(nsamples)
    )

    # Explicitly compute size factors with calculate_sf
    sf_expected <- calculate_sf(input_matrix, method = "normed_sum", verbose = FALSE)

    res <- fit_devil(
        input_matrix   = input_matrix,
        design_matrix  = design_matrix,
        overdispersion = FALSE,
        size_factors   = "normed_sum",
        CUDA           = FALSE,
        parallel.cores = 1,
        verbose        = FALSE
    )

    expect_equal(res$size_factors, sf_expected)
})

test_that("cpu_fit in Poisson mode (overdispersion=FALSE) returns theta=0", {
    set.seed(10)
    ngenes <- 4
    nsamples <- 6

    input_matrix <- matrix(rpois(ngenes * nsamples, lambda = 5),
        nrow = ngenes, ncol = nsamples
    )
    rownames(input_matrix) <- paste0("g", seq_len(ngenes))
    design_matrix <- cbind(
        intercept = 1,
        x         = rnorm(nsamples)
    )
    offset_vector <- rep(0, nsamples)

    res <- cpu_fit(
        input_matrix = input_matrix,
        design_matrix = design_matrix,
        offset_vector = offset_vector,
        init_overdispersion = NULL,
        init_beta_rough = FALSE,
        overdispersion = FALSE,
        n.cores = 1,
        max_iter = 50,
        tolerance = 1e-3,
        verbose = FALSE
    )

    expect_equal(dim(res$beta), c(ngenes, ncol(design_matrix)))
    expect_equal(res$theta, rep(0, ngenes))
    expect_equal(res$iterations$theta_iters, 0L)
})

test_that("cpu_fit respects init_beta_rough flag for beta initialization", {
    set.seed(11)
    ngenes <- 3
    nsamples <- 5

    input_matrix <- matrix(rpois(ngenes * nsamples, lambda = 10),
        nrow = ngenes, ncol = nsamples
    )
    rownames(input_matrix) <- paste0("g", seq_len(ngenes))
    design_matrix <- cbind(
        intercept = 1,
        x         = rnorm(nsamples)
    )
    offset_vector <- rep(0, nsamples)

    res <- cpu_fit(
        input_matrix = input_matrix,
        design_matrix = design_matrix,
        offset_vector = offset_vector,
        init_overdispersion = NULL,
        init_beta_rough = TRUE,
        overdispersion = FALSE,
        n.cores = 1,
        max_iter = 5,
        tolerance = 1e-3,
        verbose = FALSE
    )

    # Rough init sets beta_0[,1] = log1p(rowMeans(counts)); a few beta_fit iterations update it
    # We just check it's at least in the right ballpark and finite
    expect_true(all(is.finite(res$beta[, 1])))
    expect_equal(length(res$iterations$beta_iters), ngenes)
})

test_that("cpu_fit with overdispersion='MOM' produces non-negative theta", {
    set.seed(12)
    ngenes <- 4
    nsamples <- 6

    input_matrix <- matrix(rpois(ngenes * nsamples, lambda = 8),
        nrow = ngenes, ncol = nsamples
    )
    rownames(input_matrix) <- paste0("g", seq_len(ngenes))
    design_matrix <- cbind(
        intercept = 1,
        x         = rnorm(nsamples)
    )
    offset_vector <- rep(0, nsamples)

    res <- cpu_fit(
        input_matrix = input_matrix,
        design_matrix = design_matrix,
        offset_vector = offset_vector,
        init_overdispersion = NULL,
        init_beta_rough = FALSE,
        overdispersion = "MOM",
        n.cores = 1,
        max_iter = 50,
        tolerance = 1e-3,
        verbose = FALSE
    )

    expect_equal(length(res$theta), ngenes)
    expect_true(all(res$theta >= 0))
    expect_equal(res$iterations$theta_iters, 0L)
})

test_that("cpu_fit with overdispersion='old' returns finite theta and NA theta_iters", {
    set.seed(13)
    ngenes <- 3
    nsamples <- 7

    input_matrix <- matrix(rpois(ngenes * nsamples, lambda = 5),
        nrow = ngenes, ncol = nsamples
    )
    rownames(input_matrix) <- paste0("g", seq_len(ngenes))
    design_matrix <- cbind(
        intercept = 1,
        x         = rnorm(nsamples)
    )
    offset_vector <- rep(0, nsamples)

    res <- cpu_fit(
        input_matrix = input_matrix,
        design_matrix = design_matrix,
        offset_vector = offset_vector,
        init_overdispersion = NULL,
        init_beta_rough = FALSE,
        overdispersion = "old",
        n.cores = 1,
        max_iter = 50,
        tolerance = 1e-3,
        verbose = FALSE
    )

    expect_equal(length(res$theta), ngenes)
    expect_true(all(is.finite(res$theta)))
    expect_true(all(!is.na(res$theta))) # should not be NA
    # theta_iters is documented/implemented as NA_integer_
    expect_true(all(is.na(res$iterations$theta_iters)))
})

test_that("cpu_fit with overdispersion='new' runs when fit_overdispersion_cppp is available", {
    ns <- asNamespace("devil")
    if (!exists("fit_overdispersion_cppp", envir = ns, inherits = FALSE)) {
        skip("fit_overdispersion_cppp not available (no compiled code).")
    }

    set.seed(14)
    ngenes <- 3
    nsamples <- 8

    input_matrix <- matrix(rpois(ngenes * nsamples, lambda = 6),
        nrow = ngenes, ncol = nsamples
    )
    rownames(input_matrix) <- paste0("g", seq_len(ngenes))
    design_matrix <- cbind(
        intercept = 1,
        x         = rnorm(nsamples)
    )
    offset_vector <- rep(0, nsamples)

    res <- cpu_fit(
        input_matrix = input_matrix,
        design_matrix = design_matrix,
        offset_vector = offset_vector,
        init_overdispersion = NULL,
        init_beta_rough = FALSE,
        overdispersion = "new",
        n.cores = 1,
        max_iter = 50,
        tolerance = 1e-3,
        verbose = FALSE
    )

    expect_equal(length(res$theta), ngenes)
    expect_true(all(is.finite(res$theta)))
    expect_equal(length(res$iterations$theta_iters), ngenes)
    expect_true(all(res$iterations$theta_iters >= 0))
})
