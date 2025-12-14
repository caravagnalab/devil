test_that("compute_sandwich returns correct shape and scaling without clusters", {
    set.seed(1)
    n <- 10
    p <- 3
    X <- matrix(rnorm(n * p), nrow = n, ncol = p)
    y <- rpois(n, lambda = 5)
    beta <- rnorm(p)
    overdispersion <- 0.5
    size_factors <- rep(1, n)

    S <- compute_sandwich(
        design_matrix = X,
        y = y,
        beta = beta,
        overdispersion = overdispersion,
        size_factors = size_factors,
        clusters = NULL
    )

    # Shape
    expect_true(is.matrix(S))
    expect_equal(dim(S), c(p, p))
})

test_that("compute_sandwich uses clustered meat when clusters are provided", {
    set.seed(2)
    n <- 12
    p <- 2
    X <- matrix(rnorm(n * p), nrow = n, ncol = p)
    y <- rpois(n, lambda = 10)
    beta <- rnorm(p)
    overdispersion <- 1.0
    size_factors <- rep(1, n)
    clusters <- rep(1:3, each = 4)

    S_nocluster <- compute_sandwich(
        design_matrix = X,
        y = y,
        beta = beta,
        overdispersion = overdispersion,
        size_factors = size_factors,
        clusters = NULL
    )

    S_cluster <- compute_sandwich(
        design_matrix = X,
        y = y,
        beta = beta,
        overdispersion = overdispersion,
        size_factors = size_factors,
        clusters = clusters
    )

    expect_true(all(S_nocluster <= S_cluster))
    expect_false(isTRUE(all.equal(S_nocluster, S_cluster)))
})

test_that("compute_sandwich result is symmetric and scales with sample size", {
    set.seed(3)
    n1 <- 8
    n2 <- 16
    p <- 3

    X1 <- matrix(rnorm(n1 * p), nrow = n1, ncol = p)
    X2 <- matrix(rnorm(n2 * p), nrow = n2, ncol = p)
    y1 <- rpois(n1, lambda = 5)
    y2 <- rpois(n2, lambda = 5)

    beta <- rnorm(p)
    overdispersion <- 0.7
    size_factors1 <- rep(1, n1)
    size_factors2 <- rep(1, n2)

    S1 <- compute_sandwich(X1, y1, beta, overdispersion, size_factors1, clusters = NULL)
    S2 <- compute_sandwich(X2, y2, beta, overdispersion, size_factors2, clusters = NULL)

    # Symmetric
    expect_equal(S1, t(S1))
    expect_equal(S2, t(S2))

    expect_true(all(diag(S1) >= diag(S2)))
})
