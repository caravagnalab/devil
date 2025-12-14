test_that("calculate_sf returns unit size factors for single-row matrix", {
    Y <- matrix(1:5, nrow = 1)
    sf <- calculate_sf(Y, method = "normed_sum")

    expect_length(sf, ncol(Y))
    expect_equal(sf, rep(1, ncol(Y)))
})

test_that("calculate_sf matches method argument correctly", {
    Y <- matrix(1:12, nrow = 3)

    sf1 <- calculate_sf(Y, method = "normed_sum")
    sf2 <- calculate_sf(Y, method = "psinorm")

    expect_length(sf1, ncol(Y))
    expect_length(sf2, ncol(Y))

    expect_error(calculate_sf(Y, method = "foo"), "'arg' should be one of")
})

test_that("calculate_sf returns positive finite size factors with geometric mean 1", {
    set.seed(1)
    Y <- matrix(rpois(100, lambda = 10), nrow = 10)

    for (m in c("normed_sum", "psinorm")) {
        sf <- calculate_sf(Y, method = m)

        expect_length(sf, ncol(Y))
        expect_true(all(is.finite(sf)))
        expect_true(all(sf > 0))

        gm <- exp(mean(log(sf)))
        expect_equal(gm, 1, tolerance = 1e-8)
    }
})

test_that("calculate_sf agrees with helper functions for each method", {
    set.seed(123)
    Y <- matrix(rpois(60, lambda = 20), nrow = 6)

    expect_equal(calculate_sf(Y, method = "normed_sum"), normed_sum_sf(Y))
    expect_equal(calculate_sf(Y, method = "psinorm"), psinorm_sf(Y))
})

test_that("calculate_sf agrees with edgeR_sf when edgeR is installed", {
    skip_if_not_installed("edgeR")

    set.seed(123)
    Y <- matrix(rpois(60, lambda = 20), nrow = 6)

    sf1 <- calculate_sf(Y, method = "edgeR")
    sf2 <- edgeR_sf(Y)

    expect_equal(sf1, sf2)
})

test_that("normed_sum_sf matches manual geometric-mean normalization", {
    Y <- matrix(c(
        1, 2,
        3, 4,
        5, 6
    ), nrow = 3, byrow = TRUE)
    col_sums <- colSums(Y)
    gm <- exp(mean(log(col_sums)))
    sf_manual <- col_sums / gm

    sf <- normed_sum_sf(Y)

    expect_equal(sf, sf_manual)
})

test_that("normed_sum_sf errors when a column is all zeros", {
    Y <- matrix(c(
        0, 1,
        0, 2,
        0, 4
    ), nrow = 3, byrow = TRUE)

    expect_error(
        normed_sum_sf(Y),
        "At least one column \\(sample\\) contains all zeros"
    )
})

test_that("psinorm_sf returns positive finite size factors with geometric mean 1", {
    set.seed(42)
    Y <- matrix(rpois(200, lambda = 5), nrow = 20)

    sf <- psinorm_sf(Y)

    expect_length(sf, ncol(Y))
    expect_true(all(is.finite(sf)))
    expect_true(all(sf > 0))

    gm <- exp(mean(log(sf)))
    expect_equal(gm, 1, tolerance = 1e-8)
})

test_that("psinorm_sf is roughly monotone with respect to column scaling", {
    set.seed(123)
    base <- matrix(rpois(200, lambda = 10), nrow = 20)

    # Three columns: base, 2x, 4x
    Y <- cbind(base[, 1], 2 * base[, 1], 4 * base[, 1])

    sf <- psinorm_sf(Y)

    # They should be increasing (up to numerical noise)
    expect_true(sf[2] > sf[1])
    expect_true(sf[3] > sf[2])

    gm <- exp(mean(log(sf)))
    expect_equal(gm, 1, tolerance = 1e-8)
})

test_that("edgeR_sf returns positive finite size factors with geometric mean 1", {
    skip_if_not_installed("edgeR")

    set.seed(123)
    Y <- matrix(rpois(200, lambda = 15), nrow = 20)

    sf <- edgeR_sf(Y)

    expect_length(sf, ncol(Y))
    expect_true(all(is.finite(sf)))
    expect_true(all(sf > 0))

    gm <- exp(mean(log(sf)))
    expect_equal(gm, 1, tolerance = 1e-8)
})

test_that("edgeR_sf roughly increases with column scaling", {
    skip_if_not_installed("edgeR")

    set.seed(123)
    base <- matrix(rpois(200, lambda = 10), nrow = 20)

    Y <- cbind(base[, 1], 3 * base[, 1])
    sf <- edgeR_sf(Y)

    expect_true(sf[2] > sf[1])
})

test_that("compute_offset_matrix returns constant offset when no size_factors", {
    skip_if_not_installed("DelayedArray")

    off <- 2.5
    Y <- matrix(0, nrow = 3, ncol = 4)

    offset_mat <- compute_offset_matrix(off, Y, size_factors = NULL)
    offset_mat <- as.matrix(offset_mat)

    expect_equal(dim(offset_mat), dim(Y))
    expect_true(all(offset_mat == off))
})

test_that("compute_offset_vector returns constant vector when no size_factors", {
    off <- 1.2
    Y <- matrix(0, nrow = 5, ncol = 4)

    off_vec <- compute_offset_vector(off, Y, size_factors = NULL)

    expect_length(off_vec, ncol(Y))
    expect_true(all(off_vec == off))
})

test_that("compute_offset_vector correctly incorporates size_factors", {
    off <- 0.5
    Y <- matrix(0, nrow = 2, ncol = 3)
    sf <- c(1, 2, 4)

    off_vec <- compute_offset_vector(off, Y, size_factors = sf)

    expect_length(off_vec, ncol(Y))
    expect_equal(off_vec, off + log(sf))
})

test_that("compute_offset_matrix and compute_offset_vector are consistent per sample", {
    skip_if_not_installed("DelayedArray")

    off <- 0
    Y <- matrix(0, nrow = 3, ncol = 4)
    sf <- c(1, 2, 3, 4)

    mat <- as.matrix(compute_offset_matrix(off, Y, size_factors = sf))
    vec <- compute_offset_vector(off, Y, size_factors = sf)

    # Every row of mat should equal vec
    for (i in seq_len(nrow(mat))) {
        expect_equal(mat[i, ], vec)
    }
})
