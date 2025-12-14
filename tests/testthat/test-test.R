test_that("test_de returns tibble with correct dimensions and columns", {
    ngenes <- 5
    nsamples <- 8
    p <- 3 # number of coefficients

    set.seed(1)
    devil.fit <- list(
        input_matrix   = matrix(rpois(ngenes * nsamples, lambda = 10), nrow = ngenes),
        design_matrix  = matrix(rnorm(nsamples * p), nrow = nsamples, ncol = p),
        beta           = matrix(rnorm(ngenes * p), nrow = ngenes, ncol = p),
        overdispersion = rep(0.5, ngenes),
        size_factors   = rep(1, nsamples)
    )
    rownames(devil.fit$beta) <- paste0("g", seq_len(ngenes))

    contrast <- c(0, 1, -1)

    res <- test_de(
        devil.fit,
        contrast = contrast,
        pval_adjust_method = "BH",
        max_lfc = 10,
        clusters = NULL,
        parallel.cores = 1
    )

    expect_s3_class(res, "tbl_df")
    expect_equal(nrow(res), ngenes)
    expect_true(all(c("name", "pval", "adj_pval", "lfc") %in% names(res)))
})

test_that("test_de computes lfcs correctly and applies clipping", {
    ngenes <- 4
    nsamples <- 6
    p <- 2

    # Construct betas so we know the exact lfcs
    beta <- rbind(
        c(0, 0.5),
        c(0, -1),
        c(0, 4), # huge positive
        c(0, -5) # huge negative
    )
    rownames(beta) <- paste0("g", 1:ngenes)

    contrast <- c(0, 1) # test the second coefficient

    devil.fit <- list(
        input_matrix   = matrix(1L, nrow = ngenes, ncol = nsamples),
        design_matrix  = cbind(1, rnorm(nsamples)),
        beta           = beta,
        overdispersion = rep(0.5, ngenes),
        size_factors   = rep(1, nsamples)
    )

    max_lfc <- 2
    res <- test_de(
        devil.fit,
        contrast = contrast,
        pval_adjust_method = "BH",
        max_lfc = max_lfc,
        clusters = NULL,
        parallel.cores = 1
    )

    # Expected *unclipped* lfcs on log2 scale: beta %*% contrast divided by log(2)
    expected_raw <- as.numeric(beta %*% contrast) / log(2)
    expected_clipped <- pmin(pmax(expected_raw, -max_lfc), max_lfc)

    expect_equal(res$lfc, expected_clipped)
})

test_that("test_de uses requested p-value adjustment method", {
    ngenes <- 10
    nsamples <- 8
    p <- 2

    set.seed(2)
    devil.fit <- list(
        input_matrix   = matrix(rpois(ngenes * nsamples, lambda = 5), nrow = ngenes),
        design_matrix  = cbind(1, rnorm(nsamples)),
        beta           = matrix(rnorm(ngenes * p), nrow = ngenes, ncol = p),
        overdispersion = rep(0.5, ngenes),
        size_factors   = rep(1, nsamples)
    )
    rownames(devil.fit$beta) <- paste0("g", seq_len(ngenes))

    contrast <- c(0, 1)

    res_BH <- test_de(devil.fit, contrast, pval_adjust_method = "BH", max_lfc = 10, clusters = NULL, parallel.cores = 1)
    res_BF <- test_de(devil.fit, contrast, pval_adjust_method = "bonferroni", max_lfc = 10, clusters = NULL, parallel.cores = 1)

    # Same raw p-values, different adjustments
    expect_equal(res_BH$pval, res_BF$pval)
    expect_true(all(res_BF$adj_pval >= res_BH$adj_pval))
})

test_that("test_de converts factor clusters to numeric and runs", {
    ngenes <- 5
    nsamples <- 9
    p <- 2

    set.seed(3)
    devil.fit <- list(
        input_matrix   = matrix(rpois(ngenes * nsamples, lambda = 10), nrow = ngenes),
        design_matrix  = cbind(1, rnorm(nsamples)),
        beta           = matrix(rnorm(ngenes * p), nrow = ngenes, ncol = p),
        overdispersion = rep(0.5, ngenes),
        size_factors   = rep(1, nsamples)
    )
    rownames(devil.fit$beta) <- paste0("g", seq_len(ngenes))

    contrast <- c(0, 1)
    clusters_factor <- factor(rep(letters[1:3], each = 3))

    expect_message(
        res <- test_de(
            devil.fit, contrast,
            clusters = clusters_factor,
            parallel.cores = 1
        ),
        "Converting clusters to numeric factors"
    )

    expect_s3_class(res, "tbl_df")
    expect_equal(nrow(res), ngenes)
})

test_that("test_de returns more conservative p-values when using clustered sandwich", {
    ngenes <- 6
    nsamples <- 12
    p <- 2

    set.seed(4)
    devil.fit <- list(
        input_matrix   = matrix(rpois(ngenes * nsamples, lambda = 8), nrow = ngenes),
        design_matrix  = cbind(1, rnorm(nsamples)),
        beta           = matrix(rnorm(ngenes * p), nrow = ngenes, ncol = p),
        overdispersion = rep(0.5, ngenes),
        size_factors   = rep(1, nsamples)
    )
    rownames(devil.fit$beta) <- paste0("g", seq_len(ngenes))

    contrast <- c(0, 1)
    clusters <- rep(1:3, each = 4)

    res_nocluster <- test_de(
        devil.fit, contrast,
        clusters = NULL,
        parallel.cores = 1
    )
    res_cluster <- test_de(
        devil.fit, contrast,
        clusters = clusters,
        parallel.cores = 1
    )

    # With the sandwich, using clusters inflates the variance,
    # hence makes p-values larger. And test_de takes max(p, pnull),
    # so res_cluster$pval should be >= res_nocluster$pval elementwise.
    expect_true(all(res_cluster$pval >= res_nocluster$pval))
})
