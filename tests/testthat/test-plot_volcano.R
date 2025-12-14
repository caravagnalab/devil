test_that("plot_volcano runs without error on simple input", {
    df <- dplyr::tibble(
        name = paste0("g", 1:10),
        adj_pval = runif(10),
        lfc = rnorm(10)
    )

    p <- plot_volcano(df)
    expect_s3_class(p, "ggplot")
})

test_that("plot_volcano assigns classes correctly", {
    df <- dplyr::tibble(
        name = c("A", "B", "C", "D"),
        adj_pval = c(0.1, 0.001, 0.1, 0.001),
        lfc = c(0.2, 0.2, 2, 2)
    )

    out <- suppressMessages(
        plot_volcano(df)$data %>% dplyr::select(name, class)
    )

    expect_equal(as.character(out$class), c(
        "non-significant", # A
        "p-value", # B
        "lfc", # C
        "p-value and lfc" # D
    ))
})

test_that("plot_volcano removes NA and prints message", {
    df <- dplyr::tibble(
        name = c("g1", "g2", "g3"),
        adj_pval = c(0.01, NA, 0.2),
        lfc = c(1, 0.5, -2)
    )

    expect_message(
        p <- plot_volcano(df),
        "Warning: some of the reults are unrealiable"
    )

    expect_equal(nrow(p$data), 2)
})

test_that("plot_volcano replaces zero adj_pval with smallest non-zero value", {
    df <- dplyr::tibble(
        name = paste0("g", 1:5),
        adj_pval = c(0, 0.01, 0.02, 0, 0.03),
        lfc = rnorm(5)
    )

    expect_message(
        p <- plot_volcano(df),
        "genes have adjusted p-value equal to 0"
    )

    # Extract modified p-values
    adj <- p$data$adj_pval

    # Minimum original non-zero p-value = 0.01
    expect_true(all(adj[p$data$adj_pval > 0] >= 0.01))
})

test_that("plot_volcano centers x-axis when center=TRUE", {
    df <- dplyr::tibble(
        name = paste0("g", 1:4),
        adj_pval = runif(4),
        lfc = c(-3, -1, 1, 2)
    )

    p <- plot_volcano(df, center = TRUE)
    rng <- p@scales$scales[[2]]$limits

    expect_equal(rng, c(-3, 3))
})

test_that("plot_volcano labels only genes meeting both criteria", {
    df <- dplyr::tibble(
        name = c("A", "B", "C"),
        adj_pval = c(0.1, 0.001, 0.001),
        lfc = c(0.5, 0.5, 3)
    )

    p <- plot_volcano(df, labels = TRUE)
    lbl <- p$data$label

    expect_equal(lbl, c(NA, NA, "C"))
})

test_that("plot_volcano accepts custom colors", {
    df <- dplyr::tibble(
        name = paste0("g", 1:3),
        adj_pval = c(0.1, 0.001, 0.001),
        lfc = c(0.2, 0.2, 2)
    )

    cols <- c("red", "green", "blue", "purple")
    p <- plot_volcano(df, colors = cols)

    expect_equal(p$scales$scales[[1]]$palette(4), cols)
})

test_that("plot_volcano sets point alpha and size", {
    df <- dplyr::tibble(
        name = "g1",
        adj_pval = 0.01,
        lfc = 2
    )

    p <- plot_volcano(df, color_alpha = 0.3, point_size = 5)
    layer <- p$layers[[1]] # geom_point layer

    expect_equal(layer$aes_params$alpha, 0.3)
    expect_equal(layer$aes_params$size, 5)
})

test_that("plot_volcano leaves x-axis unconstrained when center=FALSE", {
    df <- dplyr::tibble(
        name = paste0("g", 1:4),
        adj_pval = runif(4),
        lfc = c(-4, -2, 1, 3)
    )

    p <- plot_volcano(df, center = FALSE)

    expect_null(p$coordinates$limits$x)
})

test_that("plot_volcano works when no genes are significant", {
    df <- dplyr::tibble(
        name = paste0("g", 1:5),
        adj_pval = rep(0.5, 5),
        lfc = rep(0, 5)
    )

    p <- plot_volcano(df)

    expect_s3_class(p, "ggplot")
    expect_true(all(p$data$class == "non-significant"))
})
