#' Create Volcano Plot for Differential Expression Results
#'
#' @description
#' Generates a customizable volcano plot visualizing differential expression results,
#' highlighting significant genes based on both fold change and statistical significance.
#' The plot supports various customization options including color schemes, point sizes,
#' and gene labeling.
#'
#' @details
#' The function creates a scatter plot with:
#' - X-axis: Log2 fold change
#' - Y-axis: -Log10 adjusted p-value
#' Points are colored based on significance categories:
#' 1. Non-significant: Neither p-value nor fold change threshold met
#' 2. LFC significant: Only fold change threshold met
#' 3. P-value significant: Only p-value threshold met
#' 4. Both significant: Both thresholds met
#'
#' The plot includes dashed lines indicating significance thresholds and optionally
#' labels genes meeting both significance criteria.
#'
#' @param devil.res A tibble from test_de() containing columns:
#'   - name: Gene identifiers
#'   - adj_pval: Adjusted p-values
#'   - lfc: Log2 fold changes
#' @param lfc_cut Numeric. Absolute log2 fold change threshold for significance.
#'   Default: 1
#' @param pval_cut Numeric. Adjusted p-value threshold for significance.
#'   Default: 0.05
#' @param labels Logical. Whether to label genes meeting both significance criteria.
#'   Default: TRUE
#' @param colors Character vector of length 4 specifying colors for:
#'   1. Non-significant genes
#'   2. Fold-change significant only
#'   3. P-value significant only
#'   4. Both significant
#'   Default: c("gray", "forestgreen", "steelblue", "indianred")
#' @param color_alpha Numeric between 0 and 1. Transparency level for points.
#'   Default: 0.7
#' @param point_size Numeric. Size of plotting points.
#'   Default: 1
#' @param center Logical. Whether to center the x-axis at zero.
#'   Default: TRUE
#' @param title Character. Plot title.
#'   Default: "Volcano plot"
#'
#' @return A ggplot2 object containing the volcano plot.
#'
#' @note
#' - Genes with adj_pval = 0 are assigned the smallest non-zero p-value in the dataset
#' - NA values are removed with a warning
#' - Gene labels are placed with overlap prevention
#'
#' @examples
#' set.seed(1)
#' y <- t(as.matrix(rnbinom(1000, 1, .1)))
#' fit <- devil::fit_devil(input_matrix = y, design_matrix = matrix(1, ncol = 1, nrow = 1000), verbose = T)
#' de_results <- devil::test_de(devil.fit = fit, contrast = c(1))
#'
#' # Basic volcano plot
#' plot_volcano(de_results)
#'
#' # Custom thresholds and colors
#' plot_volcano(de_results,
#'     lfc_cut = 2,
#'     pval_cut = 0.01,
#'     colors = c("grey80", "blue", "green", "red")
#' )
#'
#' # Without gene labels
#' de_results$name <- "fake gene"
#' plot_volcano(de_results, labels = FALSE)
#'
#' @export
plot_volcano <- function(
    devil.res,
    lfc_cut = 1,
    pval_cut = .05,
    labels = TRUE,
    colors = c("gray", "forestgreen", "steelblue", "indianred"),
    color_alpha = .7,
    point_size = 1,
    center = TRUE,
    title = "Volcano plot"
) {
    if (sum(is.na(devil.res))) {
        message("Warning: some of the reults are unrealiable (i.e. contains NaN)\n Those genes will not be displayed")
        devil.res <- stats::na.omit(devil.res)
    }

    d <- devil.res %>%
        dplyr::mutate(pval_filter = .data$adj_pval <= pval_cut, lfc_filter = abs(.data$lfc) >= lfc_cut) %>%
        dplyr::mutate(class = dplyr::if_else(.data$pval_filter & .data$lfc_filter, "p-value and lfc",
            dplyr::if_else(.data$pval_filter, "p-value",
                dplyr::if_else(.data$lfc_filter, "lfc", "non-significant")
            )
        )) %>%
        dplyr::mutate(class = factor(class, levels = c("non-significant", "lfc", "p-value", "p-value and lfc"))) %>%
        dplyr::mutate(label = dplyr::if_else(class == "p-value and lfc", .data$name, NA))


    if (sum(d$adj_pval == 0) > 0) {
        min_value <- min(d$adj_pval[d$adj_pval > 0])
        message(paste0(sum(d$adj_pval == 0), " genes have adjusted p-value equal to 0, will be set to ", min_value))
        d$adj_pval[d$adj_pval == 0] <- min_value
    }

    p <- d %>%
        ggplot2::ggplot(mapping = ggplot2::aes(x = .data$lfc, y = -log10(.data$adj_pval), col = .data$class, label = .data$label)) +
        ggplot2::geom_point(alpha = color_alpha, size = point_size) +
        ggplot2::theme_minimal() +
        ggplot2::labs(x = expression(Log[2] ~ FC), y = expression(-log[10] ~ Pvalue), col = "") +
        ggplot2::scale_color_manual(values = colors) +
        ggplot2::geom_vline(xintercept = c(-lfc_cut, lfc_cut), linetype = "dashed") +
        ggplot2::geom_hline(yintercept = -log10(pval_cut), linetype = "dashed") +
        ggplot2::ggtitle(title) +
        ggplot2::theme(legend.position = "bottom")

    if (center) {
        p <- p + ggplot2::xlim(c(-max(abs(d$lfc)), max(abs(d$lfc))))
    }
    if (labels) {
        p <- p + ggplot2::geom_text(col = "black", check_overlap = TRUE, na.rm = TRUE)
    }
    p
}
