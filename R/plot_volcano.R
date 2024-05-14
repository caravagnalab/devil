
#' Volcano Plot
#'
#' Plot a volcano plot for differential expression analysis.
#'
#' @param devil.res A data frame or tibble containing the results of differential expression analysis of the function `test_de`.
#' @param lfc_cut The threshold for absolute log-fold change. Genes with absolute log-fold change greater than or equal to this value are highlighted.
#' @param pval_cut The threshold for adjusted p-value. Genes with adjusted p-value less than or equal to this value are highlighted.
#' @param labels Logical indicating whether to label significant genes on the plot.
#' @param colors A vector of colors to use for different classes of genes.
#' @param color_alpha The alpha value for point colors.
#' @param point_size The size of points in the plot.
#' @param center Logical indicating whether to center the x-axis at zero.
#' @param title The title of the plot.
#'
#' @return A ggplot object representing the volcano plot.
#'
#' @details This function creates a volcano plot for visualizing differential expression analysis results.
#'          It highlights genes based on their log-fold change and adjusted p-values.
#'          Genes meeting the specified thresholds for both log-fold change and adjusted p-value are labeled.
#'
#' @export
plot_volcano <- function(
    devil.res,
    lfc_cut=1,
    pval_cut=.05,
    labels=TRUE,
    colors=c("gray", "forestgreen", "steelblue", "indianred"),
    color_alpha=.7,
    point_size=1,
    center=TRUE,
    title="Volcano plot") {

  if (sum(is.na(devil.res))) {
    message('Warning: some of the reults are unrealiable (i.e. contains NaN)\n Those genes will not be displayed')
    devil.res <- stats::na.omit(devil.res)
  }

  d <- devil.res %>%
    dplyr::mutate(pval_filter = .data$adj_pval <= pval_cut, lfc_filter = abs(.data$lfc) >= lfc_cut) %>%
    dplyr::mutate(class = dplyr::if_else(.data$pval_filter & .data$lfc_filter, "p-value and lfc",
                                        dplyr::if_else(.data$pval_filter, "p-value",
                                                      dplyr::if_else(.data$lfc_filter, "lfc", "non-significant")))) %>%
    dplyr::mutate(class = factor(class, levels = c("non-significant", "lfc", 'p-value', 'p-value and lfc'))) %>%
    dplyr::mutate(label = dplyr::if_else(class == 'p-value and lfc', .data$name, NA))


  if (sum(d$adj_pval == 0) > 0) {
    message(paste0(sum(d$adj_pval == 0), ' genes have adjusted p-value equal to 0, will be set to 1e-16'))
    d$adj_pval[d$adj_pval == 0] <- 1e-16
  }

  p <- d %>%
    ggplot2::ggplot(mapping = ggplot2::aes(x=.data$lfc, y=-log10(.data$adj_pval), col=.data$class, label=.data$label)) +
    ggplot2::geom_point(alpha = color_alpha, size=point_size) +
    ggplot2::theme_minimal() +
    ggplot2::labs(x = expression(Log[2] ~ FC), y = expression(-log[10] ~ Pvalue), col="") +
    ggplot2::scale_color_manual(values = colors) +
    ggplot2::geom_vline(xintercept = c(-lfc_cut, lfc_cut), linetype = 'dashed') +
    ggplot2::geom_hline(yintercept = -log10(pval_cut), linetype = "dashed") +
    ggplot2::ggtitle(title) +
    ggplot2::theme(legend.position = 'bottom')

  if (center) { p <- p + ggplot2::xlim(c(-max(abs(d$lfc)), max(abs(d$lfc))))}
  if (labels) { p <- p + ggplot2::geom_text(col = 'black', check_overlap = TRUE, na.rm = TRUE)}
  p
}
