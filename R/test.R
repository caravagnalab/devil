#' Test for Differential Expression
#'
#' @description
#' Performs statistical testing for differential expression using results from a fitted
#' devil model. Supports both standard and robust (clustered) variance estimation,
#' with multiple testing correction and customizable fold change thresholds.
#'
#' @details
#' The function implements the following analysis pipeline:
#' 1. Calculates log fold changes using contrast vectors/matrices
#' 2. Computes test statistics using either standard or robust variance estimation
#' 3. Calculates p-values using t-distribution with appropriate degrees of freedom
#' 4. Adjusts p-values for multiple testing
#' 5. Applies fold change thresholding
#'
#' The variance estimation can account for sample clustering (e.g., multiple samples
#' from the same patient) using a sandwich estimator for robust inference.
#'
#' @param devil.fit A fitted model object from fit_devil().
#'   Must contain beta coefficients, design matrix, and overdispersion parameters.
#' @param contrast Numeric vector or matrix specifying the comparison of interest.
#'   Length must match number of coefficients in the model.
#'   For example, c(0, 1, -1) tests difference between second and third coefficient.
#' @param pval_adjust_method Character. Method for p-value adjustment.
#'   Passed to stats::p.adjust(). Common choices:
#'   - "BH": Benjamini-Hochberg (default)
#'   - "bonferroni": Bonferroni correction
#'   - "holm": Holm's step-down method
#' @param max_lfc Numeric. Maximum absolute log2 fold change to report.
#'   Larger values are capped at ±max_lfc. Default: 10
#' @param clusters Numeric vector or factor. Sample cluster assignments for robust
#'   variance estimation. Length must match number of samples. Default: NULL
#' @param BPPARAM A \code{\link[BiocParallel]{BiocParallelParam}} object controlling
#'   parallel evaluation. Default: \code{BiocParallel::SerialParam()}.
#'
#' @return A tibble with columns:
#' \describe{
#'   \item{name}{Character. Gene identifiers from input data}
#'   \item{pval}{Numeric. Raw p-values from statistical tests}
#'   \item{adj_pval}{Numeric. Adjusted p-values after multiple testing correction}
#'   \item{lfc}{Numeric. Log2 fold changes, capped at ±max_lfc}
#' }
#'
#' @examples
#' ## Example: test_de() on a simple two-group comparison
#' set.seed(1)
#'
#' # Simulate counts (genes x cells)
#' counts <- matrix(rnbinom(1000, mu = 0.2, size = 1), nrow = 100, ncol = 10)
#' rownames(counts) <- paste0("gene", seq_len(nrow(counts)))
#'
#' # Two-group design (no intercept)
#' group <- factor(rep(c("A", "B"), each = 5))
#' design <- model.matrix(~ 0 + group)
#' colnames(design) <- levels(group)
#'
#' # Fit model
#' fit <- fit_devil(
#'     x             = counts,
#'     design_matrix = design,
#'     size_factors  = "normed_sum"
#' )
#'
#' # Test A vs B (contrast = +1*A -1*B)
#' res <- test_de(
#'     fit,
#'     contrast = c(A = 1, B = -1)
#' )
#' head(res[order(res$adj_pval), ])
#'
#' ## Example: clustered (patient-aware) variance (sandwich SE)
#'
#' patient <- factor(rep(c("P1", "P2"), each = 5))
#'
#' res_clustered <- test_de(
#'     fit,
#'     contrast = c(A = 1, B = -1),
#'     clusters = patient
#' )
#'
#' head(res_clustered[order(res_clustered$adj_pval), ])
#'
#' @export
#' @rawNamespace useDynLib(devil);
test_de <- function(devil.fit, contrast, clusters = NULL, pval_adjust_method = "BH", max_lfc = 10,
                    BPPARAM = BiocParallel::SerialParam()) {
  if (!("beta_sandwiches_null" %in% names(devil.fit))) {
    stop(
      "The fit object was created with an older version of devil and is no longer supported. ",
      "Please re-fit your model with the current version of fit_devil()."
    )
  }

  nsamples <- nrow(devil.fit$design_matrix)
  contrast  <- as.array(contrast)
  lfcs      <- (devil.fit$beta %*% contrast) %>% unlist() %>% unname() %>% c()

  p_values <- BiocParallel::bplapply(
    seq_len(nrow(devil.fit$input_matrix)),
    function(gene_idx) {
      mu_test <- lfcs[gene_idx]

      H_clust <- devil.fit$beta_sandwiches[[gene_idx]]
      H_null  <- devil.fit$beta_sandwiches_null[[gene_idx]]

      if (!is.null(H_clust) && !is.null(H_null)) {
        total_variance <- t(contrast) %*% H_clust %*% contrast
        p_clust <- 2 * stats::pt(abs(mu_test) / sqrt(total_variance), df = nsamples - 2, lower.tail = FALSE)

        total_variance <- t(contrast) %*% H_null %*% contrast
        p_null <- 2 * stats::pt(abs(mu_test) / sqrt(total_variance), df = nsamples - 2, lower.tail = FALSE)

        max(p_clust, p_null)
      } else if (!is.null(H_null)) {
        total_variance <- t(contrast) %*% H_null %*% contrast
        2 * stats::pt(abs(mu_test) / sqrt(total_variance), df = nsamples - 2, lower.tail = FALSE)
      } else {
        NA_real_
      }
    },
    BPPARAM = BPPARAM
  ) %>% unlist()

  result_df <- dplyr::tibble(
    name     = rownames(devil.fit$beta),
    pval     = p_values,
    adj_pval = stats::p.adjust(p_values, method = pval_adjust_method),
    lfc      = lfcs / log(2)
  )

  result_df %>%
    dplyr::mutate(lfc = ifelse(.data$lfc >= max_lfc, max_lfc, .data$lfc)) %>%
    dplyr::mutate(lfc = ifelse(.data$lfc <= -max_lfc, -max_lfc, .data$lfc))
}
