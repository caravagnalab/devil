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
#' @param parallel.cores Integer or NULL. Number of CPU cores for parallel processing.
#'   If NULL, uses all available cores. Default: NULL
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
#' \dontrun{
#' # Basic differential expression test
#' results <- test_de(fit, contrast = c(0, 1, -1))
#'
#' # With sample clustering and stricter fold change threshold
#' results <- test_de(fit, contrast = c(0, 1, -1),
#'                    clusters = patient_ids,
#'                    max_lfc = 5)
#' }
#'
#' @export
#' @rawNamespace useDynLib(devil);
test_de <- function(devil.fit, contrast, pval_adjust_method = "BH", max_lfc = 10, clusters = NULL, parallel.cores=NULL) {

  # Detect cores to use
  max.cores <- parallel::detectCores()
  if (is.null(parallel.cores)) {
    n.cores = max.cores
  } else {
    if (parallel.cores > max.cores) {
      message(paste0("Requested ", parallel.cores, " cores, but only ", max.cores, " available."))
    }
    n.cores = min(max.cores, parallel.cores)
  }

  # Extract necessary information
  ngenes <- nrow(devil.fit$input_matrix)
  nsamples <- nrow(devil.fit$design_matrix)
  contrast <- as.array(contrast)

  # Calculate log fold changes
  lfcs <- (devil.fit$beta %*% contrast) %>%
    unlist() %>%
    unname() %>%
    c()

  # Calculate p-values in parallel
  if (!is.null(clusters)) {

    if (!is.numeric(clusters)) {
      message("Converting clusters to numeric factors")
      clusters = as.numeric(as.factor(clusters))
    }

    p_values <- parallel::mclapply(1:nrow(devil.fit$input_matrix), function(gene_idx) {
      mu_test <- lfcs[gene_idx]

      H <- compute_sandwich(
        devil.fit$design_matrix,
        devil.fit$input_matrix[gene_idx,],
        devil.fit$beta[gene_idx,], devil.fit$overdispersion[gene_idx],
        devil.fit$size_factors,
        clusters
      )

      total_variance <- t(contrast) %*% H %*% contrast
      #1 - stats::pchisq(mu_test^2 / total_variance, df = 1)
      2 * stats::pt(abs(mu_test) / sqrt(total_variance), df = nsamples - 2, lower.tail = F)
    }, mc.cores = n.cores) %>% unlist()

  } else {
    p_values <- parallel::mclapply(1:ngenes, function(gene_idx) {
      mu_test <- lfcs[gene_idx]
      H <- compute_hessian(devil.fit$beta[gene_idx,], 1 / devil.fit$overdispersion[gene_idx], devil.fit$input_matrix[gene_idx,], devil.fit$design_matrix, devil.fit$size_factors)
      total_variance <- t(contrast) %*% H %*% contrast
      #1 - stats::pchisq(mu_test^2 / total_variance, df = 1)
      2 * stats::pt(abs(mu_test) / sqrt(total_variance), df = nsamples - 2, lower.tail = F)
    }, mc.cores = n.cores) %>% unlist()
  }

  # Create tibble with results
  result_df <- dplyr::tibble(
    name = rownames(devil.fit$beta),
    pval = p_values,
    adj_pval = stats::p.adjust(p_values, method = pval_adjust_method),
    lfc = lfcs / log(2)
  )

  # Filter results based on max_lfc
  result_df <- result_df %>%
    dplyr::mutate(lfc = ifelse(.data$lfc >= max_lfc, max_lfc, .data$lfc)) %>%
    dplyr::mutate(lfc = ifelse(.data$lfc <= -max_lfc, -max_lfc, .data$lfc))

  # if (sum(is.na(result_df))) {
  #   message('Warning: the results for some genes are unrealiable (i.e. NaN)\n This might be due to gene very lowly expressed or not expressed at all for some conditions')
  # }
  # if (sum(is.na(result_df))) {
  #   na_genes_idxs <- which(is.na(result_df$pval))
  #   dm <- as.matrix(devil.fit$design_matrix[,contrast != 0])
  #   cell_idx <- dm != 0
  #   dm <- as.matrix(dm[cell_idx,])
  #   tmp <- lapply(na_genes_idxs, function(gene_idx) {
  #     beta0 <- init_beta(t(devil.fit$input_matrix[gene_idx,cell_idx]), design_matrix = dm, offset_matrix = devil.fit$offset_matrix[gene_idx,cell_idx])
  #     new_beta <- beta_fit(devil.fit$input_matrix[gene_idx,cell_idx], X = dm, mu_beta = beta0, off = devil.fit$offset_matrix[gene_idx,cell_idx], k = 1 / devil.fit$overdispersion[gene_idx], max_iter = 500, eps = 1e-3)
  #     new_beta <- new_beta$mu_beta
  #     mu_test <- sum(new_beta %*% contrast)
  #     if (!is.null(clusters)) {
  #
  #       H <- compute_sandwich(
  #         devil.fit$design_matrix,
  #         devil.fit$input_matrix[gene_idx,],
  #         new_beta, devil.fit$overdispersion[gene_idx],
  #         devil.fit$size_factors,
  #         clusters
  #       )
  #
  #       total_variance <- t(contrast) %*% H %*% contrast
  #       new_pval <- 2 * stats::pt(abs(mu_test) / sqrt(total_variance), df = nsamples - 2, lower.tail = F)
  #     } else {
  #       H <- compute_hessian(new_beta, 1 / devil.fit$overdispersion[gene_idx], devil.fit$input_matrix[gene_idx,], devil.fit$design_matrix, devil.fit$size_factors)
  #       total_variance <- t(contrast) %*% H %*% contrast
  #       new_pval <- 2 * stats::pt(abs(mu_test) / sqrt(total_variance), df = nsamples - 2, lower.tail = F)
  #     }
  #     result_df$pval[gene_idx] <<- new_pval
  #   })
  #
  #   #message('Warning: the results for some genes are unrealiable (i.e. NaN)\n This might be due to gene very lowly expressed or not expressed at all for some conditions')
  # }

  result_df$adj_pval = stats::p.adjust(result_df$pval, method = pval_adjust_method)

  return(result_df)
}
