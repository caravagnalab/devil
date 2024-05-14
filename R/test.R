
#' Test for Differential Expression
#'
#' This function tests for differential expression between conditions using the provided contrast matrix.
#'
#' @param devil.fit The fitted devil model.
#' @param contrast The contrast matrix for differential expression testing.
#' @param pval_adjust_method Method for adjusting p-values. Default is "BH" (Benjamini & Hochberg).
#' @param max_lfc Maximum absolute log fold change to consider for filtering results. Default is 10.
#' @param clusters .
#' @return A tibble containing the results of the differential expression testing.
#' @details This function computes log fold changes and p-values for each gene in parallel and filters the results based on the maximum absolute log fold change specified.
#' @export
#' @rawNamespace useDynLib(devil);
test_de <- function(devil.fit, contrast, pval_adjust_method = "BH", max_lfc = 10, clusters = NULL) {

  if (devil.fit$input_parameters$parallel) {
    n.cores = parallel::detectCores()
  } else {
    n.cores = 1
  }

  # Extract necessary information
  ngenes <- nrow(devil.fit$input_matrix)
  contrast <- as.array(contrast)

  # Calculate log fold changes
  lfcs <- (devil.fit$beta %*% contrast) %>%
    unlist() %>%
    unname() %>%
    c()

  # Calculate p-values in parallel
  if (!is.null(clusters)) {

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
      1 - stats::pchisq(mu_test^2 / total_variance, df = 1)
    }, mc.cores = n.cores) %>% unlist()

  } else {
    p_values <- parallel::mclapply(1:ngenes, function(gene_idx) {
      mu_test <- lfcs[gene_idx]
      H <- compute_hessian(devil.fit$beta[gene_idx,], 1 / devil.fit$overdispersion[gene_idx], devil.fit$input_matrix[gene_idx,], devil.fit$design_matrix, devil.fit$size_factors)
      total_variance <- t(contrast) %*% H %*% contrast
      1 - stats::pchisq(mu_test^2 / total_variance, df = 1)
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

  if (sum(is.na(result_df))) {
    message('Warning: the results for some genes are unrealiable (i.e. NaN)\n This might be due to gene very lowly expressed or not expressed at all for some conditions')
  }

  return(result_df)
}
