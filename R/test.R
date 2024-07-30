
#' Test for Differential Expression
#'
#' This function tests for differential expression between conditions using the provided contrast matrix.
#'
#' @param devil.fit The fitted devil model.
#' @param contrast The contrast matrix for differential expression testing.
#' @param pval_adjust_method Method for adjusting p-values. Default is "BH" (Benjamini & Hochberg).
#' @param max_lfc Maximum absolute log fold change to consider for filtering results. Default is 10.
#' @param clusters Vector containing cluster id for each cell.
#'  For example, might contain donor id in a multi patient experimental setting.
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
  if (sum(is.na(result_df))) {
    na_genes_idxs <- which(is.na(result_df$pval))
    dm <- as.matrix(devil.fit$design_matrix[,contrast != 0])
    cell_idx <- dm != 0
    dm <- as.matrix(dm[cell_idx,])
    tmp <- lapply(na_genes_idxs, function(gene_idx) {
      beta0 <- devil:::init_beta(t(devil.fit$input_matrix[gene_idx,cell_idx]), design_matrix = dm, offset_matrix = devil.fit$offset_matrix[gene_idx,cell_idx])
      new_beta <- devil:::beta_fit(devil.fit$input_matrix[gene_idx,cell_idx], X = dm, mu_beta = beta0, off = devil.fit$offset_matrix[gene_idx,cell_idx], k = 1 / devil.fit$overdispersion[gene_idx], max_iter = 500, eps = 1e-3)
      new_beta <- new_beta$mu_beta
      mu_test <- sum(new_beta %*% contrast)
      if (!is.null(clusters)) {

        H <- devil:::compute_sandwich(
          devil.fit$design_matrix,
          devil.fit$input_matrix[gene_idx,],
          new_beta, devil.fit$overdispersion[gene_idx],
          devil.fit$size_factors,
          clusters
        )

        total_variance <- t(contrast) %*% H %*% contrast
        new_pval <- 2 * stats::pt(abs(mu_test) / sqrt(total_variance), df = nsamples - 2, lower.tail = F)
      } else {
        H <- devil:::compute_hessian(new_beta, 1 / devil.fit$overdispersion[gene_idx], devil.fit$input_matrix[gene_idx,], devil.fit$design_matrix, devil.fit$size_factors)
        total_variance <- t(contrast) %*% H %*% contrast
        new_pval <- 2 * stats::pt(abs(mu_test) / sqrt(total_variance), df = nsamples - 2, lower.tail = F)
      }
      result_df$pval[gene_idx] <<- new_pval
    })

    #message('Warning: the results for some genes are unrealiable (i.e. NaN)\n This might be due to gene very lowly expressed or not expressed at all for some conditions')
  }

  result_df$adj_pval = stats::p.adjust(result_df$pval, method = pval_adjust_method)

  return(result_df)
}
