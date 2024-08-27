#' Test for Differential Expression
#'
#' This function performs differential expression testing between conditions using the provided contrast matrix, based on a fitted `devil` model.
#'
#' @param devil.fit An object containing the fitted `devil` model, which is returned by the `fit_devil` function.
#' @param contrast A numeric vector or matrix specifying the contrast of interest for differential expression testing. The contrast defines the comparison between conditions.
#' @param pval_adjust_method A character string specifying the method to adjust p-values for multiple testing. (default is `"BH"` (Benjamini & Hochberg method))
#' @param max_lfc A numeric value specifying the maximum absolute log fold change to consider when filtering results. (default is `10`)
#' @param clusters An optional numeric or factor vector containing cluster IDs for each sample. This is useful in experimental settings where samples belong to different groups, such as different patients.
#'
#' @return A tibble containing the results of the differential expression testing. The tibble includes:
#' \item{name}{The gene names corresponding to the rows of the `devil.fit` model.}
#' \item{pval}{The p-values associated with the differential expression test for each gene.}
#' \item{adj_pval}{The adjusted p-values after applying the specified p-value adjustment method.}
#' \item{lfc}{The log fold changes for each gene, scaled by log base 2 and filtered by `max_lfc`.}
#'
#' @details
#' This function calculates log fold changes and p-values for each gene in parallel.
#' It first computes log fold changes by multiplying the beta coefficients from the fitted model with the specified contrast.
#' Then, it calculates p-values using either the sandwich variance estimator (if `clusters` is provided) or the Hessian matrix.
#' The results are adjusted for multiple testing using the specified p-value adjustment method.
#'
#' The results are filtered based on the specified maximum absolute log fold change (`max_lfc`), ensuring that extreme log fold changes are capped at this value.
#'
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
