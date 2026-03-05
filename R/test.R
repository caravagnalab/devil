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
#'   If NULL, uses all available cores. Default: 1
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
#'     input_matrix  = counts,
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
test_de <- function(devil.fit, contrast, clusters = NULL, pval_adjust_method = "BH", max_lfc = 10, parallel.cores = 1) {
  if (!("beta_sandwiches_null" %in% names(devil.fit))) {
    warning("You're using an old devil fit object. From next version, this functionality will be removed.")
    result_df = deprecated_test_de(devil.fit = devil.fit, contrast = contrast, pval_adjust_method = pval_adjust_method, max_lfc = max_lfc, clusters = clusters, parallel.cores = parallel.cores)
    return(result_df)
  } else {
    # Detect cores to use
    max.cores <- parallel::detectCores()
    if (is.null(parallel.cores)) {
      n.cores <- max.cores
    } else {
      if (parallel.cores > max.cores) {
        message("Requested ", parallel.cores, " cores, but only ",
                max.cores, " available.")
      }
      n.cores <- min(max.cores, parallel.cores)
    }

    ngenes <- nrow(devil.fit$input_matrix)
    nsamples <- nrow(devil.fit$design_matrix)
    contrast <- as.array(contrast)
    lfcs <- (devil.fit$beta %*% contrast) %>% unlist() %>% unname() %>% c()

    p_values = lapply(seq_len(nrow(devil.fit$input_matrix)), function(gene_idx) {
      mu_test <- lfcs[gene_idx]

      H = devil.fit$beta_sandwiches[[gene_idx]]
      if (is.null(H)) {
        H = devil.fit$beta_sandwiches_null[[gene_idx]]
        total_variance <- t(contrast) %*% H %*% contrast
        p <- 2 * stats::pt(abs(mu_test)/sqrt(total_variance), df = nsamples - 2, lower.tail = FALSE)
      } else {
        total_variance <- t(contrast) %*% H %*% contrast
        p_null <- 2 * stats::pt(abs(mu_test)/sqrt(total_variance), df = nsamples - 2, lower.tail = FALSE)

        H = devil.fit$beta_sandwiches_null[[gene_idx]]
        total_variance <- t(contrast) %*% H %*% contrast
        p <- 2 * stats::pt(abs(mu_test)/sqrt(total_variance), df = nsamples - 2, lower.tail = FALSE)
        p = max(p, p_null)
      }
      p
    }) %>% unlist()

    result_df <- dplyr::tibble(name = rownames(devil.fit$beta),
                               pval = p_values,
                               adj_pval = stats::p.adjust(p_values, method = pval_adjust_method),
                               lfc = lfcs/log(2))

    result_df <- result_df %>%
      dplyr::mutate(lfc = ifelse(.data$lfc >= max_lfc, max_lfc, .data$lfc)) %>%
      dplyr::mutate(lfc = ifelse(.data$lfc <= -max_lfc, -max_lfc, .data$lfc))

    return(result_df)
  }
}


deprecated_test_de <- function(devil.fit, contrast, pval_adjust_method = "BH", max_lfc = 10, clusters = NULL, parallel.cores = 1) {
  # Detect cores to use
  max.cores <- parallel::detectCores()
  if (is.null(parallel.cores)) {
    n.cores <- max.cores
  } else {
    if (parallel.cores > max.cores) {
      message("Requested ", parallel.cores, " cores, but only ", max.cores, " available.")
    }
    n.cores <- min(max.cores, parallel.cores)
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

  if (!is.null(clusters) & !is.numeric(clusters)) {
    message("Converting clusters to numeric factors")
    clusters <- as.numeric(as.factor(clusters))
  }

  # Calculate p-values in parallel
  p_values <- parallel::mclapply(seq_len(nrow(devil.fit$input_matrix)), function(gene_idx) {
    mu_test <- lfcs[gene_idx]
    n = dim(devil.fit$input_matrix)[2]

    b = devil:::compute_hessian(devil.fit$beta[gene_idx,],
                                devil.fit$overdispersion[gene_idx],
                                devil.fit$input_matrix[gene_idx,],
                                devil.fit$design_matrix,
                                devil.fit$size_factors)

    msimple = devil:::compute_meat(
      devil.fit$design_matrix,
      devil.fit$input_matrix[gene_idx,],
      devil.fit$beta[gene_idx,],
      devil.fit$overdispersion[gene_idx],
      devil.fit$size_factors
    )

    H = (b %*% msimple %*% b) * n
    total_variance <- t(contrast) %*% H %*% contrast
    p <- 2 * stats::pt(abs(mu_test) / sqrt(total_variance), df = nsamples - 2, lower.tail = FALSE)

    if (!is.null(clusters)) {
      m = devil:::compute_clustered_meat(devil.fit$design_matrix,
                                         devil.fit$input_matrix[gene_idx,],
                                         devil.fit$beta[gene_idx,],
                                         devil.fit$overdispersion[gene_idx],
                                         devil.fit$size_factors,
                                         clusters)
      S = (b %*% m %*% b) * n
      total_variance <- t(contrast) %*% S %*% contrast
      pnull <- 2 * stats::pt(abs(mu_test) / sqrt(total_variance), df = nsamples - 2, lower.tail = FALSE)
    } else {
      pnull <- 0
    }

    max(p, pnull)
  }, mc.cores = n.cores) %>% unlist()

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

  return(result_df)
}
