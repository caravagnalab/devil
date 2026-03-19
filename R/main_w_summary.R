#' Fit Statistical Model for Count Data
#'
#' @description
#' Fits a statistical model to count data, particularly designed for RNA-seq
#' data analysis. The function estimates regression coefficients (beta),
#' gene-wise overdispersion parameters, and normalizes data using size factors.
#' It supports both CPU and (optionally) GPU-based computation with
#' parallel processing capabilities.
#'
#' @details
#' The function implements a negative binomial regression model with
#' the following steps:
#' \enumerate{
#'   \item Computes size factors for data normalization (if requested)
#'   \item Initializes model parameters including beta coefficients and overdispersion
#'   \item Fits the regression coefficients using either CPU (parallel) or GPU computation
#'   \item Optionally fits/updates overdispersion parameters using one of several strategies
#' }
#'
#' The model fitting process uses iterative optimization with configurable convergence
#' criteria and maximum iterations. For large datasets, the GPU implementation processes
#' genes in batches for improved memory efficiency.
#'
#' @section Size Factor Methods:
#' Three normalization methods are available when \code{size_factors} is a character string:
#' \itemize{
#'   \item \code{"normed_sum"} (default): Geometric mean normalization based on library sizes.
#'     Fast and works well for most datasets.
#'   \item \code{"psinorm"}: Psi-normalization using Pareto distribution MLE.
#'     More robust to highly variable genes.
#'   \item \code{"edgeR"}: edgeR's TMM with singleton pairing method.
#'     Requires the edgeR package from Bioconductor.
#' }
#' If \code{size_factors = NULL}, no normalization is performed and all size factors are set to 1.
#'
#' @section Overdispersion Strategies:
#' The \code{overdispersion} argument controls how gene-wise overdispersion is handled:
#' \itemize{
#'   \item \code{"old"} or \code{"MLE"}: Overdispersion is fit via the original (legacy) NB
#'         MLE-based procedure (with Cox-Reid adjustment inside \code{fit_dispersion()}).
#'   \item \code{"new"} or \code{"I"}: Overdispersion is fit via the new iterative NB
#'         routine implemented in \code{fit_overdispersion_cppp()}, typically faster and
#'         more stable for large single-cell datasets.
#'   \item \code{"MOM"}: Overdispersion is estimated using a method-of-moments approach via
#'         \code{estimate_mom_dispersion_cpp()}, which is cheap and provides a rough
#'         dispersion estimate.
#'   \item \code{FALSE}: Disable overdispersion fitting and use a Poisson model
#'         (overdispersion fixed to 0).
#' }
#'
#' @param input_matrix A numeric matrix of count data (genes x samples).
#'   Rows represent genes/features, columns represent samples/cells.
#' @param design_matrix A numeric matrix of predictor variables (samples x predictors).
#'   Each row corresponds to a sample, each column to a predictor variable.
#'   Must have \code{nrow(design_matrix) == ncol(input_matrix)}.
#' @param overdispersion Character or logical. Strategy for estimating overdispersion:
#'   one of \code{"new"}, \code{"I"}, \code{"old"}, \code{"MLE"}, \code{"MOM"}, or
#'   \code{FALSE} to disable overdispersion fitting (Poisson model).
#'   Default: \code{"MOM"}.
#' @param init_overdispersion Numeric scalar or \code{NULL}. Initial value for the
#'   overdispersion parameter used as a starting point for the iterative procedures.
#'   If \code{NULL}, an initial value is estimated from the data via \code{estimate_dispersion()}.
#'   Recommended value if specified: \code{100}. Default: \code{NULL}.
#' @param init_beta_rough Logical. Whether to initialize betas in a rough but extremely fast way.
#'   Default: \code{FALSE}.
#' @param offset Numeric scalar. Value used when computing the offset vector to avoid
#'   numerical issues with zero counts. Default: \code{0}.
#' @param size_factors Character string or \code{NULL}. Method for computing normalization
#'   factors to account for different sequencing depths. Options are:
#'   \itemize{
#'     \item \code{NULL} (default): No normalization (all size factors set to 1)
#'     \item \code{"normed_sum"}: Geometric mean normalization
#'     \item \code{"psinorm"}: Psi-normalization
#'     \item \code{"edgeR"}: edgeR TMM method
#'   }
#' @param verbose Logical. Whether to print progress messages during execution.
#'   Default: \code{FALSE}.
#' @param max_iter Integer. Maximum number of iterations for parameter optimization
#'   (both beta and overdispersion routines). Default: \code{200}.
#' @param tolerance Numeric. Convergence criterion for parameter optimization.
#'   Default: \code{1e-3}.
#' @param CUDA Logical. Whether to use GPU acceleration (requires CUDA support and
#'   a compiled \code{beta_fit_gpu()} implementation). Default: \code{FALSE}.
#' @param batch_size Integer. Number of genes to process per batch in GPU mode.
#'   Only relevant if \code{CUDA = TRUE}. Default: \code{1024}.
#' @param parallel.cores Integer or \code{NULL}. Number of CPU cores for parallel
#'   processing with \code{parallel::mclapply}. If \code{NULL}, uses all available cores.
#'   Default: \code{1}.
#'
#' @return A list containing:
#' \describe{
#'   \item{beta}{Matrix of fitted coefficients (genes x predictors).}
#'   \item{overdispersion}{Vector of fitted overdispersion parameters (one per gene).}
#'   \item{iterations}{List with elements \code{beta_iters} and \code{theta_iters}
#'         giving the number of iterations used for each gene.}
#'   \item{size_factors}{Vector of computed size factors (one per sample).}
#'   \item{offset_vector}{Vector of offset values used in the model (length = number of samples).}
#'   \item{design_matrix}{Input design matrix (as provided, possibly coerced to numeric matrix).}
#'   \item{input_matrix}{Input count matrix (as provided, possibly coerced to numeric matrix).}
#'   \item{input_parameters}{List of used parameter values
#'         (\code{max_iter}, \code{tolerance}, \code{parallel.cores}).}
#' }
#'
#' @examples
#' ## Example: fit a simple two-group model
#' set.seed(1)
#'
#' # Simulate a small counts matrix (genes x cells)
#' counts <- matrix(
#'     rnbinom(1000, mu = 0.2, size = 1),
#'     nrow = 100, ncol = 10
#' )
#' rownames(counts) <- paste0("gene", seq_len(nrow(counts)))
#'
#' # Two-group design (no intercept)
#' group <- factor(rep(c("A", "B"), each = 5))
#' design <- model.matrix(~ 0 + group)
#' colnames(design) <- levels(group)
#'
#' # Fit the model
#' fit <- fit_devil(
#'     input_matrix  = counts,
#'     design_matrix = design,
#'     size_factors  = "normed_sum",
#'     verbose       = TRUE
#' )
#'
#' @export
#' @rawNamespace useDynLib(devil);
fit_devil_summary <- function(
    input_matrix,
    design_matrix,
    clusters,
    offset = 0,
    size_factors = NULL,
    verbose = FALSE,
    max_iter = 200,
    tolerance = 1e-3,
    CUDA = FALSE,
    batch_size = 1024L,
    parallel.cores = 1
) {
  # - Input parameters ----
  gene_names <- rownames(input_matrix)
  ngenes     <- nrow(input_matrix)
  nfeatures  <- ncol(design_matrix)
  
  # Check input clusters
  if (!is.null(clusters)) {
    clusters <- as.numeric(factor(clusters, levels = unique(clusters)))
    if (is.unsorted(clusters)) stop("Input data should be grouped in blocks of patients. Use the `group_data` function to group them.")
    cluster_blocks_indexes <- cumsum(rle(clusters)$lengths)
  } else {
    cluster_blocks_indexes <- NULL
  }
  
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
  
  # Check if CUDA is available
  CUDA_is_available <- FALSE
  if (CUDA) {
    if (!exists("beta_fit_gpu", mode = "function")) {
      warning(
        "CUDA support was not enabled during package installation. ",
        "The beta_fit_gpu function is not available. ",
        "Reinstall with configure.args='--with-cuda' to enable GPU acceleration: ",
        "devtools::install_github(\"caravagnalab/devil\", force=TRUE, configure.args=\"--with-cuda\"). ",
        "Falling back to CPU computation."
      )
      CUDA <- FALSE
    } else {
      message("CUDA support detected - using GPU acceleration")
      CUDA_is_available <- TRUE
    }
  }
  
  # - Compute size factors ----
  if (!is.null(size_factors)) {
    if (verbose) message("Compute size factors")
    sf <- devil:::calculate_sf(input_matrix, method = size_factors, verbose = verbose)
    
    if (!is.null(clusters)) {
      sf <- ave(sf, clusters, FUN = median)
    }
  } else {
    sf <- rep(1, nrow(design_matrix))
  }
  
  # - Compute offset vector ----
  offset_vector <- devil:::compute_offset_vector(offset, input_matrix, sf)
  
  # Compute blueprint
  blueprint = get_group_blueprint(design_matrix, offset_vector, clusters)
  
  # - GPU vs CPU branch ----
  if (CUDA & CUDA_is_available) {
    fit_res <- gpu_fit_summary(
      input_matrix = input_matrix,
      offset_vector = offset_vector,
      blueprint     = blueprint,
      nfeatures     = nfeatures,
      nsamples      = ncol(input_matrix),
      gene_names    = gene_names,
      n_clusters    = if (!is.null(cluster_blocks_indexes)) length(cluster_blocks_indexes) else 0L,
      max_iter      = max_iter,
      tolerance     = tolerance,
      batch_size    = batch_size
    )
  } else {
    
    fit_res = cpu_fit_summary(
      input_matrix = input_matrix, 
      offset_vector = offset_vector,
      blueprint = blueprint, 
      initialized_beta = initialized_beta, 
      initialized_theta = initialized_theta, 
      n.cores = n.cores, 
      max_iter = max_iter, 
      tolerance = tolerance, 
      verbose = verbose
    )
  }
  
  return(list(
    beta                 = fit_res$beta,
    beta_sandwiches_null = fit_res$beta_sandwiches_null,
    beta_sandwiches      = fit_res$beta_sandwiches,
    overdispersion       = fit_res$theta,
    iterations           = fit_res$iterations,
    size_factors         = sf,
    offset_vector        = offset_vector,
    design_matrix        = design_matrix,
    input_matrix         = input_matrix,
    input_parameters     = list(max_iter = max_iter, tolerance = tolerance, parallel.cores = n.cores)
  ))
}


# GPU fitting function
gpu_fit_summary <- function(
    input_matrix,
    offset_vector,
    blueprint,
    nfeatures,       # ncol(design_matrix)
    nsamples,        # ncol(input_matrix)
    gene_names,
    n_clusters,
    max_iter,
    tolerance,
    batch_size
) {
  ngenes      <- nrow(input_matrix)
  extra_genes <- ngenes %% batch_size
  genes_batch <- ngenes - extra_genes
  
  # ── Helper: call beta_fit_gpu_summary on a slice of input_matrix ───────────
  run_gpu <- function(y_slice, bs) {
    beta_fit_gpu_summary(
      y            = y_slice,
      X_unique     = blueprint$X_unique,
      off_unique   = blueprint$off_unique,
      mapping      = as.integer(blueprint$mapping),
      counts       = as.numeric(blueprint$counts),
      cluster_map  = as.integer(blueprint$clusters_unique),
      n_clusters   = n_clusters,
      max_iter     = max_iter,
      eps          = tolerance,
      batch_size   = bs
    )
  }
  
  # ── Helper: extract H^{-1} per gene ───────────────────────────────────────
  assemble_hessians <- function(res, n) {
    if (is.null(res$hessian_inv)) return(vector("list", n))
    lapply(seq_len(n), function(g)
      matrix(res$hessian_inv[, g], nrow = nfeatures, ncol = nfeatures)
    )
  }
  
  # ── Helper: assemble H^{-1} M H^{-1} * nsamples per gene ─────────────────
  assemble_sandwiches <- function(res, n) {
    has_sandwich <- !is.null(res$hessian_inv) && !is.null(res$meat) && n_clusters > 0
    if (!has_sandwich) return(vector("list", n))
    lapply(seq_len(n), function(g) {
      H <- matrix(res$hessian_inv[, g], nrow = nfeatures, ncol = nfeatures)
      M <- matrix(res$meat[, g],        nrow = nfeatures, ncol = nfeatures)
      (H %*% M %*% H) * nsamples
    })
  }
  
  message("Fit beta (summary), using CUDA acceleration")
  start_time <- Sys.time()
  
  res_main <- run_gpu(input_matrix[seq_len(genes_batch), , drop = FALSE], batch_size)
  
  if (extra_genes > 0) {
    res_extra <- run_gpu(
      input_matrix[(genes_batch + 1):ngenes, , drop = FALSE],
      extra_genes
    )
  }
  
  message("[TIMING] Beta fit (GPU summary): ",
          difftime(Sys.time(), start_time, units = "secs"), " secs")
  
  # ── Combine outputs across main and (optional) extra batch ────────────────
  beta  <- res_main$mu_beta
  theta <- res_main$theta
  iters <- res_main$iter
  
  if (extra_genes > 0) {
    beta  <- rbind(beta,  res_extra$mu_beta)
    theta <- c(theta,     res_extra$theta)
    iters <- c(iters,     res_extra$iter)
  }
  
  if (is.null(dim(beta))) beta <- matrix(beta, ncol = 1)
  rownames(beta) <- gene_names
  
  hessians  <- assemble_hessians (res_main, genes_batch)
  sandwiches <- assemble_sandwiches(res_main, genes_batch)
  
  if (extra_genes > 0) {
    hessians   <- c(hessians,   assemble_hessians (res_extra, extra_genes))
    sandwiches <- c(sandwiches, assemble_sandwiches(res_extra, extra_genes))
  }
  
  list(
    beta                 = beta,
    theta                = theta,
    iterations           = list(beta_iters = iters, theta_iters = 0L),
    beta_sandwiches_null = hessians,
    beta_sandwiches      = sandwiches
  )
}


# CPU fitting function
cpu_fit_summary <- function(
    input_matrix,
    offset_vector,
    blueprint,
    initialized_beta,
    initialized_theta,
    n.cores,
    max_iter,
    tolerance,
    verbose
) {
  ngenes   <- nrow(input_matrix)
  nsamples <- ncol(input_matrix)
  
  # Init beta / theta (unchanged)
  initialized_beta        <- matrix(0, nrow = ngenes, ncol = ncol(blueprint$X_unique))
  initialized_beta[, 1]   <- log1p(DelayedMatrixStats::rowMeans2(input_matrix))
  # initialized_theta       <- devil:::estimate_dispersion(input_matrix, offset_vector)
  
  # ── Pre-compute ALL gene aggregates in two matrix rowsum calls ────────────
  tY          <- t(input_matrix)                          # samples × genes
  all_y_sums  <- rowsum(tY,    blueprint$mapping)         # n_groups × n_genes
  all_y_sq    <- rowsum(tY * tY,  blueprint$mapping)         # n_groups × n_genes
  rm(tY)
  # ─────────────────────────────────────────────────────────────────────────
  
  initialized_theta = devil:::estimate_dispersion_summary(all_y_sums, all_y_sq, blueprint)
  
  fit_single_gene <- function(gene_idx) {
    y_sums    <- all_y_sums[, gene_idx]
    y_squared <- all_y_sq[, gene_idx]
    
    beta_fit <- devil:::beta_fit_efficient(
      y_sums, blueprint$counts, blueprint$X_unique,
      initialized_beta[gene_idx, ], blueprint$off_unique,
      initialized_theta[gene_idx], max_iter, tolerance
    )
    mu_beta <- beta_fit$mu_beta
    iters   <- beta_fit$iter
    
    theta    <- devil:::estimate_mom_dispersion_efficient(
      y_sums, y_squared, blueprint$X_unique, exp(blueprint$off_unique),
      blueprint$counts, mu_beta, nsamples
    )
    hessian  <- devil:::compute_hessian_efficient(
      mu_beta, theta, y_sums, blueprint$counts,
      blueprint$X_unique, exp(blueprint$off_unique)
    )
    meat     <- devil:::compute_clustered_meat_efficient(
      X_unique                 = blueprint$X_unique,
      y_sums_per_group_cluster = y_sums,
      counts_per_group_cluster = blueprint$counts,
      group_to_cluster_map     = as.integer(blueprint$clusters),
      beta                     = mu_beta,
      overdispersion           = theta,
      sf_unique_per_pair       = exp(blueprint$off_unique),
      num_clusters             = length(unique(blueprint$clusters)),
      N_total                  = nsamples
    )
    sandwich <- (hessian %*% meat %*% hessian) * nsamples
    
    list(beta = mu_beta, theta = theta,
         hessian = hessian, meat = meat, sandwich = sandwich, iters = iters)
  }
  
  result_list = lapply(1:ngenes, fit_single_gene)
  
  if (verbose) message("Aggregating results")
  
  final_beta  <- do.call(rbind, lapply(result_list, `[[`, "beta"))
  final_theta <- vapply(result_list, `[[`, numeric(1),  "theta")
  final_iters <- vapply(result_list, `[[`, integer(1),  "iters")
  hessian_list  <- lapply(result_list, `[[`, "hessian")
  meat_list <- lapply(result_list, `[[`, "meat")
  sandwich_list <- lapply(result_list, `[[`, "sandwich")
  
  rownames(final_beta) <- rownames(input_matrix)
  
  return(list(
    beta                 = final_beta,
    theta                = final_theta,
    iterations           = final_iters,
    beta_sandwiches_null = hessian_list,
    meat_list            = meat_list,
    beta_sandwiches      = sandwich_list
  ))
}

get_group_blueprint <- function(X_mat, off_vec, clusters = NULL) {
  df <- as.data.frame(X_mat)
  df$off <- off_vec
  if (!is.null(clusters)) df$cluster_id <- clusters
  
  # 1. Generate mapping
  # group_indices() assigns 1 to the 'first' unique group it finds (alphabetically/numerically)
  mapping <- df %>% 
    group_by(across(everything())) %>% 
    group_indices()
  
  # 2. Extract unique metadata in ID order (1, 2, 3...)
  # We find the first occurrence of each group ID in the mapping vector
  first_occurrences <- match(seq_len(max(mapping)), mapping)
  reduced_metadata <- df[first_occurrences, , drop = FALSE]
  
  # 3. Counts in ID order
  # tabulate is much faster than table() and always returns 1:max counts
  group_counts <- tabulate(mapping)
  
  return(list(
    mapping = mapping,
    X_unique = as.matrix(reduced_metadata[, 1:ncol(X_mat)]),
    off_unique = reduced_metadata$off,
    clusters_unique = if(!is.null(clusters)) reduced_metadata$cluster_id else NULL,
    counts = group_counts,
    n_groups = length(group_counts)
  ))
}

get_aggregate = function(y, mapping, n_groups) {
  # rowsum() computes the sum of y grouped by mapping
  # We ensure all groups are present by using the 'n_groups' logic
  y_sums <- rowsum(y, mapping)
  y_sq   <- rowsum(y**2, mapping)
  
  # rowsum() returns a matrix with rownames = group IDs. 
  # We sort by the rownames to be 100% sure we are in order 1, 2, 3...
  # (though rowsum usually maintains order if mapping is numeric)
  order_idx <- as.character(seq_len(n_groups))
  
  return(list(
    y_sums = as.vector(y_sums[order_idx, ]), 
    y_squared = as.vector(y_sq[order_idx, ])
  ))
}


estimate_dispersion_summary <- function(all_y_sums, all_y_sq, blueprint) {
  N <- sum(blueprint$counts)
  
  # Scalar: weighted mean of exp(off) over all samples
  mean_offset_inverse <- N / sum(blueprint$counts * exp(blueprint$off_unique))
  
  # Per-gene totals (colSums over groups)
  total_y  <- colSums(all_y_sums)   # n_genes
  total_sq <- colSums(all_y_sq)     # n_genes
  
  mean_counts <- total_y / N
  
  # Sample variance: (Σy² - (Σy)²/N) / (N-1)  — matches rowVars()
  variance <- (total_sq - total_y^2 / N) / (N - 1)
  
  dispersion <- (variance - mean_offset_inverse * mean_counts) / mean_counts^2
  ifelse(is.na(dispersion) | dispersion < 0, 0.01, dispersion)
}
