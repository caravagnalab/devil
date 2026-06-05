.extract_matrix_from_x <- function(x, assay.type) {
  if (is(x, "SummarizedExperiment")) {
    assay(x, assay.type)
  } else if (is.matrix(x) || is(x, "Matrix")) {
    x
  } else {
    stop("'x' must be a matrix, SummarizedExperiment, or SingleCellExperiment object.")
  }
}


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
#' If \code{size_factors} is a numeric vector of precomputed size factors (one per sample), they will be used directly.
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
#' @param x A numeric matrix of count data (genes x samples), a
#'   \code{\link[SummarizedExperiment]{SummarizedExperiment}}, or a
#'   \code{\link[SingleCellExperiment]{SingleCellExperiment}} object.
#'   Rows represent genes/features, columns represent samples/cells.
#'   When \code{x} is an SE/SCE object, the assay named by \code{assay.type} is used.
#' @param design_matrix A numeric matrix of predictor variables (samples x predictors).
#'   Each row corresponds to a sample, each column to a predictor variable.
#'   Must have \code{nrow(design_matrix) == ncol(x)}.
#' @param assay.type Character. Name of the assay to extract when \code{x} is a
#'   \code{SummarizedExperiment} or \code{SingleCellExperiment}. Default: \code{"counts"}.
#' @param clusters Vector of cluster or patient identifiers (one element per cell/sample).
#'   Cells belonging to the same cluster must be contiguous — use \code{group_data()} to
#'   sort them first. When provided, enables clustered sandwich covariance estimation.
#'   Default: \code{NULL}.
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
#' @param size_factors Character string, numeric vector, or \code{NULL}. Controls
#'   normalization for sequencing depth. Options are:
#'   \itemize{
#'     \item \code{NULL} (default): No normalization (all size factors set to 1)
#'     \item \code{"normed_sum"}: Geometric mean normalization
#'     \item \code{"psinorm"}: Psi-normalization
#'     \item \code{"edgeR"}: edgeR TMM method
#'     \item A numeric vector of length \code{ncol(x)}: precomputed size
#'       factors used directly without further normalization
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
#' @param BPPARAM A \code{\link[BiocParallel]{BiocParallelParam}} object controlling
#'   parallel evaluation. Default: \code{BiocParallel::SerialParam()}.
#'
#' @return A list containing:
#' \describe{
#'   \item{beta}{Matrix of fitted coefficients (genes x predictors).}
#'   \item{beta_sandwiches_null}{List of per-gene Hessian inverse matrices (\eqn{H^{-1}}).}
#'   \item{beta_sandwiches}{List of per-gene clustered sandwich matrices
#'         (\eqn{H^{-1} M H^{-1} \times n\_samples}); \code{NULL} entries when no
#'         \code{clusters} are provided.}
#'   \item{overdispersion}{Vector of fitted overdispersion parameters (one per gene).}
#'   \item{iterations}{List with elements \code{beta_iters} and \code{theta_iters}
#'         giving the number of iterations used for each gene.}
#'   \item{size_factors}{Vector of computed size factors (one per sample).}
#'   \item{offset_vector}{Vector of offset values used in the model (length = number of samples).}
#'   \item{design_matrix}{Input design matrix (as provided, possibly coerced to numeric matrix).}
#'   \item{input_matrix}{Input count matrix (as provided, possibly coerced to numeric matrix).}
#'   \item{input_parameters}{List of used parameter values
#'         (\code{max_iter}, \code{tolerance}).}
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
#' # Fit from a raw matrix
#' fit <- fit_devil(
#'     x             = counts,
#'     design_matrix = design,
#'     size_factors  = "normed_sum"
#' )
#'
#' ## Example: fit from a SingleCellExperiment object
#' library(SingleCellExperiment)
#' sce <- SingleCellExperiment(assays = list(counts = counts))
#' fit_sce <- fit_devil(
#'     x             = sce,
#'     design_matrix = design,
#'     assay.type    = "counts"
#' )
#'
#' @export
#' @rawNamespace useDynLib(devil);
fit_devil <- function(
    x,
    design_matrix,
    assay.type = "counts",
    clusters = NULL,
    overdispersion = "MOM",
    init_overdispersion = NULL,
    init_beta_rough = FALSE,
    offset = 0,
    size_factors = NULL,
    verbose = FALSE,
    max_iter = 200,
    tolerance = 1e-3,
    CUDA = FALSE,
    batch_size = 1024L,
    BPPARAM = BiocParallel::SerialParam()
) {
  # - Resolve input matrix from x (matrix, SummarizedExperiment, or SingleCellExperiment)
  input_matrix <- .extract_matrix_from_x(x, assay.type)

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
    if (is.character(size_factors)) {
      if (verbose) message("Compute size factors")
      sf <- calculate_sf(input_matrix, method = size_factors, verbose = verbose)  
    } else {
      if (verbose) message("Using pre-computed size factors")
      sf <- size_factors
    }
  } else {
    sf <- rep(1, nrow(design_matrix))
  }
  
  # - Compute offset vector ----
  offset_vector <- compute_offset_vector(offset, input_matrix, sf)
  
  # - GPU vs CPU branch ----
  if (CUDA & CUDA_is_available) {
    fit_res <- gpu_fit(
      input_matrix           = input_matrix,
      design_matrix          = design_matrix,
      offset_vector          = offset_vector,
      cluster_blocks_indexes = cluster_blocks_indexes,
      gene_names             = gene_names,
      nfeatures              = nfeatures,
      max_iter               = max_iter,
      tolerance              = tolerance,
      batch_size             = batch_size
    )
  } else {
    fit_res <- cpu_fit(
      input_matrix           = input_matrix,
      design_matrix          = design_matrix,
      cluster_blocks_indexes = cluster_blocks_indexes,
      offset_vector          = offset_vector,
      init_overdispersion    = init_overdispersion,
      init_beta_rough        = init_beta_rough,
      overdispersion         = overdispersion,
      BPPARAM                = BPPARAM,
      max_iter               = max_iter,
      tolerance              = tolerance,
      verbose                = verbose
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
    input_parameters     = list(max_iter = max_iter, tolerance = tolerance)
  ))
}


# GPU fitting function
gpu_fit <- function(
    input_matrix,
    design_matrix,
    offset_vector,
    cluster_blocks_indexes,
    gene_names,
    nfeatures,
    max_iter,
    tolerance,
    batch_size
) {
  ngenes      <- nrow(input_matrix)
  nsamples    <- nrow(design_matrix)
  extra_genes <- ngenes %% batch_size
  genes_batch <- ngenes - extra_genes
  
  n_clusters_gpu   <- if (!is.null(cluster_blocks_indexes)) length(cluster_blocks_indexes) else 0L
  cluster_ends_gpu <- if (!is.null(cluster_blocks_indexes)) as.integer(cluster_blocks_indexes) else integer(0)
  
  # Helper: call beta_fit_gpu on a slice of input_matrix
  run_gpu <- function(y_slice, bs) {
    beta_fit_gpu(
      y_slice,
      design_matrix,
      offset_vector,
      max_iter     = max_iter,
      eps          = tolerance,
      batch_size   = bs,
      TEST         = FALSE,
      cluster_ends = cluster_ends_gpu,
      n_clusters   = n_clusters_gpu
    )
  }
  
  # Helper: extract per-gene hessian_inv as a list of [nfeatures x nfeatures] matrices
  assemble_sandwiches_null <- function(res, n) {
    if (is.null(res$hessian_inv)) return(vector("list", n))
    lapply(seq_len(n), function(g)
      matrix(res$hessian_inv[, g], nrow = nfeatures, ncol = nfeatures)
    )
  }
  
  # Helper: assemble clustered sandwich H⁻¹ M H⁻¹ * nsamples per gene
  assemble_sandwiches <- function(res, n) {
    has_sandwich <- !is.null(res$hessian_inv) && !is.null(res$meat) && n_clusters_gpu > 0
    if (!has_sandwich) return(vector("list", n))
    lapply(seq_len(n), function(g) {
      H <- matrix(res$hessian_inv[, g], nrow = nfeatures, ncol = nfeatures)
      M <- matrix(res$meat[, g],        nrow = nfeatures, ncol = nfeatures)
      (H %*% M %*% H) * nsamples
    })
  }
  
  message("Fit beta, using CUDA acceleration")
  start_time <- Sys.time()
  
  res_main <- run_gpu(input_matrix[seq_len(genes_batch), , drop = FALSE], batch_size)
  
  if (extra_genes > 0) {
    res_extra <- run_gpu(input_matrix[(genes_batch + 1):ngenes, , drop = FALSE], extra_genes)
  }
  
  message("[TIMING] Beta fit (GPU): ", difftime(Sys.time(), start_time, units = "secs"), " secs")
  
  # Combine beta, theta, iters
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
  
  # Assemble sandwich lists
  sandwiches_null <- assemble_sandwiches_null(res_main, genes_batch)
  sandwiches      <- assemble_sandwiches(res_main, genes_batch)
  
  if (extra_genes > 0) {
    sandwiches_null <- c(sandwiches_null, assemble_sandwiches_null(res_extra, extra_genes))
    sandwiches      <- c(sandwiches,      assemble_sandwiches(res_extra,      extra_genes))
  }
  
  return(list(
    beta                 = beta,
    theta                = theta,
    iterations           = list(beta_iters = iters, theta_iters = 0L),
    beta_sandwiches_null = sandwiches_null,
    beta_sandwiches      = sandwiches
  ))
}


# CPU fitting function
cpu_fit <- function(
    input_matrix,
    design_matrix,
    cluster_blocks_indexes = NULL,
    offset_vector,
    init_overdispersion,
    init_beta_rough,
    overdispersion,
    BPPARAM = BiocParallel::SerialParam(),
    max_iter,
    tolerance,
    verbose
) {
  ngenes     <- nrow(input_matrix)
  nsamples   <- ncol(input_matrix)
  exp_offset <- exp(offset_vector)
  
  if (verbose) message("==> Initializing parameters")
  
  # Initialize dispersion
  if (verbose) message("Initialize theta")
  if (is.null(init_overdispersion)) {
    dispersion_init <- estimate_dispersion(input_matrix, offset_vector)
  } else {
    dispersion_init <- rep(init_overdispersion, ngenes)
  }
  
  # Initialize beta
  if (verbose) message("Initialize beta")
  if (isTRUE(init_beta_rough)) {
    beta_0 <- matrix(0, nrow = ngenes, ncol = ncol(design_matrix))
    beta_0[, 1] <- DelayedMatrixStats::rowMeans2(input_matrix) %>% log1p()
  } else {
    beta_0 <- init_beta(input_matrix, design_matrix, offset_vector)
  }
  
  if (verbose) message("Fitting expression coefficients and overdispersion")
  
  results_list <- BiocParallel::bplapply(
    X       = seq_len(ngenes),
    BPPARAM = BPPARAM,
    FUN     = function(i) {
      
      # Step A: Fit beta
      fit <- beta_fit(
        y        = input_matrix[i, ],
        X        = design_matrix,
        mu_beta  = beta_0[i, ],
        off      = offset_vector,
        k        = dispersion_init[i],
        max_iter = max_iter,
        eps      = tolerance
      )
      curr_beta <- fit$mu_beta
      
      # Step B: Fit theta
      curr_theta <- dispersion_init[i]
      curr_theta_iters <- NA_integer_
      if (overdispersion %in% c("old", "MLE")) {
        curr_theta <- fit_dispersion(
          beta                    = curr_beta,
          design_matrix,
          input_matrix[i, ],
          offset_vector,
          tolerance               = tolerance,
          max_iter                = max_iter,
          do_cox_reid_adjustment  = TRUE
        )
      } else if (overdispersion %in% c("new", "I")) {
        od_res <- fit_overdispersion_cppp(
          y        = input_matrix[i, ],
          X        = design_matrix,
          mu_beta  = curr_beta,
          off      = offset_vector,
          kappa    = dispersion_init[i],
          max_iter = max_iter,
          eps_theta = tolerance
        )
        curr_theta       <- od_res$kappa
        curr_theta_iters <- as.integer(od_res$kappa_iters)
      } else if (overdispersion == "MOM") {
        curr_theta <- estimate_mom_dispersion_cpp(
          count_matrix  = matrix(input_matrix[i, ], nrow = 1),
          design_matrix = design_matrix,
          beta_matrix   = matrix(curr_beta, nrow = 1),
          sf            = exp_offset
        )
        curr_theta_iters <- 0L
      } else if (isFALSE(overdispersion)) {
        curr_theta       <- 0
        curr_theta_iters <- 0L
      } else {
        stop("Unknown overdispersion mode: ", overdispersion)
      }
      
      # Step C: Hessian inverse (used as s_null, matching GPU convention)
      bread  <- compute_hessian(
        beta           = curr_beta,
        overdispersion = curr_theta,
        y              = input_matrix[i, ],
        design_matrix  = design_matrix,
        size_factors   = exp_offset
      )
      s_null <- bread   # H⁻¹, consistent with GPU beta_sandwiches_null
      
      # Step D: Clustered sandwich
      s_clust <- NULL
      if (!is.null(cluster_blocks_indexes)) {
        meat_clust <- compute_clustered_meat_fast(
          design_matrix          = design_matrix,
          y                      = input_matrix[i, ],
          beta                   = curr_beta,
          overdispersion         = curr_theta,
          size_factors           = exp_offset,
          cluster_blocks_indexes = cluster_blocks_indexes
        )
        s_clust <- (bread %*% meat_clust %*% bread) * nsamples
      }
      
      return(list(
        beta        = curr_beta,
        theta       = curr_theta,
        iter        = fit$iter,
        theta_iters = curr_theta_iters,
        s_null      = s_null,
        s_clust     = s_clust
      ))
    }
  )
  
  if (verbose) message("Aggregating results")

  final_beta        <- do.call(rbind, lapply(results_list, `[[`, "beta"))
  final_theta       <- vapply(results_list, `[[`, numeric(1),  "theta")
  final_iters       <- vapply(results_list, `[[`, integer(1),  "iter")
  final_theta_iters <- vapply(results_list, `[[`, integer(1),  "theta_iters")
  s_null_list       <- lapply(results_list, `[[`, "s_null")
  s_clust_list      <- lapply(results_list, `[[`, "s_clust")

  rownames(final_beta) <- rownames(input_matrix)

  return(list(
    beta                 = final_beta,
    theta                = final_theta,
    iterations           = list(beta_iters = final_iters, theta_iters = final_theta_iters),
    beta_sandwiches_null = s_null_list,
    beta_sandwiches      = s_clust_list
  ))
}