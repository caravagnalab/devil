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
#'         MLE-based procedure (with Cox–Reid adjustment inside \code{fit_dispersion()}).
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
#' @param input_matrix A numeric matrix of count data (genes × samples).
#'   Rows represent genes/features, columns represent samples/cells.
#' @param design_matrix A numeric matrix of predictor variables (samples × predictors).
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
#' @param init_beta_rough Logial. Whether to initialize betas in a rough but extremely fast way.
#'   Default: \code{FALSE}.
#' @param offset Numeric scalar. Value used when computing the offset vector to avoid
#'   numerical issues with zero counts. Default: \code{0}.
#' @param size_factors Character string or \code{NULL}. Method for computing normalization
#'   factors to account for different sequencing depths. Options are:
#'   \itemize{
#'     \item \code{NULL} (default):  No normalization (all size factors set to 1)
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
#'   \item{beta}{Matrix of fitted coefficients (genes × predictors).}
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
fit_devil <- function(
    input_matrix,
    design_matrix,
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
    parallel.cores = 1
) {
  # - Input parameters ----
  # Read general info about input matrix and design matrix
  gene_names <- rownames(input_matrix)
  ngenes <- nrow(input_matrix)
  nfeatures <- ncol(design_matrix)

  # Check input clusters
  if (!is.null(clusters)) {
    clusters = as.numeric(factor(clusters, levels = unique(clusters)))
    if (is.unsorted(clusters)) stop("Input data should be grouped in block of patients. Use the `group_data` function to group them")
    cluster_blocks_indexes <- cumsum(rle(clusters)$lengths)
  } else {
    cluster_blocks_indexes = NULL
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
      CUDA_is_available <- FALSE
    } else {
      message("CUDA support detected - using GPU acceleration")
      CUDA_is_available <- TRUE
    }
  }

  # - CPU and GPU common part (i.e. size_factors and offset_vectors) ----
  ## - Compute size factors ----
  if (!is.null(size_factors)) {
    if (verbose) {
      message("Compute size factors")
    }
    sf <- devil:::calculate_sf(input_matrix, method = size_factors, verbose = verbose)
  } else {
    sf <- rep(1, nrow(design_matrix))
  }

  ## - Compute offset vector ----
  offset_vector <- devil:::compute_offset_vector(offset, input_matrix, sf)

  # - Start GPU vs CPU branch ----
  if (CUDA & CUDA_is_available) {

    ## - GPU branch ----
    remainder <- ngenes %% batch_size
    extra_genes <- remainder
    genes_batch <- ngenes - extra_genes

    message("Fit beta, using CUDA acceleration")
    start_time <- Sys.time()
    res_beta_fit <- beta_fit_gpu(
      input_matrix[seq_len(genes_batch), ],
      design_matrix,
      offset_vector,
      max_iter = max_iter,
      eps = tolerance,
      batch_size = batch_size,
      TEST = FALSE
    )

    if (remainder > 0) {
      res_beta_fit_extra <- beta_fit_gpu(
        input_matrix[(genes_batch + 1):ngenes, ],
        design_matrix,
        offset_vector,
        max_iter = max_iter,
        eps = tolerance,
        batch_size = extra_genes,
        TEST = FALSE
      )
    }

    end_time <- Sys.time()
    message("[TIMING] Beta fit computing (GPU):", difftime(end_time, start_time, units = "secs"))

    # Extract beta and theta from GPU results
    beta <- res_beta_fit$mu_beta
    theta <- res_beta_fit$theta

    # gpu_k <- NULL
    # gpu_beta_init <- NULL

    if (remainder > 0) {
      beta_extra <- res_beta_fit_extra$mu_beta
      theta_extra <- res_beta_fit_extra$theta
      beta <- rbind(beta, beta_extra)
      theta <- c(theta, theta_extra)
      beta_iters <- c(res_beta_fit$iter, res_beta_fit_extra$iter)
    } else {
      beta_iters <- c(res_beta_fit$iter)
    }

    if (is.null(dim(beta))) {
      beta <- matrix(beta, ncol = 1)
    }

    rownames(beta) <- gene_names

    # Create fit_res structure to match CPU branch
    fit_res <- list(
      beta = beta,
      theta = theta,
      iterations = list(
        beta_iters = beta_iters,
        theta_iters = 0L # GPU uses MOM, no iterative fitting
      )
    )

  } else {
    ## - CPU branch ----
    fit_res <- cpu_fit(
      input_matrix = input_matrix,
      design_matrix = design_matrix,
      cluster_blocks_indexes = cluster_blocks_indexes,
      #clusters = clusters,
      offset_vector = offset_vector,
      init_overdispersion = init_overdispersion,
      init_beta_rough = init_beta_rough,
      overdispersion = overdispersion,
      n.cores = n.cores, max_iter = max_iter,
      tolerance = tolerance, verbose = verbose
    )
  }

  return(list(
    beta = fit_res$beta,
    beta_sandwiches_null = fit_res$beta_sandwiches_null,
    beta_sandwiches = fit_res$beta_sandwiches,
    overdispersion = fit_res$theta,
    iterations = fit_res$iterations,
    size_factors = sf,
    offset_vector = offset_vector,
    design_matrix = design_matrix,
    input_matrix = input_matrix,
    input_parameters = list(max_iter = max_iter, tolerance = tolerance, parallel.cores = n.cores)
  ))
}

# Inner CPU fitting function
cpu_fit <- function(
    input_matrix,
    design_matrix,
    cluster_blocks_indexes,
    #clusters,
    offset_vector,
    init_overdispersion,
    init_beta_rough,
    overdispersion,
    n.cores,
    max_iter,
    tolerance,
    verbose) {

  ngenes <- nrow(input_matrix)
  nsamples <- ncol(input_matrix)
  exp_offset <- exp(offset_vector)

  # 1. Initialization Logic
  if (verbose) message("==> Initializing parameters")

  # - Initialize dispersion ---
  if (verbose) message("Initialize theta")
  if (is.null(init_overdispersion)) {
    dispersion_init <- devil:::estimate_dispersion(input_matrix, offset_vector)
  } else {
    dispersion_init <- rep(init_overdispersion, nrow(input_matrix))
  }

  # - Initialize beta ---
  if (verbose) message("Initialize beta")
  if (isTRUE(init_beta_rough)) {
    beta_0 <- matrix(0, nrow = nrow(input_matrix), ncol = ncol(design_matrix))
    beta_0[, 1] <- DelayedMatrixStats::rowMeans2(input_matrix) %>% log1p()
  } else {
    beta_0 <- init_beta(input_matrix, design_matrix, offset_vector)
  }

  # 2. The Fused Workhorse Loop
  if (verbose) message("Fitting expression coefficients and overdispersion")

  results_list <- parallel::mclapply(
    X = seq_len(ngenes),
    mc.cores = n.cores,
    FUN = function(i) {

      # --- Step A: Fit Beta ---
      fit <- devil:::beta_fit(
        y = input_matrix[i, ],
        X = design_matrix,
        mu_beta = beta_0[i, ],
        off = offset_vector,
        k = dispersion_init[i],
        max_iter = max_iter,
        eps = tolerance
      )
      curr_beta <- fit$mu_beta

      # --- Step B: Fit Theta (if not MOM) ---
      curr_theta <- dispersion_init[i]
      if (overdispersion %in% c("old", "MLE")) {
        curr_theta = fit_dispersion(
          beta = curr_beta,
          design_matrix,
          input_matrix[i, ],
          offset_vector,
          tolerance = tolerance,
          max_iter = max_iter,
          do_cox_reid_adjustment = TRUE
        )
      } else if (overdispersion == "MOM") {
        # theta_fit <- devil:::fit_overdispersion_cppp(
        #   y = input_matrix[i, ],
        #   X = design_matrix,
        #   mu_beta = curr_beta,
        #   off = offset_vector,
        #   kappa = curr_theta,
        #   max_iter = max_iter,
        #   eps_theta = tolerance
        # )
        theta_fit <- devil:::estimate_mom_dispersion_cpp(
          count_matrix = matrix(input_matrix[i, ], nrow = 1),
          design_matrix = design_matrix,
          beta_matrix = matrix(curr_beta, nrow = 1),
          sf = exp_offset
        )
        curr_theta <- theta_fit
      } else if (isFALSE(overdispersion)) {
        curr_theta = 0
      } else {
        stop("Unknown overdispersion mode: ", overdispersion)
      }

      # --- Step C: Compute Bread (Inverse Hessian) ---
      # Use the optimized version to avoid redundant memory allocations
      bread <- devil:::compute_hessian(
        beta = curr_beta,
        overdispersion = curr_theta,
        y = input_matrix[i, ],
        design_matrix = design_matrix,
        size_factors = exp_offset
      )
      
      meat <- devil:::compute_meat(design_matrix = design_matrix, 
                                   y = input_matrix[i,], 
                                   beta = curr_beta, 
                                   overdispersion = curr_theta, 
                                   size_factors = exp_offset)
      
      s_null <- (bread %*% meat %*% bread) * nsamples

      # --- Step D: Compute Meat and Sandwich ---
      s_clust <- NULL
      if (!is.null(cluster_blocks_indexes)) {
        # Use the optimized 'meat' function
        # meat_clust <- devil:::compute_clustered_meat(
        #   design_matrix = design_matrix,
        #   y = input_matrix[i, ],
        #   beta = curr_beta,
        #   overdispersion = curr_theta,
        #   size_factors = exp_offset,
        #   clusters = as.integer(clusters)
        # )

        meat_clust <- devil:::compute_clustered_meat_fast(
          design_matrix = design_matrix,
          y = input_matrix[i, ],
          beta = curr_beta,
          overdispersion = curr_theta,
          size_factors = exp_offset,
          cluster_blocks_indexes = cluster_blocks_indexes
        )

        # Sandwich: V = B * M * B
        # Scale by nsamples to match standard sandwich asymptotics if required
        s_clust <- (bread %*% meat_clust %*% bread) * nsamples
      }

      # Return results as a compact list
      return(list(
        beta = curr_beta,
        theta = curr_theta,
        iter = fit$iter,
        s_null = s_null,   # Standard variance
        s_clust = s_clust # Cluster-robust variance
      ))
    }
  )

  # 3. Aggregate Results
  if (verbose) message("Variance estimation")

  final_beta <- do.call(rbind, lapply(results_list, `[[`, "beta"))
  final_theta <- vapply(results_list, `[[`, numeric(1), "theta")
  final_iters <- vapply(results_list, `[[`, integer(1), "iter")

  # Separate out the variance matrices
  s_null_list  <- lapply(results_list, `[[`, "s_null")
  s_clust_list <- lapply(results_list, `[[`, "s_clust")

  rownames(final_beta) <- rownames(input_matrix)

  return(list(
    beta = final_beta,
    theta = final_theta,
    iterations = final_iters,
    beta_sandwiches_null = s_null_list,
    beta_sandwiches = s_clust_list
  ))
}
