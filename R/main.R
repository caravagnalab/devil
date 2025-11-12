#' Fit Statistical Model for Count Data
#'
#' @description
#' Fits a statistical model to count data, particularly designed for RNA sequencing data analysis.
#' The function estimates multiple parameters including regression coefficients (beta),
#' overdispersion parameters, and normalizes data using size factors. It supports both CPU
#' and GPU-based computation with parallel processing capabilities.
#'
#' @details
#' The function implements a negative binomial regression model with the following steps:
#' 1. Computes size factors for data normalization (if requested)
#' 2. Initializes model parameters including beta coefficients and overdispersion
#' 3. Fits the model using either CPU (parallel) or GPU computation
#' 4. Optionally estimates overdispersion parameters
#'
#' The model fitting process uses iterative optimization with configurable convergence
#' criteria and maximum iterations. For large datasets, the GPU implementation processes
#' genes in batches for improved memory efficiency.
#'
#' @section Size Factor Methods:
#' Three normalization methods are available:
#' \itemize{
#'   \item \code{"normed_sum"} (default): Geometric mean normalization based on library sizes.
#'     Fast and works well for most datasets.
#'   \item \code{"psinorm"}: Psi-normalization using Pareto distribution MLE.
#'     More robust to highly variable genes.
#'   \item \code{"edgeR"}: edgeR's TMM with singleton pairing method.
#'     Requires the edgeR package from Bioconductor.
#' }
#'
#' @param input_matrix A numeric matrix of count data (genes × samples).
#'   Rows represent genes/features, columns represent samples/cells.
#' @param design_matrix A numeric matrix of predictor variables (samples × predictors).
#'   Each row corresponds to a sample, each column to a predictor variable.
#' @param overdispersion Logical. Whether to estimate the overdispersion parameter.
#'   Set to FALSE for Poisson regression. Either "new" or "old" Default: "new"
#' @param init_overdispersion Numeric or NULL. Initial value for overdispersion parameter.
#'   If NULL, estimates initial value from data. Recommended value if specified: 100.
#'   Default: NULL
#' @param offset Numeric. Value added to counts to avoid numerical issues with zero counts.
#'   Default: 1e-6
#' @param size_factors Character string or NULL. Method for computing normalization factors
#'   to account for different sequencing depths. Options are:
#'   \itemize{
#'     \item \code{"normed_sum"} (default): Geometric mean normalization
#'     \item \code{"psinorm"}: Psi-normalization
#'     \item \code{"edgeR"}: edgeR TMM method
#'     \item \code{NULL}: No normalization (all size factors set to 1)
#'   }
#' @param verbose Logical. Whether to print progress messages during execution.
#'   Default: FALSE
#' @param max_iter Integer. Maximum number of iterations for parameter optimization.
#'   Default: 100
#' @param tolerance Numeric. Convergence criterion for parameter optimization.
#'   Default: 1e-3
#' @param CUDA Logical. Whether to use GPU acceleration (requires CUDA support).
#'   Default: FALSE
#' @param batch_size Integer. Number of genes to process per batch in GPU mode.
#'   Only relevant if CUDA = TRUE. Default: 1024
#' @param parallel.cores Integer or NULL. Number of CPU cores for parallel processing.
#'   If NULL, uses all available cores. Default: 1
#'
#' @return A list containing:
#' \describe{
#'   \item{beta}{Matrix of fitted coefficients (genes × predictors)}
#'   \item{overdispersion}{Vector of fitted overdispersion parameters (one per gene)}
#'   \item{iterations}{Vector of iteration counts for convergence (one per gene)}
#'   \item{size_factors}{Vector of computed size factors (one per sample)}
#'   \item{offset_vector}{Vector of offset values used in the model}
#'   \item{design_matrix}{Input design matrix (as provided)}
#'   \item{input_matrix}{Input count matrix (as provided)}
#'   \item{input_parameters}{List of used parameter values (max_iter, tolerance, parallel.cores)}
#' }
#'
#' @examples
#' \dontrun{
#' # Basic usage with default parameters
#' fit <- fit_devil(counts, design)
#'
#' # Using GPU acceleration with custom batch size
#' fit <- fit_devil(counts, design, CUDA = TRUE, batch_size = 2048)
#'
#' # Disable overdispersion estimation (Poisson model)
#' fit <- fit_devil(counts, design, overdispersion = FALSE)
#' }
#'
#' @export
#' @rawNamespace useDynLib(devil);
fit_devil <- function(
    input_matrix,
    design_matrix,
    overdispersion = "new",
    init_overdispersion = NULL,
    # do_cox_reid_adjustment = TRUE,
    offset=0,
    size_factors="normed_sum",
    verbose=FALSE,
    max_iter=200,
    tolerance=1e-3,
    CUDA = FALSE,
    batch_size = 1024L,
    parallel.cores=1) {

  # Read general info about input matrix and design matrix
  gene_names <- rownames(input_matrix)
  ngenes <- nrow(input_matrix)
  nfeatures <- ncol(design_matrix)

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

  # Check if CUDA is available
  CUDA_is_available <- FALSE
  if (CUDA) {
    if (!exists("beta_fit_gpu", mode = "function")) {
      warning("CUDA support was not enabled during package installation. ",
              "The beta_fit_gpu function is not available. ",
              "Reinstall with configure.args='--with-cuda' to enable GPU acceleration. ",
              "Falling back to CPU computation.")
      CUDA <- FALSE
    } else {
      message("CUDA support detected - using GPU acceleration")
      CUDA_is_available <- TRUE
    }
  }

  # Compute size factors
  if (!is.null(size_factors)) {
    if (verbose) { message("Compute size factors") }
    sf <- calculate_sf(input_matrix, method = size_factors, verbose = verbose)
  } else {
    sf <- rep(1, nrow(design_matrix))
  }

  # Calculate offset vector
  offset_vector = compute_offset_vector(offset, input_matrix, sf)

  # Initialize overdispersion
  if (is.null(init_overdispersion)) {
    dispersion_init <- c(estimate_dispersion(input_matrix, offset_vector))
  } else {
    dispersion_init <- rep(init_overdispersion, nrow(input_matrix))
  }

  # if (verbose) { message("Initialize beta estimate") }
  # groups <- devil:::get_groups_for_model_matrix(design_matrix)

  if (verbose) { message("Initialize beta estimate") }
  beta_0 <- init_beta(input_matrix, design_matrix, offset_vector)

  if (CUDA & CUDA_is_available) {
    message("Messing with CUDA! Implementation still needed")

    remainder = ngenes %% batch_size
    extra_genes = remainder
    genes_batch = ngenes - extra_genes

    message("Fit beta CUDA")
    start_time <- Sys.time()

    res_beta_fit <- beta_fit_gpu(
      input_matrix[1:genes_batch,],
      design_matrix,
      beta_0[1:genes_batch,],
      offset_vector,
      dispersion_init[1:genes_batch],
      max_iter = max_iter,
      eps = tolerance,
      batch_size = batch_size
    )

    if (remainder > 0) {
      res_beta_fit_extra <- beta_fit_gpu(
        input_matrix[(genes_batch+1):ngenes,],
        design_matrix,
        beta_0[(genes_batch+1):ngenes,],
        offset_vector,
        dispersion_init[(genes_batch+1):ngenes],
        max_iter = max_iter,
        eps = tolerance,
        batch_size = extra_genes
      )
    }

    end_time <- Sys.time()
    message("BETA GPU RUNTIME:")
    message(as.numeric(difftime(end_time, start_time, units = "secs")))

    beta = res_beta_fit$mu_beta

    if (remainder > 0) {
      beta_extra = res_beta_fit_extra$mu_beta
      beta <- rbind(beta, beta_extra)
      beta_iters=c(res_beta_fit$iter, res_beta_fit_extra$iter)
    } else {
      beta_iters=c(res_beta_fit$iter)
    }

    if (is.null(dim(beta))) {
      beta = matrix(beta, ncol = 1)
    }

  } else {

    if (verbose) { message("Fitting beta coefficients") }

    tmp <- parallel::mclapply(1:ngenes, function(i) {
      beta_fit(input_matrix[i,], design_matrix, beta_0[i,], offset_vector, dispersion_init[i], max_iter = max_iter, eps = tolerance)
    }, mc.cores = n.cores)

    beta <- lapply(1:ngenes, function(i) { tmp[[i]]$mu_beta }) %>% do.call("rbind", .)
    rownames(beta) <- gene_names
    beta_iters <- lapply(1:ngenes, function(i) { tmp[[i]]$iter }) %>% unlist()


    # # To remove
    # y_unique_cap = as.integer(dim(input_matrix)[2] / 2)
    #
    # tmp <- parallel::mclapply(1:ngenes, function(i) {
    #   two_step_fit_cpp(input_matrix[i,], design_matrix, beta_0[i,], offset_vector, dispersion_init[i],
    #                    max_iter_beta = max_iter, max_iter_kappa = max_iter, eps_theta = tolerance, eps_beta = tolerance,
    #                    newton_max = 16, y_unique_cap = y_unique_cap, fit_overdispersion = overdispersion)
    # }, mc.cores = n.cores)
    #
    # beta <- lapply(1:ngenes, function(i) { tmp[[i]]$mu_beta }) %>% do.call("rbind", .)
    # theta = lapply(1:ngenes, function(i) { tmp[[i]]$kappa }) %>% unlist()
    #
    # if (!overdispersion) theta = dispersion_init
    #
    # rownames(beta) <- gene_names
    # beta_iters <- lapply(1:ngenes, function(i) { tmp[[i]]$beta_iters }) %>% unlist()
    # theta_iters <- lapply(1:ngenes, function(i) { tmp[[i]]$kappa_iters }) %>% unlist()

    # if (batch_size == 1) {
    #   tmp <- parallel::mclapply(1:ngenes, function(i) {
    #     two_step_fit_cpp(input_matrix[i,], design_matrix, beta_0[i,], offset_vector, dispersion_init[i],
    #                      max_iter_beta = max_iter, max_iter_kappa = max_iter, eps_theta = tolerance, eps_beta = tolerance,
    #                      newton_max = 16, y_unique_cap = y_unique_cap)
    #
    #   }, mc.cores = n.cores)
    #   beta <- lapply(1:ngenes, function(i) { tmp[[i]]$mu_beta }) %>% do.call("rbind", .)
    #   theta = lapply(1:ngenes, function(i) { tmp[[i]]$kappa }) %>% unlist()
    #   rownames(beta) <- gene_names
    #   beta_iters <- lapply(1:ngenes, function(i) { tmp[[i]]$beta_iters }) %>% unlist()
    #   theta_iters <- lapply(1:ngenes, function(i) { tmp[[i]]$kappa_iters }) %>% unlist()
    # } else {
    #   batch_size = min(ngenes, batch_size)
    #   n_batches = ceiling(ngenes / batch_size)
    #   tmp = lapply(1:n_batches, function(nb) {
    #     idxs = (batch_size*(nb-1)+1):(min((batch_size*nb),ngenes))
    #     tmp = two_step_fit_batched_cpp(t(input_matrix[idxs,]), design_matrix, t(beta_0[idxs,]), offset_vector, kappa_vec = dispersion_init[idxs],
    #                                    max_iter, max_iter, tolerance, tolerance, 16, y_unique_cap = y_unique_cap, n_threads = 1)
    #     list(mu_beta = tmp$mu_beta, kappa = tmp$kappa, beta_iters = tmp$beta_iters, kappa_iters = tmp$kappa_iters)
    #   })
    #
    #   beta = lapply(tmp, function(x) t(x$mu_beta)) %>% do.call("rbind", .)
    #   theta = lapply(tmp, function(x) x$kappa) %>% unlist()
    #   rownames(beta) <- gene_names
    #   beta_iters = lapply(tmp, function(x) x$beta_iters) %>% unlist()
    #   theta_iters = lapply(tmp, function(x) x$kappa_iters) %>% unlist()
    # }
  }

  if (!isFALSE(overdispersion)) {
    if (verbose) { message("Fit overdispersion") }


    if (overdispersion == "old") {
      theta <- parallel::mclapply(1:ngenes, function(i) {
        fit_dispersion(beta[i,], design_matrix, input_matrix[i,], offset_vector,
                       tolerance = tolerance, max_iter = max_iter, do_cox_reid_adjustment = TRUE)
      }, mc.cores = n.cores) %>% unlist()
      theta_iters = NA
    } else if (overdispersion == "new") {
      y_unique_cap = as.integer(dim(input_matrix)[2] / 2)
      tmp <- parallel::mclapply(1:ngenes, function(i) {
        fit_overdispersion_cppp(y = input_matrix[i,], X = design_matrix, mu_beta = beta[i,], off = offset_vector,
                                        kappa = dispersion_init[i], max_iter = max_iter, eps_theta = tolerance,
                                        newton_max = 16, y_unique_cap = y_unique_cap)
      }, mc.cores = n.cores)

      theta = lapply(tmp, function(x) x$kappa) %>% unlist()
      theta_iters = lapply(tmp, function(x) x$kappa_iters) %>% unlist()
    } else {
      stop()
    }
  } else {
    theta = rep(0, ngenes)
    theta_iters = 0
  }

  return(list(
    beta=beta,
    overdispersion=theta,
    iterations=list(beta_iters=beta_iters, theta_iters = theta_iters),
    size_factors=sf,
    offset_vector=offset_vector,
    design_matrix=design_matrix,
    input_matrix=input_matrix,
    input_parameters=list(max_iter=max_iter, tolerance=tolerance, parallel.cores=n.cores)
    )
  )
}

get_groups_for_model_matrix <- function(model_matrix){
  if(! lte_n_equal_rows(model_matrix, ncol(model_matrix))){
    return(NULL)
  }else{
    get_row_groups(model_matrix, n_groups = ncol(model_matrix))
  }
}

handle_input_matrix <- function(input_matrix, verbose) {
  if(is.matrix(input_matrix)){
    if(!is.numeric(input_matrix)){
      stop("The input_matrix argument must consist of numeric values and not of ", mode(input_matrix), " values")
    }
    data_mat <- input_matrix
  } else if (methods::is(input_matrix, "DelayedArray")){
    data_mat <- input_matrix
  } else if (methods::is(input_matrix, "dgCMatrix") || methods::is(input_matrix, "dgTMatrix")) {
    data_mat <- as.matrix(input_matrix)
  }else{
    stop("Cannot handle data of class '", class(input_matrix), "'.",
         "It must be of type dense matrix object (i.e., a base matrix or DelayedArray),",
         " or a container for such a matrix (for example: SummarizedExperiment).")
  }
  data_mat
}
