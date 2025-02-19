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
#' @param input_matrix A numeric matrix of count data (genes × samples).
#'   Rows represent genes/features, columns represent samples/cells.
#' @param design_matrix A numeric matrix of predictor variables (samples × predictors).
#'   Each row corresponds to a sample, each column to a predictor variable.
#' @param overdispersion Logical. Whether to estimate the overdispersion parameter.
#'   Set to FALSE for Poisson regression. Default: TRUE
#' @param init_overdispersion Numeric or NULL. Initial value for overdispersion parameter.
#'   If NULL, estimates initial value from data. Recommended value if specified: 100.
#'   Default: NULL
#' @param do_cox_reid_adjustment Logical. Whether to apply Cox-Reid adjustment in
#'   overdispersion estimation. Default: TRUE
#' @param offset Numeric. Value added to counts to avoid numerical issues with zero counts.
#'   Default: 1e-6
#' @param size_factors Logical. Whether to compute normalization factors for different
#'   sequencing depths. Default: TRUE
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
#'   If NULL, uses all available cores. Default: NULL
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
    overdispersion = TRUE,
    init_overdispersion = NULL,
    do_cox_reid_adjustment = TRUE,
    offset=1e-6,
    size_factors=TRUE,
    verbose=FALSE,
    max_iter=100,
    tolerance=1e-3,
    CUDA = FALSE,
    batch_size = 1024L,
    parallel.cores=NULL) {

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
    message("Check CUDA availability function need to be implemented")
    CUDA_is_available <- TRUE
  }

  # Compute size factors
  if (size_factors) {
    if (verbose) { message("Compute size factors") }
    sf <- devil:::calculate_sf(input_matrix, verbose = verbose)
  } else {
    sf <- rep(1, nrow(design_matrix))
  }

  # Calculate offset vector
  offset_vector = devil:::compute_offset_vector(offset, input_matrix, sf)
  #offset_matrix = devil:::compute_offset_matrix(offset, input_matrix, sf)

  # Initialize overdispersion
  if (is.null(init_overdispersion)) {
    dispersion_init <- c(devil:::estimate_dispersion(input_matrix, offset_vector))
  } else {
    dispersion_init <- rep(init_overdispersion, nrow(input_matrix))
  }

  # if (verbose) { message("Initialize beta estimate") }
  # groups <- devil:::get_groups_for_model_matrix(design_matrix)

  if (verbose) { message("Initialize beta estimate") }
  beta_0 <- devil:::init_beta(input_matrix, design_matrix, offset_vector)

  if (CUDA & CUDA_is_available) {
    message("Messing with CUDA! Implementation still needed")

    remainder = ngenes %% batch_size
    extra_genes = remainder
    genes_batch = ngenes - extra_genes

    message("Fit beta CUDA")
    start_time <- Sys.time()

    res_beta_fit <- devil:::beta_fit_gpu(
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
      res_beta_fit_extra <- devil:::beta_fit_gpu(
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
      iterations=c(res_beta_fit$iter, res_beta_fit_extra$iter)
    } else {
      iterations=c(res_beta_fit$iter)
    }

    if (is.null(dim(beta))) {
      beta = matrix(beta, ncol = 1)
    }

  } else {

    if (verbose) { message("Fit beta coefficients") }

    tmp <- parallel::mclapply(1:ngenes, function(i) {
      devil:::beta_fit(input_matrix[i,], design_matrix, beta_0[i,], offset_vector, dispersion_init[i], max_iter = max_iter, eps = tolerance)
    }, mc.cores = n.cores)

    beta <- lapply(1:ngenes, function(i) { tmp[[i]]$mu_beta }) %>% do.call("rbind", .)
    rownames(beta) <- gene_names
    iterations <- lapply(1:ngenes, function(i) { tmp[[i]]$iter }) %>% unlist()

  }

  s <- Sys.time()
  if (overdispersion) {
    if (verbose) { message("Fit overdispersion") }

    theta <- parallel::mclapply(1:ngenes, function(i) {
      devil:::fit_dispersion(beta[i,], design_matrix, input_matrix[i,], offset_vector, tolerance = tolerance, max_iter = max_iter, do_cox_reid_adjustment = do_cox_reid_adjustment)
    }, mc.cores = n.cores) %>% unlist()

  } else {
    theta = rep(0, ngenes)
  }

  return(list(
    beta=beta,
    overdispersion=theta,
    iterations=iterations,
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
