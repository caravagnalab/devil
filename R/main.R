#' Fit Model Parameters
#'
#' This function fits model parameters, including beta coefficients, the dispersion parameter,
#' and beta sigma, using the provided predictor variables (`design_matrix`) and response variable (`input_matrix`).
#' It optionally estimates overdispersion based on the fitted model.
#'
#' @description
#' `fit_devil` performs model fitting by estimating beta coefficients, dispersion parameters,
#' and beta sigma. The function uses predictor variables provided in the `design_matrix` and a response
#' variable provided in the `input_matrix`. Optional features include the estimation of overdispersion
#' and the computation of size factors. The function supports parallel processing and allows customization
#' of various parameters such as the number of iterations, convergence tolerance, and more.
#'
#' @param input_matrix A numeric matrix representing the response variable, with rows corresponding to genes and columns to samples.
#' @param design_matrix A numeric matrix representing the predictor variables, with rows corresponding to samples and columns to predictors.
#' @param overdispersion Logical value indicating whether to estimate the overdispersion parameter. (default is `TRUE`)
#' @param init_overdispersion If `NULL`, a initial overdispersion value will be estimated. If a numerical value is passed, it will be used as starting value. (default is `NULL`, recommended numerical is `100`)
#' @param do_cox_reid_adjustment .
#' @param offset A numeric vector to be included as an offset in the model. Can be used to avoid issues with non-invertible matrices.(default is `0`)
#' @param size_factors Logical value indicating whether to compute size factors for normalization. (default is `TRUE`)
#' @param verbose Logical value indicating whether to display progress messages during execution. (default is `FALSE`)
#' @param max_iter Integer specifying the maximum number of iterations allowed for the optimization process. (default is `100`)
#' @param tolerance Numeric value indicating the tolerance level for the convergence criterion. (default is `1e-3`)
#' @param CUDA Logical value indicating whether to use GPU version of the code (default is `FALSE`)
#' @param batch_size Integer specifying the number of genes that will be fit in each batch if `CUDA = TRUE`. (default is 1024)
#' @param parallel.cores Integer specifying the number of CPU cores to use for parallelization. If `NULL`, the maximum number of available cores are used. (defaults is `NULL`)
#'
#' @return A list containing the following elements:
#' \item{beta}{A matrix of fitted beta coefficients for each gene.}
#' \item{overdispersion}{A numeric vector of overdispersion parameters for each gene (if estimated).}
#' \item{iterations}{A numeric vector indicating the number of iterations taken for each gene.}
#' \item{size_factors}{A numeric vector of size factors used for normalization.}
#' \item{offset_matrix}{A numeric matrix of offset values used in the model.}
#' \item{design_matrix}{The design matrix provided as input.}
#' \item{input_matrix}{The input matrix used after processing.}
#' \item{input_parameters}{A list of input parameters used in the function, including `max_iter`, `tolerance`, and `parallel.cores`.}
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
