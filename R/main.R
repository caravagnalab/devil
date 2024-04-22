
#' Fit Model Parameters
#'
#' This function fits model parameters including beta coefficients, dispersion parameter,
#' and beta sigma using provided predictor variables (X) and response variable (y).
#' It optionally estimates overdispersion based on the fitted model.
#'
#' @description
#' This function fits model parameters including beta coefficients, dispersion parameter,
#' and beta sigma using provided predictor variables (input_matrix) and response variable (design_matrix).
#' It optionally estimates overdispersion based on the fitted model.
#'
#' @param input_matrix Vector of response variable.
#' @param design_matrix Matrix of predictor variables.
#' @param overdispersion Logical indicating whether to estimate overdispersion.
#' @param offset Offset vector to be included in the model. Defaults to 0.
#' @param size_factors Logical indicating whether to compute size factors. Defaults to TRUE.
#' @param verbose Logical indicating whether to display progress messages. Defaults to FALSE.
#' @param max_iter Maximum number of iterations for optimization. Defaults to 500.
#' @param eps Tolerance level for convergence criterion. Defaults to 1e-4.
#' @param parallel Logical indicating whether to use parallel computation. Defaults to TRUE.
#' @param use_memory Logical indicating whether the data will be stored in memory. Defaults to TRUE.
#' @return List containing fitted model parameters including beta coefficients,
#' overdispersion parameter (if estimated), and beta sigma.
#' @export
#' @rawNamespace useDynLib(devil);
fit_devil <- function(input_matrix, design_matrix, overdispersion = TRUE, offset=0, size_factors=TRUE, verbose=FALSE, max_iter=500, eps=1e-4, parallel=TRUE, use_memory=TRUE) {

  if (parallel) {
    n.cores = parallel::detectCores()
  } else {
    n.cores = 1
  }

  input_mat <- handle_input_matrix(input_matrix, use_memory, verbose=verbose)

  if (size_factors) {
    if (verbose) { message("Compute size factors") }
    sf <- calculate_sf(input_mat)
  } else {
    sf <- rep(1, nrow(design_matrix))
  }

  offset_matrix = compute_offset_matrix(offset, input_mat, sf)

  if (verbose) { message("Initialize beta estimate") }
  groups <- get_groups_for_model_matrix(design_matrix)

  if (!is.null(groups)) {
    beta_0_groups <- init_beta_groups(input_mat, groups, offset_matrix)
  }
  beta_0 <- init_beta(input_mat, design_matrix, offset_matrix)

  ngenes <- nrow(input_matrix)
  nfeatures <- ncol(design_matrix)

  if (verbose) { message("Fit beta coefficients") }

  tmp <- parallel::mclapply(1:ngenes, function(i) {
    if (!(is.null(groups))) {
      r <- beta_fit(input_mat[i,], design_matrix, beta_0_groups[i,], offset_matrix[i,], max_iter = max_iter, eps = eps)
      if (r$iter == max_iter) {
        r <- beta_fit(input_mat[i,], design_matrix, beta_0[i,], offset_matrix[i,], max_iter = max_iter, eps = eps)
      }
    } else {
      r <- beta_fit(input_mat[i,], design_matrix, beta_0[i,], offset_matrix[i,], max_iter = max_iter, eps = eps)
    }
    r
  }, mc.cores = n.cores)

  beta <- lapply(1:ngenes, function(i) {
    tmp[[i]]$mu_beta
  }) %>% do.call("rbind", .)
  rownames(beta) <- rownames(input_matrix)

  sigma <- lapply(1:ngenes, function(i) {
    tmp[[i]]$Zigma
  }) %>% do.call("cbind", .) %>% array(dim=c(nfeatures, nfeatures, ngenes))

  iterations <- lapply(1:ngenes, function(i) {
    tmp[[i]]$iter
  }) %>% unlist()

  if (overdispersion) {
    if (verbose) { message("Fit overdispersion") }

    theta <- parallel::mclapply(1:ngenes, function(i) {
      fit_dispersion(beta[i,], design_matrix, input_mat[i,], offset_matrix[i,])
    }, mc.cores = n.cores) %>% unlist()

  } else {
    theta = rep(0, ngenes)
  }

  return(list(
    beta=beta,
    overdispersion=theta,
    sigma_beta=sigma,
    iterations=iterations,
    size_factors=sf,
    offset_matrix=offset_matrix,
    design_matrix=design_matrix,
    input_matrix=input_matrix,
    input_parameters=list(max_iter=max_iter, eps=eps, parallel=parallel)
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

handle_input_matrix <- function(input_matrix, use_memory, verbose) {
  if(is.matrix(input_matrix)){
    if(!is.numeric(input_matrix)){
      stop("The input_matrix argument must consist of numeric values and not of ", mode(input_matrix), " values")
    }

    if (isTRUE(use_memory)) {
      data_mat <- input_matrix
    } else if (isFALSE(use_memory)){
      data_mat <- HDF5Array::writeHDF5Array(input_matrix)
    } else {
      stop("Illegal argument type for use_memory. Can only handle 'TRUE' or 'FALSE'")
    }

  } else if (methods::is(input_matrix, "DelayedArray")){

    if (isTRUE(use_memory)) {
      data_mat <- input_matrix
    } else if (isTRUE(use_memory)) {
      data_mat <- as.matrix(input_matrix)
    } else {
      stop("Illegal argument type for use_memory. Can only handle TRUE' or 'FALSE'")
    }

  } else if (methods::is(input_matrix, "dgCMatrix") || methods::is(input_matrix, "dgTMatrix")) {

    if (isFALSE(use_memory)) {
      data_mat <- HDF5Array::writeHDF5Array(input_matrix)
    } else if (isTRUE(use_memory)) {
      data_mat <- as.matrix(input_matrix)
    }

  }else{
    stop("Cannot handle data of class '", class(input_matrix), "'.",
         "It must be of type dense matrix object (i.e., a base matrix or DelayedArray),",
         " or a container for such a matrix (for example: SummarizedExperiment).")
  }
  data_mat
}
