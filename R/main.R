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
#' @param offset A numeric vector to be included as an offset in the model. (default is `0`)
#' @param size_factors Logical value indicating whether to compute size factors for normalization. (default is `TRUE`)
#' @param verbose Logical value indicating whether to display progress messages during execution. (default is `FALSE`)
#' @param max_iter Integer specifying the maximum number of iterations allowed for the optimization process. (default is `500`)
#' @param tolerance Numeric value indicating the tolerance level for the convergence criterion. (default is `1e-3`)
#' @param eps A small numeric value added to `input_matrix` to avoid issues with non-invertible matrices. (default is `1e-6`)
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
    offset=0,
    size_factors=TRUE,
    #avg_counts=.001,
    #min_cells=5,
    verbose=FALSE,
    max_iter=500,
    tolerance=1e-3,
    eps=1e-6,
    CUDA = FALSE,
    batch_size = 1024L,
    parallel.cores=NULL) {

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

  # Add epsilon to input_matrix to avoid non invertible matrices
  input_matrix <- input_matrix + eps

  input_mat <- devil:::handle_input_matrix(input_matrix, verbose=verbose)

  gene_names <- rownames(input_matrix)
  # counts_per_cell <- rowMeans(input_mat)
  # cell_per_genes <- rowSums(input_mat > 0)
  # filter_genes <- (counts_per_cell <= avg_counts) | (cell_per_genes <= min_cells)
  # n_low_genes <- sum(filter_genes)
  # if (n_low_genes > 0) {
  #   message(paste0("Removing ", n_low_genes, " lowly expressed genes."))
  #   input_mat <- matrix(input_mat[!filter_genes, ], ncol = nrow(design_matrix), nrow = sum(!filter_genes))
  #   input_matrix <- input_matrix[!filter_genes, ]
  #   gene_names <- gene_names[!filter_genes]
  # }

  if (size_factors) {
    if (verbose) { message("Compute size factors") }
    sf <- devil:::calculate_sf(input_matrix, verbose = verbose)
  } else {
    sf <- rep(1, nrow(design_matrix))
  }

  offset_matrix = devil:::compute_offset_matrix(offset, input_mat, sf)
  dispersion_init <- c(devil:::estimate_dispersion(input_matrix, offset_matrix))

  ngenes <- nrow(input_mat)
  nfeatures <- ncol(design_matrix)

  if (verbose) { message("Initialize beta estimate") }
  # groups <- devil:::get_groups_for_model_matrix(design_matrix)
  #
  # if (!is.null(groups)) {
  #   beta_0 <- devil:::init_beta_groups(input_mat, groups, offset_matrix)
  # } else {
  #   beta_0 <- devil:::init_beta(input_mat, design_matrix, offset_matrix)
  # }

  beta_0 <- devil:::init_beta(input_mat, design_matrix, offset_matrix)

  if (CUDA & CUDA_is_available) {
    message("Messing with CUDA! Implementation still needed")

    if (batch_size > ngenes) {
      message("Lowering batch size because of too few genes")
      batch_size <- 2^floor(log2(ngenes))
    } else if (batch_size != 2^floor(log2(batch_size))) {
      message("Converting batch size to closest power of two")
      batch_sizes <- c(2^floor(log2(batch_size)), 2^ceiling(log2(batch_size)))
      batch_sizes <- batch_sizes[batch_sizes <= ngenes]
      batch_size <- batch_sizes[which.min(abs(batch_sizes - batch_size))]
    }



    remainder = ngenes %% batch_size
    extra_genes = batch_size - remainder

    extra_input_mat <- matrix(exp(.1), nrow = extra_genes, ncol = ncol(input_mat))
    extra_offset_mat <- matrix(1, nrow = extra_genes, ncol = ncol(input_mat))
    extra_dispersion_init <- rep(1, extra_genes)
    extra_beta_0 <- matrix(.1, nrow = extra_genes, ncol = nfeatures)

    # Bind new matrices
    l_input_mat <- rbind(input_mat, extra_input_mat)
    l_offset_matrix <- rbind(offset_matrix, extra_offset_mat)
    l_beta0 <- rbind(beta_0, extra_beta_0)
    l_dispersion_init <- c(dispersion_init, extra_dispersion_init)

    stop("beta_fit_gpu not yet implemented")
    beta <- beta_fit_gpu(l_input_mat, design_matrix, l_beta0, l_offset_matrix, l_dispersion_init, max_iter = max_iter, eps = tolerance, batch_size = batch_size)
    beta <- beta[1:ngenes,]

    # tmp <- parallel::mclapply(1:(ngenes+extra_genes), function(i) {
    #   devil:::beta_fit(l_input_mat[i,], design_matrix, l_beta0[i,], l_offset_matrix[i,], l_dispersion_init[i], max_iter = max_iter, eps = tolerance)
    #   #devil:::beta_fit(input_mat[i,], design_matrix, beta_0[i,], offset_matrix[i,], 1, max_iter = max_iter, eps = tolerance)
    # }, mc.cores = n.cores)
    # beta <- lapply(1:length(tmp), function(i) { tmp[[i]]$mu_beta }) %>% do.call("rbind", .)
    # beta <- beta[1:ngenes,]

  } else {
    if (verbose) { message("Fit beta coefficients") }
    tmp <- parallel::mclapply(1:ngenes, function(i) {
      devil:::beta_fit(input_mat[i,], design_matrix, beta_0[i,], offset_matrix[i,], dispersion_init[i], max_iter = max_iter, eps = tolerance)
      #devil:::beta_fit(input_mat[i,], design_matrix, beta_0[i,], offset_matrix[i,], 1, max_iter = max_iter, eps = tolerance)
    }, mc.cores = n.cores)

    # if (!is.null(groups)) {
    #   beta_0 <- init_beta_groups(input_mat, groups, offset_matrix)
    #
    #   if (verbose) { message("Fit beta coefficients") }
    #   tmp <- parallel::mclapply(1:ngenes, function(i) {
    #     r_groups <- lapply(unique(groups), function(g) {
    #       group_filter <- groups == g
    #       y <- input_mat[i,group_filter,drop=FALSE]
    #       if (sum(y) != 0) {
    #         beta_fit_group(y,
    #                                beta_0[i,g==unique(groups),drop=T],
    #                                offset_matrix[i,group_filter, drop=F],
    #                                dispersion_init[i],
    #                                max_iter = max_iter,
    #                                eps = eps)
    #       } else {
    #         list(mu_beta = -1e08, iter = 1)
    #       }
    #
    #     })
    #     r <- list(
    #       mu_beta = do.call(cbind, lapply(r_groups, function(x) {x$mu_beta})),
    #       iter = mean(unlist(lapply(r_groups, function(x) {x$iter})))
    #     )
    #   }, mc.cores = n.cores)
    #
    # } else {
    #   beta_0 <- init_beta(input_mat, design_matrix, offset_matrix)
    #
    #   if (verbose) { message("Fit beta coefficients") }
    #   tmp <- parallel::mclapply(1:ngenes, function(i) {
    #     beta_fit(input_mat[i,], design_matrix, beta_0[i,], offset_matrix[i,], dispersion_init[i], max_iter = max_iter, eps = tolerance)
    #   }, mc.cores = n.cores)
    # }

    beta <- lapply(1:ngenes, function(i) { tmp[[i]]$mu_beta }) %>% do.call("rbind", .)
    rownames(beta) <- gene_names
    # if (!(is.null(groups))) {
    #   first_occurence_in_groups <- match(unique(groups), groups)
    #   beta <- t(solve(design_matrix[first_occurence_in_groups, ,drop=FALSE], t(beta)))
    # }

    iterations <- lapply(1:ngenes, function(i) { tmp[[i]]$iter }) %>% unlist()
  }



  if (overdispersion) {
    if (verbose) { message("Fit overdispersion") }

    theta <- parallel::mclapply(1:ngenes, function(i) {
      devil:::fit_dispersion(beta[i,], design_matrix, input_mat[i,], offset_matrix[i,], tolerance = tolerance, max_iter = max_iter)
    }, mc.cores = n.cores) %>% unlist()

  } else {
    theta = rep(0, ngenes)
  }

  return(list(
    beta=beta,
    overdispersion=theta,
    iterations=iterations,
    size_factors=sf,
    offset_matrix=offset_matrix,
    design_matrix=design_matrix,
    input_matrix=input_mat,
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
