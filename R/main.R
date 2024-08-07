
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
#' @param avg_counts Minimum value of counts per cell required to fit a gene. Default is 0.001.
#' @param min_cells Minimum value of cells with non-zero counts required to fit a gene. Default is 5.
#' @param verbose Logical indicating whether to display progress messages. Defaults to FALSE.
#' @param max_iter Maximum number of iterations for optimization. Defaults to 500.
#' @param eps Tolerance level for convergence criterion. Defaults to 1e-4.
#' @param parallel.cores Number of cores required for parallelization. Default is NULL, which use the maximum cores available.
#' @return List containing fitted model parameters including beta coefficients and overdispersion parameter (if estimated).
#' @export
#' @rawNamespace useDynLib(devil);
fit_devil <- function(input_matrix, design_matrix, overdispersion = TRUE, offset=0, size_factors=TRUE, avg_counts=.001, min_cells=5, verbose=FALSE, max_iter=500, eps=1e-4, parallel.cores=NULL) {

  max.cores <- parallel::detectCores()
  if (is.null(parallel.cores)) {
    n.cores = max.cores
  } else {
    if (parallel.cores > max.cores) {
      message(paste0("Requested ", parallel.cores, " cores, but only ", max.cores, " available."))
    }
    n.cores = min(max.cores, parallel.cores)
  }

  input_mat <- devil:::handle_input_matrix(input_matrix, verbose=verbose)

  gene_names <- rownames(input_matrix)
  counts_per_cell <- rowMeans(input_mat)
  cell_per_genes <- rowSums(input_mat > 0)
  filter_genes <- (counts_per_cell <= avg_counts) | (cell_per_genes <= min_cells)
  n_low_genes <- sum(filter_genes)
  if (n_low_genes > 0) {
    message(paste0("Removing ", n_low_genes, " lowly expressed genes."))
    input_mat <- matrix(input_mat[!filter_genes, ], ncol = nrow(design_matrix), nrow = sum(!filter_genes))
    input_matrix <- input_matrix[!filter_genes, ]
    gene_names <- gene_names[!filter_genes]
  }

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
  groups <- devil:::get_groups_for_model_matrix(design_matrix)

  if (!is.null(groups)) {
    beta_0 <- devil:::init_beta_groups(input_mat, groups, offset_matrix)

    if (verbose) { message("Fit beta coefficients") }
    tmp <- parallel::mclapply(1:ngenes, function(i) {
      r_groups <- lapply(unique(groups), function(g) {
        group_filter <- groups == g
        y <- input_mat[i,group_filter,drop=FALSE]
        if (sum(y) != 0) {
          devil:::beta_fit_group(y,
                                 beta_0[i,g==unique(groups),drop=T],
                                 offset_matrix[i,group_filter, drop=F],
                                 dispersion_init[i],
                                 max_iter = max_iter,
                                 eps = eps)
        } else {
          list(mu_beta = -1e08, iter = 1)
        }

      })
      r <- list(
        mu_beta = do.call(cbind, lapply(r_groups, function(x) {x$mu_beta})),
        iter = mean(unlist(lapply(r_groups, function(x) {x$iter})))
      )
    }, mc.cores = n.cores)

  } else {
    beta_0 <- devil:::init_beta(input_mat, design_matrix, offset_matrix)

    if (verbose) { message("Fit beta coefficients") }
    tmp <- parallel::mclapply(1:ngenes, function(i) {
      devil:::beta_fit(input_mat[i,], design_matrix, beta_0[i,], offset_matrix[i,], dispersion_init[i], max_iter = max_iter, eps = eps)
    }, mc.cores = n.cores)
  }

  beta <- lapply(1:ngenes, function(i) { tmp[[i]]$mu_beta }) %>% do.call("rbind", .)
  rownames(beta) <- gene_names

  if (!(is.null(groups))) {
    first_occurence_in_groups <- match(unique(groups), groups)
    beta <- t(solve(design_matrix[first_occurence_in_groups, ,drop=FALSE], t(beta)))
  }

  iterations <- lapply(1:ngenes, function(i) { tmp[[i]]$iter }) %>% unlist()

  if (overdispersion) {
    if (verbose) { message("Fit overdispersion") }

    theta <- parallel::mclapply(1:ngenes, function(i) {
      devil:::fit_dispersion(beta[i,], design_matrix, input_mat[i,], offset_matrix[i,])
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
    input_parameters=list(max_iter=max_iter, eps=eps, parallel.cores=n.cores)
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
