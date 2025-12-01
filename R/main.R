#' Fit Statistical Model for Count Data
#'
#' @description
#' Fits a statistical model to count data, particularly designed for RNA sequencing data analysis.
#' The function estimates regression coefficients (beta), gene-wise overdispersion parameters,
#' and normalizes data using size factors. It supports both CPU and (optionally) GPU-based
#' computation with parallel processing capabilities.
#'
#' @details
#' The function implements a negative binomial regression model with the following steps:
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
#' @param init_beta_rough Logical. Whether to initialize betas in a rough but extremely fast way.
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
#' @param profiling Logical. If \code{TRUE}, prints timing information for major
#'   steps (size factors, offset computation, initial dispersion, beta fit, theta fit).
#'   Useful for performance profiling. Default: \code{FALSE}.
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
#' @export
#' @rawNamespace useDynLib(devil);
fit_devil = function(
    input_matrix,
    design_matrix,
    overdispersion = "MOM",
    init_overdispersion = NULL,
    init_beta_rough = FALSE,
    offset=0,
    size_factors=NULL,
    verbose=FALSE,
    max_iter=200,
    tolerance=1e-3,
    CUDA = FALSE,
    batch_size = 1024L,
    parallel.cores=1,
    profiling = FALSE) {

  # - Input parameters ----
  # Read general info about input matrix and design matrix
  gene_names <- rownames(input_matrix)
  ngenes <- nrow(input_matrix)
  nfeatures <- ncol(design_matrix)

  # Detect cores to use
  max.cores <- parallel::detectCores()
  if (is.null(parallel.cores)) {
    n.cores <- max.cores
  } else {
    if (parallel.cores > max.cores) {
      message(paste0("Requested ", parallel.cores, " cores, but only ", max.cores, " available."))
    }
    n.cores <- min(max.cores, parallel.cores)
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

  # Compute size factors, common to CPU and GPU path
  if (!is.null(size_factors)) {
    if (verbose) { message("Compute size factors") }
    if (profiling) t_start <- Sys.time()
    sf <- calculate_sf(input_matrix, method = size_factors, verbose = verbose)
    if (profiling) {
      t_end <- Sys.time()
      message(sprintf("[TIMING] Size factors computation: %.3f seconds", as.numeric(difftime(t_end, t_start, units = "secs"))))
    }
  } else {
    sf <- rep(1, nrow(design_matrix))
  }

  if (profiling) {
    end = Sys.time()
    message(paste0("Size factors computing : ", end - start))
  }

  ## - Compute offset vector ----
  if (profiling) start = Sys.time()
  offset_vector = compute_offset_vector(offset, input_matrix, sf)
  if (profiling) {
    end = Sys.time()
    message(paste0("Offset vector computing : ", end - start))
  }

  # - Start GPU vs CPU branch ----
  if (profiling) start = Sys.time()
  if (CUDA & CUDA_is_available) {
    ## - GPU branch ----
    message("Messing with CUDA! Implementation still needed")

  if (CUDA & CUDA_is_available) {
    remainder = ngenes %% batch_size
    extra_genes = remainder
    genes_batch = ngenes - extra_genes

    if (verbose) { message("Fit beta coefficients (GPU)") }
    if (profiling) t_start <- Sys.time()

    res_beta_fit <- beta_fit_gpu(
      input_matrix[1:genes_batch,],
      design_matrix,
      offset_vector,
      max_iter = max_iter,
      eps = tolerance,
      batch_size = batch_size,
      TEST = TEST
    )

    if (remainder > 0) {
      res_beta_fit_extra <- beta_fit_gpu(
        input_matrix[(genes_batch+1):ngenes,],
        design_matrix,
        offset_vector,
        max_iter = max_iter,
        eps = tolerance,
        batch_size = extra_genes,
        TEST = TEST
      )
    }

    if (profiling) {
      t_end <- Sys.time()
      message(sprintf("[TIMING] Beta fitting (GPU): %.3f seconds", as.numeric(difftime(t_end, t_start, units = "secs"))))
    }

    if (verbose) { message("Process GPU results") }
    if (profiling) t_start <- Sys.time()
    beta = res_beta_fit$mu_beta
    # Get the GPU-computed theta values (MOM overdispersion)
    theta_gpu = res_beta_fit$theta

    if (remainder > 0) {
      beta_extra = res_beta_fit_extra$mu_beta
      beta <- rbind(beta, beta_extra)
      beta_iters=c(res_beta_fit$iter, res_beta_fit_extra$iter)
      # Combine theta values from both batches
      theta_gpu = c(theta_gpu, res_beta_fit_extra$theta)
    } else {
      beta_iters=c(res_beta_fit$iter)
    }
    
    # Debug: Consistency check for beta initialization (only if TEST enabled)
    if (TEST && !is.null(res_beta_fit$beta_init)) {
      if (verbose) { message("Performing GPU vs CPU beta initialization consistency check") }
      if (profiling) t_start_init <- Sys.time()
      
      # Get GPU beta_init
      beta_init_gpu = res_beta_fit$beta_init
      if (remainder > 0) {
        beta_init_gpu = rbind(beta_init_gpu, res_beta_fit_extra$beta_init)
      }
      
      # Compute CPU rough initialization for comparison
      beta_init_cpu = matrix(0, nrow = nrow(input_matrix), ncol = ncol(design_matrix))
      beta_init_cpu[,1] = rowMeans(input_matrix) %>% log1p()
      
      if (profiling) {
        t_end_init <- Sys.time()
        message(sprintf("[TIMING] CPU beta initialization (for comparison): %.3f seconds", as.numeric(difftime(t_end_init, t_start_init, units = "secs"))))
      }
      
      # Compare first column (intercept) since other columns are all zeros
      abs_diff_init <- abs(beta_init_gpu[, 1] - beta_init_cpu[, 1])
      rel_diff_init <- abs_diff_init / (abs(beta_init_cpu[, 1]) + 1e-10)
      
      message(sprintf("Beta initialization consistency check (intercept only):"))
      message(sprintf("  Mean absolute difference: %.6f", mean(abs_diff_init)))
      message(sprintf("  Median absolute difference: %.6f", median(abs_diff_init)))
      message(sprintf("  Max absolute difference: %.6f", max(abs_diff_init)))
      message(sprintf("  Mean relative difference: %.6f", mean(rel_diff_init)))
      message(sprintf("  Median relative difference: %.6f", median(rel_diff_init)))
      message(sprintf("  Max relative difference: %.6f", max(rel_diff_init)))
      message(sprintf("  Correlation: %.6f", cor(beta_init_gpu[, 1], beta_init_cpu[, 1])))
      
      # Check for potential issues
      n_large_diff_init <- sum(rel_diff_init > 0.01)  # More than 1% difference
      if (n_large_diff_init > 0) {
        warning(sprintf("Found %d genes (%.1f%%) with >1%% relative difference in beta initialization",
                       n_large_diff_init, 100 * n_large_diff_init / length(beta_init_gpu[, 1])))
      }
    }
    
    # Compute or retrieve dispersion (k) values
    if (TEST && !is.null(res_beta_fit$k)) {
      # TEST mode: use GPU-computed k values for consistency check
      if (verbose) { message("Performing GPU vs CPU dispersion consistency check") }
      dispersion_init_gpu = res_beta_fit$k
      if (remainder > 0) {
        dispersion_init_gpu = c(dispersion_init_gpu, res_beta_fit_extra$k)
      }
      
      dispersion_init_cpu <- c(estimate_dispersion(input_matrix, offset_vector))
      # Convert GPU k (which is 1/dispersion) back to dispersion for comparison
      dispersion_gpu <- 1.0 / dispersion_init_gpu
      
      # Calculate differences
      abs_diff <- abs(dispersion_gpu - dispersion_init_cpu)
      rel_diff <- abs_diff / (dispersion_init_cpu + 1e-10)  # Avoid division by zero
      
      message(sprintf("Dispersion consistency check:"))
      message(sprintf("  Mean absolute difference: %.6f", mean(abs_diff)))
      message(sprintf("  Median absolute difference: %.6f", median(abs_diff)))
      message(sprintf("  Max absolute difference: %.6f", max(abs_diff)))
      message(sprintf("  Mean relative difference: %.6f", mean(rel_diff)))
      message(sprintf("  Median relative difference: %.6f", median(rel_diff)))
      message(sprintf("  Max relative difference: %.6f", max(rel_diff)))
      message(sprintf("  Correlation: %.6f", cor(dispersion_gpu, dispersion_init_cpu)))
      
      # Check for potential issues
      n_large_diff <- sum(rel_diff > 0.1)  # More than 10% difference
      if (n_large_diff > 0) {
        warning(sprintf("Found %d genes (%.1f%%) with >10%% relative difference between GPU and CPU dispersion estimates",
                       n_large_diff, 100 * n_large_diff / length(dispersion_gpu)))
      }
      
      # Use GPU k values
      dispersion_init = dispersion_init_gpu
    } else {
      # Production mode: compute dispersion on CPU (no GPU transfer overhead)
      if (verbose) { message("Computing dispersion on CPU") }
      if (profiling) t_start_disp <- Sys.time()
      
      dispersion_init <- c(estimate_dispersion(input_matrix, offset_vector))
      
      if (profiling) {
        t_end_disp <- Sys.time()
        message(sprintf("[TIMING] CPU dispersion computation: %.3f seconds", as.numeric(difftime(t_end_disp, t_start_disp, units = "secs"))))
      }
    }
    
    # Consistency check for MOM overdispersion: compare GPU vs CPU (only if TEST enabled)
    if (overdispersion == "MOM" && TEST) {
      if (verbose) { message("Performing GPU vs CPU MOM overdispersion consistency check") }
      if (profiling) t_start <- Sys.time()
      
      # Compute MOM overdispersion on CPU for comparison
      theta_cpu = estimate_mom_dispersion_cpp(input_matrix, design_matrix, beta, sf)
      
      if (profiling) {
        t_end <- Sys.time()
        message(sprintf("[TIMING] CPU MOM overdispersion (for comparison): %.3f seconds", as.numeric(difftime(t_end, t_start, units = "secs"))))
      }
      
      # Calculate differences
      abs_diff_theta <- abs(theta_gpu - theta_cpu)
      rel_diff_theta <- abs_diff_theta / (theta_cpu + 1e-10)  # Avoid division by zero
      
      message(sprintf("MOM Overdispersion (theta) consistency check:"))
      message(sprintf("  Mean absolute difference: %.6f", mean(abs_diff_theta)))
      message(sprintf("  Median absolute difference: %.6f", median(abs_diff_theta)))
      message(sprintf("  Max absolute difference: %.6f", max(abs_diff_theta)))
      message(sprintf("  Mean relative difference: %.6f", mean(rel_diff_theta)))
      message(sprintf("  Median relative difference: %.6f", median(rel_diff_theta)))
      message(sprintf("  Max relative difference: %.6f", max(rel_diff_theta)))
      message(sprintf("  Correlation: %.6f", cor(theta_gpu, theta_cpu)))
      
      # Check for potential issues
      n_large_diff_theta <- sum(rel_diff_theta > 0.1)  # More than 10% difference
      if (n_large_diff_theta > 0) {
        warning(sprintf("Found %d genes (%.1f%%) with >10%% relative difference between GPU and CPU MOM overdispersion estimates",
                       n_large_diff_theta, 100 * n_large_diff_theta / length(theta_gpu)))
      }
    }
    
    # Use GPU-computed MOM overdispersion if available
    if (overdispersion == "MOM") {
      if (verbose) { message("Using GPU-computed MOM overdispersion (theta)") }
      theta = theta_gpu
      theta_iters = 0
    }
    
    # Consistency check for beta fitting: compare GPU vs CPU (only if TEST enabled)
    if (TEST) {
      if (verbose) { message("Performing GPU vs CPU beta fitting consistency check") }
      if (profiling) t_start_beta <- Sys.time()
      
      # Compute rough beta initialization for CPU comparison (same as GPU uses)
      beta_0_test = matrix(0, nrow = nrow(input_matrix), ncol = ncol(design_matrix))
      beta_0_test[,1] = rowMeans(input_matrix) %>% log1p()
      
      # Compute beta on CPU for comparison
      tmp_cpu <- parallel::mclapply(1:ngenes, function(i) {
        beta_fit(input_matrix[i,], design_matrix, beta_0_test[i,], offset_vector, dispersion_init[i], max_iter = max_iter, eps = tolerance)
      }, mc.cores = n.cores)
      
      beta_cpu <- lapply(1:ngenes, function(i) { tmp_cpu[[i]]$mu_beta }) %>% do.call("rbind", .)
      
      if (profiling) {
        t_end_beta <- Sys.time()
        message(sprintf("[TIMING] CPU beta fitting (for comparison): %.3f seconds", as.numeric(difftime(t_end_beta, t_start_beta, units = "secs"))))
      }
      
      # Calculate differences
      abs_diff_beta <- abs(beta - beta_cpu)
      rel_diff_beta <- abs_diff_beta / (abs(beta_cpu) + 1e-10)  # Avoid division by zero
      
      # Find problematic elements (large relative diff but check if absolute is also large)
      large_rel_idx <- which(rel_diff_beta > 1.0)  # >100% relative difference
      
      message(sprintf("Beta fitting consistency check:"))
      message(sprintf("  Mean absolute difference: %.6f", mean(abs_diff_beta)))
      message(sprintf("  Median absolute difference: %.6f", median(abs_diff_beta)))
      message(sprintf("  Max absolute difference: %.6f", max(abs_diff_beta)))
      message(sprintf("  Mean relative difference: %.6f", mean(rel_diff_beta)))
      message(sprintf("  Median relative difference: %.6f", median(rel_diff_beta)))
      message(sprintf("  Max relative difference: %.6f", max(rel_diff_beta)))
      message(sprintf("  Correlation: %.6f", cor(c(beta), c(beta_cpu))))
      
      # Additional diagnostics for large relative differences
      if (length(large_rel_idx) > 0) {
        message(sprintf("  Found %d elements with >100%% relative difference:", length(large_rel_idx)))
        message(sprintf("    - Their mean absolute difference: %.6f", mean(abs_diff_beta[large_rel_idx])))
        message(sprintf("    - Their mean |beta_cpu| value: %.6f (small values amplify relative diff)", mean(abs(c(beta_cpu)[large_rel_idx]))))
        
        # Check if large relative differences are actually problematic
        large_abs_and_rel <- sum(abs_diff_beta > 0.1 & rel_diff_beta > 0.1)
        if (large_abs_and_rel > 0) {
          warning(sprintf("Found %d elements with BOTH >0.1 absolute AND >10%% relative difference - this may indicate a real problem!", large_abs_and_rel))
        }
      }
      
      # Check for potential issues (use combined threshold)
      n_large_diff_beta <- sum(abs_diff_beta > 0.01 & rel_diff_beta > 0.01)  # More than 1% relative AND >0.01 absolute difference
      if (n_large_diff_beta > 0) {
        warning(sprintf("Found %d elements (%.1f%%) with >1%% relative difference between GPU and CPU beta estimates",
                       n_large_diff_beta, 100 * n_large_diff_beta / length(c(beta))))
      }
    }

    if (is.null(dim(beta))) {
      beta <- matrix(beta, ncol = 1)
    }
    if (profiling) {
      t_end <- Sys.time()
      message(sprintf("[TIMING] GPU results processing: %.3f seconds", as.numeric(difftime(t_end, t_start, units = "secs"))))
    }

  } else {
    ## - CPU branch ----

    fit_res = cpu_fit(input_matrix = input_matrix,
                      design_matrix = design_matrix,
                      offset_vector = offset_vector,
                      init_overdispersion = init_overdispersion,
                      init_beta_rough = init_beta_rough,
                      overdispersion = overdispersion,
                      n.cores = n.cores, max_iter = max_iter,
                      tolerance = tolerance, verbose = verbose)
  }


  if (profiling) {
    end = Sys.time()
    message(paste0("Beta fit computing : ", end - start))
  }

  return(list(
    beta=fit_res$beta,
    overdispersion=fit_res$theta,
    iterations=fit_res$iterations,
    size_factors=sf,
    offset_vector=offset_vector,
    design_matrix=design_matrix,
    input_matrix=input_matrix,
    input_parameters=list(max_iter=max_iter, tolerance=tolerance, parallel.cores=n.cores)
  ))
}

# Inner CPU fitting function
cpu_fit <- function(input_matrix, design_matrix, offset_vector,
                    init_overdispersion, init_beta_rough, overdispersion,
                    n.cores, max_iter, tolerance, verbose) {

  ngenes <- nrow(input_matrix)
  gene_names <- rownames(input_matrix)

  # - Initialize dispersion ---
  if (verbose) message("Initialize theta")
  if (is.null(init_overdispersion)) {
    dispersion_init <- estimate_dispersion(input_matrix, offset_vector)
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

  # - Fit betas ---
  if (verbose) message("Fitting beta coefficients")

  tmp <- parallel::mclapply(
    X = seq_len(ngenes),
    mc.cores = n.cores,
    FUN = function(i) {
      beta_fit(
        input_matrix[i, ],
        design_matrix,
        beta_0[i, ],
        offset_vector,
        dispersion_init[i],
        max_iter = max_iter,
        eps      = tolerance
      )
    }
  )

  beta       <- do.call(rbind, lapply(tmp, `[[`, "mu_beta"))
  beta_iters <- vapply(tmp, `[[`, integer(1L), "iter")
  rownames(beta) <- gene_names

  # - Fit theta (overdispersion) ---
  if (!isFALSE(overdispersion)) {
    if (verbose) message("Fit overdispersion (mode = ", overdispersion, ")")

    if (overdispersion %in% c("old", "MLE")) {

      theta <- parallel::mclapply(
        X = seq_len(ngenes),
        mc.cores = n.cores,
        FUN = function(i) {
          fit_dispersion(
            beta      = beta[i, ],
            design_matrix,
            input_matrix[i, ],
            offset_vector,
            tolerance = tolerance,
            max_iter  = max_iter,
            do_cox_reid_adjustment = TRUE
          )
        }
      ) %>% unlist()

      theta_iters <- NA_integer_

    } else if (overdispersion %in% c("new", "I")) {

      y_unique_cap <- as.integer(ncol(input_matrix) / 2L)

      tmp_theta <- parallel::mclapply(
        X = seq_len(ngenes),
        mc.cores = n.cores,
        FUN = function(i) {
          fit_overdispersion_cppp(
            y            = input_matrix[i, ],
            X            = design_matrix,
            mu_beta      = beta[i, ],
            off          = offset_vector,
            kappa        = dispersion_init[i],
            max_iter     = max_iter,
            eps_theta    = tolerance,
            newton_max   = 16L,
            y_unique_cap = y_unique_cap
          )
        }
      )

      theta       <- vapply(tmp_theta, `[[`, numeric(1L), "kappa")
      theta_iters <- vapply(tmp_theta, `[[`, integer(1L), "kappa_iters")

    } else if (overdispersion == "MOM") {

      theta       <- estimate_mom_dispersion_cpp(input_matrix, design_matrix, beta, exp(offset_vector))
      theta_iters <- 0L

    } else {
      stop("Unknown overdispersion mode: ", overdispersion)
    }

  } else {
    theta       <- rep(0, ngenes)
    theta_iters <- 0L
  }

  list(
    beta       = beta,
    theta      = theta,
    iterations = list(
      beta_iters  = beta_iters,
      theta_iters = theta_iters
    )
  )
}


# get_groups_for_model_matrix <- function(model_matrix){
#   if(! lte_n_equal_rows(model_matrix, ncol(model_matrix))){
#     return(NULL)
#   }else{
#     get_row_groups(model_matrix, n_groups = ncol(model_matrix))
#   }
# }
#
# handle_input_matrix <- function(input_matrix, verbose) {
#   if(is.matrix(input_matrix)){
#     if(!is.numeric(input_matrix)){
#       stop("The input_matrix argument must consist of numeric values and not of ", mode(input_matrix), " values")
#     }
#     data_mat <- input_matrix
#   } else if (methods::is(input_matrix, "DelayedArray")){
#     data_mat <- input_matrix
#   } else if (methods::is(input_matrix, "dgCMatrix") || methods::is(input_matrix, "dgTMatrix")) {
#     data_mat <- as.matrix(input_matrix)
#   }else{
#     stop("Cannot handle data of class '", class(input_matrix), "'.",
#          "It must be of type dense matrix object (i.e., a base matrix or DelayedArray),",
#          " or a container for such a matrix (for example: SummarizedExperiment).")
#   }
#   data_mat
# }


# fit_devil_old <- function(
#     input_matrix,
#     design_matrix,
#     overdispersion = "MOM",
#     init_overdispersion = NULL,
#     init_beta_rough = FALSE,
#     # do_cox_reid_adjustment = TRUE,
#     offset=0,
#     size_factors=NULL,
#     verbose=FALSE,
#     max_iter=200,
#     tolerance=1e-3,
#     CUDA = FALSE,
#     batch_size = 1024L,
#     parallel.cores=1,
#     profiling = FALSE) {
#
#   # Read general info about input matrix and design matrix
#   gene_names <- rownames(input_matrix)
#   ngenes <- nrow(input_matrix)
#   nfeatures <- ncol(design_matrix)
#
#   # Detect cores to use
#   max.cores <- parallel::detectCores()
#   if (is.null(parallel.cores)) {
#     n.cores = max.cores
#   } else {
#     if (parallel.cores > max.cores) {
#       message(paste0("Requested ", parallel.cores, " cores, but only ", max.cores, " available."))
#     }
#     n.cores = min(max.cores, parallel.cores)
#   }
#
#   # Check if CUDA is available
#   CUDA_is_available <- FALSE
#   if (CUDA) {
#     if (!exists("beta_fit_gpu", mode = "function")) {
#       warning("CUDA support was not enabled during package installation. ",
#               "The beta_fit_gpu function is not available. ",
#               "Reinstall with configure.args='--with-cuda' to enable GPU acceleration. ",
#               "Falling back to CPU computation.")
#       CUDA <- FALSE
#     } else {
#       message("CUDA support detected - using GPU acceleration")
#       CUDA_is_available <- TRUE
#     }
#   }
#
#   # Compute size factors
#   if (profiling) start = Sys.time()
#
#   if (!is.null(size_factors)) {
#     if (verbose) { message("Compute size factors") }
p#     sf <- calculate_sf(input_matrix, method = size_factors, verbose = verbose)
#   } else {
#     sf <- rep(1, nrow(design_matrix))
#   }
#
#   if (profiling) {
#     end = Sys.time()
#     message(paste0("Size factors computing : ", end - start))
#   }
#
#   # Calculate offset vector
#   if (profiling) start = Sys.time()
#   offset_vector = compute_offset_vector(offset, input_matrix, sf)
#   if (profiling) {
#     end = Sys.time()
#     message(paste0("Offset vector computing : ", end - start))
#   }
#
#   # Initialize overdispersion
#   if (profiling) start = Sys.time()
#   if (is.null(init_overdispersion)) {
#     dispersion_init <- c(estimate_dispersion(input_matrix, offset_vector))
#   } else {
#     dispersion_init <- rep(init_overdispersion, nrow(input_matrix))
#   }
#   if (profiling) {
#     end = Sys.time()
#     message(paste0("Theta init computing : ", end - start))
#   }
#
#   # if (verbose) { message("Initialize beta estimate") }
#   # groups <- devil:::get_groups_for_model_matrix(design_matrix)
#
#   if (verbose) { message("Initialize beta estimate") }
#   if (profiling) start = Sys.time()
#
#   if (init_beta_rough) {
#     beta_0 = matrix(0, nrow = nrow(input_matrix), ncol = ncol(design_matrix))
#     beta_0[,1] = rowMeans(input_matrix) %>% log1p()
#   } else {
#     beta_0 <- init_beta(input_matrix, design_matrix, offset_vector)
#   }
#   #beta_0 = initialize_beta_univariate_matrix_cpp(design_matrix, input_matrix, sf)
#   if (profiling) {
#     end = Sys.time()
#     message(paste0("Beta init computing : ", end - start))
#   }
#
#   if (profiling) start = Sys.time()
#   if (CUDA & CUDA_is_available) {
#     message("Messing with CUDA! Implementation still needed")
#
#     remainder = ngenes %% batch_size
#     extra_genes = remainder
#     genes_batch = ngenes - extra_genes
#
#     message("Fit beta CUDA")
#     start_time <- Sys.time()
#
#     res_beta_fit <- beta_fit_gpu(
#       input_matrix[1:genes_batch,],
#       design_matrix,
#       beta_0[1:genes_batch,],
#       offset_vector,
#       dispersion_init[1:genes_batch],
#       max_iter = max_iter,
#       eps = tolerance,
#       batch_size = batch_size
#     )
#
#     if (remainder > 0) {
#       res_beta_fit_extra <- beta_fit_gpu(
#         input_matrix[(genes_batch+1):ngenes,],
#         design_matrix,
#         beta_0[(genes_batch+1):ngenes,],
#         offset_vector,
#         dispersion_init[(genes_batch+1):ngenes],
#         max_iter = max_iter,
#         eps = tolerance,
#         batch_size = extra_genes
#       )
#     }
#
#     end_time <- Sys.time()
#     message("BETA GPU RUNTIME:")
#     message(as.numeric(difftime(end_time, start_time, units = "secs")))
#
#     beta = res_beta_fit$mu_beta
#
#     if (remainder > 0) {
#       beta_extra = res_beta_fit_extra$mu_beta
#       beta <- rbind(beta, beta_extra)
#       beta_iters=c(res_beta_fit$iter, res_beta_fit_extra$iter)
#     } else {
#       beta_iters=c(res_beta_fit$iter)
#     }
#
#     if (is.null(dim(beta))) {
#       beta = matrix(beta, ncol = 1)
#     }
#
#   } else {
#
#     if (verbose) { message("Fitting beta coefficients") }
#
#     tmp <- parallel::mclapply(1:ngenes, function(i) {
#       beta_fit(input_matrix[i,], design_matrix, beta_0[i,], offset_vector, dispersion_init[i], max_iter = max_iter, eps = tolerance)
#     }, mc.cores = n.cores)
#
#     beta <- lapply(1:ngenes, function(i) { tmp[[i]]$mu_beta }) %>% do.call("rbind", .)
#     rownames(beta) <- gene_names
#     beta_iters <- lapply(1:ngenes, function(i) { tmp[[i]]$iter }) %>% unlist()
#   }
#   if (profiling) {
#     end = Sys.time()
#     message(paste0("Beta fit computing : ", end - start))
#   }
#
#   if (profiling) start = Sys.time()
#   if (!isFALSE(overdispersion)) {
#     if (verbose) { message("Fit overdispersion") }
#
#     if (overdispersion %in% c("old", "MLE")) {
#       theta <- parallel::mclapply(1:ngenes, function(i) {
#         fit_dispersion(beta[i,], design_matrix, input_matrix[i,], offset_vector,
#                        tolerance = tolerance, max_iter = max_iter, do_cox_reid_adjustment = TRUE)
#       }, mc.cores = n.cores) %>% unlist()
#       theta_iters = NA
#     } else if (overdispersion %in% c("new", "I")) {
#       y_unique_cap = as.integer(dim(input_matrix)[2] / 2)
#       tmp <- parallel::mclapply(1:ngenes, function(i) {
#         fit_overdispersion_cppp(y = input_matrix[i,], X = design_matrix, mu_beta = beta[i,], off = offset_vector,
#                                 kappa = dispersion_init[i], max_iter = max_iter, eps_theta = tolerance,
#                                 newton_max = 16, y_unique_cap = y_unique_cap)
#       }, mc.cores = n.cores)
#
#       theta = lapply(tmp, function(x) x$kappa) %>% unlist()
#       theta_iters = lapply(tmp, function(x) x$kappa_iters) %>% unlist()
#     } else if (overdispersion == "MOM") {
#       theta = estimate_mom_dispersion_cpp(input_matrix, design_matrix, beta, sf)
#       theta_iters = 0
#     } else {
#       stop()
#     }
#   } else {
#     theta = rep(0, ngenes)
#     theta_iters = 0
#   }
#   if (profiling) {
#     end = Sys.time()
#     message(paste0("Theta fit computing : ", end - start))
#   }
#
#   return(list(
#     beta=beta,
#     overdispersion=theta,
#     iterations=list(beta_iters=beta_iters, theta_iters = theta_iters),
#     size_factors=sf,
#     offset_vector=offset_vector,
#     design_matrix=design_matrix,
#     input_matrix=input_matrix,
#     input_parameters=list(max_iter=max_iter, tolerance=tolerance, parallel.cores=n.cores)
#   )
#   )
# }
