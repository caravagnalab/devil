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
#' @param profiling Logical. If \code{TRUE}, prints timing information for major
#'   steps (size factors, offset computation, initial dispersion, beta fit, theta fit).
#'   Useful for performance profiling. Default: \code{FALSE}.
#' @param TEST Logical. If \code{TRUE}, enables GPU validation mode. When CUDA is enabled,
#'   this will compute ground truth results on CPU and compare with GPU results, reporting
#'   correlations and differences for init_beta, k (dispersion), theta (overdispersion),
#'   and final beta coefficients. Default: \code{FALSE}.
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
    profiling = FALSE,
    TEST = FALSE) {

  # - Input parameters ----
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
              "Reinstall with configure.args='--with-cuda' to enable GPU acceleration: ",
              "devtools::install_github(\"caravagnalab/devil\", force=TRUE, configure.args=\"--with-cuda\"). ",
              "Falling back to CPU computation.")
      CUDA <- FALSE
      CUDA_is_available <- FALSE

    } else {
      message("CUDA support detected - using GPU acceleration")
      CUDA_is_available <- TRUE
    }
  }

  # - CPU and GPU common part (i.e. size_factors and offset_vectors) ----
  ## - Compute size factors ----
  if (profiling) start = Sys.time()
  if (!is.null(size_factors)) {
    if (verbose) { message("Compute size factors") }
    sf <- calculate_sf(input_matrix, method = size_factors, verbose = verbose)
  } else {
    sf <- rep(1, nrow(design_matrix))
  }
  if (profiling) {
    end = Sys.time()
    message(paste0("[TIMING] Size factors computing : ", end - start))
  }

  ## - Compute offset vector ----
  if (profiling) start = Sys.time()
  offset_vector = compute_offset_vector(offset, input_matrix, sf)
  if (profiling) {
    end = Sys.time()
    message(paste0("[TIMING] Offset vector computing : ", end - start))
  }

  # - Start GPU vs CPU branch ----
  if (profiling) start = Sys.time()
  if (CUDA & CUDA_is_available) {
    ## - GPU branch ----

    # If TEST mode, compute CPU ground truth for validation
    cpu_ground_truth <- NULL
    if (TEST) {
      message("\n=== TEST MODE ENABLED: Computing CPU ground truth ===")
      cpu_ground_truth <- cpu_fit(
        input_matrix = input_matrix,
        design_matrix = design_matrix,
        offset_vector = offset_vector,
        init_overdispersion = init_overdispersion,
        init_beta_rough = init_beta_rough,
        overdispersion = "MOM",  # GPU uses MOM
        n.cores = n.cores,
        max_iter = max_iter,
        tolerance = tolerance,
        verbose = verbose
      )
      message("CPU ground truth computed.\n")
    }

    remainder = ngenes %% batch_size
    extra_genes = remainder
    genes_batch = ngenes - extra_genes

    message("Fit beta, using CUDA acceleration")
    start_time <- Sys.time()
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

    end_time <- Sys.time()
    message("[TIMING] Beta fit computing (GPU):")
    message(as.numeric(difftime(end_time, start_time, units = "secs")))

    # Extract beta and theta from GPU results
    beta <- res_beta_fit$mu_beta
    theta <- res_beta_fit$theta

    # Extract TEST-specific outputs if available
    gpu_k <- NULL
    gpu_beta_init <- NULL
    if (TEST && !is.null(res_beta_fit$k)) {
      gpu_k <- res_beta_fit$k
      gpu_beta_init <- res_beta_fit$beta_init
    }

    if (remainder > 0) {
      beta_extra <- res_beta_fit_extra$mu_beta
      theta_extra <- res_beta_fit_extra$theta
      beta <- rbind(beta, beta_extra)
      theta <- c(theta, theta_extra)
      beta_iters <- c(res_beta_fit$iter, res_beta_fit_extra$iter)

      if (TEST && !is.null(res_beta_fit_extra$k)) {
        gpu_k <- c(gpu_k, res_beta_fit_extra$k)
        gpu_beta_init <- rbind(gpu_beta_init, res_beta_fit_extra$beta_init)
      }
    } else {
      beta_iters <- c(res_beta_fit$iter)
    }

    if (is.null(dim(beta))) {
      beta <- matrix(beta, ncol = 1)
    }

    rownames(beta) <- gene_names

    # === GPU VALIDATION AGAINST CPU GROUND TRUTH ===
    if (TEST && !is.null(cpu_ground_truth)) {
      message("\n===============================================")
      message("=== GPU vs CPU VALIDATION RESULTS ===")
      message("===============================================\n")

      # 1. Validate init_beta (rough initialization)
      if (!is.null(gpu_beta_init)) {
        # CPU equivalent: rough init
        cpu_beta_init <- matrix(0, nrow = nrow(input_matrix), ncol = ncol(design_matrix))
        cpu_beta_init[, 1] <- log1p(rowMeans(input_matrix))

        cor_beta_init <- stats::cor(as.vector(gpu_beta_init[, 1]), as.vector(cpu_beta_init[, 1]))
        mae_beta_init <- mean(abs(gpu_beta_init[, 1] - cpu_beta_init[, 1]))
        rmse_beta_init <- sqrt(mean((gpu_beta_init[, 1] - cpu_beta_init[, 1])^2))

        message("1. BETA INITIALIZATION (rough)")
        message(sprintf("   Correlation: %.6f", cor_beta_init))
        message(sprintf("   MAE:         %.6e", mae_beta_init))
        message(sprintf("   RMSE:        %.6e\n", rmse_beta_init))
      }

      # 2. Validate k (1/dispersion_init)
      if (!is.null(gpu_k)) {
        cpu_dispersion_init <- estimate_dispersion(input_matrix, offset_vector)
        cpu_k <- 1.0 / cpu_dispersion_init

        cor_k <- stats::cor(gpu_k, cpu_k)
        mae_k <- mean(abs(gpu_k - cpu_k))
        rmse_k <- sqrt(mean((gpu_k - cpu_k)^2))
        rel_error_k <- mean(abs(gpu_k - cpu_k) / (abs(cpu_k) + 1e-10))

        message("2. DISPERSION (k = 1/theta_init)")
        message(sprintf("   Correlation:     %.6f", cor_k))
        message(sprintf("   MAE:             %.6e", mae_k))
        message(sprintf("   RMSE:            %.6e", rmse_k))
        message(sprintf("   Relative Error:  %.6f\n", rel_error_k))
      }

      # 3. Validate theta (MOM overdispersion)
      cor_theta <- stats::cor(theta, cpu_ground_truth$theta)
      mae_theta <- mean(abs(theta - cpu_ground_truth$theta))
      rmse_theta <- sqrt(mean((theta - cpu_ground_truth$theta)^2))
      rel_error_theta <- mean(abs(theta - cpu_ground_truth$theta) / (abs(cpu_ground_truth$theta) + 1e-10))

      message("3. OVERDISPERSION (MOM theta)")
      message(sprintf("   Correlation:     %.6f", cor_theta))
      message(sprintf("   MAE:             %.6e", mae_theta))
      message(sprintf("   RMSE:            %.6e", rmse_theta))
      message(sprintf("   Relative Error:  %.6f\n", rel_error_theta))

      # 4. Validate final beta coefficients
      cor_beta_all <- stats::cor(as.vector(beta), as.vector(cpu_ground_truth$beta))
      mae_beta <- mean(abs(beta - cpu_ground_truth$beta))
      rmse_beta <- sqrt(mean((beta - cpu_ground_truth$beta)^2))

      # Per-coefficient correlation
      cor_beta_per_coef <- numeric(ncol(beta))
      for (j in 1:ncol(beta)) {
        cor_beta_per_coef[j] <- stats::cor(beta[, j], cpu_ground_truth$beta[, j])
      }

      message("4. FINAL BETA COEFFICIENTS")
      message(sprintf("   Overall Correlation: %.6f", cor_beta_all))
      message(sprintf("   MAE:                 %.6e", mae_beta))
      message(sprintf("   RMSE:                %.6e", rmse_beta))
      if (ncol(beta) > 1) {
        message("   Per-coefficient correlations:")
        for (j in 1:ncol(beta)) {
          message(sprintf("      Beta[,%d]: %.6f", j, cor_beta_per_coef[j]))
        }
      }
      message("")

      message("===============================================")
      message("=== END VALIDATION ===")
      message("===============================================\n")
    }

    # Create fit_res structure to match CPU branch
    fit_res <- list(
      beta = beta,
      theta = theta,
      iterations = list(
        beta_iters = beta_iters,
        theta_iters = 0L  # GPU uses MOM, no iterative fitting
      )
    )

    # Add validation results if TEST mode
    if (TEST && !is.null(cpu_ground_truth)) {
      fit_res$test_results <- list(
        gpu_k = gpu_k,
        gpu_beta_init = gpu_beta_init,
        cpu_ground_truth = cpu_ground_truth
      )
    }

  } else {
    ## - CPU branch ----
    fit_res <- cpu_fit(input_matrix = input_matrix,
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
    message(paste0("[TIMING] Beta fit computing (CPU) : ", end - start))
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


