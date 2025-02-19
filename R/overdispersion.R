
#' Fit Dispersion Parameter for Negative Binomial Model
#'
#' @description Estimates the dispersion parameter in a negative binomial GLM using
#'   maximum likelihood estimation. Implementation from the glmGamPoi package.
#'
#' @details This implementation comes from the glmGamPoi package:
#'   https://github.com/const-ae/glmGamPoi
#'
#'   Original publication:
#'   Ahlmann-Eltze, C., Huber, W. (2020). glmGamPoi: Fitting Gamma-Poisson
#'   Generalized Linear Models on Single Cell Count Data. Bioinformatics.
#'   https://doi.org/10.1093/bioinformatics/btaa1009
#'
#' @param beta Vector of coefficient estimates
#' @param model_matrix Design matrix of predictor variables
#' @param y Vector of response variables (counts)
#' @param offset_matrix Matrix of offset values
#' @param tolerance Convergence tolerance for optimization
#' @param max_iter Maximum number of iterations for optimization
#' @param do_cox_reid_adjustment Logical indicating whether to apply Cox-Reid adjustment
#'
#' @return Estimated dispersion parameter
#'
#' @keywords internal
fit_dispersion <- function(beta, model_matrix, y, offset_matrix, tolerance, max_iter, do_cox_reid_adjustment=TRUE) {

  tab <- make_table_if_small(y, stop_if_larger = length(y)/2)

  if(all(y == 0)){ return(0) }

  mean_vector <- exp(model_matrix %*% beta + offset_matrix)
  mean_vector[mean_vector == 0] <- 1e-6
  mu <- mean(y)
  start_value <- (stats::var(y) - mu) / mu^2
  if(is.na(start_value) || start_value <= 0){
    start_value <- 0.5
  }

  far_left_value <- conventional_score_function_fast(y, mu = mean_vector, log_theta = log(1e-8),
                                                     model_matrix = model_matrix, do_cr_adj = do_cox_reid_adjustment, tab[[1]], tab[[2]])
  if(far_left_value < 0) { return(0) }

  nlminb_control_list <- list(
    iter.max = max_iter,
    rel.tol = tolerance
  )

  res <- stats::nlminb(start = log(start_value),
                       objective = function(log_theta){
                         - conventional_loglikelihood_fast(y, mu = mean_vector, log_theta = log_theta,
                                                           model_matrix = model_matrix, do_cr_adj = do_cox_reid_adjustment, tab[[1]], tab[[2]])
                       }, gradient = function(log_theta){
                         - conventional_score_function_fast(y, mu = mean_vector, log_theta = log_theta,
                                                            model_matrix = model_matrix, do_cr_adj = do_cox_reid_adjustment, tab[[1]], tab[[2]])
                       }, hessian = function(log_theta){
                         res <- conventional_deriv_score_function_fast(y, mu = mean_vector, log_theta = log_theta,
                                                                       model_matrix = model_matrix, do_cr_adj = do_cox_reid_adjustment, tab[[1]], tab[[2]])
                         matrix(- res, nrow = 1, ncol = 1)
                       }, lower = log(1e-16), upper = log(1e16),
                       control = nlminb_control_list)

  if(res$convergence != 0){
    # Do the same thing again with numerical hessian as the analytical hessian
    # is sometimes less robust than the other two functions
    res <- stats::nlminb(start = log(start_value),
                  objective = function(log_theta){
                    - conventional_loglikelihood_fast(y, mu = mean_vector, log_theta = log_theta,
                                                      model_matrix = model_matrix, do_cr_adj = do_cox_reid_adjustment, tab[[1]], tab[[2]])
                  }, gradient = function(log_theta){
                    - conventional_score_function_fast(y, mu = mean_vector, log_theta = log_theta,
                                                       model_matrix = model_matrix, do_cr_adj = do_cox_reid_adjustment, tab[[1]], tab[[2]])
                  }, lower = log(1e-16), upper = log(1e16),
                  control = nlminb_control_list)

    if(res$convergence != 0){
      # Still problematic result: do the same thing without Cox-Reid adjustment
      res <- stats::nlminb(start = log(start_value),
                    objective = function(log_theta){
                      - conventional_loglikelihood_fast(y, mu = mean_vector, log_theta = log_theta,
                                                        model_matrix = model_matrix, do_cr_adj = FALSE, tab[[1]], tab[[2]])
                    }, gradient = function(log_theta){
                      - conventional_score_function_fast(y, mu = mean_vector, log_theta = log_theta,
                                                         model_matrix = model_matrix, do_cr_adj = FALSE, tab[[1]], tab[[2]])
                    }, lower = log(1e-16), upper = log(1e16),
                    control = nlminb_control_list)
    }
  }

  return(exp(res$par))
}

#' Estimate Dispersion Parameters for Count Matrix
#'
#' @description Calculates per-gene dispersion estimates for a count matrix using
#'   a method of moments approach. Handles edge cases by setting a default high
#'   dispersion value.
#'
#' @param count_matrix Matrix of count data with genes in rows and samples in columns
#' @param offset_vector Vector of offset values for normalization
#'
#' @return Vector of dispersion estimates, one per gene
#'
#' @keywords internal
estimate_dispersion <- function(count_matrix, offset_vector) {
  mean_offset_inverse <- 1 / mean(exp(offset_vector))
  variance <- DelayedMatrixStats::rowVars(count_matrix, useNames=TRUE)
  mean_counts <- DelayedMatrixStats::rowMeans2(count_matrix, useNames=TRUE)
  dispersion <- (variance - mean_offset_inverse * mean_counts) / mean_counts^2
  ifelse(is.na(dispersion) | dispersion < 0, 100, dispersion)
}
