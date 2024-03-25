
fit_dispersion <- function(beta, model_matrix, y, offset_matrix, do_cox_reid_adjustment=TRUE) {


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
                       control = list(iter.max = 200))

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
                  control = list(iter.max = 200))

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
                    control = list(iter.max = 200))
    }
  }

  return(exp(res$par))
}
