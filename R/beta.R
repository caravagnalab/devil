
#' Initialize Beta Coefficients Using Design Matrix
#'
#' @description Initializes regression coefficients using QR decomposition
#'   of the design matrix and normalized log counts.
#'
#' @param y Count data matrix
#' @param design_matrix Matrix of predictor variables
#' @param offset_matrix Matrix of offset values
#'
#' @return Matrix of initial beta coefficients
#'
#' @keywords internal
init_beta <- function(y, design_matrix, offset_matrix) {
  qrx <- qr(design_matrix)
  Q <- qr.Q(qrx)[seq_len(nrow(design_matrix)),,drop=FALSE]
  R <- qr.R(qrx)

  #norm_log_count_mat <- t(log1p((y / exp(offset_matrix))))
  norm_log_count_mat <- log1p((t(y) / exp(offset_matrix)))
  #norm_log_count_mat <- t(log1p((y / exp(offset_matrix))))
  t(solve(R, as.matrix(t(Q) %*% norm_log_count_mat)))
}

#' Initialize Beta Coefficients Using Groups
#'
#' @description Initializes regression coefficients based on group-wise
#'   means of normalized counts.
#'
#' @param y Count data matrix
#' @param groups Vector indicating group membership
#' @param offset_matrix Matrix of offset values
#'
#' @return Matrix of initial beta coefficients by group
#'
#' @keywords internal
init_beta_groups <- function(y, groups, offset_matrix) {
  #norm_Y <- y / exp(offset_matrix)
  norm_Y <- y / exp(offset_matrix[1,])
  do.call(cbind, lapply(unique(groups), function(gr){
    log(DelayedMatrixStats::rowMeans2(norm_Y, cols = (groups == gr), useNames=TRUE))
  }))
}
