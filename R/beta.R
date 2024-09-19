
init_beta <- function(y, design_matrix, offset_matrix) {
  qrx <- qr(design_matrix)
  Q <- qr.Q(qrx)[seq_len(nrow(design_matrix)),,drop=FALSE]
  R <- qr.R(qrx)

  norm_log_count_mat <- t(log1p((y / exp(offset_matrix[1,]))))
  #norm_log_count_mat <- t(log1p((y / exp(offset_matrix))))
  t(solve(R, as.matrix(t(Q) %*% norm_log_count_mat)))
}

init_beta_groups <- function(y, groups, offset_matrix) {
  #norm_Y <- y / exp(offset_matrix)
  norm_Y <- y / exp(offset_matrix[1,])
  do.call(cbind, lapply(unique(groups), function(gr){
    log(DelayedMatrixStats::rowMeans2(norm_Y, cols = groups == gr, useNames=TRUE))
  }))
}



