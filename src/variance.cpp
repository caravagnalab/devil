#include <RcppEigen.h>
#include <cmath>
// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
using namespace Eigen;

/**
 * Computes the Hessian matrix for a generalized linear model with overdispersion
 *
 * This function calculates the inverse of the negative Hessian matrix of the log-likelihood
 * with respect to the regression coefficients (beta). It accounts for overdispersion
 * in the model using a quasi-likelihood approach.
 *
 * @param beta Vector of regression coefficients
 * @param overdispersion Overdispersion parameter (phi)
 * @param y Vector of response values
 * @param design_matrix Matrix of predictor variables
 * @param size_factors Vector of normalization factors
 * @return Inverse of the negative Hessian matrix
 */
// [[Rcpp::export]]
Eigen::MatrixXd compute_hessian(const Eigen::VectorXd& beta,
                                    const double overdispersion,
                                    const Eigen::VectorXd& y,
                                    const Eigen::MatrixXd& design_matrix,
                                    const Eigen::VectorXd& size_factors) {
  const double alpha = 1.0 / overdispersion;

  // n x 1
  Eigen::VectorXd eta = design_matrix * beta;

  // mu = sf * exp(eta)
  Eigen::ArrayXd mu = (size_factors.array() * eta.array().exp());

  // denom = 1 + alpha * mu
  Eigen::ArrayXd denom = 1.0 + alpha * mu;
  Eigen::ArrayXd gamma_sq = denom * denom;

  // scalar s_i = (y_i * alpha + 1) * mu_i / gamma_sq_i
  Eigen::ArrayXd scalar = (y.array() * alpha + 1.0).array() * mu / gamma_sq;

  // H = - X^T diag(scalar) X
  Eigen::MatrixXd H = -(design_matrix.transpose() * scalar.matrix().asDiagonal() * design_matrix);

  // Return -H^{-1}, i.e. (X^T W X)^{-1}
  return -H.inverse();
}

/**
 * Computes score residuals for a generalized linear model
 *
 * Calculates the score residuals (derivative of the log-likelihood with respect
 * to the regression coefficients) for each observation, accounting for overdispersion
 * and size factors.
 *
 * @param design_matrix Matrix of predictor variables
 * @param y Vector of response values
 * @param beta Vector of regression coefficients
 * @param overdispersion Overdispersion parameter (phi)
 * @param size_factors Vector of normalization factors
 * @return Matrix of score residuals
 */
// [[Rcpp::export]]
Eigen::MatrixXd compute_scores(const Eigen::MatrixXd& design_matrix,
                                   const Eigen::VectorXd& y,
                                   const Eigen::VectorXd& beta,
                                   const double overdispersion,
                                   const Eigen::VectorXd& size_factors) {
  const double alpha = 1.0 / overdispersion;

  Eigen::VectorXd eta = design_matrix * beta;
  Eigen::ArrayXd mu   = size_factors.array() * eta.array().exp();

  // residuals = (y - mu) / mu
  Eigen::ArrayXd residuals = (y.array() - mu) / mu;

  // weights = mu / (1 + mu / alpha)
  Eigen::ArrayXd weights = mu / (1.0 + mu / alpha);

  // wr = residuals * weights
  Eigen::ArrayXd wr = residuals * weights;

  // broadcast wr over columns: each row i of X is multiplied by wr_i
  return design_matrix.array().colwise() * wr;
}


/**
 * Computes clustered "meat" matrix for sandwich variance estimator
 *
 * Calculates the middle matrix of the sandwich variance estimator, accounting for
 * clustering in the data. This implementation uses the cluster-robust variance
 * estimation method with a finite sample adjustment.
 *
 * @param design_matrix Matrix of predictor variables
 * @param y Vector of response values
 * @param beta Vector of regression coefficients
 * @param overdispersion Overdispersion parameter (phi)
 * @param size_factors Vector of normalization factors
 * @param clusters Vector of cluster assignments
 * @return Clustered "meat" matrix for sandwich variance estimation
 */
// [[Rcpp::export]]
Eigen::MatrixXd compute_clustered_meat(const Eigen::MatrixXd& design_matrix,
                                           const Eigen::VectorXd& y,
                                           const Eigen::VectorXd& beta,
                                           const double overdispersion,
                                           const Eigen::VectorXd& size_factors,
                                           const Eigen::VectorXi& clusters) {
  const int n  = design_matrix.rows();
  const int k  = design_matrix.cols();

  // ef: n x k score contributions
  Eigen::MatrixXd ef = compute_scores(design_matrix, y, beta,
                                          overdispersion, size_factors);

  // number of clusters (assumes clusters are 1..ng)
  const int ng = clusters.maxCoeff();

  // finite-sample correction factor
  const double adj = (ng > 1) ? double(ng) / (ng - 1.0) : 1.0;

  // cluster_sums[g, :] = sum of scores in cluster g+1
  Eigen::MatrixXd cluster_sums = Eigen::MatrixXd::Zero(ng, k);

  for (int i = 0; i < n; ++i) {
    int g = clusters(i) - 1; // convert 1-based -> 0-based index
    if (g >= 0 && g < ng) {
      cluster_sums.row(g) += ef.row(i);
    }
  }

  // rval = sum_g s_g s_g^T = cluster_sums^T cluster_sums
  Eigen::MatrixXd rval = cluster_sums.transpose() * cluster_sums;

  return (adj / n) * rval;
}



/**
 * Computes unclustered "meat" matrix for sandwich variance estimator
 *
 * Calculates the middle matrix of the sandwich variance estimator without
 * accounting for clustering. This implementation forms the conventional
 * (non-robust-to-clustering) meat by summing outer products of the
 * per-observation score vectors and applying 1/n normalization.
 *
 * @param design_matrix Matrix of predictor variables (n x k)
 * @param y             Vector of response values (length n)
 * @param beta          Vector of regression coefficients (length k)
 * @param overdispersion Overdispersion parameter (phi)
 * @param size_factors  Vector of normalization/offset factors (length n)
 * @return Unclustered "meat" matrix (k x k) for sandwich variance estimation
 */
// [[Rcpp::export]]
Eigen::MatrixXd compute_meat(const Eigen::MatrixXd& design_matrix,
                             const Eigen::VectorXd& y,
                             const Eigen::VectorXd& beta,
                             const double overdispersion,
                             const Eigen::VectorXd& size_factors) {
  // n x k matrix of scores (each row is a per-observation score vector)
  Eigen::MatrixXd ef = compute_scores(design_matrix, y, beta, overdispersion, size_factors);

  int n = design_matrix.rows();
  int k = design_matrix.cols();

  Eigen::MatrixXd rval = Eigen::MatrixXd::Zero(k, k);

  // Sum outer products of each observation's score
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd si = ef.row(i).transpose();  // k x 1
    rval.noalias() += si * si.transpose();       // accumulate s_i s_i'
  }

  // Scale by 1/n to match asymptotic normalization
  return (1.0 / n) * rval;
}
