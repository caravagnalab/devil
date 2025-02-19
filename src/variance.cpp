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
Eigen::MatrixXd compute_hessian(Eigen::VectorXd beta, const double overdispersion, Eigen::VectorXd y, Eigen::MatrixXd design_matrix, Eigen::VectorXd size_factors) {
  const double alpha = 1 / overdispersion;
  const int n = y.size();
  const int p = design_matrix.cols();
  MatrixXd H = MatrixXd::Zero(p, p);
  double k, gamma_sq;

  for (int sample_idx = 0; sample_idx < n; ++sample_idx) {
    double yi = y(sample_idx);
    VectorXd design_v = design_matrix.row(sample_idx);

    MatrixXd xij = design_v * design_v.transpose();
    k = size_factors(sample_idx) * std::exp(design_v.dot(beta));
    gamma_sq = std::pow(1 + alpha * k, 2);

    MatrixXd new_term = -((yi * alpha + 1) * xij * k / gamma_sq);
    H += new_term;
  }

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
Eigen::MatrixXd compute_scores(Eigen::MatrixXd& design_matrix, Eigen::VectorXd& y, Eigen::VectorXd& beta, double overdispersion, Eigen::VectorXd& size_factors) {
  MatrixXd xmat = design_matrix;
  double alpha = 1.0 / overdispersion;

  VectorXd mu = size_factors.array() * (xmat * beta).array().exp();
  VectorXd residuals = (y - mu).array() / mu.array();
  VectorXd weights = mu.array().square() / (mu.array() + mu.array().square() / alpha);
  VectorXd wr = residuals.array() * weights.array();

  return xmat.array().colwise() * wr.array();
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
Eigen::MatrixXd compute_clustered_meat(Eigen::MatrixXd design_matrix, Eigen::VectorXd y, Eigen::VectorXd beta, double overdispersion, Eigen::VectorXd size_factors, Eigen::VectorXi clusters) {

    Eigen::MatrixXd ef = compute_scores(design_matrix, y, beta, overdispersion, size_factors);
    int k = design_matrix.cols();
    int n = design_matrix.rows();
    int ng = clusters.maxCoeff();
    double adj = double(ng) / (ng - 1);
    if (ng == 1) {adj = 1;}

    Eigen::MatrixXd rval = Eigen::MatrixXd::Zero(k, k);

    for (int j = 0; j < ng; ++j) {
      Eigen::VectorXi mask = (clusters.array() == (j + 1)).cast<int>();
      Eigen::MatrixXd ef_j = (ef.array().colwise() * mask.cast<double>().array()).colwise().sum();
      rval += adj * ef_j.transpose() * ef_j;
    }

    return rval / n;
}
