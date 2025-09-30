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
  const int n = y.size();
  const int p = design_matrix.cols();
  MatrixXd H = MatrixXd::Zero(p, p);

  for (int sample_idx = 0; sample_idx < n; ++sample_idx) {
    double yi = y(sample_idx);
    VectorXd design_v = design_matrix.row(sample_idx);

    double eta = design_v.dot(beta);
    double k = size_factors(sample_idx) * std::exp(eta);
    double denom = 1.0 + alpha * k;
    double gamma_sq = denom * denom;

    double scalar = (yi * alpha + 1.0) * k / gamma_sq;
    H.noalias() -= scalar * (design_v * design_v.transpose());
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
Eigen::MatrixXd compute_scores(const Eigen::MatrixXd& design_matrix,
                               const Eigen::VectorXd& y,
                               const Eigen::VectorXd& beta,
                               const double overdispersion,
                               const Eigen::VectorXd& size_factors) {
    double alpha = 1.0 / overdispersion;

    // Vectorized computation
    VectorXd eta = design_matrix * beta;
    VectorXd mu = size_factors.array() * eta.array().exp();
    VectorXd residuals = (y.array() - mu.array()) / mu.array();
    VectorXd weights = mu.array() / (1.0 + mu.array() / alpha);
    VectorXd wr = residuals.array() * weights.array();

    return design_matrix.array().colwise() * wr.array();
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

    Eigen::MatrixXd ef = compute_scores(design_matrix, y, beta, overdispersion, size_factors);
    int k = design_matrix.cols();
    int n = design_matrix.rows();
    int ng = clusters.maxCoeff();
    double adj = (ng > 1) ? double(ng) / (ng - 1.0) : 1.0;

    Eigen::MatrixXd rval = Eigen::MatrixXd::Zero(k, k);

    for (int j = 1; j <= ng; ++j) {
      // Sum rows directly without creating mask array
      Eigen::VectorXd ef_sum = Eigen::VectorXd::Zero(k);
      for (int i = 0; i < n; ++i) {
        if (clusters(i) == j) {
          ef_sum += ef.row(i).transpose();
        }
      }
      rval.noalias() += ef_sum * ef_sum.transpose();
    }

    return (adj / n) * rval;
}
