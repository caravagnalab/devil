#include <RcppEigen.h>
#include <cmath>

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace Eigen;

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
