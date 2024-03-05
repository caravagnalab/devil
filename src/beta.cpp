#include <RcppEigen.h>
#include <iostream>

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace Eigen;

// [[Rcpp::export]]
Eigen::VectorXd init_beta(Eigen::VectorXd y, Eigen::MatrixXd X) {
  VectorXd norm_log_count_mat = y.array().log1p();
  return X.colPivHouseholderQr().solve(norm_log_count_mat);
}

// [[Rcpp::export]]
List beta_fit(Eigen::VectorXd y, Eigen::MatrixXd X, Eigen::VectorXd mu_beta, Eigen::VectorXd off, int max_iter, float eps) {
  int cols = X.cols();
  int rows = X.rows();
  Eigen::VectorXd delta = Eigen::VectorXd::Zero(cols);
  Eigen::MatrixXd inv_sigma_beta_const = 0.01 * Eigen::MatrixXd::Identity(cols, cols);
  Eigen::MatrixXd Zigma = Eigen::MatrixXd::Identity(cols, cols);
  VectorXd mu_g = VectorXd::Zero(rows);
  VectorXd w_q = VectorXd::Zero(rows);

  bool converged = 0;
  int iter = 0;
  while (!converged && iter < max_iter) {
    w_q = (-X * mu_beta - off).array().exp();
    mu_g = (1 + y.array()) / (1 + w_q.array());
    Zigma = (X.transpose() * (mu_g.array() * w_q.array()).matrix().asDiagonal() * X).inverse();
    delta = Zigma * (X.transpose() * (mu_g.array() * w_q.array() - 1).matrix());
    mu_beta += delta;
    converged = delta.cwiseAbs().maxCoeff() < eps;
    iter++;
  }

  w_q = (-X * mu_beta - off).array().exp();
  mu_g = (1 + y.array()) / (1 + w_q.array());

  Zigma = (X.transpose() * (mu_g.array() * w_q.array()).matrix().asDiagonal() * X).inverse();

  // Return both mu_beta and Zigma as a List
  return List::create(Named("mu_beta") = mu_beta, Named("Zigma") = Zigma, Named("iter") = iter);
}

// Check how many unique rows are in a matrix and if this number is less than or equal to n
// This is important to determine if the model can be solved by group averages
// (ie. the numer of unique rows == number of columns)
// [[Rcpp::export]]
bool lte_n_equal_rows(const NumericMatrix& matrix, int n, double tolerance = 1e-10) {
  NumericMatrix reference_matrix(n, matrix.ncol());
  size_t n_matches = 0;
  for(size_t row_idx = 0; row_idx < matrix.nrow(); row_idx++){
    bool matched = false;
    NumericMatrix::ConstRow vec = matrix(row_idx, _);
    for(size_t ref_idx = 0; ref_idx < n_matches; ref_idx++){
      NumericMatrix::Row ref_vec  = reference_matrix(ref_idx, _);
      if(sum(abs(vec - ref_vec)) < tolerance){
        matched = true;
        break;
      }
    }
    if(! matched){
      ++n_matches;
      if(n_matches > n){
        return false;
      }
      reference_matrix(n_matches - 1, _) = vec;
    }
  }
  return true;
}

// [[Rcpp::export]]
IntegerVector get_row_groups(const NumericMatrix& matrix, int n_groups, double tolerance = 1e-10) {
  NumericMatrix reference_matrix(n_groups, matrix.ncol());
  IntegerVector groups(matrix.nrow());
  size_t n_matches = 0;
  for(size_t row_idx = 0; row_idx < matrix.nrow(); row_idx++){
    bool matched = false;
    NumericMatrix::ConstRow vec = matrix(row_idx, _);
    for(size_t ref_idx = 0; ref_idx < n_matches; ref_idx++){
      NumericMatrix::Row ref_vec  = reference_matrix(ref_idx, _);
      if(sum(abs(vec - ref_vec)) < tolerance){
        groups(row_idx) = ref_idx;
        matched = true;
        break;
      }
    }
    if(! matched){
      groups(row_idx) = n_matches;
      reference_matrix(n_matches, _) = vec;
      ++n_matches;
    }
  }
  return groups + 1;
}
// // [[Rcpp::export]]
// List beta_fit_global(Eigen::MatrixXd Y, Eigen::MatrixXd X, Eigen::VectorXd MU_BETA, int max_iter, float eps) {
//   int nfeatures = X.cols();
//   int ncells = X.rows();
//   int ngenes = Y.rows();
//
//   Eigen::MatrixXd BETA(ngenes, nfeatures);
//   std::vector<MatrixXd> SIGMA(ngenes, MatrixXd(nfeatures, nfeatures));
//
//   for (int i=0; i<ngenes; i++) {
//     List fit_result = beta_fit(Y.row(i), X, MU_BETA.row(i), max_iter, eps);
//
//     BETA.row(i) = as<VectorXd>(fit_result["mu_beta"]);
//     SIGMA[i] = as<MatrixXd>(fit_result["Zigma"]);
//
//   }
//
//   return List::create(Named("Beta") = BETA, Named("Sigma") = SIGMA);
// }
