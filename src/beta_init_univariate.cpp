#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

/**
 * Univariate Poisson log-link initializer (matrix version)
 */
// [[Rcpp::export]]
Rcpp::NumericMatrix initialize_beta_univariate_matrix_cpp(const Rcpp::NumericMatrix &X,
                                                         const Rcpp::NumericMatrix &Y,
                                                         const Rcpp::NumericVector &sf) {
 const int n = X.nrow();
 const int p = X.ncol();
 const int G = Y.nrow();

 if (Y.ncol() != n) {
   Rcpp::stop("ncol(Y) must match nrow(X)");
 }
 if (sf.size() != n) {
   Rcpp::stop("length(sf) must match nrow(X)");
 }

 // Map R data to Eigen
 Map<const MatrixXd> X_e(X.begin(), n, p);  // n x p
 Map<const MatrixXd> Y_e(Y.begin(), G, n);  // G x n
 Map<const VectorXd> sf_e(sf.begin(), n);   // n

 // Transpose Y for convenience: n x G
 MatrixXd Yt = Y_e.transpose();            // n x G

 // Precompute log_rate = log((Y + 0.5) / sf) for all genes
 MatrixXd log_rate(n, G);
 for (int i = 0; i < n; ++i) {
   const double log_sf_i = std::log(sf_e[i]);
   for (int g = 0; g < G; ++g) {
     log_rate(i, g) = std::log(Yt(i, g) + 0.5) - log_sf_i;
   }
 }

 // Output beta: G x p
 MatrixXd beta(G, p);
 beta.setZero();

 // Detect intercept in first column: all ~ 1
 bool is_intercept_first = false;
 const double tol = 1e-10;
 if (p > 0) {
   VectorXd diff = X_e.col(0).array() - 1.0;
   is_intercept_first = (diff.cwiseAbs().maxCoeff() < tol);
 }

 // Intercept: beta_0g = log(sum_i Y_gi / sum_i sf)
 const double sum_sf = sf_e.sum();
 const double eps_int = 1e-8;  // guard against log(0)
 if (is_intercept_first) {
   VectorXd y_row_sums = Y_e.rowwise().sum();   // G x 1
   VectorXd beta0 =
     ((y_row_sums.array() + eps_int) / (sum_sf + eps_int)).log();
   beta.col(0) = beta0;
 }

 // Precompute column means of X (for centering)
 VectorXd x_mean = X_e.colwise().mean();        // p

 // For each covariate, fit univariate regression per gene
 for (int j = 0; j < p; ++j) {
   // If this is the intercept column and we already set it, skip
   if (j == 0 && is_intercept_first) {
     continue;
   }

   // Center covariate j
   VectorXd xj = X_e.col(j);
   VectorXd x_centered = xj.array() - x_mean[j];

   // Denominator: sum_i (x_centered_i^2)
   double denom = x_centered.squaredNorm();
   if (denom <= 0.0) {
     denom = 1e-12; // protect against constant column
   }

   // Numerator per gene: sum_i x_centered_i * log_rate_{ig}
   // x_centered^T (1 x n) * log_rate (n x G) = 1 x G
   RowVectorXd num_row = x_centered.transpose() * log_rate;

   // Beta_jg = num_g / denom, for all genes g
   beta.col(j) = (num_row / denom).transpose();  // G x 1
 }

 return Rcpp::wrap(beta);
}
