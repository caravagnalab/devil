#include <RcppEigen.h>
#include <iostream>
#include <vector>
#include <map>
// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
using namespace Eigen;

// [[Rcpp::export]]
std::vector<Eigen::MatrixXd> precompute_outer(Eigen::MatrixXd X_unique) {
  // 1. Precompute K outer products x_k * x_k^T — once for all genes
  int K = X_unique.rows();
  int p = X_unique.cols();
  std::vector<Eigen::MatrixXd> outer(K);
  for (int k = 0; k < K; k++) {
    outer[k] = X_unique.row(k).transpose() * X_unique.row(k);  // p×p
  }
  
  return(outer);
}

// Assuming 'groups' maps each cell to a unique row index 0...U-1
// and 'X_unique' is a U x M matrix of those patterns.
// [[Rcpp::export]]
List beta_fit_fast(
    Eigen::VectorXd y,
    Eigen::MatrixXd X_unique, 
    Eigen::VectorXi idx,      
    Eigen::VectorXd off,
    Rcpp::List outer_list,
    Eigen::VectorXd mu_beta,
    float k, 
    int max_iter, 
    float eps) {
  
  int N = y.size();
  int K = X_unique.rows();
  int p = X_unique.cols();
  k = 1.0 / k;
  
  // Convert List to a usable C++ structure
  std::vector<Eigen::MatrixXd> outer(K);
  for(int i = 0; i < K; i++) {
    outer[i] = Rcpp::as<Eigen::MatrixXd>(outer_list[i]);
  }
  
  // Per-group quantities (length K)
  Eigen::VectorXd exp_neg_off = (-off).array().exp();
  Eigen::VectorXd exp_neg_Xb(K);   // exp(-x_k · mu_beta)
  Eigen::VectorXd group_wsum(K);   // Σ_{i∈k} w_i  (scalar weight sum)
  Eigen::VectorXd group_resid(K);  // Σ_{i∈k} (mu_g_i * w_i - 1)
  
  Eigen::VectorXd delta = Eigen::VectorXd::Zero(p);
  Eigen::MatrixXd Zigma = Eigen::MatrixXd::Identity(p, p);
  
  bool converged = false;
  int iter = 0;
  
  while (!converged && iter < max_iter) {
    
    // --- Step 1: group-level exp(-x_k · mu_beta), O(K·p) ---
    exp_neg_Xb = (-X_unique * mu_beta).array().exp();
    
    // --- Step 2: scatter over cells, accumulate group sums, O(N) ---
    group_wsum.setZero();
    group_resid.setZero();
    
    for (int i = 0; i < N; i++) {
      int k_i = idx[i];
      double w_i   = exp_neg_Xb[k_i] * exp_neg_off[i];
      double mu_g_i = (k + y[i]) / (1.0 + k * w_i);
      double mw_i  = mu_g_i * w_i;         // shared in both sums
      group_wsum[k_i]  += mw_i;
      group_resid[k_i] += mw_i - 1.0;
    }
    
    // --- Step 3: Gram matrix assembly, O(K·p²) ---
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(p, p);
    for (int kk = 0; kk < K; kk++) {
      G += group_wsum[kk] * outer[kk];
    }
    
    // --- Step 4: gradient assembly, O(K·p) ---
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(p);
    for (int kk = 0; kk < K; kk++) {
      grad += group_resid[kk] * X_unique.row(kk).transpose();
    }
    
    // --- Step 5: Newton step ---
    Zigma = (k * G).inverse();
    delta = Zigma * (k * grad);
    mu_beta += delta;
    
    converged = delta.cwiseAbs().maxCoeff() < eps;
    if (delta[0] != delta[0]) converged = true;
    iter++;
  }
  
  return List::create(Named("mu_beta") = mu_beta, Named("iter") = iter);
}

// [[Rcpp::export]]
List beta_fit_fast_optimized(
    const Eigen::VectorXd& y,
    const Eigen::MatrixXd& X_unique,
    const Eigen::VectorXi& idx,
    const Eigen::VectorXd& off,
    const Rcpp::List& outer_list, // Pass as List
    Eigen::VectorXd mu_beta,
    double k, 
    int max_iter, 
    double eps) {
  
  int N = y.size();
  int K = X_unique.rows();
  int p = X_unique.cols();
  double k_inv = 1.0 / k;
  
  // 1. Pre-convert List to a C++ vector
  std::vector<Eigen::MatrixXd> outer(K);
  for(int i = 0; i < K; i++) {
    outer[i] = Rcpp::as<Eigen::MatrixXd>(outer_list[i]);
  }
  
  // 2. Pre-allocate work variables to avoid re-allocation in the loop
  Eigen::VectorXd exp_neg_off = (-off).array().exp();
  Eigen::VectorXd exp_neg_Xb(K);
  Eigen::VectorXd group_wsum(K);
  Eigen::VectorXd group_resid(K);
  Eigen::MatrixXd G(p, p);
  Eigen::VectorXd grad(p);
  Eigen::VectorXd delta(p);
  
  // Pre-cache X_unique transpose for Step 4
  Eigen::MatrixXd XT = X_unique.transpose();
  
  int iter = 0;
  bool converged = false;
  
  while (!converged && iter < max_iter) {
    exp_neg_Xb = (-X_unique * mu_beta).array().exp();
    
    group_wsum.setZero();
    group_resid.setZero();
    
    for (int i = 0; i < N; i++) {
      int k_i = idx[i];
      double w_i   = exp_neg_Xb[k_i] * exp_neg_off[i];
      double mw_i  = ((k_inv + y[i]) / (1.0 + k_inv * w_i)) * w_i;
      
      group_wsum[k_i]  += mw_i;
      group_resid[k_i] += (mw_i - 1.0);
    }
    
    G.setZero();
    for (int kk = 0; kk < K; kk++) {
      G.noalias() += group_wsum[kk] * outer[kk];
    }
    
    grad.noalias() = XT * group_resid;
    delta = (k_inv * G).ldlt().solve(k_inv * grad);
    
    mu_beta += delta;
    
    converged = delta.cwiseAbs().maxCoeff() < eps;
    if (std::isnan(delta[0])) break; 
    iter++;
  }
  
  return List::create(Named("mu_beta") = mu_beta, Named("iter") = iter);
}

// [[Rcpp::export]]
List compress_design_matrix(const Eigen::MatrixXd& X) {
  int N = X.rows();
  int M = X.cols();
  
  std::map<std::vector<double>, int> row_to_id;
  std::vector<int> groups(N);
  std::vector<std::vector<double>> unique_rows_vec;
  
  int current_id = 0;
  
  for (int i = 0; i < N; ++i) {
    // Safe copy: Eigen is column-major so row data is not contiguous
    std::vector<double> row_vec(M);
    for (int j = 0; j < M; ++j) {
      row_vec[j] = X(i, j);
    }
    
    auto it = row_to_id.find(row_vec);
    if (it == row_to_id.end()) {
      row_to_id[row_vec] = current_id;
      unique_rows_vec.push_back(row_vec);
      groups[i] = current_id;
      current_id++;
    } else {
      groups[i] = it->second;  // reuse iterator, avoids second map lookup
    }
  }
  
  // Convert unique rows back to Eigen matrix
  int U = unique_rows_vec.size();
  Eigen::MatrixXd X_unique(U, M);
  for (int i = 0; i < U; ++i) {
    for (int j = 0; j < M; ++j) {
      X_unique(i, j) = unique_rows_vec[i][j];
    }
  }
  
  return List::create(Named("X_unique") = X_unique, Named("groups") = groups);
}