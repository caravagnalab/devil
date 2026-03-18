
#include <RcppEigen.h>
#include <map>
#include <vector>

// [[Rcpp::depends(RcppEigen)]]
using namespace Eigen;
using namespace Rcpp;

// Pre-processing function to collapse the data
// [[Rcpp::export]]
List preprocess_data(Eigen::VectorXd y, Eigen::MatrixXd X, Eigen::VectorXd off) {
  int n = X.rows();
  int p = X.cols();
  
  // Map to store unique (row + offset) -> (sum_y, count)
  // Using vector<double> as key for the map
  std::map<std::pair<std::vector<double>, double>, std::pair<double, int>> groups;
  
  for (int i = 0; i < n; ++i) {
    std::vector<double> row_vec(X.row(i).data(), X.row(i).data() + p);
    auto key = std::make_pair(row_vec, off[i]);
    
    if (groups.find(key) == groups.end()) {
      groups[key] = {y[i], 1};
    } else {
      groups[key].first += y[i];
      groups[key].second += 1;
    }
  }
  
  int m = groups.size();
  MatrixXd X_unique(m, p);
  VectorXd off_unique(m);
  VectorXd y_sums(m);
  VectorXd counts(m);
  
  int idx = 0;
  for (auto const& [key, val] : groups) {
    for (int j = 0; j < p; ++j) X_unique(idx, j) = key.first[j];
    off_unique[idx] = key.second;
    y_sums[idx] = val.first;
    counts[idx] = (double)val.second;
    idx++;
  }
  
  return List::create(
    Named("X") = X_unique,
    Named("off") = off_unique,
    Named("y_sums") = y_sums,
    Named("counts") = counts
  );
}

// [[Rcpp::export]]
List beta_fit_efficient(Eigen::VectorXd y_sums, 
                        Eigen::VectorXd counts, 
                        Eigen::MatrixXd X, 
                        Eigen::VectorXd mu_beta, 
                        Eigen::VectorXd off, float k, 
                        int max_iter, float eps) {
  int cols = X.cols();
  int rows = X.rows(); // This is now M (unique rows)
  float inv_k = 1.0 / k;
  
  VectorXd delta = VectorXd::Zero(cols);
  MatrixXd Zigma = MatrixXd::Identity(cols, cols);
  VectorXd mu_g_sum = VectorXd::Zero(rows);
  VectorXd w_q = VectorXd::Zero(rows);
  VectorXd weight_diag = VectorXd::Zero(rows);
  
  bool converged = false;
  int iter = 0;
  
  while (!converged && iter < max_iter) {
    // 1. Calculate w_q for unique groups
    w_q = (-X * mu_beta - off).array().exp();
    
    // 2. Calculate the collapsed mu_g sum for each group
    // S_mu = (n*k + sum_y) / (1 + k*w_q)
    mu_g_sum = (counts.array() * inv_k + y_sums.array()) / (1 + inv_k * w_q.array());
    
    // 3. Weight vector for the Diagonal (mu_g_sum * w_q)
    weight_diag = mu_g_sum.array() * w_q.array();
    
    // 4. Zigma calculation using the group weights
    // Hessian = k * X' * diag(weight_diag) * X
    Zigma = (inv_k * X.transpose() * weight_diag.asDiagonal() * X).inverse();
    
    // 5. Gradient calculation
    // Grad = k * X' * (weight_diag - counts)
    delta = Zigma * (inv_k * X.transpose() * (weight_diag.array() - counts.array()).matrix());
    
    mu_beta += delta;
    converged = delta.cwiseAbs().maxCoeff() < eps;
    iter++;
    
    if (delta[0] != delta[0]) { // NaN check
      converged = true;
    }
  }
  
  return List::create(
    Named("mu_beta") = mu_beta, 
    Named("iter") = iter,
    Named("Zigma") = Zigma
  );
}


// [[Rcpp::export]]
double estimate_mom_dispersion_efficient(
    const Eigen::VectorXd& y_sums_vec,    // M
    const Eigen::VectorXd& y_sq_sums_vec, // M
    const Eigen::MatrixXd& X_unique,      // M x P
    const Eigen::VectorXd& sf_unique,     // M
    const Eigen::VectorXd& counts,        // M
    const Eigen::VectorXd& beta_g,        // P
    int N_total) {
  
  int M = X_unique.rows();
  int P = X_unique.cols();
  
  // N_total must be the total number of cells (n in the second function)
  double corr = static_cast<double>(N_total) / (N_total - P);
  
  double total_num = 0.0;
  double total_den = 0.0;
  
  // Linear predictor: eta = X * beta
  VectorXd eta = X_unique * beta_g;
  
  for (int m = 0; m < M; ++m) {
    // Ensure this matches your 'sf' logic
    // If your blueprint saved 'off', you might need exp(off_unique[m])
    double mu = sf_unique[m] * std::exp(eta[m]); 
    
    double n_u      = counts[m];
    double sum_y    = y_sums_vec[m];
    double sum_y_sq = y_sq_sums_vec[m];
    
    // This is the expansion of sum((y - mu)^2)
    double sse = sum_y_sq - (2.0 * mu * sum_y) + (n_u * mu * mu);
    
    total_num += sse - (n_u * mu); // This is sum((y-mu)^2 - mu)
    total_den += n_u * (mu * mu);  // This is sum(mu^2)
  }
  
  double theta = 0.0;
  // Use a more robust check for the denominator
  if (total_den > 0.0) {
    theta = corr * (total_num / total_den);
    if (theta < 0.0) theta = 0.0;
  }
  
  return theta;
}

// [[Rcpp::export]]
Eigen::MatrixXd compute_hessian_efficient(const Eigen::VectorXd& beta,
                                   const double overdispersion,
                                   const Eigen::VectorXd& y_sums,    // M (sum of y for each unique group)
                                   const Eigen::VectorXd& counts,    // M (number of cells in each unique group)
                                   const Eigen::MatrixXd& X_unique,  // M x P
                                   const Eigen::VectorXd& sf_unique) // M
{
  const double alpha = overdispersion;
  int M = X_unique.rows();
  int P = X_unique.cols();
  
  // 1. Compute eta and mu for the M unique groups
  Eigen::VectorXd eta = X_unique * beta;
  Eigen::ArrayXd mu = sf_unique.array() * eta.array().exp();
  
  // 2. Compute group-wise scalars
  // denom = (1 + alpha * mu)^2
  Eigen::ArrayXd denom_sq = (1.0 + alpha * mu).square();
  
  // Group Weight = mu / denom_sq * (alpha * y_sum + count)
  // This represents the sum of individual scalars for all observations in the group
  Eigen::ArrayXd group_weights = (mu / denom_sq) * (alpha * y_sums.array() + counts.array());
  
  // 3. Compute H = X_unique^T * diag(group_weights) * X_unique
  // This is much faster because X_unique is M x P instead of N x P
  Eigen::MatrixXd H = X_unique.transpose() * group_weights.matrix().asDiagonal() * X_unique;
  
  // 4. Regularize and Invert
  // Added a small epsilon to the diagonal to ensure stability
  H.diagonal().array() += 1e-9;
  
  return H.inverse();
}

// [[Rcpp::export]]
Eigen::MatrixXd compute_scores_efficient(const Eigen::MatrixXd& X_unique,
                                             const Eigen::VectorXd& y_sums,
                                             const Eigen::VectorXd& counts,
                                             const Eigen::VectorXd& beta,
                                             const double overdispersion,
                                             const Eigen::VectorXd& sf_unique) {
  const double alpha = 1.0 / overdispersion;
  int M = X_unique.rows();
  
  Eigen::VectorXd eta = X_unique * beta;
  Eigen::ArrayXd mu   = sf_unique.array() * eta.array().exp();
  
  // Group-level weighted residual sum
  // wr_sum = (y_sum - n*mu) / mu * (mu / (1 + mu/alpha))
  // Simplified: (y_sum - n*mu) / (1 + mu/alpha)
  Eigen::ArrayXd wr_sum = (y_sums.array() - counts.array() * mu) / (1.0 + mu / alpha);
  
  // Return M x K matrix: each unique row x_u multiplied by its aggregate residual
  return X_unique.array().colwise() * wr_sum;
}

// [[Rcpp::export]]
Eigen::MatrixXd compute_clustered_meat_efficient(const Eigen::MatrixXd& X_unique,
                                                     const Eigen::VectorXd& y_sums_per_group_cluster,
                                                     const Eigen::VectorXd& counts_per_group_cluster,
                                                     const Eigen::VectorXi& group_to_cluster_map, // length = total group-cluster pairs
                                                     const Eigen::VectorXd& beta,
                                                     const double overdispersion,
                                                     const Eigen::VectorXd& sf_unique_per_pair,
                                                     int num_clusters,
                                                     int N_total) {
  const int k = X_unique.cols();
  
  // 1. Compute scores for each unique group-cluster combination
  // Here, we assume each row of X_unique is mapped to a cluster
  Eigen::MatrixXd ef_grouped = compute_scores_efficient(X_unique, 
                                                            y_sums_per_group_cluster, 
                                                            counts_per_group_cluster, 
                                                            beta, 
                                                            overdispersion, 
                                                            sf_unique_per_pair);
  
  // 2. Aggregate group scores into cluster scores
  Eigen::MatrixXd cluster_sums = Eigen::MatrixXd::Zero(num_clusters, k);
  for (int i = 0; i < ef_grouped.rows(); ++i) {
    int cluster_idx = group_to_cluster_map(i) - 1; 
    cluster_sums.row(cluster_idx) += ef_grouped.row(i);
  }
  
  // 3. Sandwich "Meat" calculation
  const double adj = (num_clusters > 1) ? double(num_clusters) / (num_clusters - 1.0) : 1.0;
  Eigen::MatrixXd rval = cluster_sums.transpose() * cluster_sums;
  
  return (adj / N_total) * rval;
}

// [[Rcpp::export]]
Rcpp::List aggregate_vector_by_group_efficient(const Eigen::VectorXd& y, 
                                               const Eigen::VectorXi& mapping, 
                                               int n_groups) {
  int n = y.size();
  
  // Vectors to hold the sums for the unique groups
  VectorXd y_sums = VectorXd::Zero(n_groups);
  VectorXd y_sq_sums = VectorXd::Zero(n_groups);
  
  for (int i = 0; i < n; ++i) {
    int g_idx = mapping[i] - 1; // R 1-based to C++ 0-based
    
    double val = y[i];
    y_sums[g_idx] += val;
    y_sq_sums[g_idx] += val * val;
  }
  
  return Rcpp::List::create(
    Rcpp::Named("y_sums") = y_sums,
    Rcpp::Named("y_sq_sums") = y_sq_sums
  );
}

#ifdef USE_CUDA
// [[Rcpp::export]]
List beta_fit_gpu_summary(
    Eigen::MatrixXf        y,             // [genes x cells]  — raw counts, same as beta_fit_gpu
    Eigen::MatrixXf        X_unique,      // [M x features]   — unique design rows from blueprint
    Eigen::VectorXf        off_unique,    // [M]              — log(sf) per unique group
    Eigen::VectorXi        mapping,       // [N] 1-based      — cell -> group index
    Eigen::VectorXf        counts,        // [M]              — cells per group
    Eigen::VectorXi        cluster_map,   // [M] 1-based      — group -> cluster index
    int                    n_clusters,
    int                    max_iter,
    float                  eps,
    int                    batch_size)
{
  auto t1 = std::chrono::high_resolution_clock::now();
  
  // ── Transpose to col-major layouts expected by the CUDA function ─────────
  // y:        [genes x cells] -> [cells x genes]  (N x G, col-major)
  // X_unique: [M x features]  -> [features x M]   (P x M, col-major)
  auto y_float = y.transpose().eval();
  auto X_float = X_unique.transpose().eval();
  
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "TIME Reorder cost "
            << std::chrono::duration<double, std::milli>(t2 - t1).count()
            << " ms" << std::endl;
  
  // ── Delegate to the CUDA implementation ──────────────────────────────────
  std::vector<int> iterations(y.rows());   // one entry per gene, filled by the function
  
  t1 = std::chrono::high_resolution_clock::now();
  
  auto result = beta_fit_gpu_external_summary(
    y_float,       // [N x G]  col-major
    X_float,       // [P x M]  col-major
    off_unique,    // [M]
    mapping,       // [N] int
    counts,        // [M]
    cluster_map,   // [M] int
    n_clusters,
    max_iter, eps, batch_size);
  
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "TIME Compute cost "
            << std::chrono::duration<double, std::milli>(t2 - t1).count()
            << " ms" << std::endl;
  
  // ── Build return list ─────────────────────────────────────────────────────
  // The summary version always has sandwich components when n_clusters > 0,
  // and never has TEST-mode debug fields (k, beta_init) — there is no rough
  // per-cell initialisation path to debug here.
  if (n_clusters > 0) {
    return List::create(
      Named("mu_beta")     = result.beta.cast<double>().transpose(),
      Named("theta")       = result.theta.cast<double>(),
      Named("hessian_inv") = result.hessian_inv.cast<double>(),
      Named("meat")        = result.meat.cast<double>(),
      Named("iter")        = iterations);
  } else {
    return List::create(
      Named("mu_beta")     = result.beta.cast<double>().transpose(),
      Named("theta")       = result.theta.cast<double>(),
      Named("iter")        = iterations);
  }
}
#endif // USE_CUDA