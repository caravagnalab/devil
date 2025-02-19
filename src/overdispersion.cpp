// #include <Rcpp.h>
#include <RcppArmadillo.h>

using namespace Rcpp;
// [[Rcpp::depends(RcppArmadillo)]]

/*
 * Functions originally from DESeq2:
 * Love, M.I., Huber, W., Anders, S. (2014)
 * "Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2"
 * Genome Biology, 15:550.
 * https://github.com/mikelove/DESeq2/blob/master/src/DESeq2.cpp
 *
 * This code was copied from glmGamPoi, which adapted the original DESeq2 code
 * with some modifications:
 * https://github.com/const-ae/glmGamPoi
 */

// This correction factor is necessary to avoid estimates of
// theta that are basically +Inf. The problem is that for
// some combination of the y, mu, and X the term
// lgamma(1/theta) and the log(det(t(X) %*% W %*% X))
// with W = diag(1/(1/mu + theta)) canceled each other
// exactly out for large theta.
const double cr_correction_factor = 0.99;

// Creates a frequency table of values in vector x, but only if the number of unique values
//is less than or equal to stop_if_larger. This is used for optimization in likelihood calculations
// when dealing with count data.

// [[Rcpp::export]]
List make_table_if_small(const NumericVector& x, int stop_if_larger){
  std::unordered_map<long, size_t> counts;
  counts.reserve(stop_if_larger);
  for (double v : x){
    ++counts[(long) v];
    if(counts.size() > stop_if_larger){
      return List::create(NumericVector::create(), NumericVector::create());
    }
  }
  NumericVector keys(counts.size());
  NumericVector values(counts.size());
  transform(counts.begin(), counts.end(), keys.begin(), [](std::pair<int, size_t> pair){return (double) pair.first;});
  transform(counts.begin(), counts.end(), values.begin(), [](std::pair<int, size_t> pair){return (double) pair.second;});
  return List::create(keys, values);
}

// this function returns the log posterior of dispersion parameter alpha, for negative binomial variables
// given the counts y, the expected means mu, the design matrix x (used for calculating the Cox-Reid adjustment),
// and the parameters for the normal prior on log alpha

// [[Rcpp::export]]
double conventional_loglikelihood_fast(NumericVector y, NumericVector mu, double log_theta, const arma::mat& model_matrix, bool do_cr_adj,
                                       NumericVector unique_counts = NumericVector::create(),
                                       NumericVector count_frequencies = NumericVector::create()) {
  double theta = exp(log_theta);
  double cr_term = 0.0;
  if(do_cr_adj){
    arma::vec w_diag = 1.0 / (1.0 / mu + theta);
    arma::mat b = model_matrix.t() * (model_matrix.each_col() % w_diag);
    // cr_term = -0.5 * log(det(b)) * cr_correction_factor;
    arma::mat L, U, P;
    arma::lu(L, U, P, b);
    double ld = sum(log(arma::diagvec(L)));
    arma::vec u_diag = arma::diagvec(U);
    for(double e : u_diag){
      ld += e < 1e-50 ? log(1e-50) : log(e);
    }
    cr_term = -0.5 * ld * cr_correction_factor;
  }
  double theta_neg1 = R_pow_di(theta, -1);
  double lgamma_term = 0;
  // If summarized counts are available use those to calculate sum(lgamma(y + theta_neg1))
  if(unique_counts.size() > 0 && unique_counts.size() == count_frequencies.size()){
    for(size_t iter = 0; iter < count_frequencies.size(); ++iter){
      lgamma_term += count_frequencies[iter] * lgamma(unique_counts[iter] + theta_neg1);
    }
  }else{
    lgamma_term = sum(lgamma(y + theta_neg1));
  }
  lgamma_term -=  y.size() * lgamma(theta_neg1);
  double ll_part = 0.0;
  for(size_t i = 0; i < y.size(); ++i){
    ll_part += (-y[i] - theta_neg1) * log(mu[i] + theta_neg1);
  }
  ll_part -= y.size() * theta_neg1 * log(theta);
  return lgamma_term + ll_part + cr_term;
}


// this function returns the derivative of the log posterior with respect to the log of the
// dispersion parameter alpha, given the same inputs as the previous function

// [[Rcpp::export]]
double conventional_score_function_fast(NumericVector y, NumericVector mu, double log_theta, const arma::mat& model_matrix, bool do_cr_adj,
                                        NumericVector unique_counts = NumericVector::create(),
                                        NumericVector count_frequencies = NumericVector::create()) {
  double theta = exp(log_theta);
  double theta_neg1 = 1.0 / theta;

  double cr_term = 0.0;
  if(do_cr_adj){
    arma::vec w_diag = 1.0 / (1.0 / mu + theta);
    arma::vec dw_diag = -1 * w_diag % w_diag;
    arma::mat b = model_matrix.t() * (model_matrix.each_col() % w_diag);
    arma::mat db = model_matrix.t() * (model_matrix.each_col() % dw_diag);
    // The diag(1e-6) protects against singular matrices
    arma::mat b_inv = inv_sympd(b + arma::eye(b.n_rows, b.n_cols) * 1e-6);
    cr_term = -0.5 * trace(b_inv * db) * cr_correction_factor;
  }


  double digamma_term = 0;
  // If summarized counts are available use those to calculate sum(digamma(y + theta_neg1))
  if(unique_counts.size() > 0 && unique_counts.size() == count_frequencies.size()){
    double max_y = 0.0;
    double sum_y = 0.0;
    double sum_prod_y = 0.0;
    for(size_t iter = 0; iter < count_frequencies.size(); ++iter){
      digamma_term += count_frequencies[iter] * Rf_digamma(unique_counts[iter] + theta_neg1);
      sum_y += count_frequencies[iter] * unique_counts[iter];
      sum_prod_y += count_frequencies[iter] * (unique_counts[iter] - 1) * unique_counts[iter];
      max_y = std::max(max_y, unique_counts[iter]);
    }
    double corr = theta_neg1 > 1e5 ? sum_prod_y / (2 * theta_neg1) : 0.0;
    if(max_y * 1e6 < theta_neg1){
      // This approximation is based on the fact that for large x
      // (sum(digamma(y + x))  - length(y) * digamma(x)) * x \approx sum(y)
      // Due to numerical imprecision the digamma_term reaches sum(y) sometimes
      // quicker than the ll_term, thus I subtract the first term of the
      // Laurent series expansion at x -> inf
      digamma_term = sum_y - corr;
    }else{
      digamma_term -= y.size() * Rf_digamma(theta_neg1);
      digamma_term *= theta_neg1;
      digamma_term = std::min(digamma_term, sum_y - corr);
    }
  }else{
    double max_y = 0.0;
    double sum_y = 0.0;
    double sum_prod_y = 0.0;
    for(size_t iter = 0; iter < y.size(); ++iter){
      digamma_term += Rf_digamma(y[iter] + theta_neg1);
      sum_y += y[iter];
      sum_prod_y += (y[iter] - 1) * y[iter];
      max_y = std::max(max_y, y[iter]);
    }
    double corr = theta_neg1 > 1e5 ? sum_prod_y / (2 * theta_neg1) : 0.0;
    if(max_y * 1e6 < theta_neg1){
      digamma_term = sum_y - corr;
    }else{
      digamma_term -= y.size() * Rf_digamma(theta_neg1);
      digamma_term *= theta_neg1;

      digamma_term = std::min(digamma_term, sum_y - corr);
    }
  }

  double ll_part = 0.0;
  for(size_t i = 0; i < y.size(); ++i){
    double mu_theta = (mu[i] * theta);
    if(mu_theta < 1e-10){
      ll_part += mu_theta * mu_theta * (1 / (1 + mu_theta) - 0.5);
    }else if(mu_theta < 1e-4){
      // The bounds are based on the Taylor expansion of log(1 + x) for x = 0.
      double inv = 1 / (1 + mu_theta);
      double upper_bound = mu_theta * mu_theta * inv;
      double lower_bound = mu_theta * mu_theta * (inv - 0.5);
      double suggest = (log(1 + mu_theta) - mu[i] / (mu[i] + theta_neg1)) ;
      ll_part +=  std::max(std::min(suggest, upper_bound), lower_bound);
    }else{
      ll_part += log(1 + mu_theta)  - mu[i] / (mu[i] + theta_neg1);
    }
    ll_part += y[i] / (mu[i] + theta_neg1);
  }
  ll_part *= theta_neg1;
  return ll_part - digamma_term + cr_term * theta;
}



// this function returns the second derivative of the log posterior with respect to the log of the
// dispersion parameter alpha, given the same inputs as the previous function

// [[Rcpp::export]]
double conventional_deriv_score_function_fast(NumericVector y, NumericVector mu, double log_theta, const arma::mat& model_matrix, bool do_cr_adj,
                                              NumericVector unique_counts = NumericVector::create(),
                                              NumericVector count_frequencies = NumericVector::create()) {
  double theta = exp(log_theta);
  double cr_term = 0.0;
  double cr_term2 = 0.0;
  if(do_cr_adj){
    arma::vec w_diag = 1/(1/mu + theta);
    arma::vec dw_diag = -1 * w_diag % w_diag;
    arma::vec d2w_diag = -2 * dw_diag % w_diag;

    arma::mat b = model_matrix.t() * (model_matrix.each_col() % w_diag);
    arma::mat db = model_matrix.t() * (model_matrix.each_col() % dw_diag);
    arma::mat d2b = model_matrix.t() * (model_matrix.each_col() % d2w_diag);
    // The diag(1e-6) protects against singular matrices
    arma::mat b_inv = inv_sympd(b + arma::eye(b.n_rows, b.n_cols) * 1e-6);
    arma::mat d_i_db = b_inv * db;
    double ddetb = trace(d_i_db);
    double d2detb = ((R_pow_di(ddetb, 2) - trace(d_i_db * d_i_db) + trace(b_inv * d2b)) );
    cr_term = (0.5 * R_pow_di(ddetb, 2) - 0.5 * d2detb)  * cr_correction_factor;
    cr_term2 = -0.5 * ddetb * cr_correction_factor;
  }

  double theta_neg1 = R_pow_di(theta, -1);
  double theta_neg2 = R_pow_di(theta, -2);
  double digamma_term = 0.0;
  double trigamma_term = 0.0;

  // If summarized counts are available use those to calculate sum(digamma()) and sum(trigamma())
  if(unique_counts.size() > 0 && unique_counts.size() == count_frequencies.size()){
    for(size_t iter = 0; iter < count_frequencies.size(); ++iter){
      digamma_term += count_frequencies[iter] * Rf_digamma(unique_counts[iter] + theta_neg1);
      trigamma_term += count_frequencies[iter] * Rf_trigamma(unique_counts[iter] + theta_neg1);
    }
    trigamma_term *= theta_neg2;

    digamma_term -= y.size() * Rf_digamma(theta_neg1);
    trigamma_term -=  theta_neg2 * y.size() * Rf_trigamma(theta_neg1);
  }else{
    digamma_term = sum(digamma(y + theta_neg1));
    digamma_term -= y.size() * Rf_digamma(theta_neg1);

    trigamma_term = theta_neg2 * sum(trigamma(y + theta_neg1));
    trigamma_term -=  theta_neg2 * y.size() * Rf_trigamma(theta_neg1);
  }

  double ll_part_1 = 0.0;
  double ll_part_2 = 0.0;
  for(size_t i = 0; i < y.size(); ++i){
    ll_part_1 += log(1 + mu[i] * theta) + (y[i] - mu[i]) / (mu[i] + theta_neg1);
    ll_part_2 += (mu[i] * mu[i] * theta + y[i]) / (1 + mu[i] * theta) / (1 + mu[i] * theta);
  }
  double ll_part = -2 * theta_neg1 * (ll_part_1 - digamma_term) + (ll_part_2 + trigamma_term);

  double res = ll_part + cr_term * R_pow_di(theta, 2) + (ll_part_1 - digamma_term) * theta_neg1 + cr_term2 * theta;
  return res;
}
