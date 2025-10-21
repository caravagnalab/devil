#include <RcppEigen.h>
#include <unordered_map>
#include <cmath>            // std::log, std::sqrt, std::lgamma
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(BH)]]

#ifdef _OPENMP
#include <omp.h>
#endif

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>

using namespace Rcpp;
using namespace Eigen;

// Bring in Boost scalar digamma/trigamma (optional: could instead fully qualify each call)
using boost::math::digamma;
using boost::math::trigamma;

// ======================= Gauss–Hermite (physicists) with caching =======================
struct GHTable { VectorXd X, W; };

inline GHTable& get_gh(int n){
  static std::unordered_map<int, GHTable> cache;
  auto it = cache.find(n);
  if (it != cache.end()) return it->second;

  // Golub–Welsch for Hermite (weight e^{-x^2})
  MatrixXd J = MatrixXd::Zero(n, n);
  for (int i = 0; i < n - 1; ++i) {
    double b = std::sqrt(0.5 * (i + 1));
    J(i, i + 1) = b;
    J(i + 1, i) = b;
  }
  SelfAdjointEigenSolver<MatrixXd> es(J);
  VectorXd x = es.eigenvalues();
  MatrixXd V = es.eigenvectors();
  VectorXd w(n);
  const double c = std::sqrt(M_PI);   // ok on clang/libc++; portable alternative: const double c = std::sqrt(acos(-1.0));
  for (int i = 0; i < n; ++i) {
    double wi = c * V(0, i) * V(0, i);
    w(i) = (wi < 0) ? 0.0 : wi; // numerical safety
  }

  cache.emplace(n, GHTable{std::move(x), std::move(w)});
  return cache.find(n)->second;
}

// ======================= h(x) split and Laplace mode =======================
struct LaplacePoint {
  double mu0, sigma0, h_mu0;
};

inline double g (double s, double t, double x){
  return s * (x * std::log(x) - std::lgamma(x)) - t * x;
}

// NOTE: use Boost scalar digamma/trigamma (or fully qualify: boost::math::digamma(x))
inline double g1(double s, double t, double x){
  return s * (std::log(x) + 1.0 - digamma(x)) - t;
}
inline double g2(double s, double x){
  return s * (1.0/x - trigamma(x));
}


inline LaplacePoint find_mode(double p, double s, double t,
                              int max_newton, double newton_tol,
                              double x0){
  auto hp  = [&](double x){ return p/x + g1(s,t,x); };
  auto hpp = [&](double x){ return -p/(x*x) + g2(s,x); };

  double x = std::max(1e-9, x0);
  for (int i = 0; i < max_newton; ++i){
    double g1v = hp(x);
    if (std::abs(g1v) < newton_tol) break;
    double g2v = hpp(x);
    double step = g1v / g2v;
    double xn = x - step;
    if (!(xn>0.0) || !std::isfinite(xn)) xn = 0.5 * x;
    if (std::abs(xn - x) < newton_tol * (1.0 + std::abs(x))) { x = xn; break; }
    x = xn;
  }
  double curv = -hpp(x);
  if (!(curv>0.0) || !std::isfinite(curv))
    Rcpp::stop("Mode-finding failed: h''(mu0) >= 0 or non-finite.");

  double mu0    = std::max(1e-12, x);
  double sigma0 = std::sqrt(1.0/curv);
  double h_mu0  = p*std::log(mu0) + g(s,t,mu0);

  return {mu0, sigma0, h_mu0};
}

inline double log_integral_at(double p, double s, double t,
                              int gh_order,
                              const GHTable& gh,
                              const LaplacePoint& lp)
{
  const double root2 = std::sqrt(2.0);
  const double scale = lp.sigma0 * root2;

  // Two-pass log-sum-exp without heap allocations
  double max_lt = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < gh_order; ++i){
    double u  = gh.X(i);
    double xx = lp.mu0 + scale*u;
    if (xx <= 0.0 || !std::isfinite(xx)) continue;
    double r  = (p*std::log(xx) + g(s,t,xx)) - lp.h_mu0 + u*u;
    double lt = std::log(gh.W(i)) + r;
    if (lt > max_lt) max_lt = lt;
  }
  long double acc = 0.0L;
  for (int i = 0; i < gh_order; ++i){
    double u  = gh.X(i);
    double xx = lp.mu0 + scale*u;
    if (xx <= 0.0 || !std::isfinite(xx)) continue;
    double r  = (p*std::log(xx) + g(s,t,xx)) - lp.h_mu0 + u*u;
    double lt = std::log(gh.W(i)) + r;
    acc += std::exp(lt - max_lt);
  }
  if (!(acc > 0.0L)) Rcpp::stop("GH quadrature underflow.");
  double logI = max_lt + std::log((double)acc);
  return lp.h_mu0 + std::log(scale) + logI;
}

// ======================= Public GH wrappers =======================
// [[Rcpp::export]]
double H_log_gh_cpp(double p, double s, double t,
                    int gh_order      = 32,
                    int max_newton    = 8,
                    double newton_tol = 1e-12)
{
  if (!(t > s)) Rcpp::stop("Convergence requires t > s (got t=%.6g, s=%.6g).", t, s);
  if (gh_order < 4) gh_order = 4;
  GHTable& gh = get_gh(gh_order);
  double x0 = std::max(1e-3, (p + s) / (t - s));
  LaplacePoint lp = find_mode(p, s, t, max_newton, newton_tol, x0);
  return log_integral_at(p, s, t, gh_order, gh, lp);
}

// [[Rcpp::export]]
Rcpp::NumericVector H_log_gh_pmhalf(double s, double t,
                                    int gh_order      = 32,
                                    int max_newton    = 8,
                                    double newton_tol = 1e-12)
{
  if (!(t > s)) Rcpp::stop("Convergence requires t > s (got t=%.6g, s=%.6g).", t, s);
  if (gh_order < 4) gh_order = 4;
  GHTable& gh = get_gh(gh_order);

  double p1 = +0.5, p2 = -0.5;
  double x0_1 = std::max(1e-3, (p1 + s) / (t - s));
  LaplacePoint lp1 = find_mode(p1, s, t, max_newton, newton_tol, x0_1);
  LaplacePoint lp2 = find_mode(p2, s, t, max_newton, newton_tol, lp1.mu0);

  double logH_p = log_integral_at(p1, s, t, gh_order, gh, lp1);
  double logH_m = log_integral_at(p2, s, t, gh_order, gh, lp2);
  return NumericVector::create(_["logH_p"]=logH_p, _["logH_m"]=logH_m);
}

// ======================= Small frequency table (on Eigen) =======================
/*
 Returns {keys, counts}. If unique values exceed cap, returns two empty vectors.
 Keys are int64 obtained by static_cast<long long>(y[i]) (assumes integer-valued doubles).
 */
static inline List make_table_if_small_eigen(const VectorXd y, int cap){
  std::unordered_map<long long, size_t> cnt;
  cnt.reserve(static_cast<size_t>(cap * 1.3));
  const int n = (int)y.size();
  for (int i = 0; i < n; ++i){
    long long k = static_cast<long long>(y[i]);
    auto it = cnt.find(k);
    if (it == cnt.end()){
      cnt.emplace(k, 1u);
      if ((int)cnt.size() > cap){
        return List::create(NumericVector(), NumericVector()); // empty => fallback
      }
    } else {
      ++(it->second);
    }
  }
  NumericVector keys(cnt.size()), vals(cnt.size());
  size_t j = 0;
  for (const auto& kv : cnt){
    keys[j] = static_cast<double>(kv.first);
    vals[j] = static_cast<double>(kv.second);
    ++j;
  }
  return List::create(keys, vals);
}

// [[Rcpp::export]]
List two_step_fit_cpp(const Eigen::VectorXd y,
                      const Eigen::MatrixXd X,
                      Eigen::VectorXd mu_beta,             // updated in-place
                      const Eigen::VectorXd off,
                      double kappa,
                      int max_iter_beta,
                      int max_iter_kappa,
                      double eps_theta,
                      double eps_beta,
                      bool  fit_overdispersion = true,
                      int    newton_max = 16,
                      int    y_unique_cap = 4096) {
  const int n = X.rows();
  const int p = X.cols();
  if (off.size() != n) stop("off length mismatch.");

  // internal: k = 1/kappa
  double k = 1.0 / kappa;

  const MatrixXd Xt = X.transpose();

  VectorXd w_q(n), mu_g(n), delta(p);

  // ---------- Precompute exp(-off) once ----------
  VectorXd exp_neg_off = (-off.array()).exp().matrix();

  // ---------- Phase 1: optimize beta | k ----------
  bool beta_converged = false;
  int  it_beta = 0;

  // Start: w_q = exp(-X*beta - off) = exp(-off) * exp(-(X*beta))
  VectorXd Xbeta = X * mu_beta;
  w_q = (-(Xbeta).array()).exp().matrix();
  w_q.array() *= exp_neg_off.array();

  for (it_beta = 0; it_beta < max_iter_beta; ++it_beta) {
    // mu_g and helpers at current beta
    // mu_g = (k + y) / (1 + k*w_q)
    VectorXd denom = (1.0 + k * w_q.array()).matrix();
    mu_g = (k + y.array()) / denom.array();
    VectorXd mu_g_wq = (mu_g.array() * w_q.array()).matrix(); // weight vector

    // A = X' * (k * diag(mu_g_wq)) * X without forming diag:
    VectorXd wcol = (k * mu_g_wq).matrix();
    MatrixXd Xw = X;
    Xw.array().colwise() *= wcol.array();

    MatrixXd A(p, p);
    A.noalias() = Xt * Xw;

    // rhs = k * X' * (mu_g_wq - 1)
    VectorXd rhs = (mu_g_wq.array() - 1.0).matrix();
    rhs *= k;
    rhs = Xt * rhs;

    Eigen::LLT<MatrixXd> llt(A);
    if (llt.info() != Eigen::Success) {
      A.diagonal().array() += 1e-10;      // tiny jitter only if needed
      llt.compute(A);
      if (llt.info() != Eigen::Success) stop("Cholesky failed (beta phase).");
    }
    delta = llt.solve(rhs);
    if (!delta.allFinite()) stop("Non-finite beta step.");

    mu_beta += delta;

    double beta_change = delta.cwiseAbs().maxCoeff();

    // ---------- Incremental update of w_q ----------
    // w_q *= exp( -X*delta )
    VectorXd Xdelta = X * (-delta);
    w_q.array() *= Xdelta.array().exp();

    if (beta_change < eps_beta) { beta_converged = true; break; }
  }

  // ---------------------------------------------------------------------------
  // EARLY RETURN if we don't want to fit overdispersion
  // ---------------------------------------------------------------------------
  if (!fit_overdispersion) {
    return List::create(
      Named("mu_beta")         = mu_beta,
      Named("kappa")           = R_NaReal,
      Named("beta_converged")  = beta_converged,
      Named("kappa_converged") = false,
      Named("beta_iters")      = it_beta + 1,
      Named("kappa_iters")     = 0,
      Named("used_y_table")    = false
    );
  }

  // ---------- Build (optional) frequency table on y for κ phase ----------
  List table = make_table_if_small_eigen(y, y_unique_cap);
  NumericVector y_keys = table[0];
  NumericVector y_cnts = table[1];
  const bool use_table = (y_keys.size() > 0);

  // Recompute linear predictor and mean_vector = exp(X %*% beta + off)
  VectorXd Xbeta_final = X * mu_beta;
  VectorXd mean_vector = (Xbeta_final.array() + off.array()).exp().matrix();
  // Avoid exact zeros (for safety)
  for (int i = 0; i < n; ++i) {
    if (!(mean_vector[i] > 0.0) || !std::isfinite(mean_vector[i])) mean_vector[i] = 1e-6;
  }

  // Method-of-moments overdispersion (kappa_mom = (Var(Y) - mean(Y)) / mean(Y)^2)
  // Then k = 1 / kappa_mom
  const double y_mean = y.mean(); // Eigen mean (double)
  // unbiased sample variance; if n==1, fall back to 0
  double y_var = 0.0;
  if (n > 1) {
    const VectorXd yc = y.array() - y_mean;
    y_var = (yc.array().square().sum()) / static_cast<double>(n - 1);
  }

  double kappa_mom = (y_var - y_mean) / (y_mean * y_mean);
  if (!std::isfinite(kappa_mom) || kappa_mom <= 0.0) {
    kappa_mom = 0.01;
  }

  // Convert to internal scale k = 1 / kappa, then clamp to the same bounds used later
  double k_mom = 1.0 / kappa_mom;

  // Use the same safeguard range you'll define for the kappa phase
  const double k_min_mom = 1e-4;
  const double k_max_mom = 1e4;
  if (!std::isfinite(k_mom)) { k_mom = k_min_mom; }
  k = std::max(k_min_mom, std::min(k_max_mom, k_mom));

  // ---------- Phase 2: optimize k | beta (beta frozen) ----------
  bool kappa_converged = false;
  int  it_kappa = 0;
  const double Xbeta_sum = (X * mu_beta).sum();

  // Parameters for safeguards (MUST be before lambdas) ----------
  const double k_min = 1e-4;
  const double k_max = 1e4;
  const double jump_factor = 10;
  int bad_accel = 0;

  // Fixed-point map and residual -------------------------------
  auto F_k = [&](double kk)->double {
    double sum_digamma = 0.0;
    if (use_table) {
      for (int j = 0; j < y_keys.size(); ++j)
        sum_digamma += y_cnts[j] * digamma(kk + y_keys[j]);
    } else {
      for (int i = 0; i < n; ++i)
        sum_digamma += digamma(kk + y[i]);
    }

    double sum_log1p = 0.0, sum_mu_g_wq = 0.0;
    for (int i = 0; i < n; ++i) {
      const double wi = w_q[i], yi = y[i];
      const double kw = kk * wi, den = 1.0 + kw;
      sum_log1p   += std::log1p(kw);
      sum_mu_g_wq += (kk + yi) * wi / den;
    }

    const double C1 = Xbeta_sum - (sum_digamma - sum_log1p) + sum_mu_g_wq;
    if (!(C1 > n))
      stop("kappa update requires C1 > n (got C1=%.6g, n=%d).", C1, n);

    Rcpp::NumericVector both = H_log_gh_pmhalf(n, C1, newton_max, newton_max, eps_theta);
    const double logH_p = both["logH_p"];
    const double logH_m = both["logH_m"];
    return std::exp(logH_p - logH_m);
  };

  auto f_of_k = [&](double kk)->double {
    double kn = F_k(kk);
    return std::log(kn) - std::log(kk);
  };

  // Numerical derivative helper
  auto f_prime_k = [&](double kk, double fk) -> double {
    double h = std::max(1e-8, 1e-6 * kk);
    return (f_of_k(kk + h) - fk) / h;
  };

  // Adaptive bracket search
  auto find_bracket = [&](double k_center) -> std::pair<double,double> {
    double f_center = f_of_k(k_center);

    for (int expand = 1; expand <= 10; ++expand) {
      double factor = std::pow(2.0, expand);
      double lo = std::max(k_min, k_center / factor);
      double hi = std::min(k_max, k_center * factor);

      double flo = f_of_k(lo);
      if (flo * f_center < 0) return {lo, k_center};

      double fhi = f_of_k(hi);
      if (f_center * fhi < 0) return {k_center, hi};
      if (flo * fhi < 0) return {lo, hi};
    }
    return {-1.0, -1.0};
  };

  // Brent (unchanged) --------------------------------
  auto brent = [&](double a, double b, double fa, double fb)->double {
    double c=a, fc=fa, d=b-a, e=d;
    const double tol = eps_theta, eps_m = std::numeric_limits<double>::epsilon();
    for (int it=0; it<60; ++it) {
      if (std::fabs(fc) < std::fabs(fb)) { a=b; b=c; c=a; fa=fb; fb=fc; fc=fa; }
      double tol1 = 2.0*eps_m*std::fabs(b) + 0.5*tol;
      double xm   = 0.5*(c - b);
      if (std::fabs(xm) <= tol1 || fb == 0.0) return b;

      double s, p, q;
      if (std::fabs(e) >= tol1 && std::fabs(fa) > std::fabs(fb)) {
        s = fb/fa;
        if (a==c) { p = 2.0*xm*s; q = 1.0 - s; }
        else {
          double r = fb/fc, t = fa/fc;
          p = s*(2.0*xm*t*(t-r) - (b-a)*(r-1.0));
          q = (t-1.0)*(r-1.0)*(s-1.0);
        }
        if (p>0) q = -q;
        p = std::fabs(p);
        if (2.0*p < std::min(3.0*xm*q - std::fabs(tol1*q), std::fabs(e*q))) {
          e = d; d = p/q;
        } else { d = xm; e = d; }
      } else { d = xm; e = d; }
      a = b; fa = fb;
      b += (std::fabs(d) > tol1 ? d : (xm>0 ? tol1 : -tol1));
      fb = f_of_k(b);
      if ( (fb>0 && fc>0) || (fb<0 && fc<0) ) { c=a; fc=fa; e=d=b-a; }
    }
    return b;
  };

  // Initialize
  double k0 = std::max(k_min, std::min(k_max, k));
  double f0 = f_of_k(k0);
  if (!std::isfinite(f0)) stop("kappa residual not finite at start.");

  // Try one Newton step for better initialization
  double fp0 = f_prime_k(k0, f0);
  if (std::isfinite(fp0) && std::fabs(fp0) > 1e-10) {
    double k_newton = k0 - f0 / fp0;
    k_newton = std::max(k_min, std::min(k_max, k_newton));
    double f_newton = f_of_k(k_newton);

    if (std::isfinite(f_newton) && std::fabs(f_newton) < std::fabs(f0)) {
      k0 = k_newton;
      f0 = f_newton;
    }
  }

  double k1 = F_k(k0);
  k1 = std::min(k1, jump_factor*std::max(k0,1.0));
  k1 = std::max(k_min, std::min(k_max, k1));
  double f1 = f_of_k(k1);

  // If we already bracket, use Brent immediately
  if (f0*f1 < 0) {
    k = brent(std::min(k0,k1), std::max(k0,k1), f0, f1);
    kappa_converged = true;
  } else {
    // Aitken with fallbacks
    for (it_kappa = 0; it_kappa < max_iter_kappa; ++it_kappa) {
      // Adaptive jump factor
      double adaptive_jump = jump_factor / (1.0 + 0.1 * it_kappa);

      double k2 = F_k(k1);
      k2 = std::min(k2, adaptive_jump*std::max(k1,1.0));
      k2 = std::max(k_min, std::min(k_max, k2));
      double f2 = f_of_k(k2);

      // Aitken Δ² proposal
      double d1 = k1 - k0, d2 = k2 - k1, denom = (d2 - d1);
      double k_acc = (std::fabs(denom) > 1e-14) ? k0 - (d1*d1)/denom : k2;

      if (!(k_acc > 0) || !std::isfinite(k_acc))
        k_acc = k2;
      k_acc = std::min(k_acc, adaptive_jump*std::max(k2,1.0));
      k_acc = std::max(k_min, std::min(k_max, k_acc));

      // Residual decrease with backtracking
      double k_prop = k_acc;
      double f_prop = f_of_k(k_prop);
      int bt = 0;
      while ( (std::fabs(f_prop) >= std::fabs(f1) || !std::isfinite(f_prop)) && bt < 8 ) {
        k_prop = 0.5*(k_prop + k1);
        f_prop = f_of_k(k_prop);
        ++bt;
      }

      // Check if Aitken improved
      if (!(std::fabs(f_prop) < std::fabs(f1))) {
        bad_accel++;

        // STRATEGY 1: Try secant if we have two distinct points
        if (bad_accel == 1 && std::fabs(f1 - f0) > 1e-14) {
          double k_sec = k1 - f1 * (k1 - k0) / (f1 - f0);
          k_sec = std::max(k_min, std::min(k_max, k_sec));

          double f_sec = f_of_k(k_sec);
          if (std::isfinite(f_sec) && std::fabs(f_sec) < std::fabs(f1)) {
            k_prop = k_sec;
            f_prop = f_sec;
            bad_accel = 0;
          }
        }

        // STRATEGY 2: Try to bracket from recent points
        if (bad_accel >= 1) {
          double a = std::min(k1, k2), b = std::max(k1, k2);
          double fa = f1, fb = f2;
          if (fa*fb < 0) {
            k = brent(a, b, fa, fb);
            kappa_converged = true;
            break;
          }
        }

        // STRATEGY 3: Adaptive bracket search
        if (bad_accel >= 1) {
          auto bracket = find_bracket(k1);
          if (bracket.first > 0) {
            double flo = f_of_k(bracket.first);
            double fhi = f_of_k(bracket.second);
            k = brent(bracket.first, bracket.second, flo, fhi);
            kappa_converged = true;
            break;
          }
        }

        // Fallback to plain fixed-point step
        k_prop = k2;
        f_prop = f2;
      } else {
        bad_accel = 0;
      }

      // Check convergence
      if (std::fabs(k_prop - k1) < eps_theta * (1.0 + k1)) {
        k = k_prop;
        kappa_converged = true;
        break;
      }

      // Shift window
      k0 = k1; f0 = f1;
      k1 = k_prop; f1 = f_prop;
    }

    if (!kappa_converged) k = std::max(k_min, k1);
  }

  // Output on user scale
  const double kappa_out = 1.0 / k;

  return List::create(
    Named("mu_beta")         = mu_beta,
    Named("kappa")           = kappa_out,
    Named("beta_converged")  = beta_converged,
    Named("kappa_converged") = kappa_converged,
    Named("beta_iters")      = it_beta + 1,
    Named("kappa_iters")     = it_kappa + 1,
    Named("used_y_table")    = use_table
  );
}


// [[Rcpp::export]]
List two_step_fit_batched_cpp(
    const Eigen::MatrixXd Y,           // n x n_replicates matrix (each column is a replicate)
    const Eigen::MatrixXd X,           // Shared design matrix (n x p)
    Eigen::MatrixXd mu_beta_mat,       // p x n_replicates matrix (each column is initial beta)
    const Eigen::VectorXd off,         // Shared offset (length n)
    Eigen::VectorXd kappa_vec,         // Initial kappa for each replicate (length n_replicates)
    int max_iter_beta,
    int max_iter_kappa,
    double eps_theta,
    double eps_beta,
    int newton_max = 16,
    int y_unique_cap = 4096,
    int n_threads = 1)
{
  const int n = X.rows();
  const int p = X.cols();
  const int n_replicates = Y.cols();

  // Validate inputs
  if (Y.rows() != n)
    stop("Y must have %d rows to match X", n);
  if (mu_beta_mat.rows() != p)
    stop("mu_beta_mat must have %d rows to match X columns", p);
  if (mu_beta_mat.cols() != n_replicates)
    stop("mu_beta_mat must have %d columns to match Y", n_replicates);
  if (kappa_vec.size() != n_replicates)
    stop("kappa_vec length must match number of replicates");
  if (off.size() != n)
    stop("off length mismatch");

  // Pre-allocate output containers
  MatrixXd mu_beta_out = mu_beta_mat;  // Will be updated in place
  VectorXd kappa_out(n_replicates);
  std::vector<bool> beta_converged_out(n_replicates);
  std::vector<bool> kappa_converged_out(n_replicates);
  std::vector<int> beta_iters_out(n_replicates);
  std::vector<int> kappa_iters_out(n_replicates);
  std::vector<bool> used_y_table_out(n_replicates);

  // Precompute shared quantities
  const MatrixXd Xt = X.transpose();
  const VectorXd exp_neg_off = (-off.array()).exp().matrix();

  // Set number of threads
#ifdef _OPENMP
  if (n_threads > 0) omp_set_num_threads(n_threads);
#endif

  // Process each replicate in parallel
#pragma omp parallel for schedule(dynamic) if(n_replicates > 1)
  for (int r = 0; r < n_replicates; ++r) {

    // Extract data for this replicate
    const VectorXd y = Y.col(r);
    VectorXd mu_beta = mu_beta_out.col(r);
    double kappa = kappa_vec[r];
    double k = 1.0 / kappa;

    // Working arrays for this replicate
    VectorXd w_q(n), mu_g(n), delta(p);

    // ---------- Phase 1: optimize beta | k ----------
    bool beta_converged = false;
    int it_beta = 0;

    VectorXd Xbeta = X * mu_beta;
    w_q = (-(Xbeta).array()).exp().matrix();
    w_q.array() *= exp_neg_off.array();

    for (it_beta = 0; it_beta < max_iter_beta; ++it_beta) {
      VectorXd denom = (1.0 + k * w_q.array()).matrix();
      mu_g = (k + y.array()) / denom.array();
      VectorXd mu_g_wq = (mu_g.array() * w_q.array()).matrix();

      VectorXd wcol = (k * mu_g_wq).matrix();
      MatrixXd Xw = X;
      Xw.array().colwise() *= wcol.array();

      MatrixXd A(p, p);
      A.noalias() = Xt * Xw;

      VectorXd rhs = (mu_g_wq.array() - 1.0).matrix();
      rhs *= k;
      rhs = Xt * rhs;

      Eigen::LLT<MatrixXd> llt(A);
      if (llt.info() != Eigen::Success) {
        A.diagonal().array() += 1e-10;
        llt.compute(A);
        if (llt.info() != Eigen::Success) {
          Rcpp::warning("Cholesky failed (beta phase) in replicate %d", r+1);
          beta_converged = false;
          break;
        }
      }
      delta = llt.solve(rhs);
      if (!delta.allFinite()) {
        Rcpp::warning("Non-finite beta step in replicate %d", r+1);
        beta_converged = false;
        break;
      }

      mu_beta += delta;

      double beta_change = delta.cwiseAbs().maxCoeff();

      VectorXd Xdelta = X * (-delta);
      w_q.array() *= Xdelta.array().exp();

      if (beta_change < eps_beta) {
        beta_converged = true;
        break;
      }
    }

    // ---------- Build frequency table on y for κ phase ----------
    List table = make_table_if_small_eigen(y, y_unique_cap);
    NumericVector y_keys = table[0];
    NumericVector y_cnts = table[1];
    const bool use_table = (y_keys.size() > 0);

    // Recompute linear predictor and mean_vector = exp(X %*% beta + off)
    VectorXd Xbeta_final = X * mu_beta;
    VectorXd mean_vector = (Xbeta_final.array() + off.array()).exp().matrix();
    // Avoid exact zeros (for safety)
    for (int i = 0; i < n; ++i) {
      if (!(mean_vector[i] > 0.0) || !std::isfinite(mean_vector[i])) mean_vector[i] = 1e-6;
    }

    // Method-of-moments overdispersion (kappa_mom = (Var(Y) - mean(Y)) / mean(Y)^2)
    // Then k = 1 / kappa_mom
    const double y_mean = y.mean(); // Eigen mean (double)
    // unbiased sample variance; if n==1, fall back to 0
    double y_var = 0.0;
    if (n > 1) {
      const VectorXd yc = y.array() - y_mean;
      y_var = (yc.array().square().sum()) / static_cast<double>(n - 1);
    }

    double kappa_mom = (y_var - y_mean) / (y_mean * y_mean);
    if (!std::isfinite(kappa_mom) || kappa_mom <= 0.0) {
      kappa_mom = 0.01;
    }

    // Convert to internal scale k = 1 / kappa, then clamp to the same bounds used later
    double k_mom = 1.0 / kappa_mom;

    // Use the same safeguard range you'll define for the kappa phase
    const double k_min_mom = 1e-4;
    const double k_max_mom = 1e4;
    if (!std::isfinite(k_mom)) { k_mom = k_min_mom; }
    k = std::max(k_min_mom, std::min(k_max_mom, k_mom));

    // ---------- Phase 2: optimize k | beta ----------
    bool kappa_converged = false;
    int it_kappa = 0;
    const double Xbeta_sum = (X * mu_beta).sum();

    // Parameters for safeguards
    const double k_min = 1e-4;
    const double k_max = 1e4;
    const double jump_factor = 10;
    int bad_accel = 0;

    // Fixed-point map and residual
    auto F_k = [&](double kk)->double {
      double sum_digamma = 0.0;
      if (use_table) {
        for (int j = 0; j < y_keys.size(); ++j)
          sum_digamma += y_cnts[j] * digamma(kk + y_keys[j]);
      } else {
        for (int i = 0; i < n; ++i)
          sum_digamma += digamma(kk + y[i]);
      }

      double sum_log1p = 0.0, sum_mu_g_wq = 0.0;
      for (int i = 0; i < n; ++i) {
        const double wi = w_q[i], yi = y[i];
        const double kw = kk * wi, den = 1.0 + kw;
        sum_log1p   += std::log1p(kw);
        sum_mu_g_wq += (kk + yi) * wi / den;
      }

      const double C1 = Xbeta_sum - (sum_digamma - sum_log1p) + sum_mu_g_wq;
      if (!(C1 > n)) {
        Rcpp::warning("kappa update requires C1 > n (got C1=%.6g, n=%d) in replicate %d",
                      C1, n, r+1);
        return kk;  // Return current value as fallback
      }

      Rcpp::NumericVector both = H_log_gh_pmhalf(n, C1, newton_max, newton_max, eps_theta);
      const double logH_p = both["logH_p"];
      const double logH_m = both["logH_m"];
      return std::exp(logH_p - logH_m);
    };

    auto f_of_k = [&](double kk)->double {
      double kn = F_k(kk);
      return std::log(kn) - std::log(kk);
    };

    // Numerical derivative helper
    auto f_prime_k = [&](double kk, double fk) -> double {
      double h = std::max(1e-8, 1e-6 * kk);
      return (f_of_k(kk + h) - fk) / h;
    };

    // Adaptive bracket search
    auto find_bracket = [&](double k_center) -> std::pair<double,double> {
      double f_center = f_of_k(k_center);

      for (int expand = 1; expand <= 10; ++expand) {
        double factor = std::pow(2.0, expand);
        double lo = std::max(k_min, k_center / factor);
        double hi = std::min(k_max, k_center * factor);

        double flo = f_of_k(lo);
        if (flo * f_center < 0) return {lo, k_center};

        double fhi = f_of_k(hi);
        if (f_center * fhi < 0) return {k_center, hi};
        if (flo * fhi < 0) return {lo, hi};
      }
      return {-1.0, -1.0};
    };

    // Brent's method
    auto brent = [&](double a, double b, double fa, double fb)->double {
      double c=a, fc=fa, d=b-a, e=d;
      const double tol = eps_theta, eps_m = std::numeric_limits<double>::epsilon();
      for (int it=0; it<60; ++it) {
        if (std::fabs(fc) < std::fabs(fb)) { a=b; b=c; c=a; fa=fb; fb=fc; fc=fa; }
        double tol1 = 2.0*eps_m*std::fabs(b) + 0.5*tol;
        double xm   = 0.5*(c - b);
        if (std::fabs(xm) <= tol1 || fb == 0.0) return b;

        double s, p, q;
        if (std::fabs(e) >= tol1 && std::fabs(fa) > std::fabs(fb)) {
          s = fb/fa;
          if (a==c) { p = 2.0*xm*s; q = 1.0 - s; }
          else {
            double r = fb/fc, t = fa/fc;
            p = s*(2.0*xm*t*(t-r) - (b-a)*(r-1.0));
            q = (t-1.0)*(r-1.0)*(s-1.0);
          }
          if (p>0) q = -q;
          p = std::fabs(p);
          if (2.0*p < std::min(3.0*xm*q - std::fabs(tol1*q), std::fabs(e*q))) {
            e = d; d = p/q;
          } else { d = xm; e = d; }
        } else { d = xm; e = d; }
        a = b; fa = fb;
        b += (std::fabs(d) > tol1 ? d : (xm>0 ? tol1 : -tol1));
        fb = f_of_k(b);
        if ( (fb>0 && fc>0) || (fb<0 && fc<0) ) { c=a; fc=fa; e=d=b-a; }
      }
      return b;
    };

    // Initialize
    double k0 = std::max(k_min, std::min(k_max, k));
    double f0 = f_of_k(k0);
    if (!std::isfinite(f0)) {
      Rcpp::warning("kappa residual not finite at start in replicate %d", r+1);
      k0 = 1.0;  // Fallback to k=1
      f0 = f_of_k(k0);
    }

    // Try one Newton step for better initialization
    double fp0 = f_prime_k(k0, f0);
    if (std::isfinite(fp0) && std::fabs(fp0) > 1e-10) {
      double k_newton = k0 - f0 / fp0;
      k_newton = std::max(k_min, std::min(k_max, k_newton));
      double f_newton = f_of_k(k_newton);

      if (std::isfinite(f_newton) && std::fabs(f_newton) < std::fabs(f0)) {
        k0 = k_newton;
        f0 = f_newton;
      }
    }

    double k1 = F_k(k0);
    k1 = std::min(k1, jump_factor*std::max(k0,1.0));
    k1 = std::max(k_min, std::min(k_max, k1));
    double f1 = f_of_k(k1);

    // If we already bracket, use Brent immediately
    if (f0*f1 < 0) {
      k = brent(std::min(k0,k1), std::max(k0,k1), f0, f1);
      kappa_converged = true;
    } else {
      // Aitken with fallbacks
      for (it_kappa = 0; it_kappa < max_iter_kappa; ++it_kappa) {
        // Adaptive jump factor
        double adaptive_jump = jump_factor / (1.0 + 0.1 * it_kappa);

        double k2 = F_k(k1);
        k2 = std::min(k2, adaptive_jump*std::max(k1,1.0));
        k2 = std::max(k_min, std::min(k_max, k2));
        double f2 = f_of_k(k2);

        // Aitken Δ² proposal
        double d1 = k1 - k0, d2 = k2 - k1, denom = (d2 - d1);
        double k_acc = (std::fabs(denom) > 1e-14) ? k0 - (d1*d1)/denom : k2;

        if (!(k_acc > 0) || !std::isfinite(k_acc))
          k_acc = k2;
        k_acc = std::min(k_acc, adaptive_jump*std::max(k2,1.0));
        k_acc = std::max(k_min, std::min(k_max, k_acc));

        // Residual decrease with backtracking
        double k_prop = k_acc;
        double f_prop = f_of_k(k_prop);
        int bt = 0;
        while ( (std::fabs(f_prop) >= std::fabs(f1) || !std::isfinite(f_prop)) && bt < 8 ) {
          k_prop = 0.5*(k_prop + k1);
          f_prop = f_of_k(k_prop);
          ++bt;
        }

        // Check if Aitken improved
        if (!(std::fabs(f_prop) < std::fabs(f1))) {
          bad_accel++;

          // STRATEGY 1: Try secant if we have two distinct points
          if (bad_accel == 1 && std::fabs(f1 - f0) > 1e-14) {
            double k_sec = k1 - f1 * (k1 - k0) / (f1 - f0);
            k_sec = std::max(k_min, std::min(k_max, k_sec));

            double f_sec = f_of_k(k_sec);
            if (std::isfinite(f_sec) && std::fabs(f_sec) < std::fabs(f1)) {
              k_prop = k_sec;
              f_prop = f_sec;
              bad_accel = 0;
            }
          }

          // STRATEGY 2: Try to bracket from recent points
          if (bad_accel >= 1) {
            double a = std::min(k1, k2), b = std::max(k1, k2);
            double fa = f1, fb = f2;
            if (fa*fb < 0) {
              k = brent(a, b, fa, fb);
              kappa_converged = true;
              break;
            }
          }

          // STRATEGY 3: Adaptive bracket search
          if (bad_accel >= 1) {
            auto bracket = find_bracket(k1);
            if (bracket.first > 0) {
              double flo = f_of_k(bracket.first);
              double fhi = f_of_k(bracket.second);
              k = brent(bracket.first, bracket.second, flo, fhi);
              kappa_converged = true;
              break;
            }
          }

          // Fallback to plain fixed-point step
          k_prop = k2;
          f_prop = f2;
        } else {
          bad_accel = 0;
        }

        // Check convergence
        if (std::fabs(k_prop - k1) < eps_theta * (1.0 + k1)) {
          k = k_prop;
          kappa_converged = true;
          break;
        }

        // Shift window
        k0 = k1; f0 = f1;
        k1 = k_prop; f1 = f_prop;
      }

      if (!kappa_converged) k = std::max(k_min, k1);
    }

    // Store results (thread-safe: each thread writes to different index)
    mu_beta_out.col(r) = mu_beta;
    kappa_out[r] = 1.0 / k;
    beta_converged_out[r] = beta_converged;
    kappa_converged_out[r] = kappa_converged;
    beta_iters_out[r] = it_beta + 1;
    kappa_iters_out[r] = it_kappa + 1;
    used_y_table_out[r] = use_table;
  }

  // Convert std::vector<bool> to LogicalVector for R
  LogicalVector beta_conv_r(n_replicates);
  LogicalVector kappa_conv_r(n_replicates);
  LogicalVector used_table_r(n_replicates);
  IntegerVector beta_iters_r(n_replicates);
  IntegerVector kappa_iters_r(n_replicates);

  for (int r = 0; r < n_replicates; ++r) {
    beta_conv_r[r] = beta_converged_out[r];
    kappa_conv_r[r] = kappa_converged_out[r];
    used_table_r[r] = used_y_table_out[r];
    beta_iters_r[r] = beta_iters_out[r];
    kappa_iters_r[r] = kappa_iters_out[r];
  }

  return List::create(
    Named("mu_beta") = mu_beta_out,
    Named("kappa") = kappa_out,
    Named("beta_converged") = beta_conv_r,
    Named("kappa_converged") = kappa_conv_r,
    Named("beta_iters") = beta_iters_r,
    Named("kappa_iters") = kappa_iters_r,
    Named("used_y_table") = used_table_r
  );
}
