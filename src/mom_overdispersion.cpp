
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector estimate_mom_dispersion_cpp(const NumericMatrix& count_matrix,
                                            const NumericMatrix& design_matrix,
                                            const NumericMatrix& beta_matrix,
                                            const NumericVector& sf) {
  int G         = count_matrix.nrow();   // genes
  int n         = count_matrix.ncol();   // cells
  int n_design  = design_matrix.nrow();
  int p         = design_matrix.ncol();

  if (n_design != n)
    stop("design_matrix must have nrow == ncol(count_matrix)");
  if (beta_matrix.nrow() != G)
    stop("beta_matrix must have nrow == nrow(count_matrix)");
  if (beta_matrix.ncol() != p)
    stop("beta_matrix must have ncol == ncol(design_matrix)");
  if (sf.size() != n)
    stop("sf must have length ncol(count_matrix)");

  double corr = static_cast<double>(n) / (n - p);

  NumericVector theta(G);

  for (int g = 0; g < G; ++g) {
    double num = 0.0;
    double den = 0.0;

    for (int j = 0; j < n; ++j) {
      // linear predictor for gene g, cell j
      double eta = 0.0;
      for (int k = 0; k < p; ++k) {
        eta += design_matrix(j, k) * beta_matrix(g, k);
      }

      double mu   = sf[j] * std::exp(eta);
      double y    = count_matrix(g, j);
      double diff = y - mu;

      num += diff * diff - mu;
      den += mu * mu;
    }

    double th = 0.0;
    if (den > 0.0) {
      th = corr * num / den;
      if (th < 0.0) th = 0.0;  // truncate to non-negative
    }
    theta[g] = th;
  }

  return theta;
}
