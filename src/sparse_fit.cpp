#include <RcppEigen.h>
#include <map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
// [[Rcpp::depends(RcppEigen)]]
using namespace Eigen;

// Force inlining on GCC/Clang so helpers stay in the same instruction stream
// as the gene loop (prevents call overhead, enables cross-function Eigen opts).
#if defined(__GNUC__) || defined(__clang__)
#  define DEVIL_INLINE __attribute__((always_inline)) inline
#else
#  define DEVIL_INLINE inline
#endif

// ── Design-matrix compression ─────────────────────────────────────────────────
// Finds K unique rows of X and assigns each of the N cells a 0-based group id.
// Also precomputes K outer products x_k x_k^T (reused in every IRLS iteration
// and in the Hessian, reducing the dominant O(N·p²) cost to O(K·p²) + O(N)).
// Called once per cpu_fit_sparse_cpp() invocation; results are read-only
// inside the gene loop and safe to share across OpenMP threads.
static void compress_design(
    const MatrixXd& X,
    MatrixXd& X_unique,
    VectorXi& idx,
    std::vector<MatrixXd>& outers)
{
    int N = X.rows(), p = X.cols();
    std::map<std::vector<double>, int> row_to_id;
    std::vector<std::vector<double>> unique_rows;
    idx.resize(N);

    for (int i = 0; i < N; ++i) {
        std::vector<double> rv(p);
        for (int j = 0; j < p; ++j) rv[j] = X(i, j);
        auto [it, inserted] = row_to_id.emplace(rv, (int)unique_rows.size());
        if (inserted) unique_rows.push_back(rv);
        idx[i] = it->second;
    }

    int K = (int)unique_rows.size();
    X_unique.resize(K, p);
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < p; ++j)
            X_unique(k, j) = unique_rows[k][j];

    outers.resize(K);
    for (int k = 0; k < K; ++k)
        outers[k] = X_unique.row(k).transpose() * X_unique.row(k);  // p×p
}

// ── Compressed IRLS ───────────────────────────────────────────────────────────
// Same NB-IRLS as beta_fit() but exploits the K<<N unique-row structure of X.
// Per iteration:
//   • K exp() calls instead of N  (dominant transcendental cost)
//   • O(N)    scatter to accumulate per-group weight sums
//   • O(K·p²) Gram matrix via precomputed outer products
//   • O(K·p)  gradient via K unique rows
// k_disp: overdispersion theta; precision k = 1/theta used internally.
// exp_neg_off: exp(-offset) per cell, precomputed before the gene loop.
DEVIL_INLINE static VectorXd irls_gene_fast(
    const VectorXd& y,
    const MatrixXd& X_unique,
    const MatrixXd& X_unique_T,          // transpose cached once outside gene loop
    const VectorXi& idx,
    const std::vector<MatrixXd>& outers,
    const VectorXd& exp_neg_off,
    VectorXd beta,
    float k_disp, int max_iter, float eps, int& iter_out)
{
    double k_inv = (k_disp > 0.0f) ? 1.0 / k_disp : 1e8;
    int N = y.size(), K = X_unique.rows(), p = X_unique.cols();

    VectorXd exp_neg_Xb(K);
    VectorXd group_wsum(K), group_resid(K);
    MatrixXd G(p, p);
    VectorXd grad(p), delta(p);

    iter_out = 0;
    bool converged = false;

    while (!converged && iter_out < max_iter) {
        // Step 1: K exp() calls (vs N in the dense path)
        exp_neg_Xb = (-X_unique * beta).array().exp();

        // Step 2: O(N) scatter — accumulate per-group weight and residual sums
        group_wsum.setZero();
        group_resid.setZero();
        for (int i = 0; i < N; ++i) {
            int ki = idx[i];
            double w_i  = exp_neg_Xb[ki] * exp_neg_off[i];
            double mw_i = ((k_inv + y[i]) / (1.0 + k_inv * w_i)) * w_i;
            group_wsum[ki]  += mw_i;
            group_resid[ki] += mw_i - 1.0;
        }

        // Step 3: O(K·p²) Gram matrix via precomputed outer products
        G.setZero();
        for (int k = 0; k < K; ++k)
            G.noalias() += group_wsum[k] * outers[k];

        // Step 4: O(K·p) gradient
        grad.noalias() = X_unique_T * group_resid;

        // Step 5: Newton step (LDLT more stable than explicit inverse)
        delta = (k_inv * G).ldlt().solve(k_inv * grad);
        beta += delta;

        converged = std::isnan(delta[0]) || delta.cwiseAbs().maxCoeff() < eps;
        ++iter_out;
    }
    return beta;
}

// ── Compressed Hessian inverse ────────────────────────────────────────────────
// Replaces O(N·p²) X^T diag(s) X with O(K·p²) sum_k(s_k * outer_k)
// where s_k = sum_{i in group k} s_i (O(N) scatter).
// sf: exp(offset) per cell, precomputed before the gene loop.
DEVIL_INLINE static MatrixXd hessian_inv_gene_fast(
    const VectorXd& beta, double alpha,
    const VectorXd& y,
    const MatrixXd& X_unique,
    const VectorXi& idx,
    const std::vector<MatrixXd>& outers,
    const VectorXd& sf)
{
    int N = y.size(), K = X_unique.rows(), p = X_unique.cols();

    // K group exp(X_unique * beta), then scatter sf_i per cell
    VectorXd exp_Xb_k = (X_unique * beta).array().exp();

    VectorXd group_s_sum = VectorXd::Zero(K);
    for (int i = 0; i < N; ++i) {
        double mu_i  = sf[i] * exp_Xb_k[idx[i]];
        double denom = 1.0 + alpha * mu_i;
        group_s_sum[idx[i]] += (y[i] * alpha + 1.0) * mu_i / (denom * denom);
    }

    MatrixXd H = MatrixXd::Zero(p, p);
    for (int k = 0; k < K; ++k)
        H.noalias() -= group_s_sum[k] * outers[k];

    return (-H).inverse();
}

// ── Compressed MOM overdispersion ─────────────────────────────────────────────
// O(K·p) matrix mul for group exp values + O(N) scatter for residuals.
DEVIL_INLINE static double mom_disp_gene_fast(
    const VectorXd& y,
    const MatrixXd& X_unique,
    const VectorXi& idx,
    const VectorXd& beta,
    const VectorXd& sf, int p_pred)
{
    int N = y.size(), K = X_unique.rows();

    VectorXd exp_Xb_k = (X_unique * beta).array().exp();

    double num = 0.0, den = 0.0;
    for (int i = 0; i < N; ++i) {
        double mu_i = sf[i] * exp_Xb_k[idx[i]];
        double res  = y[i] - mu_i;
        num += res * res - mu_i;
        den += mu_i * mu_i;
    }
    if (den <= 0.0) return 0.0;
    double th = (static_cast<double>(N) / (N - p_pred)) * num / den;
    return (th < 0.0) ? 0.0 : th;
}

// ── Clustered sandwich meat ───────────────────────────────────────────────────
// Cluster structure is orthogonal to design-row groups, so we keep the full
// N-row design matrix here.  This is called once per gene (not per IRLS iter)
// so its O(N·p) cost is dominated by the O(N·p²·iters) IRLS savings.
DEVIL_INLINE static MatrixXd clustered_meat_gene(
    const MatrixXd& X, const VectorXd& y,
    const VectorXd& beta, double overdispersion,
    const VectorXd& sf, const VectorXi& cluster_ends)
{
    int n = X.rows(), k = X.cols(), ng = cluster_ends.size();
    MatrixXd cluster_sums = MatrixXd::Zero(ng, k);
    int start = 0;
    for (int g = 0; g < ng; ++g) {
        int end = cluster_ends(g), bsz = end - start;
        if (bsz > 0) {
            auto Xb  = X.block(start, 0, bsz, k);
            auto yb  = y.segment(start, bsz).array();
            auto sfb = sf.segment(start, bsz).array();
            ArrayXd mu  = sfb * (Xb * beta).array().exp();
            VectorXd wr = ((yb - mu) / (1.0 + mu * overdispersion)).matrix();
            cluster_sums.row(g) = Xb.transpose() * wr;
        }
        start = end;
    }
    double adj = (ng > 1) ? double(ng) / (ng - 1.0) : 1.0;
    return (adj / n) * (cluster_sums.transpose() * cluster_sums);
}

// ── Exported function ─────────────────────────────────────────────────────────

//' Fit NB model for all genes using a sparse count matrix (CPU)
//'
//' @param sparse_count_matrix A dgCMatrix (genes x cells).
//' @param design_matrix Dense numeric matrix (cells x predictors).
//' @param offset_vector Numeric vector of length ncells.
//' @param init_dispersion Numeric vector of initial overdispersion per gene.
//' @param beta_init Numeric matrix of initial beta coefficients (genes x predictors).
//' @param fit_mom Logical: TRUE for MOM overdispersion, FALSE for Poisson (theta=0).
//' @param cluster_blocks_indexes Integer vector of cumulative cluster end indices, or NULL.
//' @param max_iter Maximum IRLS iterations.
//' @param tolerance Convergence tolerance.
//' @param n_threads Number of OpenMP threads (default 1).
//' @return A list with beta, theta, iter, beta_sandwiches_null, beta_sandwiches.
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List cpu_fit_sparse_cpp(
    SEXP                  sparse_count_matrix,
    Eigen::MatrixXd       design_matrix,
    Eigen::VectorXd       offset_vector,
    Eigen::VectorXd       init_dispersion,
    Eigen::MatrixXd       beta_init,
    bool                  fit_mom,
    Rcpp::Nullable<Rcpp::IntegerVector> cluster_blocks_indexes,
    int                   max_iter,
    double                tolerance,
    int                   n_threads = 1
) {
    // Map dgCMatrix slots without copying data
    Rcpp::S4            Y_s4(sparse_count_matrix);
    Rcpp::IntegerVector dims  = Y_s4.slot("Dim");
    Rcpp::IntegerVector p_vec = Y_s4.slot("p");
    Rcpp::IntegerVector i_vec = Y_s4.slot("i");
    Rcpp::NumericVector x_vec = Y_s4.slot("x");

    int ngenes = dims[0], ncells = dims[1], nnz = x_vec.size();

    // Column-major (CSC) map — zero copy
    Eigen::Map<const SparseMatrix<double>> Y_csc(
        ngenes, ncells, nnz,
        p_vec.begin(), i_vec.begin(), x_vec.begin());

    // Convert CSC → CSR once (O(nnz)) for efficient per-gene row access
    SparseMatrix<double, RowMajor> Y_csr(Y_csc);

    // ── Precompute shared quantities (once, before gene loop) ────────────────
    // Design matrix compression: K unique rows, cell→group mapping, K outer products
    MatrixXd X_unique;
    VectorXi idx;
    std::vector<MatrixXd> outers;
    compress_design(design_matrix, X_unique, idx, outers);
    MatrixXd X_unique_T = X_unique.transpose();   // p×K, avoids per-gene transpose

    int K = X_unique.rows();
    int npred = design_matrix.cols();
    float tol_f = static_cast<float>(tolerance);

    // exp(±offset): sf for hessian/MOM; exp_neg_off for IRLS scatter
    VectorXd sf          = offset_vector.array().exp();
    VectorXd exp_neg_off = (-offset_vector).array().exp();

    // Cluster info
    bool has_clusters = !cluster_blocks_indexes.isNull();
    VectorXi cl_blocks;
    if (has_clusters)
        cl_blocks = Rcpp::as<VectorXi>(cluster_blocks_indexes.get());

    // Pre-allocate outputs (separate per-gene vectors for thread safety)
    MatrixXd            beta_result(ngenes, npred);
    VectorXd            theta_result(ngenes);
    Rcpp::IntegerVector iter_result(ngenes);
    std::vector<MatrixXd> s_null(ngenes), s_clust(ngenes);

#ifdef _OPENMP
    omp_set_num_threads(n_threads);
    #pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int gene = 0; gene < ngenes; ++gene) {
        // Materialise only this gene's sparse row as a dense vector
        VectorXd y_i = VectorXd(Y_csr.row(gene));

        int iters = 0;
        VectorXd beta_i = irls_gene_fast(
            y_i, X_unique, X_unique_T, idx, outers, exp_neg_off,
            beta_init.row(gene).transpose(),
            static_cast<float>(init_dispersion(gene)),
            max_iter, tol_f, iters);

        double theta_i = 0.0;
        if (fit_mom)
            theta_i = mom_disp_gene_fast(y_i, X_unique, idx, beta_i, sf, npred);

        MatrixXd H_inv = hessian_inv_gene_fast(
            beta_i, theta_i, y_i, X_unique, idx, outers, sf);

        beta_result.row(gene) = beta_i.transpose();
        theta_result(gene)    = theta_i;
        iter_result(gene)     = iters;
        s_null[gene]          = H_inv;

        if (has_clusters) {
            MatrixXd M    = clustered_meat_gene(
                design_matrix, y_i, beta_i, theta_i, sf, cl_blocks);
            s_clust[gene] = (H_inv * M * H_inv) * ncells;
        }
    }

    Rcpp::List r_s_null(ngenes), r_s_clust(ngenes);
    for (int gene = 0; gene < ngenes; ++gene) {
        r_s_null[gene]  = Rcpp::wrap(s_null[gene]);
        r_s_clust[gene] = has_clusters ? Rcpp::wrap(s_clust[gene]) : R_NilValue;
    }

    return Rcpp::List::create(
        Rcpp::Named("beta")                 = beta_result,
        Rcpp::Named("theta")                = theta_result,
        Rcpp::Named("iter")                 = iter_result,
        Rcpp::Named("beta_sandwiches_null") = r_s_null,
        Rcpp::Named("beta_sandwiches")      = r_s_clust
    );
}
