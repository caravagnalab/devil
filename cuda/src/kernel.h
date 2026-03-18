__global__ void expGPU(float *A, float *B,float *C,std::size_t elem_count,std::size_t cells);

  __global__ void process2D(float *k, float* Y,float* w_q,float* mu_g, int genes, int cells);

  __global__ void elementWise(float *mu_g, float *w_g, std::size_t elem_count);

  __global__ void elementWiseSub(float *mu_g, std::size_t elem_count);

__global__ void final1D(float *mu_beta,float* delta, std::size_t elem_count);

// Kernels for dispersion estimation
__global__ void compute_row_means(const float* mat, float* means, int rows, int cols);

__global__ void compute_row_variances(const float* mat, const float* means, float* vars, int rows, int cols);

__global__ void compute_dispersion(const float* means, const float* vars, float offset_inv, float* k, int rows);

// Kernels for MOM overdispersion estimation
__global__ void compute_mu_from_eta(const float* eta, const float* offset, 
                                     float* mu, int genes, int cells);

__global__ void compute_mom_components(const float* Y, const float* mu,
                                        float* diff_sq_minus_mu, float* mu_sq,
                                        int genes, int cells);

__global__ void compute_theta_from_num_den(const float* num, const float* den,
                                            float corr, float* theta, int genes);

// Rough beta initialization
__global__ void init_beta_rough_kernel(const float* means, float* beta, 
                                        int genes, int features);

// Hessian weight: s_gi = (y*theta + 1)*mu / (1 + theta*mu)^2
// theta[g] is overdispersion (phi), mu already = sf*exp(eta)
__global__ void compute_hessian_weights(
    const float* theta, const float* Y, const float* mu_g,
    float* hess_w, int genesBatch, int cells);

// Cluster sums and score residuals computed inline:
// S[g, cl, f] = sum_{c in cl} [(y-mu)/(1+mu*theta)] * X[f,c]
// Output layout: [features x n_clusters x genesBatch] col-major
__global__ void compute_cluster_sums_and_scores(
    const float* Y, const float* mu,
    const float* X, const float* theta,
    const int* cluster_ends,
    float* out,
    int genesBatch, int cells, int features, int n_clusters);

// In-place negate
__global__ void negate_kernel(float* x, int n);

// ── Summary-space kernels ─────────────────────────────────────────────────────

__global__ void aggregate_y_by_group(
    const float* __restrict__ Y,
    const int*   __restrict__ mapping,
    float*       __restrict__ y_sums,
    float*       __restrict__ y_sq_sums,
    int N, int M, int genesBatch);

__global__ void process2D_summary(
    const float* __restrict__ k,
    const float* __restrict__ y_sums,
    const float* __restrict__ counts,
    const float* __restrict__ w_q,
    float*       __restrict__ weight,
    int genesBatch, int M);

__global__ void elementWiseSub_summary(
    float*       __restrict__ weight,
    const float* __restrict__ counts,
    int genesBatch, int M);

__global__ void compute_mom_components_summary(
    const float* __restrict__ y_sums,
    const float* __restrict__ y_sq_sums,
    const float* __restrict__ mu_mom,
    const float* __restrict__ counts,
    float*       __restrict__ d_num,
    float*       __restrict__ d_den,
    int genesBatch, int M);

__global__ void compute_hessian_weights_summary(
    const float* __restrict__ theta,
    const float* __restrict__ y_sums,
    const float* __restrict__ counts,
    const float* __restrict__ mu_mom,
    float*       __restrict__ hess_w,
    int genesBatch, int M);

__global__ void compute_cluster_sums_summary(
    const float* __restrict__ y_sums,
    const float* __restrict__ counts,
    const float* __restrict__ mu_mom,
    const float* __restrict__ X_unique,
    const float* __restrict__ theta,
    const int*   __restrict__ cluster_map,
    float*       __restrict__ cluster_sums,
    int genesBatch, int M, int features, int n_clusters);

__global__ void expGPU_neg(const float* __restrict__ eta,
                           const float* __restrict__ off,
                           float*       __restrict__ w_q,
                           int total, int M, int genesBatch);