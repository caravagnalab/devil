__global__ void expGPU(float *A, float *B,float *C,std::size_t elem_count,std::size_t cells) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x < elem_count) {
      //printf("TH id %d %f \n",x,A[x]);
      C[x] = exp(-A[x] - B[x % cells]);
    }
}

__global__ void process2D(float *k, float* Y,float* w_q,float* mu_g, int genes, int cells) {
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < genes && j < cells) {
      const int ij = i * cells + j;
      mu_g[ij] =( k[i]+Y[ij] )/( 1+(k[i]*w_q[ij]) );
    }
}

__global__ void elementWise(float *mu_g, float *w_g, std::size_t elem_count) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x<elem_count)
      mu_g[x]=mu_g[x]*w_g[x];
}

__global__ void elementWiseSub(float *mu_g, std::size_t elem_count) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x<elem_count)
      mu_g[x]=mu_g[x]-1;
}

__global__ void final1D(float *mu_beta,float* delta, std::size_t elem_count) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x<elem_count)
      mu_beta[x]=mu_beta[x]+delta[x];
}

// Compute row means: mean[i] = sum(mat[i,:]) / cols
// Uses double precision for intermediate calculations
__global__ void compute_row_means(const float* mat, float* means, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        //WARNING: DONT CHANGE TO FLOAT, KEEP DOUBLE PRECISION FOR ACCURACY, if you want implement kahan summation, it works, tested! 
        double sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            sum += (double)mat[row * cols + col];
        }
        means[row] = (float)(sum / cols);
    }
}

// Compute row variances: var[i] = sum((mat[i,:] - mean[i])^2) / (cols - 1)
// Uses sample variance (n-1 denominator) to match R's rowVars behavior
// Uses double precision for intermediate calculations
__global__ void compute_row_variances(const float* mat, const float* means, float* vars, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        //WARNING: DONT CHANGE TO FLOAT, KEEP DOUBLE PRECISION FOR ACCURACY, if you want implement kahan summation, it works, tested!
        double sum_sq = 0.0;
        double mean = (double)means[row];
        for (int col = 0; col < cols; ++col) {
            double diff = (double)mat[row * cols + col] - mean;
            sum_sq += diff * diff;
        }
        // Use sample variance (n-1) to match R's rowVars
        vars[row] = (float)(sum_sq / (cols - 1));
    }
}

// Compute dispersion: k[i] = (var[i] - offset_inv * mean[i]) / (mean[i]^2)
// k[i] = max(0.01, k[i])  (clamp to avoid negative/invalid values)
__global__ void compute_dispersion(const float* means, const float* vars, float offset_inv, float* k, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        float mean = means[i];
        float var = vars[i];
        float disp = (var - offset_inv * mean) / (mean * mean);
        
        // Clamp to 0.01 if NaN or negative
        if (isnan(disp) || disp < 0.01f) {
            disp = 0.01f;
        }
        
        // Store 1/k as that's what the algorithm uses
        k[i] = 1.0f / disp;
    }
}

// Compute mu from eta: mu[g,j] = exp(eta[g,j] + offset[j])
// Note: offset contains log(size_factors), eta is linear predictor
// Therefore: mu = sf * exp(eta) = exp(log(sf) + eta) = exp(eta + offset)
__global__ void compute_mu_from_eta(const float* eta, const float* offset, 
                                     float* mu, int genes, int cells) {
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (g < genes && j < cells) {
      int idx_row = g * cells + j;
      int idx_col = g + j * genes;
      mu[idx_col] = expf(eta[idx_row] + offset[j]);
    }
}

// Compute MOM components: diff_sq_minus_mu[g,j] = (Y[g,j] - mu[g,j])² - mu[g,j]
//                         mu_sq[g,j] = mu[g,j]²
__global__ void compute_mom_components(const float* Y, const float* mu,
                                        float* diff_sq_minus_mu, float* mu_sq,
                                        int genes, int cells) {
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (g < genes && j < cells) {
        int idx = g * cells + j;
        float y_val = Y[idx];
        float mu_val = mu[idx];
        float diff = y_val - mu_val;
        
        diff_sq_minus_mu[idx] = diff * diff - mu_val;
        mu_sq[idx] = mu_val * mu_val;
    }
}

// Final theta computation: theta[g] = max(0, corr * num[g] / den[g])
__global__ void compute_theta_from_num_den(const float* num, const float* den,
                                            float corr, float* theta, int genes) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (g < genes) {
        float th = 0.0f;
        if (den[g] > 0.0f) {
            th = corr * num[g] / den[g];
            if (th < 0.0f) th = 0.0f;
        }
        theta[g] = th;
    }
}

// Rough beta initialization: beta[g, 0] = log1p(mean[g]), beta[g, f>0] = 0
__global__ void init_beta_rough_kernel(const float* means, float* beta, 
                                        int genes, int features) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < genes) {
        // Set first column to log1p(mean)
        beta[g * features] = log1pf(means[g]);
        // Set remaining columns to zero
        for (int f = 1; f < features; ++f) {
            beta[g * features + f] = 0.0f;
        }
    }
}

__global__ void compute_hessian_weights(
    const float* theta, const float* Y, const float* mu_g,
    float* hess_w, int genesBatch, int cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= genesBatch * cells) return;
    int g       = idx / cells;         // row-major: gene is slow axis
    int c       = idx % cells;         // row-major: cell is fast axis
    float alpha = theta[g];
    float mu    = mu_g[g * cells + c]; // row-major: [genesBatch x cells]
    float y     = Y   [g * cells + c]; // row-major: [genesBatch x cells]
    float denom = 1.0f + alpha * mu;
    hess_w[idx] = (y * alpha + 1.0f) * mu / (denom * denom);
}

__global__ void compute_cluster_sums_and_scores(
    const float* Y,            // [genesBatch x cells] col-major: Y[g,c] = g + c*genesBatch
    const float* mu,           // [genesBatch x cells] col-major: same layout
    const float* X,            // [features x cells]   col-major: X[f,c] = f + c*features
    const float* theta,        // [genesBatch]
    const int*   cluster_ends, // [n_clusters] cumulative end indices
    float*       out,          // [features x n_clusters x genesBatch] col-major
    int genesBatch, int cells, int features, int n_clusters)
{
    int f  = blockIdx.x * blockDim.x + threadIdx.x;
    int cl = blockIdx.y * blockDim.y + threadIdx.y;
    int g  = blockIdx.z;
    if (f >= features || cl >= n_clusters || g >= genesBatch) return;

    int start  = (cl == 0) ? 0 : cluster_ends[cl - 1];
    int end    = cluster_ends[cl];
    float th   = theta[g];
    float sum  = 0.0f;

    for (int c = start; c < end; ++c) {
        float mu_gc    = mu[g * cells + c];        // row-major: [genesBatch x cells]
        float y_gc     = Y [g * cells + c];        // row-major: [genesBatch x cells]
        float score_gc = (y_gc - mu_gc) / (1.0f + mu_gc * th);
        sum += score_gc * X[f + c * features];     // col-major: [features x cells] — unchanged
    }

    // Output: [features x n_clusters x genesBatch] col-major
    // element [f, cl, g] = f + cl*features + g*features*n_clusters
    out[f + cl * features + g * features * n_clusters] = sum;
}

__global__ void negate_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = -x[i];
}

// Reduce raw Y [N x genesBatch] col-major into y_sums and y_sq_sums
// [M x genesBatch] col-major, both zeroed before this call.
// mapping is 1-based, length N.
__global__ void aggregate_y_by_group(
    const float* __restrict__ Y,          // [N x genesBatch] col-major
    const int*   __restrict__ mapping,    // [N] 1-based group index
    float*       __restrict__ y_sums,     // [M x genesBatch] col-major
    float*       __restrict__ y_sq_sums,  // [M x genesBatch] col-major
    int N, int M, int genesBatch)
{
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  if (g >= genesBatch || n >= N) return;

  int m   = mapping[n] - 1;              // 0-based
  float v = Y[n + g * N];
  atomicAdd(&y_sums   [m * genesBatch + g], v);
  atomicAdd(&y_sq_sums[m * genesBatch + g], v * v);
}

// IRLS inner step: compute weight = mu_g_sum * w_q in M-space.
// Replaces process2D + elementWise for the summary case.
// w_q    [genesBatch x M] col-major (output of expGPU in M-space)
// y_sums [M x genesBatch] col-major
// counts [M]
// weight [genesBatch x M] col-major  ← output
__global__ void process2D_summary(
    const float* __restrict__ k,       // [genesBatch]  inv-dispersion
    const float* __restrict__ y_sums,  // [M x genesBatch] col-major
    const float* __restrict__ counts,  // [M]
    const float* __restrict__ w_q,     // [genesBatch x M] col-major
    float*       __restrict__ weight,  // [genesBatch x M] col-major
    int genesBatch, int M)
{
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (g >= genesBatch || m >= M) return;

  float inv_k = k[g];
  float wq    = w_q [g + m * genesBatch];         // [genesBatch x M] col-major
  float ys    = y_sums[g * M + m];                // wait — see note below
  // NOTE on layout: y_sums is [M x genesBatch] col-major → element (m,g) = y_sums[m + g*M]
  // Re-index correctly:
  ys          = y_sums[m + g * M];
  float cnt   = counts[m];
  float mu_gs = (cnt * inv_k + ys) / (1.0f + inv_k * wq);
  weight[g + m * genesBatch] = mu_gs * wq;
}

// Subtract counts[m] from weight[g,m] in-place (gradient step, replaces elementWiseSub).
__global__ void elementWiseSub_summary(
    float*       __restrict__ weight,  // [genesBatch x M] col-major, in-place
    const float* __restrict__ counts,  // [M]
    int genesBatch, int M)
{
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (g >= genesBatch || m >= M) return;
  weight[g + m * genesBatch] -= counts[m];
}

// MOM components in M-space. Replaces compute_mom_components.
// mu_mom [genesBatch x M] col-major  (sf * exp(eta), already computed)
// y_sums, y_sq_sums [M x genesBatch] col-major
// counts [M]
// d_num, d_den [genesBatch x M] col-major  ← outputs
__global__ void compute_mom_components_summary(
    const float* __restrict__ y_sums,    // [M x genesBatch]
    const float* __restrict__ y_sq_sums, // [M x genesBatch]
    const float* __restrict__ mu_mom,    // [genesBatch x M]
    const float* __restrict__ counts,    // [M]
    float*       __restrict__ d_num,     // [genesBatch x M]
    float*       __restrict__ d_den,     // [genesBatch x M]
    int genesBatch, int M)
{
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (g >= genesBatch || m >= M) return;

  float mu  = mu_mom [g + m * genesBatch];
  float cnt = counts [m];
  float ys  = y_sums [m + g * M];
  float ys2 = y_sq_sums[m + g * M];

  float sse = ys2 - 2.0f * mu * ys + cnt * mu * mu;
  d_num[g + m * genesBatch] = sse - cnt * mu;
  d_den[g + m * genesBatch] = cnt * mu * mu;
}

// Hessian weights in M-space. Replaces compute_hessian_weights.
// weight_out = mu / (1 + alpha*mu)^2 * (alpha*y_sum + count)
__global__ void compute_hessian_weights_summary(
    const float* __restrict__ theta,     // [genesBatch]
    const float* __restrict__ y_sums,    // [M x genesBatch]
    const float* __restrict__ counts,    // [M]
    const float* __restrict__ mu_mom,    // [genesBatch x M]
    float*       __restrict__ hess_w,    // [genesBatch x M]
    int genesBatch, int M)
{
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (g >= genesBatch || m >= M) return;

  float alpha = theta[g];
  float mu    = mu_mom [g + m * genesBatch];
  float denom = 1.0f + alpha * mu;
  float ys    = y_sums [m + g * M];
  float cnt   = counts [m];
  hess_w[g + m * genesBatch] = (mu / (denom * denom)) * (alpha * ys + cnt);
}

// Clustered meat in M-space.
// Each group m belongs to exactly one cluster given by cluster_map[m] (1-based).
// Computes cluster_sums[f, cl, g] += score(g,m) * X[f,m]
// where score(g,m) = (y_sums[m,g] - counts[m]*mu[g,m]) / (1 + mu[g,m]*theta[g])
// Output cluster_sums: [features x n_clusters x genesBatch] col-major
__global__ void compute_cluster_sums_summary(
    const float* __restrict__ y_sums,       // [M x genesBatch]
    const float* __restrict__ counts,       // [M]
    const float* __restrict__ mu_mom,       // [genesBatch x M]
    const float* __restrict__ X_unique,     // [features x M] col-major
    const float* __restrict__ theta,        // [genesBatch]
    const int*   __restrict__ cluster_map,  // [M] 1-based cluster index
    float*       __restrict__ cluster_sums, // [features x n_clusters x genesBatch]
    int genesBatch, int M, int features, int n_clusters)
{
  int f  = blockIdx.x * blockDim.x + threadIdx.x;
  int g  = blockIdx.y * blockDim.y + threadIdx.y;
  if (f >= features || g >= genesBatch) return;

  for (int m = 0; m < M; ++m) {
    int cl   = cluster_map[m] - 1;          // 0-based cluster
    float mu = mu_mom [g + m * genesBatch];
    float ys = y_sums [m + g * M];
    float cnt= counts [m];
    float th = theta  [g];
    float score = (ys - cnt * mu) / (1.0f + mu * th);
    float xfm   = X_unique[f + m * features]; // [features x M] col-major
    // cluster_sums layout: [features x n_clusters x genesBatch] col-major
    // index: f + cl*features + g*features*n_clusters
    atomicAdd(&cluster_sums[f + cl * features + g * features * n_clusters],
              score * xfm);
  }
}

__global__ void expGPU_neg(const float* __restrict__ eta,
                            const float* __restrict__ off,
                            float*       __restrict__ w_q,
                            int total, int M, int genesBatch)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int g = idx / M;                              // gene index
  int m = idx % M;                              // group index
  // eta:  [genesBatch x M] row-major  → element (g,m) = eta[g*M + m] = eta[idx]
  // w_q:  [genesBatch x M] col-major  → element (g,m) = w_q[g + m*genesBatch]
  w_q[g + m * genesBatch] = expf(-eta[idx] - off[m]);
}