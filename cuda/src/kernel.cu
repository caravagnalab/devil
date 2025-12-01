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
        int idx = g * cells + j;
        mu[idx] = expf(eta[idx] + offset[j]);
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
