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
