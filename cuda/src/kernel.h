__global__ void expGPU(float *A, float *B,float *C,std::size_t elem_count,std::size_t cells);

  __global__ void process2D(float *k, float* Y,float* w_q,float* mu_g, int genes, int cells);

  __global__ void elementWise(float *mu_g, float *w_g, std::size_t elem_count);

  __global__ void elementWiseSub(float *mu_g, std::size_t elem_count);

__global__ void final1D(float *mu_beta,float* delta, std::size_t elem_count);

// Kernels for dispersion estimation
__global__ void compute_row_means(const float* mat, float* means, int rows, int cols);

__global__ void compute_row_variances(const float* mat, const float* means, float* vars, int rows, int cols);

__global__ void compute_dispersion(const float* means, const float* vars, float offset_inv, float* k, int rows);
