#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thread>
#include "einsum.hpp"
#include "kernel.h"
#include "inverse.hpp"
#include "utils.hpp"
#include "cutensor.h"
#include <omp.h>
#include <Eigen/Dense>
#include <list>
#include "batch.hpp"

template <typename T>
struct CudaDeleter {
  void operator()(T* ptr) const {
    cudaFree(ptr);
  }
};

__global__ void printMatrix(const int rows,const int cols, float* const matrix) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%2.4f ", matrix[j*rows+i]);
    }
    printf("\n");
  }
}


__global__ void printMatrixT(const int rows,const int cols, float* const matrix) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%2.3f ", matrix[j*rows+i]);
    }
    printf("\n");
  }
}

template<typename T>
void toGPU(T vec,float* const vec_gpu) {
  CUDA_CHECK( cudaMemcpy(vec_gpu, vec.data(), vec.size() * sizeof(float), cudaMemcpyHostToDevice) );
}

BatchResult
beta_fit_gpu_external(
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const
  &Y_host,
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const
  &X_host,
  Eigen::VectorXf const  &offset_host,
  int max_iter, float eps, int batch_size,
  std::vector<int>& iterations, bool TEST,
  const std::vector<int>& cluster_ends,
  int n_clusters) {
  
  /******************************
   * Shape definition
   ******************************/
  
  const std::size_t genes(Y_host.cols());
  const std::size_t cells(X_host.cols());
  const std::size_t features(X_host.rows());
  if (TEST) {
    std::cout << "X {"<<X_host.rows()<<","<<X_host.cols() <<"}\n";
    std::cout << "Y {" << Y_host.rows() << "," << Y_host.cols() << "}\n";
    std::cout << "offset {"<<offset_host.size()<<", 1" <<"}\n";
    std::cout << "Genes" << genes <<std::endl;
    std::cout << "Cells " << cells <<std::endl;
    std::cout << "Features " << features <<std::endl;
  }
  std::size_t genesBatch = batch_size;
  
  // Calculate mean(exp(offset_vector)) once on CPU
  double offset_sum = 0.0f;
  for (int i = 0; i < offset_host.size(); i++) {
    offset_sum += exp(offset_host[i]);
  }
  float offset_inv = 1.0f / (offset_sum / offset_host.size());
  
  std::vector<float> mu_beta_final(genes*features, 0.0);
  std::vector<float> k_final;
  if (TEST) {
    k_final.resize(genes, 0.0);  // Store final k values only if TEST mode
  }
  std::vector<float> theta_final(genes, 0.0);  // Store final theta (MOM overdispersion) values
  std::vector<float> hessian_final(genes * features * features, 0.0f);
  std::vector<float> meat_final(genes * features * features, 0.0f);
  std::vector<float> beta_init_final;
  if (TEST) {
    beta_init_final.resize(genes*features, 0.0);  // Debug: store initial beta values only if TEST mode
  }
  
  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  std::cout << "Detected " << deviceCount << " GPU(s)" << std::endl;
  for(int gpu=0;gpu<deviceCount;++gpu) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Device " << gpu << ": " << deviceProp.name << std::endl;
    std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
  }
  
  omp_set_num_threads(deviceCount);
  
  /************************
   * Create array of handles, variables, pointer, object and so on. 
   ***********************/
  std::vector<cublasHandle_t> cublasH(deviceCount);
  std::vector<cutensorHandle_t> cutensorH(deviceCount);
  
  //this will call a dummy constructor, don't worry about initialization!
  std::vector<EinsumWrapper> einsum_cg_tmp2(deviceCount);
  std::vector<EinsumWrapper> einsum_A(deviceCount);
  std::vector<EinsumWrapper> einsum_B(deviceCount);
  std::vector<EinsumWrapper> einsum_Bk(deviceCount);
  std::vector<EinsumWrapper> einsum_C(deviceCount);
  std::vector<EinsumWrapper> einsum_last(deviceCount);
  std::vector<EinsumWrapper> einsum_delta(deviceCount);
  
  std::vector<float*> X(deviceCount);
  std::vector<float*> Y(deviceCount);
  std::vector<float*> offset(deviceCount);
  std::vector<float*> mu_beta(deviceCount);
  std::vector<float *> k(deviceCount);
  std::vector<float *> cg_tmp(deviceCount);
  std::vector<float *> w_q(deviceCount);
  std::vector<float *> mu_g(deviceCount);
  std::vector<float *> workspace(deviceCount);
  
  std::vector<float **> Zigma_pointer(deviceCount);
  std::vector<float **> Bk_pointer(deviceCount);
  std::vector<float *> Zigma(deviceCount);
  
  // Allocate pivot and info arrays once per device
  std::vector<int *> pivot(deviceCount, nullptr);
  std::vector<int *> info(deviceCount, nullptr);
  
  // Allocate vectors for temporary buffers (per-device)
  std::vector<float *> d_means(deviceCount);
  std::vector<float *> d_vars(deviceCount);
  std::vector<float *> d_mu_mom(deviceCount);
  std::vector<float *> d_diff_sq_minus_mu(deviceCount);
  std::vector<float *> d_mu_sq(deviceCount);
  std::vector<float *> d_num(deviceCount);
  std::vector<float *> d_den(deviceCount);
  std::vector<float *> d_theta(deviceCount);
  std::vector<float *> d_ones(deviceCount);
  
  std::vector<float *> cg_tmp2(deviceCount);
  std::vector<float *>    A(deviceCount);
  std::vector<float *>    B(deviceCount);
  std::vector<float *>    C(deviceCount);
  std::vector<float *>    Bk(deviceCount);
  std::vector<float *>    delta(deviceCount);
  std::vector<float *>    last(deviceCount);
  
  // NEW: for Hessian and meat
  std::vector<float*> d_hess_w(deviceCount);
  // std::vector<float*> d_score_r(deviceCount);
  std::vector<float*> d_meat(deviceCount);
  std::vector<float*> d_cluster_sums(deviceCount);
  // Upload cluster_ends once (same for all batches and all devices)
  int* d_cluster_ends = nullptr;
  if (n_clusters > 0) {
    CUDA_CHECK(cudaMalloc(&d_cluster_ends, n_clusters * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_cluster_ends, cluster_ends.data(),
                          n_clusters * sizeof(int), cudaMemcpyHostToDevice));
  }
  
#pragma omp parallel default(shared)//shared(einsum_offsetT,einsum_cg_tmp2,einsum_w_qT,einsum_A,einsum_B,einsum_Bk,einsum_C,einsum_last,einsum_delta,cublasH,cutensorH,Zigma_pointer,Bk_pointer,Zigma,w_qT,offsetT,cg_tmp2,A,B,C,Bk,delta,last,X,Y,offset,k,w_q,mu_g)
{
  std::size_t BatchCount{genes/genesBatch};
  
  /****************************
   * Select the device
   ***************************/
  int me{omp_get_thread_num()};
  
  CUDA_CHECK(cudaSetDevice(me));
  /******************************
   * Create handlers and setup
   ******************************/
  CUBLAS_CHECK(cublasCreate(&(cublasH[me])));
  CUTENSOR_CHECK( cutensorCreate( &(cutensorH[me]) ) );
  constexpr int32_t numCachelines = 1024;
  CUTENSOR_CHECK( cutensorHandleResizePlanCache(cutensorH[me], numCachelines) );
  /********************************
   * Allocate and copy X and offset on each device, since it is const do it now
   *******************************/
  CUDA_CHECK( cudaMalloc((void**)&X[me], features*cells*sizeof(float)) );
  toGPU(X_host, X[me]);
  //same offset for each genes
  CUDA_CHECK( cudaMalloc((void**)&offset[me], 1*cells*sizeof(float)) );
  toGPU(offset_host, offset[me]);
  /*********************************
   * Allocate Y,offset,K,mu_beta, but use genesBatch as size, not genes
   ********************************/
  CUDA_CHECK( cudaMalloc((void**)&Y[me], cells*genesBatch*sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void**)&mu_beta[me], genesBatch*features*sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void**)&k[me], genesBatch*sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void**)&w_q[me], genesBatch*cells*sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void**)&mu_g[me], genesBatch*cells*sizeof(float)) );
  CUDA_CHECK(cudaMalloc(&d_hess_w[me],      genesBatch * cells              * sizeof(float)));
  // CUDA_CHECK(cudaMalloc(&d_score_r[me],     genesBatch * cells              * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_meat[me],        genesBatch * features * features * sizeof(float)));
  // CUDA_CHECK(cudaMalloc(&d_cluster_sums[me],genesBatch * n_clusters * features * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cluster_sums[me], features * n_clusters * genesBatch * sizeof(float)));
  
  // Allocate temporary buffers for dispersion calculation
  CUDA_CHECK( cudaMalloc((void**)&d_means[me], genesBatch*sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void**)&d_vars[me], genesBatch*sizeof(float)) );
  
  // Allocate buffers for MOM overdispersion calculation
  CUDA_CHECK( cudaMalloc((void**)&d_mu_mom[me], genesBatch * cells * sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void**)&d_diff_sq_minus_mu[me], genesBatch * cells * sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void**)&d_mu_sq[me], genesBatch * cells * sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void**)&d_num[me], genesBatch * sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void**)&d_den[me], genesBatch * sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void**)&d_theta[me], genesBatch * sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void**)&d_ones[me], cells * sizeof(float)) );
  
  // Initialize ones vector
  std::vector<float> ones_host(cells, 1.0f);
  CUDA_CHECK( cudaMemcpy(d_ones[me], ones_host.data(), cells * sizeof(float), cudaMemcpyHostToDevice) );
  
  /*********************************
   * Initialize the Tensor object, this doesn't allocate nothing ! 
   ********************************/
  einsum_cg_tmp2[me] = EinsumWrapper(std::string{"ik,jk->ji"},
  {(int)cells, (int)features},
  {(int)genesBatch, (int)features}); //E" CORRETTO
  einsum_A[me] = EinsumWrapper(std::string{"cf,gc->cfg"},
  {(int)cells, (int)features},
  {(int)genesBatch, (int)cells}); // ASSUMIAMO CHE SIA CORRETTO COSI, HYP
  einsum_B[me] = EinsumWrapper( std::string{"cfg,ck->gkf"},
  {(int)cells, (int)features, (int)genesBatch},
  {(int)cells, (int)features});  // ASSUMIAMO CHE SIA CORRETTO COSI, HYP
  einsum_Bk[me] = EinsumWrapper ( std::string{"gfc,g->gfc"},
  {(int)genesBatch, (int)features, (int)features},
  {(int)genesBatch}); // ASSUMIAMO CORRETTO
  einsum_C[me] = EinsumWrapper ( std::string{"cf,gc->gf"},
  {(int)cells, (int)features}, 
  {(int)genesBatch, (int)cells}); // ASSUMIAMO CORRETTO
  einsum_last[me] = EinsumWrapper ( std::string{"g,gf->gf"},
  {(int)genesBatch},
  {(int)genesBatch, (int)features}); //ok
  einsum_delta[me] =
    EinsumWrapper(std::string{"gfk,gk->gf"},
    {(int)genesBatch, (int)features, (int)features},
    {(int)genesBatch, (int)features});
  
  /******************************
   * This allocate the output tensor space
   ******************************/
  cg_tmp2[me] = einsum_cg_tmp2[me].allocate_output();
  A[me] = einsum_A[me].allocate_output();
  B[me] = einsum_B[me].allocate_output();
  C[me] = einsum_C[me].allocate_output();
  Bk[me] = einsum_Bk[me].allocate_output();
  delta[me] = einsum_delta[me].allocate_output();
  last[me] = einsum_last[me].allocate_output();
  
  /******************************
   * This allocate the workspace
   ******************************/
  std::list<int> workspace_size;
  workspace_size.push_back(einsum_cg_tmp2[me].workspace_size());
  workspace_size.push_back(einsum_A[me].workspace_size());
  workspace_size.push_back(einsum_B[me].workspace_size());
  workspace_size.push_back(einsum_C[me].workspace_size());
  workspace_size.push_back(einsum_Bk[me].workspace_size());
  workspace_size.push_back(einsum_delta[me].workspace_size());
  workspace_size.push_back(einsum_last[me].workspace_size());
  auto maxSize =
    * std::max_element(workspace_size.begin(), workspace_size.end());
    
    CUDA_CHECK(cudaMalloc((void **)&workspace[me], maxSize));
    
    /******************************
     * Allocate Zigma, The array of pointer to Zigma and Bk
     ******************************/
    // Use Managed memory to simply set the addresses
    CUDA_CHECK(cudaMallocManaged((void **) &(Zigma_pointer[me]), genesBatch * sizeof(float*)) );
    CUDA_CHECK(cudaMallocManaged((void **) &(Bk_pointer[me]), genesBatch * sizeof(float*)) );
    CUDA_CHECK(cudaMalloc((void **) &(Zigma[me]), sizeof(float) * features * features * genesBatch));
    for (int i = 0; i < genesBatch; ++i) {
      Zigma_pointer[me][i] = Zigma[me] + features * features * i;
      Bk_pointer[me][i] = Bk[me] + features * features * i;
    }
    
    /******************************
     * Allocate pivot and info for matrix inversion
     ******************************/
    CUDA_CHECK(cudaMallocManaged(&pivot[me], sizeof(int)*features*genesBatch));
    CUDA_CHECK(cudaMallocManaged(&info[me], sizeof(int)*genesBatch));
    
    cudaDeviceSynchronize();
    
#pragma omp single
{
  std::size_t free_mem, total_mem;
  // Get the amount of free and total memory
  CUDA_CHECK( cudaMemGetInfo(&free_mem, &total_mem) );
  std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB"
            << std::endl;
  std::cout << "Used memory: " << (total_mem-free_mem) / (1024 * 1024) << " MB" << std::endl;
  std::cout << "Total memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;
}

#pragma omp single
{ //here we will generate the work ! 
  for (int i = 0; i < BatchCount; ++i) {
#pragma omp task default(shared)
{
  int me=omp_get_thread_num();
  // Ensure we're using the correct device for this thread
  CUDA_CHECK(cudaSetDevice(me));
  
  // copy the necessary data!
  // lo copiamo all'inizio
  /*
   CUDA_CHECK(cudaMemcpy(
   offset[me],
   offset_host.data() + i  * genesBatch * cells,
   genesBatch * cells * sizeof(float), cudaMemcpyHostToDevice)); //CORRETTO
   */
  
  // Copy Y data for this batch
  CUDA_CHECK(cudaMemcpy(
      Y[me], Y_host.data() +  i * genesBatch * cells ,
      genesBatch * cells * sizeof(float), cudaMemcpyHostToDevice));
  
  // Initialize beta with rough approximation on GPU
  dim3 threads1D(256);
  dim3 blocks1D_genes((genesBatch + threads1D.x - 1) / threads1D.x);
  
  // Compute row means of Y for beta initialization
  compute_row_means<<<blocks1D_genes, threads1D>>>(Y[me], d_means[me], genesBatch, cells);
  
  // Initialize beta: beta[g, 0] = log1p(mean[g]), beta[g, f>0] = 0
  init_beta_rough_kernel<<<blocks1D_genes, threads1D>>>(d_means[me], mu_beta[me],
                                                        genesBatch, features);
  
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Debug: Copy initialized beta back to host for consistency check (only if TEST mode)
  if (TEST) {
    CUDA_CHECK(cudaMemcpy(beta_init_final.data() + i * genesBatch * features,
                          mu_beta[me],
                                 genesBatch * features * sizeof(float),
                                 cudaMemcpyDeviceToHost));
  }
  
  // Calculate dispersion (k) on GPU for this batch
  // Compute row means of Y
  // TODO: check if this is necessary, this should be computed above !!!! However is not expensive..
  compute_row_means<<<blocks1D_genes, threads1D>>>(Y[me], d_means[me], genesBatch, cells);
  
  // Compute row variances of Y
  compute_row_variances<<<blocks1D_genes, threads1D>>>(Y[me], d_means[me], d_vars[me], genesBatch, cells);
  
  // Compute dispersion and store in k (already stores 1/dispersion)
  compute_dispersion<<<blocks1D_genes, threads1D>>>(d_means[me], d_vars[me], offset_inv, k[me], genesBatch);
  
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Copy k back to host only if TEST mode enabled
  if (TEST) {
    CUDA_CHECK(cudaMemcpy(k_final.data() + i * genesBatch,
                          k[me],
                           genesBatch * sizeof(float),
                           cudaMemcpyDeviceToHost));
  }
  
  //set something to zero, required ? BOH,sicuro non falliremo per sta cosa qui
  CUDA_CHECK( cudaMemset(w_q[me], 0, genesBatch * cells * sizeof(float)));
  CUDA_CHECK( cudaMemset(mu_g[me], 0, genesBatch*cells*sizeof(float)));
  
  /******************************
   * Initialize norm s.t. the initial check is always True , set iter to 0,
   * measure start time.
   ******************************/
  //I know, there is a narrow conversion here.
  float norm{eps+1};
  std::size_t iter{0};
  auto t1 = std::chrono::high_resolution_clock::now();
  while ((iter < max_iter) && (norm > eps)) {
    ++iter;
    einsum_cg_tmp2[me].execute(cutensorH[me], X[me], mu_beta[me],workspace[me]);
    dim3 threads1D(256);
    dim3 blocks1D((genesBatch * cells + threads1D.x - 1) / threads1D.x);
    expGPU<<<blocks1D, threads1D>>>(cg_tmp2[me], offset[me], w_q[me],
                                    genesBatch * cells,cells);
    dim3 threads2D(16,16);
    dim3 blocks2D((cells + threads2D.x - 1) / threads2D.x,
                  (genesBatch + threads2D.y - 1) / threads2D.y);
    process2D<<<blocks2D, threads2D>>>(k[me], Y[me], w_q[me],
                                       mu_g[me],
                                           genesBatch, cells);
    elementWise<<<blocks1D, threads1D>>>(mu_g[me], w_q[me],
                                         genesBatch * cells);
    einsum_A[me].execute(cutensorH[me], X[me], mu_g[me],workspace[me]);
    einsum_B[me].execute(cutensorH[me], A[me], X[me],workspace[me]);
    einsum_Bk[me].execute(cutensorH[me], B[me], k[me],workspace[me]);
    inverseMatrix2(cublasH[me], Bk_pointer[me], Zigma_pointer[me],
                   features, genesBatch, pivot[me], info[me]);
    elementWiseSub<<<blocks1D,threads1D>>>(mu_g[me], genesBatch*cells);
    einsum_C[me].execute(cutensorH[me], X[me], mu_g[me],workspace[me]);
    einsum_last[me].execute(cutensorH[me], k[me], C[me],workspace[me]);
    einsum_delta[me].execute(cutensorH[me], Zigma[me], last[me],workspace[me]);
    final1D<<<blocks1D, threads1D>>>(mu_beta[me], delta[me],
                                     genesBatch * features);
    int max_id;
    CUBLAS_CHECK(cublasIsamax(cublasH[me], genesBatch * features,
                              delta[me], 1, &max_id));
    //FORTRAN INDEX, start from 1;
    --max_id;
    CUDA_CHECK(cudaMemcpy(&norm, delta[me] + max_id, sizeof(float),
                          cudaMemcpyDeviceToHost));
    norm = std::abs(norm);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto elapsed{t2 - t1};
  
  /*
   std::cout
   << std::chrono::duration<double, std::milli>(elapsed).count() /
   iter
   << " ms [avg iter time]" << std::endl;
   */
  
  /***********************************
   * Compute MOM overdispersion (theta) after beta converges
   **********************************/
  // 1. Compute eta = mu_beta @ X (reuse einsum, cg_tmp2 already contains this)
  einsum_cg_tmp2[me].execute(cutensorH[me], X[me], mu_beta[me], workspace[me]);
  
  // 2. Compute mu = sf * exp(eta)
  dim3 threads2D_mom(16, 16);
  dim3 blocks2D_mom((cells + 15) / 16, (genesBatch + 15) / 16);
  compute_mu_from_eta<<<blocks2D_mom, threads2D_mom>>>(cg_tmp2[me], offset[me], 
                                                       d_mu_mom[me], genesBatch, cells);
  
  // 3. Compute MOM components: (Y-mu)² - mu and mu²
  compute_mom_components<<<blocks2D_mom, threads2D_mom>>>(Y[me], d_mu_mom[me],
                                                          d_diff_sq_minus_mu[me], d_mu_sq[me],
                                                                                         genesBatch, cells);
  
  // 4. Compute row sums using cuBLAS GEMV: num = matrix @ ones, den = matrix @ ones
  const float alpha = 1.0f;
  const float beta_zero = 0.0f;
  
  CUBLAS_CHECK(cublasSgemv(cublasH[me], CUBLAS_OP_T, 
                           cells, genesBatch,
                           &alpha, 
                           d_diff_sq_minus_mu[me], cells,
                           d_ones[me], 1,
                           &beta_zero, 
                           d_num[me], 1));
  
  CUBLAS_CHECK(cublasSgemv(cublasH[me], CUBLAS_OP_T,
                           cells, genesBatch,
                           &alpha,
                           d_mu_sq[me], cells,
                           d_ones[me], 1,
                           &beta_zero,
                           d_den[me], 1));
  
  // 5. Compute theta = corr * num / den
  float corr = (float)cells / (cells - features);
  dim3 threads1D_mom(256);
  dim3 blocks1D_mom((genesBatch + 255) / 256);
  compute_theta_from_num_den<<<blocks1D_mom, threads1D_mom>>>(d_num[me], d_den[me], corr, 
                                                              d_theta[me], genesBatch);
  
  CUDA_CHECK(cudaDeviceSynchronize());
  
  /***********************************
   * Hessian inverse and clustered meat
   * mu_g here still holds sf*exp(eta) from the last IRLS iteration
   **********************************/
  {
    dim3 t1d(256);
    dim3 b1d((genesBatch * cells + 255) / 256);
    
    // ── Hessian (always computed) ─────────────────────────────────────────
    // 1. Weights: s_gi = (y*theta+1)*mu / (1+theta*mu)^2  using MOM theta
    compute_hessian_weights<<<b1d, t1d>>>(d_theta[me], Y[me], d_mu_mom[me], d_hess_w[me], genesBatch, cells);
    
    // 2. A[me] = "cf,gc->cfg": X ⊗ hess_w
    einsum_A[me].execute(cutensorH[me], X[me], d_hess_w[me], workspace[me]);
    
    // 3. B[me] = "cfg,ck->gkf": gives X^T diag(w) X per gene
    einsum_B[me].execute(cutensorH[me], A[me], X[me], workspace[me]);
    
    // 4. Negate B in-place (H = -X^T W X)
    // negate_kernel<<<(genesBatch*features*features+255)/256, 256>>>(B[me], genesBatch*features*features);
    
    // 5. Copy B → Bk, invert: Zigma[me] = (X^T W X)^{-1}
    CUDA_CHECK(cudaMemcpy(Bk[me], B[me],
                          genesBatch * features * features * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    inverseMatrix2(cublasH[me], Bk_pointer[me], Zigma_pointer[me],
                   features, genesBatch, pivot[me], info[me]);
    
    // ── Meat (only if clusters provided) ─────────────────────────────────
    if (n_clusters > 0) {
      // 6. Cluster sums with inline score: S[f,cl,g] = sum_{c in cl} (y-mu)/(1+mu*theta) * X[f,c]
      // Output layout: [features x n_clusters x genesBatch] col-major
      dim3 t_meat(16, 4, 1);
      dim3 b_meat((features + 15)/16, (n_clusters + 3)/4, genesBatch);
      compute_cluster_sums_and_scores<<<b_meat, t_meat>>>(
          Y[me], d_mu_mom[me], X[me], d_theta[me], d_cluster_ends, d_cluster_sums[me],
                                                                                 genesBatch, cells, features, n_clusters);
      
      // 7. meat[g] = (adj/n) * S[g] * S[g]^T  via batched SGEMM
      // S[g] is [features x n_clusters], stride = features*n_clusters
      float adj_over_n = ((float)n_clusters / (float)(n_clusters - 1)) / (float)cells;
      const float zero = 0.0f;
      CUBLAS_CHECK(cublasSgemmStridedBatched(
          cublasH[me],
                 CUBLAS_OP_N, CUBLAS_OP_T,
                 features, features, n_clusters,
                 &adj_over_n,
                 d_cluster_sums[me], features,
                 (long long)features * n_clusters,
                 d_cluster_sums[me], features,
                 (long long)features * n_clusters,
                 &zero,
                 d_meat[me], features,
                 (long long)features * features,
                 genesBatch));
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // ── Copy to host ──────────────────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(
        hessian_final.data() + i * genesBatch * features * features,
        Zigma[me],
             genesBatch * features * features * sizeof(float),
             cudaMemcpyDeviceToHost));
    
    if (n_clusters > 0) {
      CUDA_CHECK(cudaMemcpy(
          meat_final.data() + i * genesBatch * features * features,
          d_meat[me],
                genesBatch * features * features * sizeof(float),
                cudaMemcpyDeviceToHost));
    }
  }
  
  /***********************************
   * Copy beta, k, theta, and iterations to host
   **********************************/
  CUDA_CHECK(cudaMemcpy(mu_beta_final.data() + i * genesBatch * features,
                        mu_beta[me], 
                               genesBatch*features* sizeof(float),
                               cudaMemcpyDeviceToHost));
  
  // Copy k back to host only if TEST mode enabled
  if (TEST) {
    CUDA_CHECK(cudaMemcpy(k_final.data() + i * genesBatch,
                          k[me],
                           genesBatch * sizeof(float),
                           cudaMemcpyDeviceToHost));
  }
  
  CUDA_CHECK(cudaMemcpy(theta_final.data() + i * genesBatch,
                        d_theta[me],
                               genesBatch * sizeof(float),
                               cudaMemcpyDeviceToHost));
  
  std::fill(iterations.begin()  +  i *genesBatch , iterations.begin() +  +  (i+1) *genesBatch, iter);
  
}
  }
}
/*********************
 * Free Cuda Memory
 ********************/
// Free pivot and info if allocated
if (pivot[me] != nullptr) {
  CUDA_CHECK(cudaFree(pivot[me]));
}
if (info[me] != nullptr) {
  CUDA_CHECK(cudaFree(info[me]));
}
CUDA_CHECK(cudaFree(d_means[me]));
CUDA_CHECK(cudaFree(d_vars[me]));
CUDA_CHECK(cudaFree(d_mu_mom[me]));
CUDA_CHECK(cudaFree(d_diff_sq_minus_mu[me]));
CUDA_CHECK(cudaFree(d_mu_sq[me]));
CUDA_CHECK(cudaFree(d_num[me]));
CUDA_CHECK(cudaFree(d_den[me]));
CUDA_CHECK(cudaFree(d_theta[me]));
CUDA_CHECK(cudaFree(d_ones[me]));
CUDA_CHECK(cudaFree(Zigma[me]));
CUDA_CHECK(cudaFree(Bk_pointer[me]));
CUDA_CHECK(cudaFree(Zigma_pointer[me]));
CUDA_CHECK(cudaFree(cg_tmp2[me]));
CUDA_CHECK(cudaFree(A[me]));
CUDA_CHECK(cudaFree(B[me]));
CUDA_CHECK(cudaFree(C[me]));
CUDA_CHECK(cudaFree(Bk[me]));
CUDA_CHECK(cudaFree(delta[me]));
CUDA_CHECK(cudaFree(last[me]));
CUDA_CHECK(cudaFree(X[me]));
CUDA_CHECK(cudaFree(Y[me]));
CUDA_CHECK(cudaFree(offset[me]));
CUDA_CHECK(cudaFree(mu_beta[me]));
CUDA_CHECK(cudaFree(w_q[me]));
CUDA_CHECK(cudaFree(mu_g[me]));
CUDA_CHECK(cudaFree(k[me]));
CUDA_CHECK(cudaFree(workspace[me]));
CUDA_CHECK(cudaFree(d_hess_w[me]));
// CUDA_CHECK(cudaFree(d_score_r[me]));
CUDA_CHECK(cudaFree(d_meat[me]));
CUDA_CHECK(cudaFree(d_cluster_sums[me]));
/*********************
 * Destroy handles
 ********************/
CUBLAS_CHECK( cublasDestroy(cublasH[me]) );
CUTENSOR_CHECK( cutensorDestroy(cutensorH[me]) );
}

if (d_cluster_ends != nullptr) CUDA_CHECK(cudaFree(d_cluster_ends));

Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> result_beta(mu_beta_final.data(), features, genes);
Eigen::Map<Eigen::VectorXf> result_theta(theta_final.data(), genes);
Eigen::Map<Eigen::MatrixXf> r_hess(hessian_final.data(), features * features, genes);
Eigen::Map<Eigen::MatrixXf> r_meat(meat_final.data(),    features * features, genes);

BatchResult result;
result.beta = result_beta;
result.theta = result_theta;
result.hessian_inv = r_hess;
result.meat        = r_meat;

// Only populate k if TEST mode is enabled
if (TEST) {
  Eigen::Map<Eigen::VectorXf> result_k(k_final.data(), genes);
  result.k = result_k;
} else {
  // Return empty vector when not in TEST mode
  result.k = Eigen::VectorXf(0);
}

// Only populate beta_init if TEST mode is enabled
if (TEST) {
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> result_beta_init(beta_init_final.data(), features, genes);
  result.beta_init = result_beta_init;
} else {
  // Return empty matrix when not in TEST mode
  result.beta_init = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>(0, 0);
}

return result;
}



// ─────────────────────────────────────────────────────────────────────────────
// Summary-space version: operates on M unique groups instead of N cells.
// Inputs that are new vs. beta_fit_gpu_external:
//   X_unique_host   [P x M] col-major  (design rows for unique groups)
//   off_unique_host [M]                (log size-factor per unique group)
//   mapping_host    [N] int 1-based    (cell -> group index)
//   counts_host     [M]                (number of cells per group)
//   cluster_map_host[M] int 1-based    (group -> cluster index)
// Y_host is still [N x G] col-major — only used for the initial aggregation.
// ─────────────────────────────────────────────────────────────────────────────
BatchResult
beta_fit_gpu_external_summary(
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const& Y_host,
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const& X_unique_host,
  Eigen::VectorXf const& off_unique_host,
  Eigen::VectorXi const& mapping_host,
  Eigen::VectorXf const& counts_host,
  Eigen::VectorXi const& cluster_map_host,
  int n_clusters,
  int max_iter, float eps, int batch_size)
{
  /******************************
   * Shape definition
   * CHANGE vs. original: cells = N (for aggregation only),
   *                      groups = M (the working dimension for all kernels)
   ******************************/
  const std::size_t genes    = Y_host.cols();
  const std::size_t cells    = Y_host.rows();           // N — only for aggregation
  const std::size_t groups   = X_unique_host.cols();    // M
  const std::size_t features = X_unique_host.rows();    // P
  std::size_t genesBatch     = static_cast<std::size_t>(batch_size);
  
  std::cout << "X_unique {" << features << "," << groups << "}\n";
  std::cout << "Y {"        << cells    << "," << genes  << "}\n";
  std::cout << "M (groups) = " << groups << ", N (cells) = " << cells << "\n";
  
  // Host-side output buffers — identical to original
  std::vector<float> mu_beta_final(genes * features, 0.0f);
  std::vector<float> theta_final  (genes,            0.0f);
  std::vector<float> hessian_final(genes * features * features, 0.0f);
  std::vector<float> meat_final   (genes * features * features, 0.0f);
  
  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  std::cout << "Detected " << deviceCount << " GPU(s)" << std::endl;
  for (int gpu = 0; gpu < deviceCount; ++gpu) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu);
    std::cout << "Device " << gpu << ": " << prop.name
              << " (cc " << prop.major << "." << prop.minor << ")\n";
  }
  omp_set_num_threads(deviceCount);
  
  /************************
   * Per-device handle / pointer arrays
   * CHANGE: w_q, mu_g, d_hess_w, d_mu_mom, d_diff_sq_minus_mu, d_mu_sq
   *         are now [genesBatch x M] instead of [genesBatch x N].
   *         d_ones is [M] instead of [N].
   *         New arrays: d_y_sums, d_y_sq_sums [M x genesBatch],
   *                     d_counts [M], d_cluster_map [M] int.
   ***********************/
  std::vector<cublasHandle_t>  cublasH (deviceCount);
  std::vector<cutensorHandle_t> cutensorH(deviceCount);
  
  std::vector<EinsumWrapper> einsum_cg_tmp2(deviceCount);
  std::vector<EinsumWrapper> einsum_A      (deviceCount);
  std::vector<EinsumWrapper> einsum_B      (deviceCount);
  std::vector<EinsumWrapper> einsum_Bk     (deviceCount);
  std::vector<EinsumWrapper> einsum_C      (deviceCount);
  std::vector<EinsumWrapper> einsum_last   (deviceCount);
  std::vector<EinsumWrapper> einsum_delta  (deviceCount);
  
  // Static per-device buffers (uploaded once, same for all batches)
  std::vector<float*> X      (deviceCount);  // [features x M]
  std::vector<float*> d_off  (deviceCount);  // [M]  exp(off_unique)
  std::vector<float*> d_counts(deviceCount); // [M]
  std::vector<float*> d_sf   (deviceCount);
  // std::vector<int*>   d_mapping(deviceCount);// [N]  for aggregation
  std::vector<int*>   d_cluster_map(deviceCount); // [M]
  
  // Per-batch device buffers
  // std::vector<float*> Y_dev     (deviceCount); // [N x genesBatch] col-major (raw, for aggregation)
  std::vector<float*> mu_beta   (deviceCount); // [genesBatch x features]
  std::vector<float*> k         (deviceCount); // [genesBatch]
  std::vector<float*> w_q       (deviceCount); // [genesBatch x M]
  std::vector<float*> mu_g      (deviceCount); // [genesBatch x M]  (weight in IRLS)
  std::vector<float*> d_y_sums  (deviceCount); // [M x genesBatch]
  std::vector<float*> d_y_sq_sums(deviceCount);// [M x genesBatch]
  std::vector<float*> d_mu_mom  (deviceCount); // [genesBatch x M]
  std::vector<float*> d_num     (deviceCount); // [genesBatch x M]  MOM numerator
  std::vector<float*> d_den     (deviceCount); // [genesBatch x M]  MOM denominator
  std::vector<float*> d_theta   (deviceCount); // [genesBatch]
  std::vector<float*> d_ones    (deviceCount); // [M]  for GEMV reduction
  std::vector<float*> d_hess_w  (deviceCount); // [genesBatch x M]
  std::vector<float*> d_meat    (deviceCount); // [genesBatch x features x features]
  std::vector<float*> d_cluster_sums(deviceCount); // [features x n_clusters x genesBatch]
  
  // IRLS temporaries (same role as original)
  std::vector<float*> cg_tmp2(deviceCount);
  std::vector<float*> A      (deviceCount);
  std::vector<float*> B      (deviceCount);
  std::vector<float*> C      (deviceCount);
  std::vector<float*> Bk     (deviceCount);
  std::vector<float*> delta  (deviceCount);
  std::vector<float*> last   (deviceCount);
  std::vector<float*> workspace(deviceCount);
  
  std::vector<float**> Zigma_pointer(deviceCount);
  std::vector<float**> Bk_pointer   (deviceCount);
  std::vector<float*>  Zigma         (deviceCount);
  std::vector<int*>    pivot(deviceCount, nullptr);
  std::vector<int*>    info (deviceCount, nullptr);
  
  // CHANGE: offset_inv is not needed for IRLS (offset is per-group, applied in expGPU).
  // We still compute it for the legacy k-initialisation kernel which expects a scalar.
  // We derive it from off_unique (exp then mean).
  double off_sum = 0.0;
  for (int m = 0; m < (int)off_unique_host.size(); ++m)
    off_sum += std::exp(off_unique_host[m]);
  float offset_inv = 1.0f / (float)(off_sum / groups);
  
  // Pre-compute exp(off_unique) on host for uploading to device as d_off
  std::vector<float> sf_unique_host(groups);
  for (int m = 0; m < (int)groups; ++m)
    sf_unique_host[m] = std::exp(off_unique_host[m]);
  
  // ── CPU pre-aggregation ───────────────────────────────────────────────────
  // Aggregate all of Y into y_sums and y_sq_sums in M-space, once.
  // Layout: [M x G] col-major, i.e. element (m, g) = index m + g*M.
  // Also compute per-gene means and initial k values so the GPU batch loop
  // no longer needs raw Y, Y_dev, d_mapping, or any aggregation kernels.
  const int G = (int)genes;
  const int M = (int)groups;
  
  std::vector<float> all_y_sums   (G * M, 0.0f);
  std::vector<float> all_y_sq_sums(G * M, 0.0f);
  
  for (int n = 0; n < (int)cells; ++n) {
    int m = mapping_host[n] - 1;                   // 0-based group index
    for (int g = 0; g < G; ++g) {
      float v = Y_host(n, g);                      // [N x G] col-major
      all_y_sums   [m + g * M] += v;
      all_y_sq_sums[m + g * M] += v * v;
    }
  }
  
  // Per-gene mean, variance, and initial k — all derived from the aggregated sums.
  std::vector<float> gene_means(G, 0.0f);
  std::vector<float> gene_k    (G, 0.0f);
  
  for (int g = 0; g < G; ++g) {
    float s = 0.0f, s2 = 0.0f;
    for (int m = 0; m < M; ++m) {
      s  += all_y_sums   [m + g * M];
      s2 += all_y_sq_sums[m + g * M];
    }
    float mean = s / (float)cells;
    float var  = (s2 - (float)cells * mean * mean) / ((float)cells - 1.0f);
    float disp = (var - offset_inv * mean) / (mean * mean + 1e-8f);
    gene_means[g] = mean;
    gene_k    [g] = 1.0f / std::max(0.01f, disp);
  }
  // ─────────────────────────────────────────────────────────────────────────
  
#pragma omp parallel default(shared)
{
  std::size_t BatchCount = genes / genesBatch;
  int me = omp_get_thread_num();
  CUDA_CHECK(cudaSetDevice(me));
  
  CUBLAS_CHECK(cublasCreate(&cublasH[me]));
  CUTENSOR_CHECK(cutensorCreate(&cutensorH[me]));
  constexpr int32_t numCachelines = 1024;
  CUTENSOR_CHECK(cutensorHandleResizePlanCache(cutensorH[me], numCachelines));
  
  /******************************
   * Upload static data (once per device)
   * CHANGE: X is [features x M], offset is [M], plus new counts, mapping, cluster_map.
   ******************************/
  CUDA_CHECK(cudaMalloc((void**)&X[me],          features * groups * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(X[me], X_unique_host.data(),
                        features * groups * sizeof(float), cudaMemcpyHostToDevice));
  
  // d_off holds log(sf) — the raw off_unique values — for expGPU_neg in the IRLS
  CUDA_CHECK(cudaMalloc((void**)&d_off[me], groups * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_off[me], off_unique_host.data(),
                        groups * sizeof(float), cudaMemcpyHostToDevice));
  
  // d_sf holds exp(off_unique) = sf — for compute_mu_from_eta in the MOM step
  CUDA_CHECK(cudaMalloc((void**)&d_sf[me],  groups * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_sf[me], sf_unique_host.data(),
                        groups * sizeof(float), cudaMemcpyHostToDevice));
  
  CUDA_CHECK(cudaMalloc((void**)&d_counts[me],    groups * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_counts[me], counts_host.data(),
                        groups * sizeof(float), cudaMemcpyHostToDevice));
  
  // CUDA_CHECK(cudaMalloc((void**)&d_mapping[me],   cells * sizeof(int)));
  // CUDA_CHECK(cudaMemcpy(d_mapping[me], mapping_host.data(),
  //                       cells * sizeof(int), cudaMemcpyHostToDevice));
  
  if (n_clusters > 0) {
    CUDA_CHECK(cudaMalloc((void**)&d_cluster_map[me], groups * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_cluster_map[me], cluster_map_host.data(),
                          groups * sizeof(int), cudaMemcpyHostToDevice));
  } else {
    d_cluster_map[me] = nullptr;
  }
  
  /******************************
   * Per-batch allocations
   * CHANGE: everything that was [genesBatch x cells] is now [genesBatch x M].
   *         d_y_sums / d_y_sq_sums are new [M x genesBatch].
   *         Y_dev stays [N x genesBatch] — used only during aggregation.
   ******************************/
  // CUDA_CHECK(cudaMalloc((void**)&Y_dev[me],       cells * genesBatch * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&mu_beta[me],      genesBatch * features * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&k[me],            genesBatch * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&w_q[me],          genesBatch * groups * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&mu_g[me],         genesBatch * groups * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_y_sums[me],     groups * genesBatch * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_y_sq_sums[me],  groups * genesBatch * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_mu_mom[me],     genesBatch * groups * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_num[me],        genesBatch * groups * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_den[me],        genesBatch * groups * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_theta[me],      genesBatch * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_ones[me],       groups * sizeof(float)));  // CHANGE: M not N
  CUDA_CHECK(cudaMalloc((void**)&d_hess_w[me],     genesBatch * groups * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_meat[me],       genesBatch * features * features * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_cluster_sums[me],
                        features * n_clusters * genesBatch * sizeof(float)));
  
  // Initialise ones vector [M]
  std::vector<float> ones_host(groups, 1.0f);
  CUDA_CHECK(cudaMemcpy(d_ones[me], ones_host.data(),
                        groups * sizeof(float), cudaMemcpyHostToDevice));
  
  /******************************
   * EinsumWrapper setup
   * CHANGE: every dimension that was `cells` is now `groups` (= M).
   * Subscript strings are IDENTICAL to the original.
   ******************************/
  einsum_cg_tmp2[me] = EinsumWrapper("ik,jk->ji",
  {(int)groups,   (int)features},
  {(int)genesBatch,(int)features});
  
  einsum_A[me] = EinsumWrapper("cf,gc->cfg",
  {(int)groups,   (int)features},
  {(int)genesBatch,(int)groups});
  
  einsum_B[me] = EinsumWrapper("cfg,ck->gkf",
  {(int)groups,   (int)features, (int)genesBatch},
  {(int)groups,   (int)features});
  
  einsum_Bk[me] = EinsumWrapper("gfc,g->gfc",
  {(int)genesBatch,(int)features,(int)features},
  {(int)genesBatch});
  
  einsum_C[me] = EinsumWrapper("cf,gc->gf",
  {(int)groups,   (int)features},
  {(int)genesBatch,(int)groups});
  
  einsum_last[me] = EinsumWrapper("g,gf->gf",
  {(int)genesBatch},
  {(int)genesBatch,(int)features});
  
  einsum_delta[me] = EinsumWrapper("gfk,gk->gf",
  {(int)genesBatch,(int)features,(int)features},
  {(int)genesBatch,(int)features});
  
  // Allocate output tensors
  cg_tmp2[me]  = einsum_cg_tmp2[me].allocate_output();
  A[me]        = einsum_A[me].allocate_output();
  B[me]        = einsum_B[me].allocate_output();
  C[me]        = einsum_C[me].allocate_output();
  Bk[me]       = einsum_Bk[me].allocate_output();
  delta[me]    = einsum_delta[me].allocate_output();
  last[me]     = einsum_last[me].allocate_output();
  
  // Workspace (max over all einsums)
  std::list<int> ws_sizes;
  ws_sizes.push_back(einsum_cg_tmp2[me].workspace_size());
  ws_sizes.push_back(einsum_A[me].workspace_size());
  ws_sizes.push_back(einsum_B[me].workspace_size());
  ws_sizes.push_back(einsum_C[me].workspace_size());
  ws_sizes.push_back(einsum_Bk[me].workspace_size());
  ws_sizes.push_back(einsum_delta[me].workspace_size());
  ws_sizes.push_back(einsum_last[me].workspace_size());
  CUDA_CHECK(cudaMalloc((void**)&workspace[me],
                        *std::max_element(ws_sizes.begin(), ws_sizes.end())));
  
  // Zigma / Bk pointer arrays (unchanged)
  CUDA_CHECK(cudaMallocManaged((void**)&Zigma_pointer[me], genesBatch * sizeof(float*)));
  CUDA_CHECK(cudaMallocManaged((void**)&Bk_pointer[me],    genesBatch * sizeof(float*)));
  CUDA_CHECK(cudaMalloc((void**)&Zigma[me], features * features * genesBatch * sizeof(float)));
  for (int i = 0; i < (int)genesBatch; ++i) {
    Zigma_pointer[me][i] = Zigma[me] + features * features * i;
    Bk_pointer[me][i]    = Bk[me]   + features * features * i;
  }
  CUDA_CHECK(cudaMallocManaged(&pivot[me], sizeof(int) * features * genesBatch));
  CUDA_CHECK(cudaMallocManaged(&info[me],  sizeof(int) * genesBatch));
  
  cudaDeviceSynchronize();
  
#pragma omp single
{
  std::size_t free_mem, total_mem;
  CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
  std::cout << "Free memory: "  << free_mem / (1024*1024) << " MB\n";
  std::cout << "Used memory: "  << (total_mem-free_mem)/(1024*1024) << " MB\n";
  std::cout << "Total memory: " << total_mem / (1024*1024) << " MB\n";
}

#pragma omp single
{
  for (int i = 0; i < (int)BatchCount; ++i) {
#pragma omp task default(shared)
{
  int me = omp_get_thread_num();
  CUDA_CHECK(cudaSetDevice(me));
  
  /******************************
   * Upload pre-aggregated sums for this batch.
   * Transfer is [M x genesBatch] floats — far smaller than [N x genesBatch].
   * Layout: all_y_sums is [M x G] col-major; the slice for batch i starts
   * at offset i*genesBatch*M and has length genesBatch*M.
   ******************************/
  CUDA_CHECK(cudaMemcpy(d_y_sums[me],
                        all_y_sums.data()    + (std::size_t)i * genesBatch * M,
                        genesBatch * M * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y_sq_sums[me],
                        all_y_sq_sums.data() + (std::size_t)i * genesBatch * M,
                        genesBatch * M * sizeof(float), cudaMemcpyHostToDevice));
  
  /******************************
   * Beta initialisation from pre-computed gene means.
   * Upload the genesBatch-slice of gene_means into d_theta (used as scratch),
   * then call init_beta_rough_kernel exactly as before.
   ******************************/
  dim3 t1d(256);
  dim3 b1d_genes((genesBatch + 255) / 256);
  float* d_means_tmp = d_theta[me];   // safe: overwritten after IRLS
  CUDA_CHECK(cudaMemcpy(d_means_tmp,
                        gene_means.data() + (std::size_t)i * genesBatch,
                        genesBatch * sizeof(float), cudaMemcpyHostToDevice));
  init_beta_rough_kernel<<<b1d_genes, t1d>>>(d_means_tmp, mu_beta[me],
                                             genesBatch, features);
  
  /******************************
   * Initial k from pre-computed gene_k values.
   ******************************/
  CUDA_CHECK(cudaMemcpy(k[me],
                        gene_k.data() + (std::size_t)i * genesBatch,
                        genesBatch * sizeof(float), cudaMemcpyHostToDevice));
  
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Zero working buffers
  CUDA_CHECK(cudaMemset(w_q[me],  0, genesBatch * groups * sizeof(float)));
  CUDA_CHECK(cudaMemset(mu_g[me], 0, genesBatch * groups * sizeof(float)));
  
  /******************************
   * IRLS loop
   * CHANGE: process2D + elementWise replaced by process2D_summary.
   *         elementWiseSub replaced by elementWiseSub_summary.
   *         All 2D launch configs use (genesBatch, M) instead of (genesBatch, N).
   *         expGPU receives d_off (size M) and total size = genesBatch*M.
   *         Einsums are called identically — dims changed at construction time.
   ******************************/
  float norm = eps + 1.0f;
  std::size_t iter = 0;
  while (iter < (std::size_t)max_iter && norm > eps) {
    ++iter;
    
    // eta = X * mu_beta  →  cg_tmp2 [genesBatch x M]
    einsum_cg_tmp2[me].execute(cutensorH[me], X[me], mu_beta[me], workspace[me]);
    
    // w_q = exp(eta + off_unique)  [genesBatch x M]
    // AFTER
    dim3 t1(256);
    dim3 b1((genesBatch * groups + 255) / 256);
    expGPU_neg<<<b1, t1>>>(cg_tmp2[me], d_off[me], w_q[me],
                           genesBatch * groups, groups, genesBatch);
    
    // weight = mu_g_sum * w_q, in M-space
    dim3 t2(16, 16);
    dim3 b2((genesBatch + t2.x - 1) / t2.x,
            (groups     + t2.y - 1) / t2.y);
    process2D_summary<<<b2, t2>>>(k[me], d_y_sums[me], d_counts[me],
                                  w_q[me], mu_g[me],
                                               genesBatch, groups);
    
    // Hessian: A = X ⊗ weight, B = A^T X, Bk = k*B
    einsum_A[me].execute(cutensorH[me], X[me], mu_g[me], workspace[me]);
    einsum_B[me].execute(cutensorH[me], A[me], X[me],    workspace[me]);
    einsum_Bk[me].execute(cutensorH[me], B[me], k[me],   workspace[me]);
    inverseMatrix2(cublasH[me], Bk_pointer[me], Zigma_pointer[me],
                   features, genesBatch, pivot[me], info[me]);
    
    // Gradient: weight -= counts[m]  (in-place)
    elementWiseSub_summary<<<b2, t2>>>(mu_g[me], d_counts[me],
                                       genesBatch, groups);
    
    // C = X^T * (weight - counts)
    einsum_C[me].execute(cutensorH[me], X[me], mu_g[me], workspace[me]);
    einsum_last[me].execute(cutensorH[me], k[me], C[me], workspace[me]);
    einsum_delta[me].execute(cutensorH[me], Zigma[me], last[me], workspace[me]);
    
    // mu_beta += delta
    final1D<<<b1, t1>>>(mu_beta[me], delta[me], genesBatch * features);
    
    // Convergence check (unchanged)
    int max_id;
    CUBLAS_CHECK(cublasIsamax(cublasH[me], genesBatch * features,
                              delta[me], 1, &max_id));
    --max_id;
    CUDA_CHECK(cudaMemcpy(&norm, delta[me] + max_id, sizeof(float),
                          cudaMemcpyDeviceToHost));
    norm = std::abs(norm);
  }
  
  /******************************
   * MOM overdispersion in M-space
   * CHANGE: compute_mom_components replaced by compute_mom_components_summary.
   *         compute_mu_from_eta still works — it just produces [genesBatch x M]
   *         using the M-length d_off instead of N-length offset.
   *         GEMV reduction dimensions change from N to M.
   ******************************/
  // 1. eta already in cg_tmp2; recompute mu_mom = sf * exp(eta) in M-space
  einsum_cg_tmp2[me].execute(cutensorH[me], X[me], mu_beta[me], workspace[me]);
  // AFTER
  {
    dim3 t1(256);
    dim3 b1((genesBatch * groups + 255) / 256);
    compute_mu_from_eta_rowmajor<<<b1, t1>>>(cg_tmp2[me], d_sf[me],
                                             d_mu_mom[me], genesBatch, groups);
  }
  
  // 2. MOM numerator and denominator in M-space
  {
    dim3 t2(16, 16);
    dim3 b2((genesBatch + t2.x - 1) / t2.x,
            (groups     + t2.y - 1) / t2.y);
    compute_mom_components_summary<<<b2, t2>>>(
        d_y_sums[me], d_y_sq_sums[me], d_mu_mom[me], d_counts[me],
                                                             d_num[me], d_den[me],
                                                                             genesBatch, groups);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // 3. Row-reduce num and den [genesBatch x M] -> [genesBatch] via GEMV with ones [M]
  // CHANGE: leading dimension and vector length are M not N
  {
    const float alpha1 = 1.0f, beta0 = 0.0f;
    CUBLAS_CHECK(cublasSgemv(cublasH[me], CUBLAS_OP_N,
                             genesBatch, groups,
                             &alpha1,
                             d_num[me], genesBatch,
                             d_ones[me], 1,
                             &beta0,
                             d_theta[me], 1));
    // Temporarily store den sum in k (it is a genesBatch buffer we can reuse here)
    CUBLAS_CHECK(cublasSgemv(cublasH[me], CUBLAS_OP_N,
                             genesBatch, groups,
                             &alpha1,
                             d_den[me], genesBatch,
                             d_ones[me], 1,
                             &beta0,
                             k[me], 1));
  }
  
  // 4. theta = corr * num_sum / den_sum
  // CHANGE: correction factor uses N_total (cells) and P (features)
  {
    float corr = (float)cells / ((float)cells - (float)features);
    dim3 t1d(256);
    dim3 b1d((genesBatch + 255) / 256);
    compute_theta_from_num_den<<<b1d, t1d>>>(
        d_theta[me],  // holds num_sum
               k[me],        // holds den_sum
                corr, d_theta[me], genesBatch);
    // NOTE: compute_theta_from_num_den reads num/den and writes theta;
    //       if its signature is (num, den, corr, theta_out, G) and
    //       num == theta_out (aliased), check that your kernel handles
    //       the in-place case, or use a separate d_num_sum scratch buffer.
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Re-init k from theta for any downstream code that reads k
  // (In the original k was 1/dispersion used in IRLS; now we have the final theta.)
  // Nothing downstream reads k after this point, so we leave it as den_sum scratch.
  
  /******************************
   * Hessian inverse and clustered meat in M-space
   * CHANGE: compute_hessian_weights replaced by compute_hessian_weights_summary.
   *         compute_cluster_sums_and_scores replaced by compute_cluster_sums_summary.
   *         All 2D configs use M not N.
   *         batched SGEMM for meat is IDENTICAL.
   ******************************/
  {
    dim3 t2(16, 16);
    dim3 b2((genesBatch + t2.x - 1) / t2.x,
            (groups     + t2.y - 1) / t2.y);
    
    // Hessian weights in M-space
    compute_hessian_weights_summary<<<b2, t2>>>(
        d_theta[me], d_y_sums[me], d_counts[me], d_mu_mom[me],
                                                         d_hess_w[me],
                                                                 genesBatch, groups);
    
    // A = X ⊗ hess_w, B = A^T X  (einsums unchanged)
    einsum_A[me].execute(cutensorH[me], X[me], d_hess_w[me], workspace[me]);
    einsum_B[me].execute(cutensorH[me], A[me], X[me],        workspace[me]);
    
    // Copy B -> Bk, invert (unchanged)
    CUDA_CHECK(cudaMemcpy(Bk[me], B[me],
                          genesBatch * features * features * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    inverseMatrix2(cublasH[me], Bk_pointer[me], Zigma_pointer[me],
                   features, genesBatch, pivot[me], info[me]);
    
    // Clustered meat in M-space
    if (n_clusters > 0) {
      CUDA_CHECK(cudaMemset(d_cluster_sums[me], 0,
                            features * n_clusters * genesBatch * sizeof(float)));
      dim3 t_m(16, 16);
      dim3 b_m((features   + t_m.x - 1) / t_m.x,
               (genesBatch + t_m.y - 1) / t_m.y);
      // CHANGE: one kernel call iterates over M groups internally (loop in kernel)
      compute_cluster_sums_summary<<<b_m, t_m>>>(
          d_y_sums[me], d_counts[me], d_mu_mom[me],
                                              X[me], d_theta[me], d_cluster_map[me],
                                                                               d_cluster_sums[me],
                                                                                             genesBatch, groups, features, n_clusters);
      
      // meat = (adj/N) * S * S^T  — IDENTICAL to original
      float adj_over_n = ((float)n_clusters / (float)(n_clusters - 1)) / (float)cells;
      const float zero = 0.0f;
      CUBLAS_CHECK(cublasSgemmStridedBatched(
          cublasH[me],
                 CUBLAS_OP_N, CUBLAS_OP_T,
                 features, features, n_clusters,
                 &adj_over_n,
                 d_cluster_sums[me], features, (long long)features * n_clusters,
                 d_cluster_sums[me], features, (long long)features * n_clusters,
                 &zero,
                 d_meat[me], features, (long long)features * features,
                 genesBatch));
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy Hessian inverse and meat to host (unchanged offsets)
    CUDA_CHECK(cudaMemcpy(
        hessian_final.data() + i * genesBatch * features * features,
        Zigma[me],
             genesBatch * features * features * sizeof(float),
             cudaMemcpyDeviceToHost));
    
    if (n_clusters > 0) {
      CUDA_CHECK(cudaMemcpy(
          meat_final.data() + i * genesBatch * features * features,
          d_meat[me],
                genesBatch * features * features * sizeof(float),
                cudaMemcpyDeviceToHost));
    }
  }
  
  /******************************
   * Copy beta and theta to host (unchanged)
   ******************************/
  CUDA_CHECK(cudaMemcpy(
      mu_beta_final.data() + i * genesBatch * features,
      mu_beta[me],
             genesBatch * features * sizeof(float),
             cudaMemcpyDeviceToHost));
  
  CUDA_CHECK(cudaMemcpy(
      theta_final.data() + i * genesBatch,
      d_theta[me],
             genesBatch * sizeof(float),
             cudaMemcpyDeviceToHost));
  
} // end omp task
  } // end batch loop
} // end omp single

/******************************
 * Free per-device memory
 * CHANGE: free new buffers; remove d_means / d_vars (not separately allocated here).
 ******************************/
if (pivot[me])         CUDA_CHECK(cudaFree(pivot[me]));
if (info[me])          CUDA_CHECK(cudaFree(info[me]));
CUDA_CHECK(cudaFree(X[me]));
CUDA_CHECK(cudaFree(d_off[me]));
CUDA_CHECK(cudaFree(d_sf[me]));
CUDA_CHECK(cudaFree(d_counts[me]));
// CUDA_CHECK(cudaFree(d_mapping[me]));
if (d_cluster_map[me]) CUDA_CHECK(cudaFree(d_cluster_map[me]));
// CUDA_CHECK(cudaFree(Y_dev[me]));
CUDA_CHECK(cudaFree(mu_beta[me]));
CUDA_CHECK(cudaFree(k[me]));
CUDA_CHECK(cudaFree(w_q[me]));
CUDA_CHECK(cudaFree(mu_g[me]));
CUDA_CHECK(cudaFree(d_y_sums[me]));
CUDA_CHECK(cudaFree(d_y_sq_sums[me]));
CUDA_CHECK(cudaFree(d_mu_mom[me]));
CUDA_CHECK(cudaFree(d_num[me]));
CUDA_CHECK(cudaFree(d_den[me]));
CUDA_CHECK(cudaFree(d_theta[me]));
CUDA_CHECK(cudaFree(d_ones[me]));
CUDA_CHECK(cudaFree(d_hess_w[me]));
CUDA_CHECK(cudaFree(d_meat[me]));
CUDA_CHECK(cudaFree(d_cluster_sums[me]));
CUDA_CHECK(cudaFree(Zigma[me]));
CUDA_CHECK(cudaFree(Bk_pointer[me]));
CUDA_CHECK(cudaFree(Zigma_pointer[me]));
CUDA_CHECK(cudaFree(cg_tmp2[me]));
CUDA_CHECK(cudaFree(A[me]));
CUDA_CHECK(cudaFree(B[me]));
CUDA_CHECK(cudaFree(C[me]));
CUDA_CHECK(cudaFree(Bk[me]));
CUDA_CHECK(cudaFree(delta[me]));
CUDA_CHECK(cudaFree(last[me]));
CUDA_CHECK(cudaFree(workspace[me]));
CUBLAS_CHECK(cublasDestroy(cublasH[me]));
CUTENSOR_CHECK(cutensorDestroy(cutensorH[me]));

} // end omp parallel

/******************************
 * Assemble and return BatchResult (identical to original)
 ******************************/
Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
result_beta(mu_beta_final.data(), features, genes);
Eigen::Map<Eigen::VectorXf>   result_theta(theta_final.data(), genes);
Eigen::Map<Eigen::MatrixXf>   r_hess(hessian_final.data(), features * features, genes);
Eigen::Map<Eigen::MatrixXf>   r_meat(meat_final.data(),    features * features, genes);

BatchResult result;
result.beta        = result_beta;
result.theta       = result_theta;
result.hessian_inv = r_hess;
result.meat        = r_meat;
result.k           = Eigen::VectorXf(0);
result.beta_init   = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::ColMajor>(0, 0);
return result;
}