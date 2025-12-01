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
    std::vector<int>& iterations, bool TEST) {

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
	    // Compute row means of Y (recompute since d_means was used for beta init)
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
      /*********************
       * Destroy handles
       ********************/
      CUBLAS_CHECK( cublasDestroy(cublasH[me]) );
      CUTENSOR_CHECK( cutensorDestroy(cutensorH[me]) );
    }

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> result_beta(mu_beta_final.data(), features, genes);
    Eigen::Map<Eigen::VectorXf> result_theta(theta_final.data(), genes);
    
    BatchResult result;
    result.beta = result_beta;
    result.theta = result_theta;
    
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

