#include <cuda_runtime.h>
#include <cuda.h>
#include <cusolverDn.h>
#include "utils.hpp"
#include <cassert>
#include <iostream>
#include <ostream>

__global__ void initIdentityGPU(float *Matrix, int rows, int cols,float alpha) {
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  if(y < rows && x < cols) {
    if(x == y)
      Matrix[y*cols+x] = 1*alpha;
    else
      Matrix[y*cols+x] = 0;
  }
}


void inverseMatrix2(cublasHandle_t cublasH, float *A_device[], float *A_inv_device[], int n, int batchSize, int*& pivot, int*& info){

  //factorize !
  CUBLAS_CHECK( cublasSgetrfBatched(cublasH,
                                   n,
                                   A_device,
                                   n,
                                   pivot,
                                   info,
				   batchSize) );
  for(int i=0;i<batchSize;++i) {
    if (info[i] != 0 ) {
      CUDA_CHECK(cudaDeviceSynchronize());
      std::cerr<< "Info value: " << info[i] << std::endl;
      assert(0 == info[i]);
    }
  }

  cublasSgetriBatched(cublasH, n, A_device, n, pivot, A_inv_device, n, info, batchSize);
  for(int i=0;i<batchSize;++i) {
    if (info[i] != 0 ) {
      CUDA_CHECK(cudaDeviceSynchronize());
      std::cerr<< "Info value: " << info[i] << std::endl;
      assert(0 == info[i]);
    }
  }
  
  // Don't free here - caller manages memory
  
  return; 
  
}
