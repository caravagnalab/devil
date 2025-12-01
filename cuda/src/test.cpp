#include <gtest/gtest.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include <iostream>
#include <type_traits>
#include <vector>

#include "utils.hpp"
#include "inverse.hpp"
#include "cutensor.h"
#include "einsum.hpp"
#include  <string>


void invertExactBatch(std::vector<float> vec_exact,std::vector<float> vec_inverse_exact,int col) {
  int batchSize=2;
  float *vec;
  float *vec_inverse;

  float **vec_pointer;
  float **vec_inverse_pointer;
  
  int *pivot = nullptr;
  int *info = nullptr;

  int col2 = col * col;
  //alloc su device la memoria
  CUDA_CHECK(cudaMallocManaged(&vec, col2 * sizeof(float) * batchSize));
  CUDA_CHECK(cudaMallocManaged(&vec_inverse, col2 * sizeof(float) * batchSize));
  //alloco array di matrici
  CUDA_CHECK(cudaMallocManaged(&vec_pointer, sizeof(float *) * batchSize));
  CUDA_CHECK(cudaMallocManaged(&vec_inverse_pointer, sizeof(float*) * batchSize));
  
  // Allocate pivot and info arrays
  CUDA_CHECK(cudaMallocManaged(&pivot, sizeof(int) * col * batchSize));
  CUDA_CHECK(cudaMallocManaged(&info, sizeof(int) * batchSize));
  
  for (int j =0;j<batchSize;++j) {
    for (int i = 0; i < col2; ++i)
      vec[i + j * col2] = vec_exact[i];
    vec_inverse_pointer[j] = vec_inverse + j * col2;
    vec_pointer[j]=vec+j*col2;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cublasHandle_t cublasH;
  cublasCreate(&cublasH);
  inverseMatrix2(cublasH, vec_pointer, vec_inverse_pointer, col, batchSize, pivot, info);
  CUDA_CHECK(cudaDeviceSynchronize());
  for(int j=0;j<batchSize;++j) {
  for(int i=0;i<col2;++i) 
    EXPECT_NEAR(vec_inverse[i+j*col2], vec_inverse_exact[i], 2E-6);
  }
  // Free allocated memory
  CUDA_CHECK(cudaFree(pivot));
  CUDA_CHECK(cudaFree(info));
  CUDA_CHECK( cudaFree(vec) );
  CUDA_CHECK( cudaFree(vec_inverse) );
  CUDA_CHECK( cudaFree(vec_pointer) );
  CUDA_CHECK( cudaFree(vec_inverse_pointer) );
  cublasDestroy(cublasH);
}

float* einSum(std::vector<float> a, std::vector<int> a_shape,
            std::vector<float> b, std::vector<int> b_shape,
            std::string s) {
  
  cutensorHandle_t handle;
  cutensorCreate(&handle);
  /**********************
   * Setup planCache (optional)
   **********************/
  constexpr int32_t numCachelines = 1024;
  CUTENSOR_CHECK(cutensorHandleResizePlanCache(handle, numCachelines));
  //the output allocation is managed by einsum!
  float* output;

  float *a_device;
  int size_a = 1;
  for (auto dim : a_shape) {
    size_a=size_a*dim;
  }
  int size_b = 1;
  for (auto dim : b_shape) {
    size_b=size_b*dim;
  }
  CUDA_CHECK( cudaMallocManaged(&a_device, size_a * sizeof(float)));
  for (int i = 0; i < size_a; ++i)
    a_device[i] = a[i];
  
  float *b_device;
  CUDA_CHECK( cudaMallocManaged(&b_device, size_b*sizeof(float)) );
  for (int i = 0; i < size_b; ++i)
    b_device[i]=b[i];
  output =(float*) general_einsum(handle, a_shape, b_shape, a_device,b_device, s.c_str()); 
  cudaDeviceSynchronize();
  CUTENSOR_CHECK(cutensorDestroy(handle));
  CUDA_CHECK(cudaFree(a_device));
  CUDA_CHECK(cudaFree(b_device));
  return output;
}

float* einSumWrapped(std::vector<float> a, std::vector<int> a_shape,
            std::vector<float> b, std::vector<int> b_shape,
            std::string s) {
  EinsumWrapper einsum{s, a_shape, b_shape};
  float * output = einsum.allocate_output();
  float * workspace;
  CUDA_CHECK(cudaMalloc((void **)&workspace, einsum.workspace_size()));

  
  int size_a = 1;
  for (auto dim : a_shape) {
    size_a=size_a*dim;
  }
  int size_b = 1;
  for (auto dim : b_shape) {
    size_b=size_b*dim;
  }
  float *a_device;
  CUDA_CHECK( cudaMallocManaged(&a_device, size_a * sizeof(float)));
  for (int i = 0; i < size_a; ++i)
    a_device[i] = a[i];
  
  float *b_device;
  CUDA_CHECK( cudaMallocManaged(&b_device, size_b*sizeof(float)) );
  for (int i = 0; i < size_b; ++i)
    b_device[i]=b[i];

  cutensorHandle_t handle;
  cutensorCreate(&handle);
  einsum.execute(handle,a_device,b_device,workspace);
  cudaDeviceSynchronize();
  CUTENSOR_CHECK(cutensorDestroy(handle));
  CUDA_CHECK(cudaFree(a_device));
  CUDA_CHECK(cudaFree(b_device));
  return output;
}


TEST(EinSum, Test1) {
  std::vector<float> a = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
  std::vector<int> a_shape = {2,3};
  std::vector<float> b = {7.0f,8.0f,9.0f,10.0f,11.0f,12.0f};
  std::vector<int> b_shape = {2, 3};
  std::vector<float> result_exact = {120.0f, 165.0f, 168.0f,
                                     231.0f, 216.0f, 297.0f};
  float *result_device =
      (float *)einSum(a, a_shape, b, b_shape, std::string{"ei,jk->ij"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};
TEST(EinSumWrapped, Test1) {
  std::vector<float> a = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
  std::vector<int> a_shape = {2,3};
  std::vector<float> b = {7.0f,8.0f,9.0f,10.0f,11.0f,12.0f};
  std::vector<int> b_shape = {2, 3};
  std::vector<float> result_exact = {120.0f, 165.0f, 168.0f,
                                     231.0f, 216.0f, 297.0f};
  float *result_device =
      (float *)einSumWrapped(a, a_shape, b, b_shape, std::string{"ei,jk->ij"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};

TEST(EinSum, Test2) {
  std::vector<float> a = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
  std::vector<int> a_shape = {2,3};
  std::vector<float> b = {7.0f,8.0f,9.0f,10.0f,11.0f,12.0f};
  std::vector<int> b_shape = {2, 3};
  std::vector<float> result_exact = {7.0f, 16.0f,27.0f,40.0f,55.0f,72.0f};
  float *result_device =
      (float *)einSum(a, a_shape, b, b_shape, std::string{"ij,ij->ij"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
  CUDA_CHECK(cudaFree(result_device));
};
TEST(EinSumWrapped, Test2) {
  std::vector<float> a = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
  std::vector<int> a_shape = {2,3};
  std::vector<float> b = {7.0f,8.0f,9.0f,10.0f,11.0f,12.0f};
  std::vector<int> b_shape = {2, 3};
  std::vector<float> result_exact = {7.0f, 16.0f,27.0f,40.0f,55.0f,72.0f};
  float *result_device =
      (float *)einSumWrapped(a, a_shape, b, b_shape, std::string{"ij,ij->ij"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
  CUDA_CHECK(cudaFree(result_device));
};
TEST(EinSum, Test3) {
  std::vector<float> a = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
  std::vector<int> a_shape = {2,3};
  std::vector<float> b = {7.0f,8.0f,9.0f,10.0f,11.0f,12.0f};
  std::vector<int> b_shape = {2, 3};
  std::vector<float> result_exact = {144.0f, 495.0f};
  
  float *result_device =
      (float *)einSum(a, a_shape, b, b_shape, std::string{"ij,ik->i"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};
TEST(EinSumWrapped, Test3) {
  std::vector<float> a = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
  std::vector<int> a_shape = {2,3};
  std::vector<float> b = {7.0f,8.0f,9.0f,10.0f,11.0f,12.0f};
  std::vector<int> b_shape = {2, 3};
  std::vector<float> result_exact = {144.0f, 495.0f};
  
  float *result_device =
      (float *)einSumWrapped(a, a_shape, b, b_shape, std::string{"ij,ik->i"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};

TEST(EinSum, Test4) {
  std::vector<float> a = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
  std::vector<int> a_shape = {2,3};
  std::vector<float> b = {7.0f,8.0f,9.0f,10.0f,11.0f,12.0f};
  std::vector<int> b_shape = {2, 3};
  std::vector<float> result_exact = {50.0f, 122.0f,
                                     68.0f, 167.0f};
  float *result_device =
      (float *)einSum(a, a_shape, b, b_shape, std::string{"fc,gc->gf"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};
TEST(EinSumWrapped, Test4) {
  std::vector<float> a = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
  std::vector<int> a_shape = {2,3};
  std::vector<float> b = {7.0f,8.0f,9.0f,10.0f,11.0f,12.0f};
  std::vector<int> b_shape = {2, 3};
  std::vector<float> result_exact = {50.0f, 122.0f,
                                     68.0f, 167.0f};
  float *result_device =
      (float *)einSumWrapped(a, a_shape, b, b_shape, std::string{"fc,gc->gf"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};

TEST(EinSum, Test5) {
    std::vector<float> a = {
      1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,
      1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
    std::vector<int> a_shape = {2,2,3};
  std::vector<float> b = {7.0f,8.0f,9.0f,10.0f,11.0f,12.0f};
  std::vector<int> b_shape = {2, 3};
  std::vector<float> result_exact = {50.0f, 122.0f,
                                     68.0f, 167.0f};
  float *result_device =
      (float *)einSum(a, a_shape, b, b_shape, std::string{"ijk,ik->ij"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};
TEST(EinSumWrapped, Test5) {
    std::vector<float> a = {
      1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,
      1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
    std::vector<int> a_shape = {2,2,3};
  std::vector<float> b = {7.0f,8.0f,9.0f,10.0f,11.0f,12.0f};
  std::vector<int> b_shape = {2, 3};
  std::vector<float> result_exact = {50.0f, 122.0f,
                                     68.0f, 167.0f};
  float *result_device =
      (float *)einSum(a, a_shape, b, b_shape, std::string{"ijk,ik->ij"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};

TEST(EinSum, Transpose1) {
    std::vector<float> a = { 1.0f,2.0f,3.0f,4.0f};
    std::vector<int> a_shape = {2,2};
  std::vector<float> b = {1.0};
  std::vector<int> b_shape = {};
  std::vector<float> result_exact = {1.0f, 3.0f,
                                     2.0f, 4.0f};
  float *result_device =
    (float *)einSum(a, a_shape, b, b_shape, std::string{"ji->ij"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};
TEST(EinSumWrapped, Transpose1) {
    std::vector<float> a = { 1.0f,2.0f,3.0f,4.0f};
    std::vector<int> a_shape = {2,2};
  std::vector<float> b = {1.0};
  std::vector<int> b_shape = {};
  std::vector<float> result_exact = {1.0f, 3.0f,
                                     2.0f, 4.0f};
  float *result_device =
    (float *)einSumWrapped(a, a_shape, b, b_shape, std::string{"ji->ij"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};

TEST(EinSum, Transpose2) {
    std::vector<float> a = {1.0f,2.0f,
			    3.0f,4.0f, 5.0f,6.0f};
    std::vector<int> a_shape = {3,2};
  std::vector<float> b = {1.0};
  std::vector<int> b_shape = {};
  std::vector<float> result_exact = {1.0f, 3.0f,5.0f,
                                     2.0f, 4.0f,6.0f};
  float *result_device =
    (float *)einSum(a, a_shape, b, b_shape, std::string{"ji->ij"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};
TEST(EinSumWrapped, Transpose2) {
    std::vector<float> a = {1.0f,2.0f,
			    3.0f,4.0f, 5.0f,6.0f};
    std::vector<int> a_shape = {3,2};
  std::vector<float> b = {1.0};
  std::vector<int> b_shape = {};
  std::vector<float> result_exact = {1.0f, 3.0f,5.0f,
                                     2.0f, 4.0f,6.0f};
  float *result_device =
    (float *)einSumWrapped(a, a_shape, b, b_shape, std::string{"ji->ij"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};


TEST(EinSum, Broadcast1) {
    std::vector<float> a = {1.0f,2.0f,3.0f};
    std::vector<int> a_shape = {3};
    std::vector<float> b = {1.0f,2.0f,3.0f,4,5,6};
    std::vector<int> b_shape = {2,3};
    std::vector<float> result_exact = {6,15,12,30,18,45};
  float *result_device =
    (float *)einSum(a, a_shape, b, b_shape, std::string{"i,jk->ij"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};
TEST(EinSumWrapped, Broadcast1) {
    std::vector<float> a = {1.0f,2.0f,3.0f};
    std::vector<int> a_shape = {3};
    std::vector<float> b = {1.0f,2.0f,3.0f,4,5,6};
    std::vector<int> b_shape = {2,3};
    std::vector<float> result_exact = {6,15,12,30,18,45};
  float *result_device =
    (float *)einSum(a, a_shape, b, b_shape, std::string{"i,jk->ij"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};

TEST(EinSum, Broadcast2) {
  std::vector<float> a = {1.0f,2.0f,3.0f,4,5,6,1.0f,2.0f,3.0f,4,5,6, };
  std::vector<int> a_shape = {2,2,3};
  std::vector<float> b = {7,8,9,10,11,12};
    std::vector<int> b_shape = {2,3};
    std::vector<float> result_exact = {7,16,27,40,55,72,7,16,27,40,55,72};
  float *result_device =
    (float *)einSum(a, a_shape, b, b_shape, std::string{"ijk,jk->ijk"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};

TEST(EinSumWrapped, Broadcast2) {
  std::vector<float> a = {1.0f,2.0f,3.0f,4,5,6,1.0f,2.0f,3.0f,4,5,6, };
  std::vector<int> a_shape = {2,2,3};
  std::vector<float> b = {7,8,9,10,11,12};
    std::vector<int> b_shape = {2,3};
    std::vector<float> result_exact = {7,16,27,40,55,72,7,16,27,40,55,72};
  float *result_device =
    (float *)einSumWrapped(a, a_shape, b, b_shape, std::string{"ijk,jk->ijk"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};

TEST(EinSum, Broadcast3) {
  std::vector<float> a = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29};
  std::vector<int> a_shape = {2,3,5};
  std::vector<float> b = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
    std::vector<int> b_shape = {5,4};
    std::vector<float> result_exact = { 120,  130,  140,  150, 320,  355,  390,  425, 520,  580,  640,  700,720,  805,  890,  975, 920, 1030, 1140, 1250,1120, 1255, 1390, 1525};
  float *result_device =
    (float *)einSum(a, a_shape, b, b_shape, std::string{"gfc,ck->gfk"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};
TEST(EinSumWrapped, Broadcast3) {
  std::vector<float> a = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29};
  std::vector<int> a_shape = {2,3,5};
  std::vector<float> b = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
    std::vector<int> b_shape = {5,4};
    std::vector<float> result_exact = { 120,  130,  140,  150, 320,  355,  390,  425, 520,  580,  640,  700,720,  805,  890,  975, 920, 1030, 1140, 1250,1120, 1255, 1390, 1525};
  float *result_device =
    (float *)einSumWrapped(a, a_shape, b, b_shape, std::string{"gfc,ck->gfk"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
};

TEST(EinSum, Broadcast4) {
  std::vector<float> a = {8, 3, 3, 0, 3, 0, 5, 8, 1, 8, 4, 8, 9, 6, 7, 0, 9, 6, 9, 9, 3, 0, 7, 4, 7, 8, 9, 7, 7, 1, 9, 5, 0, 6, 2, 4};
  std::vector<int> a_shape = {6,2,3};
  std::vector<float> b = {6, 8, 9, 4, 2, 7};
  std::vector<int> b_shape = {6};
  std::vector<float> result_exact ={48, 18, 18, 0, 18, 0, 40, 64, 8, 64, 32, 64, 81, 54, 63, 0, 81, 54, 36, 36, 12, 0, 28, 16, 14, 16, 18, 14, 14, 2, 63, 35, 0, 42, 14, 28};

  float *result_device =
    (float *)einSum(a, a_shape, b, b_shape, std::string{"gfc,g->gfc"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
}



TEST(EinSumWrapped, Broadcast4) {
  std::vector<float> a = {8, 3, 3, 0, 3, 0, 5, 8, 1, 8, 4, 8, 9, 6, 7, 0, 9, 6, 9, 9, 3, 0, 7, 4, 7, 8, 9, 7, 7, 1, 9, 5, 0, 6, 2, 4};
  std::vector<int> a_shape = {6,2,3};
  std::vector<float> b = {6, 8, 9, 4, 2, 7};
  std::vector<int> b_shape = {6};
  std::vector<float> result_exact ={48, 18, 18, 0, 18, 0, 40, 64, 8, 64, 32, 64, 81, 54, 63, 0, 81, 54, 36, 36, 12, 0, 28, 16, 14, 16, 18, 14, 14, 2, 63, 35, 0, 42, 14, 28};

  float *result_device =
    (float *)einSumWrapped(a, a_shape, b, b_shape, std::string{"gfc,g->gfc"});
  std::vector<float> result(result_exact.size());
  cudaMemcpy(&result[0], result_device, result_exact.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < result_exact.size(); i++)
    EXPECT_NEAR(result[i], result_exact[i], 1E-6);
    CUDA_CHECK(cudaFree(result_device));
}



TEST(InvertMatrix, Exact3X3Batch) {
  invertExactBatch(
      std::vector<float>{2.0f, 0.0f, -1.0f, 5.0f, 1.0f, 0.0f, 0.0f, 1.0f, 3.0f},
      std::vector<float>{3.0f,  -1.0f, 1.0f,  -15.0f, 6.0f,
                               -5.0f, 5.0f,  -2.0f, 2.0f},
      3);
};


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

