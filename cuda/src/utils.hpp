#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <iostream>
#include <stdio.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define CUSOLVER_CHECK(call)                                                   \
  {                                                                            \
    cusolverStatus_t err = call;                                               \
    if (err != CUSOLVER_STATUS_SUCCESS) {                                      \
      std::cerr << "cuSOLVER error in file '" << __FILE__ << "' in line "      \
                << __LINE__ << ": " << cusolverGetErrorString(err)             \
                << std::endl;                                                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

inline const char* cusolverGetErrorString(cusolverStatus_t status) {
    switch (status) {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_MAPPING_ERROR:
            return "CUSOLVER_STATUS_MAPPING_ERROR";
        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        default:
            return "Unknown cuSOLVER error";
    }
}

#define CUTENSOR_CHECK(x) { const auto err = x;\
    if (err != CUTENSOR_STATUS_SUCCESS) {printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__); exit(-1); } }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA_CHECK: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


#define CUBLAS_CHECK(call)  { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error in file '" << __FILE__ << "' at line " << __LINE__ << ": " << cublasGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} 

// Function to convert cuBLAS error codes to strings
inline const char* cublasGetErrorString(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        default:
            return "Unknown cuBLAS error";
    }
}


