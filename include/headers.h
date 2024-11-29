#ifndef headers_H_
#define headers_H_
// Include generic C++ headers
#include <fenv.h>
#include <fstream>
#include <iostream>

// include cuda thrust vectors
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <cublas_v2.h>

#include <BLASWrappers.h>
#define CHECK_CUDA(func)                                         \
  {                                                              \
    cudaError_t status = (func);                                 \
    if (status != cudaSuccess)                                   \
    {                                                            \
      printf("CUDA API failed at line %d with error: %s (%d)\n", \
             __LINE__, cudaGetErrorString(status), status);      \
      exit(EXIT_FAILURE);                                        \
    }                                                            \
  }

#define CHECK_CUSPARSE(func)                                         \
  {                                                                  \
    cusparseStatus_t status = (func);                                \
    if (status != CUSPARSE_STATUS_SUCCESS)                           \
    {                                                                \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
             __LINE__, cusparseGetErrorString(status), status);      \
      exit(EXIT_FAILURE);                                            \
    }                                                                \
  }
#define CHECK_CUBLAS(func)                                                           \
  {                                                                                  \
    cublasStatus_t status = (func);                                                  \
    if (status != CUBLAS_STATUS_SUCCESS)                                             \
    {                                                                                \
      printf(                                                                        \
          "cuBLAS error on or before line number %d in file: %s. Error code: %d.\n", \
          __LINE__,                                                                  \
          __FILE__,                                                                  \
          status);                                                                   \
    }                                                                                \
  }

#endif
