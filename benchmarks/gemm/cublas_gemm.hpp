// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: cublas gemm

#include "common.h"

cublasHandle_t getCublasHandle(cudaStream_t stream) {
    cublasHandle_t handle = nullptr;
    CG_CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    CG_CHECK_CUBLAS_ERROR(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    CG_CHECK_CUBLAS_ERROR(cublasSetStream(handle, stream));

    return handle;
}

template <typename T>
void gemmCublas(const T *A, const T *B, T *C, T *D, size_t M, size_t N, size_t K, float alpha, float beta,
                cudaStream_t stream) {
    CG_CHECK_EQ(C, D);

    cudaDataType A_type = std::is_same_v<T, cutlass::bfloat16_t> ? CUDA_R_16BF : CUDA_R_16F;
    cudaDataType B_type = std::is_same_v<T, cutlass::bfloat16_t> ? CUDA_R_16BF : CUDA_R_16F;
    cudaDataType C_type = std::is_same_v<T, cutlass::bfloat16_t> ? CUDA_R_16BF : CUDA_R_16F;

    static cublasHandle_t handle = getCublasHandle(stream);

    CG_CHECK_CUBLAS_ERROR(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                                       reinterpret_cast<const void *>(B), B_type, K, reinterpret_cast<const void *>(A),
                                       A_type, K, &beta, reinterpret_cast<void *>(C), C_type, N, CUBLAS_COMPUTE_32F,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
