// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: cublasLt gemm

#include "common.h"

cublasLtHandle_t getCublasLtHandle() {
    cublasLtHandle_t handle = nullptr;
    CG_CHECK_CUBLAS_ERROR(cublasLtCreate(&handle));

    return handle;
}

cublasLtMatmulDesc_t getCublasLtMatmulDesc() {
    cublasLtMatmulDesc_t desc = nullptr;
    CG_CHECK_CUBLAS_ERROR(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t A_op = CUBLAS_OP_T;
    cublasOperation_t B_op = CUBLAS_OP_N;
    CG_CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &A_op, sizeof(A_op)));
    CG_CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &B_op, sizeof(B_op)));

    return desc;
}

cublasLtMatrixLayout_t getCublasLtMatrixLayout(cudaDataType dtype, size_t row, size_t col, size_t ld) {
    cublasLtMatrixLayout_t layout = nullptr;
    CG_CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutCreate(&layout, dtype, row, col, ld));

    return layout;
}

template <typename T>
void gemmCublasLt(const T *A, const T *B, T *C, T *D, size_t M, size_t N, size_t K, float alpha, float beta,
                  cudaStream_t stream) {
    cudaDataType A_type = std::is_same_v<T, cutlass::bfloat16_t> ? CUDA_R_16BF : CUDA_R_16F;
    cudaDataType B_type = std::is_same_v<T, cutlass::bfloat16_t> ? CUDA_R_16BF : CUDA_R_16F;
    cudaDataType C_type = std::is_same_v<T, cutlass::bfloat16_t> ? CUDA_R_16BF : CUDA_R_16F;
    cudaDataType D_type = std::is_same_v<T, cutlass::bfloat16_t> ? CUDA_R_16BF : CUDA_R_16F;

    static cublasLtHandle_t handle = getCublasLtHandle();
    static cublasLtMatmulDesc_t desc = getCublasLtMatmulDesc();
    static cublasLtMatrixLayout_t A_layout = getCublasLtMatrixLayout(A_type, K, M, K);
    static cublasLtMatrixLayout_t B_layout = getCublasLtMatrixLayout(B_type, K, N, K);
    static cublasLtMatrixLayout_t C_layout = getCublasLtMatrixLayout(C_type, N, M, N);
    static cublasLtMatrixLayout_t D_layout = getCublasLtMatrixLayout(D_type, N, M, N);
    static cublasLtMatmulAlgo_t *algo = nullptr;
    static void *workspace = nullptr;
    static size_t workspace_size = 0;

    CG_CHECK_CUBLAS_ERROR(cublasLtMatmul(handle, desc, &alpha, reinterpret_cast<const void *>(B), B_layout,
                                         reinterpret_cast<const void *>(A), A_layout, &beta,
                                         reinterpret_cast<void *>(C), C_layout, reinterpret_cast<void *>(D), D_layout,
                                         algo, workspace, workspace_size, stream));
}
