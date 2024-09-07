// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: cutlass gemm

#include "gemm.h"
#include "gemm_impl.h"

template void gemm<cutlass::bfloat16_t>(const cutlass::bfloat16_t *A, const cutlass::bfloat16_t *B,
                                        cutlass::bfloat16_t *C, cutlass::bfloat16_t *D, size_t M, size_t N, size_t K,
                                        float alpha, float beta, cudaStream_t stream);

template void gemm<cutlass::half_t>(const cutlass::half_t *A, const cutlass::half_t *B, cutlass::half_t *C,
                                    cutlass::half_t *D, size_t M, size_t N, size_t K, float alpha, float beta,
                                    cudaStream_t stream);
