// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: cutlass gemm impl

#pragma once

#include "gemm_sm80.h"

template <typename T>
void gemm(const T *A, const T *B, T *C, T *D, size_t M, size_t N, size_t K, float alpha, float beta,
          cudaStream_t stream) {
    dispatch_gemm_sm80<T>(A, B, C, D, M, N, K, alpha, beta, stream);
}
