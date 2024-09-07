// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: cutlass gemm

#pragma once

#include "driver_types.h"

/**
 * @brief cutlass gemm api: D = alpha * (A * B) + beta * C
 *
 * @tparam T
 * @param A [M, K], row major
 * @param B [K, N], col major
 * @param C [M, N], row major
 * @param D [M, N], row major
 * @param M
 * @param N
 * @param K
 * @param alpha
 * @param beta
 * @param stream
 */
template <typename T>
void gemm(const T *A, const T *B, T *C, T *D, size_t M, size_t N, size_t K, float alpha, float beta,
          cudaStream_t stream);
