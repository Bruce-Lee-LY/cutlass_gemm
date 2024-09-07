// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: cutlass gemm sm80

#pragma once

#include "common.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/device_memory.h"

template <typename T, size_t cta_m, size_t cta_n, size_t cta_k, size_t warp_m, size_t warp_n, size_t warp_k,
          size_t stages, size_t split_k = 1>
void gemm_sm80_kernel(const T *A, const T *B, T *C, T *D, size_t M, size_t N, size_t K, float alpha, float beta,
                      cudaStream_t stream) {
    using ElementInputA = T;
    using ElementInputB = T;
    using ElementOutput = T;
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;
    using ElementAccumulator = float;

    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;

    using ThreadblockShape = cutlass::gemm::GemmShape<cta_m, cta_n, cta_k>;
    using WarpShape = cutlass::gemm::GemmShape<warp_m, warp_n, warp_k>;
    using MMAOpShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    constexpr static bool is_split = split_k > 1;

    using ElementComputeEpilogue = ElementAccumulator;
    using EpilogueOp =
        cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                     ElementAccumulator, ElementComputeEpilogue>;

    using Gemm = typename cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator,
        MMAOp, SmArch, ThreadblockShape, WarpShape, MMAOpShape, EpilogueOp, SwizzleThreadBlock, stages, 8, 8, is_split>;

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    const size_t &lda = std::is_same_v<LayoutInputA, cutlass::layout::RowMajor> ? K : M;
    const size_t &ldb = std::is_same_v<LayoutInputB, cutlass::layout::RowMajor> ? N : K;
    const size_t &ldc = std::is_same_v<LayoutOutput, cutlass::layout::RowMajor> ? N : M;
    const size_t &ldd = ldc;
    cutlass::TensorRef<const ElementInputA, LayoutInputA> tensor_a_ref(reinterpret_cast<const ElementInputA *>(A), lda);
    cutlass::TensorRef<const ElementInputB, LayoutInputB> tensor_b_ref(reinterpret_cast<const ElementInputB *>(B), ldb);
    cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c_ref(reinterpret_cast<ElementOutput *>(C), ldc);
    cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_d_ref(reinterpret_cast<ElementOutput *>(D), ldd);

    typename Gemm::Arguments arguments{problem_size, tensor_a_ref,  tensor_b_ref, tensor_c_ref,
                                       tensor_d_ref, {alpha, beta}, split_k};

    void *workspace = nullptr;

    Gemm gemm_op;
    if constexpr (is_split) {
        size_t workspace_size = gemm_op.get_workspace_size(arguments);
        cutlass::device_memory::allocation<uint8_t> workspace_mem =
            cutlass::device_memory::allocation<uint8_t>(workspace_size);
        workspace = workspace_mem.get();
    }

    CG_CHECK_CUTLASS_ERROR(gemm_op.can_implement(arguments));
    CG_CHECK_CUTLASS_ERROR(gemm_op.initialize(arguments, workspace, stream));
    CG_CHECK_CUTLASS_ERROR(gemm_op.run(stream));
}

template <typename T>
void dispatch_gemm_sm80(const T *A, const T *B, T *C, T *D, size_t M, size_t N, size_t K, float alpha, float beta,
                        cudaStream_t stream) {
    if (M < 64) {
        gemm_sm80_kernel<T, 32, 128, 32, 32, 32, 32, 7>(A, B, C, D, M, N, K, alpha, beta, stream);
    } else if (M < 128) {
        if (K < 6144) {
            gemm_sm80_kernel<T, 32, 128, 32, 32, 32, 32, 7>(A, B, C, D, M, N, K, alpha, beta, stream);
        } else {
            gemm_sm80_kernel<T, 64, 128, 32, 32, 32, 32, 6>(A, B, C, D, M, N, K, alpha, beta, stream);
        }
    } else if (M < 256) {
        if (K < 6144) {
            gemm_sm80_kernel<T, 64, 128, 32, 32, 32, 32, 6>(A, B, C, D, M, N, K, alpha, beta, stream);

        } else {
            gemm_sm80_kernel<T, 128, 128, 32, 32, 64, 32, 5>(A, B, C, D, M, N, K, alpha, beta, stream);
        }
    } else if (M < 768) {
        if (K < 6144) {
            gemm_sm80_kernel<T, 128, 128, 32, 32, 64, 32, 5>(A, B, C, D, M, N, K, alpha, beta, stream);
        } else {
            gemm_sm80_kernel<T, 128, 256, 32, 64, 64, 32, 4>(A, B, C, D, M, N, K, alpha, beta, stream);
        }
    } else if (M < 1536) {
        if (K < 6144) {
            gemm_sm80_kernel<T, 128, 256, 32, 64, 64, 32, 4>(A, B, C, D, M, N, K, alpha, beta, stream);
        } else {
            gemm_sm80_kernel<T, 128, 128, 64, 64, 64, 64, 3>(A, B, C, D, M, N, K, alpha, beta, stream);
        }
    } else if (M < 2560) {
        if (K < 6144) {
            gemm_sm80_kernel<T, 128, 128, 64, 64, 64, 64, 3>(A, B, C, D, M, N, K, alpha, beta, stream);
        } else {
            gemm_sm80_kernel<T, 128, 256, 32, 64, 64, 32, 4>(A, B, C, D, M, N, K, alpha, beta, stream);
        }
    } else {
        if (M < 12288) {
            gemm_sm80_kernel<T, 128, 256, 32, 64, 64, 32, 4>(A, B, C, D, M, N, K, alpha, beta, stream);
        } else {
            gemm_sm80_kernel<T, 256, 128, 32, 64, 64, 32, 4>(A, B, C, D, M, N, K, alpha, beta, stream);
        }
    }
}
