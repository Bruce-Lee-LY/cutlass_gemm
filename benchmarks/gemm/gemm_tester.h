// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: gemm tester

#pragma once

#include <memory>

#include "cuda_timer.h"
#include "matrix.h"

template <typename T>
class GemmTester {
public:
    explicit GemmTester(size_t M = 512, size_t N = 2048, size_t K = 1024, float alpha = 1.0, float beta = 0.0,
                        cudaStream_t stream = nullptr, size_t warmup_iterations = 1, size_t profiling_iterations = 10,
                        size_t sleep_duration = 100, bool enable_check = false)
        : m_M(M),
          m_N(N),
          m_K(K),
          m_alpha(alpha),
          m_beta(beta),
          m_stream(stream),
          m_warmup_iterations(warmup_iterations),
          m_profiling_iterations(profiling_iterations),
          m_sleep_duration(sleep_duration),
          m_enable_check(enable_check) {
        CG_CHECK_GT(m_M, 0);
        CG_CHECK_GT(m_N, 0);
        CG_CHECK_GT(m_K, 0);
        CG_CHECK_GT(m_warmup_iterations, 0);
        CG_CHECK_GT(m_profiling_iterations, 0);
        CG_CHECK_GT(m_sleep_duration, 0);

        m_A = std::make_shared<Matrix<T>>(m_M, m_K, "Matrix A");
        CG_CHECK(m_A);
        m_B = std::make_shared<Matrix<T>>(m_K, m_N, "Matrix B");
        CG_CHECK(m_B);
        m_C = std::make_shared<Matrix<T>>(m_M, m_N, "Matrix C");
        CG_CHECK(m_C);
        m_D = std::make_shared<Matrix<T>>(m_M, m_N, "Matrix D");
        CG_CHECK(m_D);
        m_base = std::make_shared<Matrix<T>>(m_M, m_N, "Matrix Base");
        CG_CHECK(m_base);
        m_D->copyDevice(m_C.get());
        m_base->copyDevice(m_C.get());

        m_cuda_timer = std::make_shared<CudaTimer>(m_stream);
        CG_CHECK(m_cuda_timer);

        if (m_enable_check) {
            m_cuda_timer->start();
            gemm_cublas(m_A->getDevPtr(), m_B->getDevPtr(), m_base->getDevPtr(), m_M, m_N, m_K, m_alpha, m_beta,
                        m_stream);
            CLOG("Cublas-Gemm use: %.3f ms", m_cuda_timer->end());
            m_base->moveToHost();
            m_base->memSetDevice();
        }
    }

    ~GemmTester() {}

    template <typename Func>
    void evaluate(Func &&gemm, const std::string &name) {
        CLOG("----------------- Evaluating %s -----------------", name.c_str());
        usleep(m_sleep_duration * 1000);
        m_D->copyDevice(m_C.get());

        // warm up
        m_cuda_timer->start();
        for (size_t i = 0; i < m_warmup_iterations; ++i) {
            gemm(m_A->getDevPtr(), m_B->getDevPtr(), m_D->getDevPtr(), m_D->getDevPtr(), m_M, m_N, m_K, m_alpha, m_beta,
                 m_stream);
        }
        m_warmup_time = static_cast<double>(m_cuda_timer->end()) / static_cast<double>(m_warmup_iterations);
        CLOG("Warm up time: %.3f ms", m_warmup_time);

        if (m_enable_check) {
            m_D->moveToHost();
            m_D->checkValue(m_base.get());
        }

        profile(std::forward<Func>(gemm), name);
    }

private:
    void gemm_cublas(const T *A, const T *B, T *C, size_t M, size_t N, size_t K, float alpha, float beta,
                     cudaStream_t stream) {
        cublasHandle_t handle = nullptr;
        CG_CHECK_CUBLAS_ERROR(cublasCreate(&handle));
        CG_CHECK_CUBLAS_ERROR(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
        CG_CHECK_CUBLAS_ERROR(cublasSetStream(handle, stream));

        cudaDataType A_type = std::is_same_v<T, cutlass::bfloat16_t> ? CUDA_R_16BF : CUDA_R_16F;
        cudaDataType B_type = std::is_same_v<T, cutlass::bfloat16_t> ? CUDA_R_16BF : CUDA_R_16F;
        cudaDataType C_type = std::is_same_v<T, cutlass::bfloat16_t> ? CUDA_R_16BF : CUDA_R_16F;

        CG_CHECK_CUBLAS_ERROR(
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, reinterpret_cast<const void *>(B), B_type,
                         K, reinterpret_cast<const void *>(A), A_type, K, &beta, reinterpret_cast<void *>(C), C_type, N,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CG_CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    }

    template <typename Func>
    void profile(Func &&gemm, const std::string &name) {
        m_cuda_timer->start();
        for (size_t i = 0; i < m_profiling_iterations; ++i) {
            gemm(m_A->getDevPtr(), m_B->getDevPtr(), m_D->getDevPtr(), m_D->getDevPtr(), m_M, m_N, m_K, m_alpha, m_beta,
                 m_stream);
        }
        m_profiling_time = static_cast<double>(m_cuda_timer->end()) / static_cast<double>(m_profiling_iterations);
        m_throughput = static_cast<double>(m_M * m_N * m_K * 2 * 1e-12) / static_cast<double>(m_profiling_time * 1e-3);

        if ((std::abs(m_base_time) <= 1e-6) && (std::abs(m_base_throughput) <= 1e-6)) {
            m_base_time = m_profiling_time;
            m_base_throughput = m_throughput;
        }

        CLOG("%s exit, profiling time: %.3f ms (%.2f%%), throughput: %.3f TFLOPS (%.2f%%)", name.c_str(),
             m_profiling_time, m_profiling_time / m_base_time * 100, m_throughput,
             m_throughput / m_base_throughput * 100);
    }

    const size_t m_M = 512;
    const size_t m_N = 2048;
    const size_t m_K = 1024;
    const float m_alpha = 1.0;
    const float m_beta = 0.0;
    const cudaStream_t m_stream = nullptr;
    const size_t m_warmup_iterations = 1;
    const size_t m_profiling_iterations = 10;
    const size_t m_sleep_duration = 100;
    const bool m_enable_check = false;

    std::shared_ptr<Matrix<T>> m_A = nullptr;     // row major, M * K
    std::shared_ptr<Matrix<T>> m_B = nullptr;     // col major, K * N
    std::shared_ptr<Matrix<T>> m_C = nullptr;     // row major, M * N, raw data
    std::shared_ptr<Matrix<T>> m_D = nullptr;     // row major, M * N, compute result
    std::shared_ptr<Matrix<T>> m_base = nullptr;  // row major, M * N, base result, init matrix D before each gemm

    std::shared_ptr<CudaTimer> m_cuda_timer = nullptr;

    double m_warmup_time = 0.0;
    double m_profiling_time = 0.0;
    double m_throughput = 0.0;
    double m_base_time = 0.0;        // cublas tensor op default
    double m_base_throughput = 0.0;  // cublas tensor op default

    CG_DISALLOW_COPY_AND_ASSIGN(GemmTester);
};
