// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: cuda timer

#pragma once

#include "common.h"

class CudaTimer {
public:
    CudaTimer(cudaStream_t stream = nullptr) : m_stream(stream) {
        CG_CHECK_CUDART_ERROR(cudaEventCreate(&m_start));
        CG_CHECK(m_start);
        CG_CHECK_CUDART_ERROR(cudaEventCreate(&m_end));
        CG_CHECK(m_end);
    }

    ~CudaTimer() {
        if (m_start) {
            CG_CHECK_CUDART_ERROR(cudaEventDestroy(m_start));
            m_start = nullptr;
        }

        if (m_end) {
            CG_CHECK_CUDART_ERROR(cudaEventDestroy(m_end));
            m_end = nullptr;
        }
    }

    void start() {
        CG_CHECK_CUDART_ERROR(cudaEventRecord(m_start, m_stream));
    }

    float end() {
        CG_CHECK_CUDART_ERROR(cudaEventRecord(m_end, m_stream));
        CG_CHECK_CUDART_ERROR(cudaEventSynchronize(m_end));
        CG_CHECK_CUDART_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_end));

        return m_elapsed_time;
    }

private:
    const cudaStream_t m_stream = nullptr;

    cudaEvent_t m_start = nullptr;
    cudaEvent_t m_end = nullptr;
    float m_elapsed_time = 0.0;

    CG_DISALLOW_COPY_AND_ASSIGN(CudaTimer);
};
