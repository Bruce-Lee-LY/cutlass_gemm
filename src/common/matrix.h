// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: matrix

#pragma once

#include <random>

#include "common.h"

template <typename T>
class Matrix {
public:
    Matrix(size_t row, size_t col, const std::string &name = "Matrix", float min = -1.0, float max = 1.0)
        : m_row(row), m_col(col), m_name(name), m_min(min), m_max(max) {
        CG_CHECK_GT(m_row, 0);
        CG_CHECK_GT(m_col, 0);

        m_elem_num = m_row * m_col;
        CG_CHECK_GT(m_elem_num, 0);

        m_host_ptr = new T[m_elem_num];
        CG_CHECK(m_host_ptr);
        CG_CHECK_CUDART_ERROR(cudaMalloc((void **)&m_dev_ptr, m_elem_num * sizeof(T)));
        CG_CHECK(m_dev_ptr);

        std::random_device rd;
        std::default_random_engine engine{rd()};
        std::uniform_real_distribution<float> uniform(m_min, m_max);
        for (size_t i = 0; i < m_elem_num; ++i) {
            m_host_ptr[i] = static_cast<T>(uniform(engine));
        }

        CG_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr, m_elem_num * sizeof(T), cudaMemcpyHostToDevice));

        CLOG("%s: %zu * %zu, cpu: %p, gpu: %p", m_name.c_str(), m_row, m_col, m_host_ptr, m_dev_ptr);
    }

    ~Matrix() {
        if (m_host_ptr) {
            delete[] m_host_ptr;
            m_host_ptr = nullptr;
        }

        if (m_dev_ptr) {
            CG_CHECK_CUDART_ERROR(cudaFree((void *)m_dev_ptr));
            m_dev_ptr = nullptr;
        }
    }

    size_t getRow() const {
        return m_row;
    }

    size_t getCol() const {
        return m_col;
    }

    size_t getElemNum() const {
        return m_elem_num;
    }

    T *getHostPtr() const {
        return m_host_ptr;
    }

    T *getDevPtr() const {
        return m_dev_ptr;
    }

    void moveToHost() {
        CG_CHECK_CUDART_ERROR(cudaMemcpy(m_host_ptr, m_dev_ptr, m_elem_num * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void moveToDevice() {
        CG_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr, m_elem_num * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copyHost(Matrix<T> *base) {
        CG_CHECK(base);
        CG_CHECK_EQ(m_row, base->getRow());
        CG_CHECK_EQ(m_col, base->getCol());

        memcpy(m_host_ptr, base->getHostPtr(), m_elem_num * sizeof(T));
    }

    void copyDevice(Matrix<T> *base) {
        CG_CHECK(base);
        CG_CHECK_EQ(m_row, base->getRow());
        CG_CHECK_EQ(m_col, base->getCol());

        CG_CHECK_CUDART_ERROR(
            cudaMemcpy(m_dev_ptr, base->getDevPtr(), m_elem_num * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    void memSetHost() {
        memset(m_host_ptr, 0, m_elem_num * sizeof(T));
    }

    void memSetDevice() {
        CG_CHECK_CUDART_ERROR(cudaMemset(m_dev_ptr, 0, m_elem_num * sizeof(T)));
    }

    void checkValue(Matrix<T> *base) {
        CG_CHECK(base);
        CG_CHECK_EQ(m_row, base->getRow());
        CG_CHECK_EQ(m_col, base->getCol());

        m_max_diff = 0.0;
        m_avg_diff = 0.0;
        double diff = 0.0;
        for (size_t i = 0; i < m_elem_num; ++i) {
            diff = static_cast<double>(
                std::abs(static_cast<float>(m_host_ptr[i]) - static_cast<float>(base->getHostPtr()[i])));
            m_max_diff = std::max(m_max_diff, diff);
            m_avg_diff += diff;
        }

        m_avg_diff /= static_cast<double>(m_elem_num);

        CLOG("Max diff: %f, avg diff: %f", m_max_diff, m_avg_diff);
    }

private:
    const size_t m_row = 0;
    const size_t m_col = 0;
    const std::string m_name = "Matrix";
    // the threshold of the random matrix will affect the difference of the gemm results
    const float m_min = -1.0;
    const float m_max = 1.0;

    size_t m_elem_num = 0;
    T *m_host_ptr = nullptr;
    T *m_dev_ptr = nullptr;

    double m_max_diff = 0.0;
    double m_avg_diff = 0.0;

    CG_DISALLOW_COPY_AND_ASSIGN(Matrix);
};
