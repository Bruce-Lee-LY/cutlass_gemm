// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: common macro

#pragma once

#include <stdint.h>

#include <algorithm>

#include "cublasLt.h"
#include "cublas_v2.h"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "logging.h"
#include "util.h"

#define CG_LIKELY(x) __builtin_expect(!!(x), 1)
#define CG_UNLIKELY(x) __builtin_expect(!!(x), 0)

#define CG_CHECK(x)                       \
    do {                                  \
        if (CG_UNLIKELY(!(x))) {          \
            CLOG("Check failed: %s", #x); \
            exit(EXIT_FAILURE);           \
        }                                 \
    } while (0)

#define CG_CHECK_EQ(x, y) CG_CHECK((x) == (y))
#define CG_CHECK_NE(x, y) CG_CHECK((x) != (y))
#define CG_CHECK_LE(x, y) CG_CHECK((x) <= (y))
#define CG_CHECK_LT(x, y) CG_CHECK((x) < (y))
#define CG_CHECK_GE(x, y) CG_CHECK((x) >= (y))
#define CG_CHECK_GT(x, y) CG_CHECK((x) > (y))

#define CG_DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName &) = delete;      \
    void operator=(const TypeName &) = delete

#define CG_CHECK_CUDART_ERROR(_expr_)                                                             \
    do {                                                                                          \
        cudaError_t _ret_ = _expr_;                                                               \
        if (CG_UNLIKELY(_ret_ != cudaSuccess)) {                                                  \
            const char *_err_str_ = cudaGetErrorName(_ret_);                                      \
            int _rt_version_ = 0;                                                                 \
            cudaRuntimeGetVersion(&_rt_version_);                                                 \
            int _driver_version_ = 0;                                                             \
            cudaDriverGetVersion(&_driver_version_);                                              \
            CLOG("CUDA Runtime API error = %04d \"%s\", runtime version: %d, driver version: %d", \
                 static_cast<int>(_ret_), _err_str_, _rt_version_, _driver_version_);             \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

#define CG_CHECK_CUBLAS_ERROR(_expr_)                                                                     \
    do {                                                                                                  \
        cublasStatus_t _ret_ = _expr_;                                                                    \
        if (CG_UNLIKELY(_ret_ != CUBLAS_STATUS_SUCCESS)) {                                                \
            size_t _rt_version_ = cublasGetCudartVersion();                                               \
            CLOG("CUBLAS API error = %04d, runtime version: %zu", static_cast<int>(_ret_), _rt_version_); \
            exit(EXIT_FAILURE);                                                                           \
        }                                                                                                 \
    } while (0)

#define CG_CHECK_CUTLASS_ERROR(_expr_)                                                   \
    do {                                                                                 \
        cutlass::Status _ret_ = _expr_;                                                  \
        if (CG_UNLIKELY(_ret_ != cutlass::Status::kSuccess)) {                           \
            const char *_err_str_ = cutlassGetStatusString(_ret_);                       \
            CLOG("CUTLASS API error = %04d \"%s\"", static_cast<int>(_ret_), _err_str_); \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

#define CG_UNUSED(x) (void)(x)
