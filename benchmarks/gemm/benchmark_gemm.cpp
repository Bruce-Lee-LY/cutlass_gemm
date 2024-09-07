// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: benchmark gemm

#include "cublas_gemm.hpp"
#include "cublaslt_gemm.hpp"
#include "gemm.h"
#include "gemm_tester.h"
#include "gflags/gflags.h"
#include "omp.h"

DEFINE_uint32(M, 512, "M");
DEFINE_uint32(N, 2048, "N");
DEFINE_uint32(K, 1024, "K");
DEFINE_double(alpha, 1.0, "alpha");
DEFINE_double(beta, 0.0, "beta");
DEFINE_bool(is_bf16, false, "data type of A, B, C and D");
DEFINE_uint32(warmup_iterations, 1, "warmup iteration numbers and average the result");
DEFINE_uint32(profiling_iterations, 10, "profiling iteration numbers and average the result");
DEFINE_uint32(sleep_duration, 100, "sleep_milliseconds between profiling");
DEFINE_bool(enable_check, false, "check the GPU result against the cublas result");
DEFINE_uint32(cpu_procs, omp_get_num_procs(), "processor num used of CPU");
DEFINE_uint32(gpu_rank, 0, "the used GPU rank");

template <typename T>
void test_gemm(size_t M = 512, size_t N = 2048, size_t K = 1024, float alpha = 1.0, float beta = 0.0,
               cudaStream_t stream = nullptr, size_t warmup_iterations = 1, size_t profiling_iterations = 10,
               size_t sleep_duration = 100, bool enable_check = false) {
    GemmTester<T> tester(M, N, K, alpha, beta, stream, warmup_iterations, profiling_iterations, sleep_duration,
                         enable_check);
    tester.evaluate(gemmCublas<T>, "Cublas-Gemm");
    tester.evaluate(gemmCublasLt<T>, "CublasLt-Gemm");
    tester.evaluate(gemm<T>, "Cutlass-Gemm");
}

int main(int argc, char *argv[]) {
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    omp_set_num_threads(FLAGS_cpu_procs);
    CG_CHECK_CUDART_ERROR(cudaSetDevice(FLAGS_gpu_rank));

    cudaDeviceProp dev_prop;
    CG_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, FLAGS_gpu_rank));
    CLOG("CUTLASS GEMM start with %u CPU processes on the %u-th GPU: %s", FLAGS_cpu_procs, FLAGS_gpu_rank,
         dev_prop.name);

    int driver_version = 0;
    int runtime_version = 0;
    CG_CHECK_CUDART_ERROR(cudaDriverGetVersion(&driver_version));
    CG_CHECK_CUDART_ERROR(cudaRuntimeGetVersion(&runtime_version));
    CLOG("CUDA driver version / runtime version: %d.%d / %d.%d", driver_version / 1000, (driver_version % 100) / 10,
         runtime_version / 1000, (runtime_version % 100) / 10);
    CLOG("CUDA capability major/minor version number: %d.%d", dev_prop.major, dev_prop.minor);
    CLOG("%d multiprocessors, %d CUDA cores/MP: %d CUDA cores", dev_prop.multiProcessorCount,
         convert_SM_to_cores(dev_prop.major, dev_prop.minor),
         convert_SM_to_cores(dev_prop.major, dev_prop.minor) * dev_prop.multiProcessorCount);
    CLOG("GPU max clock rate: %.0f MHz (%0.2f GHz)", static_cast<double>(dev_prop.clockRate) * 1e-3,
         static_cast<double>(dev_prop.clockRate) * 1e-6);
    CLOG("Memory clock rate: %.0f MHz (%0.2f GHz)", static_cast<double>(dev_prop.memoryClockRate) * 1e-3,
         static_cast<double>(dev_prop.memoryClockRate) * 1e-6);
    CLOG("Memory bus width: %d-bit", dev_prop.memoryBusWidth);
    CLOG("Total amount of global memory: %.0f MBytes (%zu Bytes)",
         static_cast<double>(dev_prop.totalGlobalMem) / 1048576, dev_prop.totalGlobalMem);
    CLOG("Total amount of constant memory: %.0f KBytes (%zu Bytes)", static_cast<double>(dev_prop.totalConstMem) / 1024,
         dev_prop.totalConstMem);
    CLOG("Total amount of shared memory per block: %.0f KBytes (%zu Bytes)",
         static_cast<double>(dev_prop.sharedMemPerBlock) / 1024, dev_prop.sharedMemPerBlock);
    CLOG("Total shared memory per multiprocessor: %.0f KBytes (%zu Bytes)",
         static_cast<double>(dev_prop.sharedMemPerMultiprocessor) / 1024, dev_prop.sharedMemPerMultiprocessor);
    CLOG("L2 cache size: %.0f KBytes (%d Bytes)", static_cast<double>(dev_prop.l2CacheSize) / 1024,
         dev_prop.l2CacheSize);
    CLOG("Total number of registers available per block: %d", dev_prop.regsPerBlock);
    CLOG("Warp size: %d", dev_prop.warpSize);
    CLOG("Max number of threads per multiprocessor: %d", dev_prop.maxThreadsPerMultiProcessor);
    CLOG("Max number of threads per block: %d", dev_prop.maxThreadsPerBlock);
    CLOG("Max dimension size of a thread block (x,y,z): (%d, %d, %d)", dev_prop.maxThreadsDim[0],
         dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
    CLOG("Max dimension size of a grid size (x,y,z): (%d, %d, %d)", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1],
         dev_prop.maxGridSize[2]);

    cudaStream_t stream = nullptr;

    CLOG("A (%u x %u) * B (%u x %u) = C (%u x %u)", FLAGS_M, FLAGS_K, FLAGS_K, FLAGS_N, FLAGS_M, FLAGS_N);
    CLOG(
        "Profiling: alpha: %f, beta: %f, stream: %p, is bf16: %d, warmup iterations: %u, profiling iterations: %u, "
        "sleep duration: %u ms, enable check: %d",
        FLAGS_alpha, FLAGS_beta, stream, FLAGS_is_bf16, FLAGS_warmup_iterations, FLAGS_profiling_iterations,
        FLAGS_sleep_duration, FLAGS_enable_check);

    if (FLAGS_is_bf16) {
        test_gemm<cutlass::bfloat16_t>(FLAGS_M, FLAGS_N, FLAGS_K, FLAGS_alpha, FLAGS_beta, stream,
                                       FLAGS_warmup_iterations, FLAGS_profiling_iterations, FLAGS_sleep_duration,
                                       FLAGS_enable_check);
    } else {
        test_gemm<cutlass::half_t>(FLAGS_M, FLAGS_N, FLAGS_K, FLAGS_alpha, FLAGS_beta, stream, FLAGS_warmup_iterations,
                                   FLAGS_profiling_iterations, FLAGS_sleep_duration, FLAGS_enable_check);
    }

    GFLAGS_NAMESPACE::ShutDownCommandLineFlags();

    CLOG("Done");

    return 0;
}
