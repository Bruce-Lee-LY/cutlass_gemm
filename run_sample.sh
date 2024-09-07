# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 20:42:28 on Sun, Feb 12, 2023
#
# Description: run sample script

#!/bin/bash

set -euo pipefail

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

rm -rf log ncu && mkdir -p log ncu

# $1: M, $2: N, $3: K
evaluate_gemm() {
    echo "Evaluating $1 * $2 * $3"
    $WORK_PATH/output/bin/benchmark_gemm -M=$1 -N=$2 -K=$3 -alpha=1.0 -beta=0.0 -is_bf16=false -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > log/gemm_${1}_${2}_${3}.log 2>&1
    sleep 3
}

# $1: M, $2: N, $3: K
ncu_gemm() {
    echo "NCU $1 * $2 * $3"
    sudo ncu --set full --target-processes all --force-overwrite -o ncu/gemm_${1}_${2}_${3} $WORK_PATH/output/bin/benchmark_gemm -M=$1 -N=$2 -K=$3 -alpha=1.0 -beta=0.0 -is_bf16=false -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_gemm_${1}_${2}_${3}.log 2>&1
    sleep 3
}

# $1:N, $2: K
benchmark_gemm() {
    Ms=(1 2 4 8 16 32 64 128 256 512 1024 2048 3072 4096 5120 6144 7168 8192)
    N=$1
    K=$2

    for M in ${Ms[@]};
    do
        evaluate_gemm $M $N $K
        # ncu_gemm $M $N $K
    done
}

# FP16
nohup $WORK_PATH/output/bin/benchmark_gemm -M=512 -N=2048 -K=1024 -alpha=1.0 -beta=0.0 -is_bf16=false -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/gemm_512_2048_1024.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/gemm_512_2048_1024 $WORK_PATH/output/bin/gemm -M=512 -N=2048 -K=1024 -alpha=1.0 -beta=0.0 -is_bf16=false -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_gemm_512_2048_1024.log 2>&1

# BF16
# nohup $WORK_PATH/output/bin/benchmark_gemm -M=512 -N=2048 -K=1024 -alpha=1.0 -beta=0.0 -is_bf16=true -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/gemm_512_2048_1024.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/gemm_512_2048_1024 $WORK_PATH/output/bin/gemm -M=512 -N=2048 -K=1024 -alpha=1.0 -beta=0.0 -is_bf16=true -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_gemm_512_2048_1024.log 2>&1

# beta == 1
# nohup $WORK_PATH/output/bin/benchmark_gemm -M=512 -N=2048 -K=1024 -alpha=1.0 -beta=1.0 -is_bf16=false -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/gemm_512_2048_1024.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/gemm_512_2048_1024 $WORK_PATH/output/bin/gemm -M=512 -N=2048 -K=1024 -alpha=1.0 -beta=0.0 -is_bf16=false -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_gemm_512_2048_1024.log 2>&1

# benchmark_gemm 4096 4096
# benchmark_gemm 8192 8192
