#!/bin/bash
set -e

BASE_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR=$BASE_DIR/build
LOG_DIR=$BASE_DIR/log

rm -rf $BUILD_DIR
if [ ! -d $BUILD_DIR ]; then
    mkdir $BUILD_DIR
fi

if [ ! -d $LOG_DIR ]; then
    mkdir $LOG_DIR
fi

cd $BUILD_DIR

datetime=$(date +%Y%m%d_%H%M%S)
log_file="$LOG_DIR/build_$datetime.log"
if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++-19 \
    -DCMAKE_C_COMPILER=/usr/bin/clang-19 \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.0 \
    -DCUDA_NVCC_EXECUTABLE=/usr/local/cuda-13.0/bin/nvcc \
    -DCUDA_HOST_COMPILER=/usr/bin/clang-19 \
    -DCMAKE_INSTALL_PREFIX=../install \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
    -DCUTLASS_NVCC_ARCHS="89;90a;100a" \
    2>&1 | tee $log_file

