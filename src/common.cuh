#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                   \
                    cudaGetErrorString(err_), __FILE__, __LINE__);         \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }
