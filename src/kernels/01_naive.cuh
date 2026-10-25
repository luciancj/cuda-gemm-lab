#pragma once

#include "../common.cuh"

// Kernel 1: one thread per element of C.
//
// threadIdx.x walks *down* a column of C, so the 32 threads of a warp read A
// rows that are K floats apart and write C elements that are N floats apart.
// Every A load and C store is uncoalesced: each thread gets its own memory
// transaction.
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

inline void run_sgemm_naive(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C) {
    const dim3 block(32, 32);
    const dim3 grid(ceil_div(M, 32), ceil_div(N, 32));
    sgemm_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
