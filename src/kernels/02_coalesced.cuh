#pragma once

#include "../common.cuh"

// Kernel 2: same work per thread as kernel 1, but threadIdx.x now walks
// *along* a row of C.
//
// Within a warp, B loads and C stores hit 32 consecutive floats (one 128-byte
// transaction), and every thread reads the same A element (a broadcast). The
// arithmetic is unchanged; only the thread-to-data mapping moved.
__global__ void sgemm_coalesced(int M, int N, int K, float alpha,
                                const float* A, const float* B, float beta,
                                float* C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

inline void run_sgemm_coalesced(int M, int N, int K, float alpha,
                                const float* A, const float* B, float beta,
                                float* C) {
    const dim3 block(32, 32);
    const dim3 grid(ceil_div(N, 32), ceil_div(M, 32));
    sgemm_coalesced<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
