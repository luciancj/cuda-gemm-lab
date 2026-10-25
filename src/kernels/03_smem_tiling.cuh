#pragma once

#include "../common.cuh"

// Kernel 3: stage 32x32 tiles of A and B in shared memory.
//
// Each element loaded from global memory is now reused by 32 threads, which
// cuts DRAM traffic by 32x. The new bottleneck is the inner loop: every FMA
// still needs two shared-memory loads, so the kernel becomes bound by shared
// memory bandwidth rather than DRAM bandwidth.
constexpr int kSmemTile = 32;

__global__ void sgemm_smem(int M, int N, int K, float alpha, const float* A,
                           const float* B, float beta, float* C) {
    __shared__ float As[kSmemTile][kSmemTile];
    __shared__ float Bs[kSmemTile][kSmemTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kSmemTile + ty;
    const int col = blockIdx.x * kSmemTile + tx;

    float acc = 0.0f;
    for (int t = 0; t < K; t += kSmemTile) {
        // Zero-pad out-of-range elements so the inner loop needs no branches.
        As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.0f;
        Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < kSmemTile; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

inline void run_sgemm_smem(int M, int N, int K, float alpha, const float* A,
                           const float* B, float beta, float* C) {
    const dim3 block(kSmemTile, kSmemTile);
    const dim3 grid(ceil_div(N, kSmemTile), ceil_div(M, kSmemTile));
    sgemm_smem<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
