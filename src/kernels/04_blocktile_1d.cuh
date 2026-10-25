#pragma once

#include "../common.cuh"

// Kernel 4: each thread computes a TM x 1 column strip of C.
//
// In the inner loop a thread loads one B value into a register and reuses it
// for TM FMAs, so shared-memory loads per FMA drop from 2 to (1 + TM) / TM.
// More work per thread also means fewer threads per block and more
// instruction-level parallelism.
//
// Requires M % BM == 0, N % BN == 0, K % BK == 0 (checked by the harness).
template <int BM, int BN, int BK, int TM>
__global__ void __launch_bounds__((BM * BN) / TM)
    sgemm_blocktile_1d(int M, int N, int K, float alpha, const float* A,
                       const float* B, float beta, float* C) {
    constexpr int kThreads = (BM * BN) / TM;
    // One thread loads exactly one element of each tile.
    static_assert(BM * BK == kThreads, "A tile must have one element per thread");
    static_assert(BN * BK == kThreads, "B tile must have one element per thread");

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    float acc[TM] = {0.0f};

    for (int bk = 0; bk < K; bk += BK) {
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();
        A += BK;
        B += BK * N;

        for (int dot = 0; dot < BK; ++dot) {
            const float b = Bs[dot * BN + threadCol];
            for (int i = 0; i < TM; ++i) {
                acc[i] += As[(threadRow * TM + i) * BK + dot] * b;
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        const int idx = (threadRow * TM + i) * N + threadCol;
        C[idx] = alpha * acc[i] + beta * C[idx];
    }
}

inline void run_sgemm_blocktile_1d(int M, int N, int K, float alpha,
                                   const float* A, const float* B, float beta,
                                   float* C) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
    const dim3 block((BM * BN) / TM);
    const dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    sgemm_blocktile_1d<BM, BN, BK, TM>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
