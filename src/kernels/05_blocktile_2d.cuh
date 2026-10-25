#pragma once

#include "../common.cuh"

// Kernel 5: each thread computes a TM x TN tile of C as an outer product.
//
// Per inner-loop step a thread loads TM values of A and TN values of B into
// registers and does TM * TN FMAs: 16 shared loads for 64 FMAs with 8x8. The
// 128x128 block tile also raises global arithmetic intensity: each BK step
// moves (BM + BN) * BK floats and does 2 * BM * BN * BK FLOPs, i.e.
// 32 FLOP/byte, past the ridge point of most GPUs (T4: ~25 FLOP/byte).
//
// Requires M % BM == 0, N % BN == 0, K % BK == 0 (checked by the harness).
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
    sgemm_blocktile_2d(int M, int N, int K, float alpha, const float* A,
                       const float* B, float beta, float* C) {
    constexpr int kThreads = (BM * BN) / (TM * TN);
    static_assert(kThreads % BK == 0 && BM % (kThreads / BK) == 0,
                  "threads must tile the A block evenly");
    static_assert(kThreads % BN == 0 && BK % (kThreads / BN) == 0,
                  "threads must tile the B block evenly");

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // There are fewer threads than tile elements, so each thread loads
    // several rows, `stride` rows apart.
    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    constexpr int strideA = kThreads / BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;
    constexpr int strideB = kThreads / BN;

    float acc[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (int bk = 0; bk < K; bk += BK) {
        for (int off = 0; off < BM; off += strideA) {
            As[(innerRowA + off) * BK + innerColA] =
                A[(innerRowA + off) * K + innerColA];
        }
        for (int off = 0; off < BK; off += strideB) {
            Bs[(innerRowB + off) * BN + innerColB] =
                B[(innerRowB + off) * N + innerColB];
        }
        __syncthreads();
        A += BK;
        B += BK * N;

        for (int dot = 0; dot < BK; ++dot) {
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dot];
            }
            for (int j = 0; j < TN; ++j) {
                regN[j] = Bs[dot * BN + threadCol * TN + j];
            }
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    acc[i * TN + j] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            const int idx = (threadRow * TM + i) * N + threadCol * TN + j;
            C[idx] = alpha * acc[i * TN + j] + beta * C[idx];
        }
    }
}

inline void run_sgemm_blocktile_2d(int M, int N, int K, float alpha,
                                   const float* A, const float* B, float beta,
                                   float* C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    const dim3 block((BM * BN) / (TM * TN));
    const dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    sgemm_blocktile_2d<BM, BN, BK, TM, TN>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
