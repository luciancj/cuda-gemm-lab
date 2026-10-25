#pragma once

#include "../common.cuh"

// Kernel 6: kernel 5 plus 128-bit memory instructions.
//
// - Global loads/stores use float4, so one instruction moves 16 bytes
//   (LDG.128 / STG.128) instead of four.
// - The A tile is stored transposed in shared memory (As[k][m]). The TM
//   values a thread needs for one `dot` step are then contiguous, so the
//   compiler can fetch them with LDS.128 instead of TM scalar loads.
//
// Requires M % BM == 0, N % BN == 0, K % BK == 0 (checked by the harness);
// that also guarantees every float4 access is 16-byte aligned.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
    sgemm_vectorized(int M, int N, int K, float alpha, const float* A,
                     const float* B, float beta, float* C) {
    constexpr int kThreads = (BM * BN) / (TM * TN);
    static_assert(BK % 4 == 0 && BN % 4 == 0 && TN % 4 == 0,
                  "float4 accesses need multiples of 4");
    static_assert(kThreads % (BK / 4) == 0 && BM % (kThreads / (BK / 4)) == 0,
                  "threads must tile the A block evenly");
    static_assert(kThreads % (BN / 4) == 0 && BK % (kThreads / (BN / 4)) == 0,
                  "threads must tile the B block evenly");

    __shared__ __align__(16) float As[BK * BM];  // transposed: As[k * BM + m]
    __shared__ __align__(16) float Bs[BK * BN];

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    constexpr int strideA = kThreads / (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);
    constexpr int strideB = kThreads / (BN / 4);

    float acc[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (int bk = 0; bk < K; bk += BK) {
        for (int off = 0; off < BM; off += strideA) {
            const int m = innerRowA + off;
            const float4 tmp =
                *reinterpret_cast<const float4*>(&A[m * K + innerColA * 4]);
            As[(innerColA * 4 + 0) * BM + m] = tmp.x;
            As[(innerColA * 4 + 1) * BM + m] = tmp.y;
            As[(innerColA * 4 + 2) * BM + m] = tmp.z;
            As[(innerColA * 4 + 3) * BM + m] = tmp.w;
        }
        for (int off = 0; off < BK; off += strideB) {
            const int k = innerRowB + off;
            *reinterpret_cast<float4*>(&Bs[k * BN + innerColB * 4]) =
                *reinterpret_cast<const float4*>(&B[k * N + innerColB * 4]);
        }
        __syncthreads();
        A += BK;
        B += BK * N;

        for (int dot = 0; dot < BK; ++dot) {
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[dot * BM + threadRow * TM + i];
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
        for (int j = 0; j < TN; j += 4) {
            float4* dst = reinterpret_cast<float4*>(
                &C[(threadRow * TM + i) * N + threadCol * TN + j]);
            float4 tmp = *dst;
            tmp.x = alpha * acc[i * TN + j + 0] + beta * tmp.x;
            tmp.y = alpha * acc[i * TN + j + 1] + beta * tmp.y;
            tmp.z = alpha * acc[i * TN + j + 2] + beta * tmp.z;
            tmp.w = alpha * acc[i * TN + j + 3] + beta * tmp.w;
            *dst = tmp;
        }
    }
}

inline void run_sgemm_vectorized(int M, int N, int K, float alpha,
                                 const float* A, const float* B, float beta,
                                 float* C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    const dim3 block((BM * BN) / (TM * TN));
    const dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    sgemm_vectorized<BM, BN, BK, TM, TN>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
