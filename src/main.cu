// Correctness + benchmark harness for the SGEMM kernel ladder.
//
// usage: sgemm <kernel|all> [size ...]
//   kernel  0 = cuBLAS, 1..6 = the kernels in src/kernels/
//   size    N (square) or MxNxK; default: 256 512 1024 2048 4096
//
// Computes C = alpha * A * B + beta * C with row-major matrices. Every kernel
// is checked against cuBLAS before it is timed. Results go to stdout as CSV,
// diagnostics to stderr. The exit code is 1 if any check fails.

#include <cublas_v2.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "common.cuh"
#include "kernels/01_naive.cuh"
#include "kernels/02_coalesced.cuh"
#include "kernels/03_smem_tiling.cuh"
#include "kernels/04_blocktile_1d.cuh"
#include "kernels/05_blocktile_2d.cuh"
#include "kernels/06_vectorized.cuh"

#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t st_ = (call);                                       \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_,        \
                    __FILE__, __LINE__);                                   \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

namespace {

constexpr float kAlpha = 1.5f;
constexpr float kBeta = 0.5f;  // < 1 so repeated timing runs don't blow up C
constexpr int kWarmup = 2;
constexpr int kReps = 10;

cublasHandle_t g_cublas;

void run_cublas(int M, int N, int K, float alpha, const float* A,
                const float* B, float beta, float* C) {
    // cuBLAS is column-major. A row-major MxK buffer *is* a column-major KxM
    // matrix (its transpose), so computing C^T = B^T * A^T in column-major
    // gives row-major C = A * B without any explicit transposes.
    CUBLAS_CHECK(cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             &alpha, B, N, A, K, &beta, C, N));
}

using RunFn = void (*)(int, int, int, float, const float*, const float*, float,
                       float*);

struct Kernel {
    const char* name;
    RunFn run;
    int tile_m, tile_n, tile_k;  // M, N, K must be multiples of these
};

const Kernel kKernels[] = {
    {"cublas", run_cublas, 1, 1, 1},
    {"naive", run_sgemm_naive, 1, 1, 1},
    {"coalesced", run_sgemm_coalesced, 1, 1, 1},
    {"smem_tiling", run_sgemm_smem, 1, 1, 1},
    {"blocktile_1d", run_sgemm_blocktile_1d, 64, 64, 8},
    {"blocktile_2d", run_sgemm_blocktile_2d, 128, 128, 8},
    {"vectorized", run_sgemm_vectorized, 128, 128, 8},
};
constexpr int kNumKernels = sizeof(kKernels) / sizeof(kKernels[0]);

struct Shape {
    int m, n, k;
};

bool parse_shape(const char* s, Shape* out) {
    int m = 0, n = 0, k = 0;
    const int got = sscanf(s, "%dx%dx%d", &m, &n, &k);
    if (got == 1) {
        n = k = m;
    } else if (got != 3) {
        return false;
    }
    if (m <= 0 || n <= 0 || k <= 0) return false;
    *out = {m, n, k};
    return true;
}

float time_ms(const Kernel& kernel, Shape s, const float* A, const float* B,
              float* C) {
    for (int i = 0; i < kWarmup; ++i) {
        kernel.run(s.m, s.n, s.k, kAlpha, A, B, kBeta, C);
    }
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kReps; ++i) {
        kernel.run(s.m, s.n, s.k, kAlpha, A, B, kBeta, C);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / kReps;
}

// Indexing bugs produce O(1) errors; FP32 reordering stays far below this.
float max_abs_err(const std::vector<float>& got, const std::vector<float>& ref,
                  bool* ok) {
    float worst = 0.0f;
    *ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - ref[i]);
        if (!(diff <= 1e-2f + 1e-3f * std::fabs(ref[i]))) *ok = false;  // NaN fails too
        if (diff > worst) worst = diff;
    }
    return worst;
}

void usage() {
    fprintf(stderr, "usage: sgemm <kernel|all> [N | MxNxK ...]\nkernels:\n");
    for (int i = 0; i < kNumKernels; ++i) {
        fprintf(stderr, "  %d  %s\n", i, kKernels[i].name);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 2;
    }

    int first = 0, last = kNumKernels - 1;
    if (strcmp(argv[1], "all") != 0) {
        char* end = nullptr;
        const long id = strtol(argv[1], &end, 10);
        if (*end != '\0' || id < 0 || id >= kNumKernels) {
            usage();
            return 2;
        }
        first = last = (int)id;
    }

    std::vector<Shape> shapes;
    for (int i = 2; i < argc; ++i) {
        Shape s;
        if (!parse_shape(argv[i], &s)) {
            fprintf(stderr, "bad size: %s\n", argv[i]);
            return 2;
        }
        shapes.push_back(s);
    }
    if (shapes.empty()) {
        for (int n : {256, 512, 1024, 2048, 4096}) shapes.push_back({n, n, n});
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "# GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    CUBLAS_CHECK(cublasCreate(&g_cublas));
    // Plain FP32 (no TF32 tensor cores), so the comparison is like for like.
    CUBLAS_CHECK(cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH));

    printf("kernel_id,kernel,M,N,K,ms,gflops,pct_cublas,max_abs_err\n");

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    bool all_ok = true;

    for (const Shape& s : shapes) {
        const size_t bytesA = sizeof(float) * s.m * s.k;
        const size_t bytesB = sizeof(float) * s.k * s.n;
        const size_t bytesC = sizeof(float) * s.m * s.n;
        std::vector<float> hA(s.m * s.k), hB(s.k * s.n), hC(s.m * s.n);
        std::vector<float> hRef(s.m * s.n), hOut(s.m * s.n);
        for (float& x : hA) x = dist(rng);
        for (float& x : hB) x = dist(rng);
        for (float& x : hC) x = dist(rng);

        float *dA, *dB, *dC0, *dOut;
        CUDA_CHECK(cudaMalloc(&dA, bytesA));
        CUDA_CHECK(cudaMalloc(&dB, bytesB));
        CUDA_CHECK(cudaMalloc(&dC0, bytesC));
        CUDA_CHECK(cudaMalloc(&dOut, bytesC));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dC0, hC.data(), bytesC, cudaMemcpyHostToDevice));

        // Reference result and baseline timing come from cuBLAS.
        CUDA_CHECK(cudaMemcpy(dOut, dC0, bytesC, cudaMemcpyDeviceToDevice));
        run_cublas(s.m, s.n, s.k, kAlpha, dA, dB, kBeta, dOut);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hRef.data(), dOut, bytesC, cudaMemcpyDeviceToHost));
        const float cublas_ms = time_ms(kKernels[0], s, dA, dB, dOut);
        const double flops = 2.0 * s.m * s.n * s.k;

        for (int id = first; id <= last; ++id) {
            const Kernel& kernel = kKernels[id];
            if (s.m % kernel.tile_m || s.n % kernel.tile_n || s.k % kernel.tile_k) {
                fprintf(stderr, "# skip %s at %dx%dx%d: needs multiples of %dx%dx%d\n",
                        kernel.name, s.m, s.n, s.k, kernel.tile_m,
                        kernel.tile_n, kernel.tile_k);
                continue;
            }

            float err = 0.0f;
            if (id != 0) {
                CUDA_CHECK(cudaMemcpy(dOut, dC0, bytesC, cudaMemcpyDeviceToDevice));
                kernel.run(s.m, s.n, s.k, kAlpha, dA, dB, kBeta, dOut);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, bytesC, cudaMemcpyDeviceToHost));
                bool ok;
                err = max_abs_err(hOut, hRef, &ok);
                if (!ok) {
                    fprintf(stderr, "FAIL %s at %dx%dx%d: max_abs_err=%g\n",
                            kernel.name, s.m, s.n, s.k, err);
                    all_ok = false;
                    continue;
                }
            }

            const float ms = id == 0 ? cublas_ms : time_ms(kernel, s, dA, dB, dOut);
            printf("%d,%s,%d,%d,%d,%.4f,%.1f,%.1f,%.2e\n", id, kernel.name, s.m,
                   s.n, s.k, ms, flops / (ms * 1e6), 100.0 * cublas_ms / ms, err);
            fflush(stdout);
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC0));
        CUDA_CHECK(cudaFree(dOut));
    }

    CUBLAS_CHECK(cublasDestroy(g_cublas));
    return all_ok ? 0 : 1;
}
