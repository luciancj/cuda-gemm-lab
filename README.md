# cuda-gemm-lab

Hand-written CUDA SGEMM (`C = αAB + βC`, FP32, row-major), taken from a naive
kernel to near-cuBLAS speed in six steps. Each step removes one specific
bottleneck. A harness checks every kernel against cuBLAS before timing it.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luciancj/cuda-gemm-lab/blob/main/notebooks/colab.ipynb)

## The ladder

| # | Kernel | Idea | Bottleneck it removes |
|---|---|---|---|
| 1 | [naive](src/kernels/01_naive.cuh) | one thread per output | (baseline) |
| 2 | [coalesced](src/kernels/02_coalesced.cuh) | map `threadIdx.x` to columns | uncoalesced global loads/stores |
| 3 | [smem_tiling](src/kernels/03_smem_tiling.cuh) | 32×32 tiles in shared memory | DRAM bandwidth (32× reuse per load) |
| 4 | [blocktile_1d](src/kernels/04_blocktile_1d.cuh) | each thread computes 8 outputs | shared-memory loads per FMA (2 → 1.125) |
| 5 | [blocktile_2d](src/kernels/05_blocktile_2d.cuh) | 8×8 register tile, outer product | shared-memory loads per FMA (→ 0.25) |
| 6 | [vectorized](src/kernels/06_vectorized.cuh) | `float4` loads, transposed A tile | instruction count for memory ops |

## Why each step matters (roofline view)

A T4 peaks at ~8.1 TFLOPS FP32 and ~320 GB/s DRAM, so a kernel needs about
**25 FLOP/byte** to be compute-bound rather than memory-bound.

- **Kernels 1–2** load two floats from global memory per FMA: 0.25 FLOP/byte.
  Even with perfect coalescing that caps them at ~80 GFLOPS from DRAM. Caches
  help a little, but they are memory-bound by design.
- **Kernel 3** reuses each global load 32 times, about 8 FLOP/byte from DRAM.
  The limit moves into the inner loop, which still does two shared-memory
  loads per FMA, so it is now bound by shared-memory bandwidth.
- **Kernels 4–5** move reuse into registers. With a 128×128 block tile, each
  `BK` step moves `(128+128)·8` floats and does `2·128·128·8` FLOPs, which is
  **32 FLOP/byte**. That passes the ridge point, so the kernel is now limited
  by compute and latency hiding, not memory.
- **Kernel 6** issues fewer, wider memory instructions (`LDG.128`, `LDS.128`).
  This frees issue slots for FMAs.

## Results

<!-- Run `make bench` (or the Colab notebook), then paste the table that
     scripts/plot.py prints here, along with results/bench.png. -->

_GPU: TBD. Fill in from `make bench`._

| # | kernel | GFLOPS @ 4096 | % of cuBLAS |
|---|---|---|---|
| 0 | cublas | – | 100% |
| 1 | naive | – | – |
| 2 | coalesced | – | – |
| 3 | smem_tiling | – | – |
| 4 | blocktile_1d | – | – |
| 5 | blocktile_2d | – | – |
| 6 | vectorized | – | – |

## Run it

On Colab, open the notebook above and pick a GPU runtime. It detects the GPU
architecture, builds, tests, benchmarks and plots.

Locally, with the CUDA toolkit installed:

```sh
make ARCH=sm_86            # your GPU's compute capability
make test ARCH=sm_86       # every kernel vs cuBLAS on square, non-square and odd shapes
make bench ARCH=sm_86      # writes results/bench.csv
python scripts/plot.py results/bench.csv
./build/sgemm 5 2048x1024x512   # one kernel, one shape
```

## Correctness

- cuBLAS is the reference, run in plain FP32 math mode with no TF32, so the
  comparison is like for like.
- `make test` runs non-square shapes (to catch swapped M/N/K indexing) and a
  97×131×77 shape (to exercise the bounds checks in kernels 1–3).
- Kernels 4–6 assume `M`, `N`, `K` are multiples of the block tile. The
  harness skips shapes that aren't, rather than reading out of bounds.

## Next steps

- **Warp tiling:** add a warp-level tile between block and thread tiles to
  improve register reuse and cut shared-memory bank conflicts.
- **Double buffering with `cp.async`:** overlap loading the next tile with
  computing on the current one.
- **Autotuning:** search over `BM/BN/BK/TM/TN` for each GPU.
- **Tensor cores:** a WMMA/TF32 variant, compared against cuBLAS with TF32
  enabled.

## Credits

The kernel progression follows Simon Boehm's
[How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM).
The code, harness and analysis here are my own.
