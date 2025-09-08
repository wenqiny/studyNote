# GEMM cublass vs cutlass
This md will try to compare cublas and cutlass perf, and try to tune cutlass to make its perf close to cublas

We will use a matmul like `(1, 4096) @ (4096, 4096)` for `bf16` as a case, it's a **memory-bound** matmul.

## Therotical throughtput
On A100, the memory bandwidth is **1560 GB/s**, so the avg time cost should be:

$$
\frac{(1 \times 4096 + 4096 \times 4096) \times 2\ (\text{bytes})}{1560 \times 10^9 (\text{bytes per second})} = 0.0000215\ \text{s} = 0.0215\ \text{ms} = 21.5\ \text{us} 
$$

On A100, the memory bandwidth is **760 GB/s**, so the avg time cost should be:

$$
\frac{(1 \times 4096 + 4096 \times 4096) \times 2\ (\text{bytes})}{760 \times 10^9 (\text{bytes per second})} = 0.0000441\ \text{s} = 0.0441\ \text{ms} = 44.1\ \text{us} 
$$

## Cublas code
<details>
  <summary>Cublas code</summary>

```cpp
#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <random>
#include <algorithm>

#define CHECK_CUDA(call) do {                               \
  cudaError_t _e = (call);                                  \
  if (_e != cudaSuccess) {                                  \
    std::cerr << "CUDA Error " << int(_e) << " : "          \
              << cudaGetErrorString(_e) << " at "           \
              << __FILE__ << ":" << __LINE__ << std::endl;  \
    std::exit(1);                                           \
  }                                                         \
} while(0)

void checkCublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    // Problem size
    const int M = 1;
    const int K = 4096;
    const int N = 4096;

    int iters = 500;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        }
    }

    // Host data
    std::vector<float> hA_f(M * K);
    std::vector<float> hB_f(K * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : hA_f) v = dist(rng);
    for (auto& v : hB_f) v = dist(rng);

    std::vector<__nv_bfloat16> hA(M * K);
    std::vector<__nv_bfloat16> hB(K * N);
    for (int i = 0; i < M*K; ++i) hA[i] = __float2bfloat16(hA_f[i]);
    for (int i = 0; i < K*N; ++i) hB[i] = __float2bfloat16(hB_f[i]);

    std::vector<float> hC(M * N, 0.0f);
    std::vector<float> hD(M * N, 0.0f);

    // Device buffers
    __nv_bfloat16* dA = nullptr;
    __nv_bfloat16* dB = nullptr;
    float* dC = nullptr;
    float* dD = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&dA, M*K*sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc((void**)&dB, K*N*sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc((void**)&dC, M*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dD, M*N*sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), M*K*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), K*N*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hC.data(), M*N*sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;

    // Warmup
    for (int i = 0; i < 10; ++i) {
        checkCublas(cublasGemmEx(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 dB, CUDA_R_16BF, N,
                                 dA, CUDA_R_16BF, K,
                                 &beta,
                                 dD, CUDA_R_32F, N,
                                 CUBLAS_COMPUTE_32F_FAST_16BF,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        checkCublas(cublasGemmEx(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 dB, CUDA_R_16BF, N,
                                 dA, CUDA_R_16BF, K,
                                 &beta,
                                 dD, CUDA_R_32F, N,
                                 CUBLAS_COMPUTE_32F_FAST_16BF,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iters;

    // Compute GFLOPs + bandwidth
    double flops = 2.0 * double(M) * double(N) * double(K);
    double gflops = (flops / (avg_ms * 1e-3)) / 1e9;

    double bytesA = double(M) * K * sizeof(__nv_bfloat16);
    double bytesB = double(K) * N * sizeof(__nv_bfloat16);
    double bytesD = double(M) * N * sizeof(float);
    double gbps = (bytesA + bytesB + bytesD) / (avg_ms * 1e-3) / 1e9;

    // Copy back result
    CHECK_CUDA(cudaMemcpy(hD.data(), dD, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    // Log (same as CUTLASS version)
    std::cout << "BF16 GEMM -> FP32 output (M,N,K)=("
              << M << "," << N << "," << K << ")\n"
              << "iters=" << iters << "\n"
              << "Avg time: " << avg_ms << " ms\n"
              << "Perf: " << gflops << " GFLOP/s\n"
              << "BW:   " << gbps  << " GB/s\n";

    std::cout << "D[0..3]: ";
    for (int i = 0; i < std::min(N, 4); ++i) {
        std::cout << hD[i] << (i+1<4 ? ", " : "\n");
    }

    // Cleanup
    cublasDestroy(handle);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dD);

    return 0;
}
```
</details>

Please build it with `nvcc -arch=sm_80 -lcublas matmul-cublas.cu`.

Its outputs on A100:
```
BF16 GEMM -> FP32 output (M,N,K)=(1,4096,4096)
iters=500
Avg time: 0.0320061 ms
Perf: 1048.37 GFLOP/s
BW:   1049.14 GB/s
D[0..3]: -7.62002, -0.401527, -0.434216, -6.12647
```

It's close to `0.02` ms therotical max output.

On RTX3080:
```
BF16 GEMM -> FP32 output (M,N,K)=(1,4096,4096)
iters=500
Avg time: 0.0516403 ms
Perf: 649.772 GFLOP/s
BW:   650.248 GB/s
D[0..3]: -7.62001, -0.401532, -0.434211, -6.12645
```

## Cutlass code
<details>
  <summary>Cutlass code</summary>

```cpp
// Build example:
//   nvcc -O3 -std=c++17 -arch=sm_80 \
//     -I/path/to/cutlass/include \
//     gemm_bf16_fp32out.cu -o gemm_bf16_fp32out
//
// Run:
//   ./gemm_bf16_fp32out --iters 500 --splitK 8
//
// Notes:
// - A, B are bf16
// - Accumulate in fp32
// - Output D (and C) in fp32
// - Optimized for A100 (SM80 Tensor Cores)

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <cassert>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/arch/arch.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/numeric_types.h"

#define CHECK_CUDA(call) do {                               \
  cudaError_t _e = (call);                                  \
  if (_e != cudaSuccess) {                                  \
    std::cerr << "CUDA Error " << int(_e) << " : "          \
              << cudaGetErrorString(_e) << " at "           \
              << __FILE__ << ":" << __LINE__ << std::endl;  \
    std::exit(1);                                           \
  }                                                         \
} while(0)

// --- Types ---
using ElementInputA = cutlass::bfloat16_t;
using ElementInputB = cutlass::bfloat16_t;
using ElementAccumulator = float;     // accumulation in fp32
using ElementCompute = float;         // alpha, beta
using ElementOutput = float;          // output in fp32

// Layouts: row-major
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

// Math operator class and architecture
using OpClass   = cutlass::arch::OpClassTensorOp;
using SmArch    = cutlass::arch::Sm80;

// Tile configuration
using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 32>;
using WarpShape        = cutlass::gemm::GemmShape<32, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

// Epilogue: LinearCombination with fp32 outputs
// Vector length 4 -> 16B stores (aligned for fp32)
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
  ElementOutput, 4, ElementAccumulator, ElementCompute
>;

using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int Stages = 3;

// GEMM definition
using GemmBF16_FP32Out = cutlass::gemm::device::Gemm<
  ElementInputA, LayoutA,
  ElementInputB, LayoutB,
  ElementOutput, LayoutC,        // C and D are fp32
  ElementAccumulator,
  OpClass, SmArch,
  ThreadblockShape, WarpShape, InstructionShape,
  EpilogueOp, Swizzle, Stages
>;

// Helper: convert float -> bf16
static inline ElementInputA f32_to_bf16(float x) {
  return ElementInputA(x);
}

int main(int argc, char** argv) {
  int iters = 500;
  int split_k_slices = 1;

  // CLI args
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--splitK") && i + 1 < argc) {
      split_k_slices = std::max(1, std::atoi(argv[++i]));
    } else if (!std::strcmp(argv[i], "--help")) {
      std::cout << "Usage: " << argv[0] << " [--iters N] [--splitK S]\n";
      return 0;
    }
  }

  // Problem size (1 x 4096) * (4096 x 4096)
  int M = 1, N = 4096, K = 4096;
  int lda = K;       // row-major A: leading dim = K
  int ldb = N;       // row-major B: leading dim = N
  int ldc = N;       // row-major C/D: leading dim = N

  size_t bytesA = size_t(M) * K * sizeof(ElementInputA);
  size_t bytesB = size_t(K) * N * sizeof(ElementInputB);
  size_t bytesC = size_t(M) * N * sizeof(ElementOutput);
  size_t bytesD = size_t(M) * N * sizeof(ElementOutput);

  // Host buffers
  std::vector<ElementInputA> hA(M * K);
  std::vector<ElementInputB> hB(K * N);
  std::vector<ElementOutput> hC(M * N);
  std::vector<ElementOutput> hD(M * N);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int i = 0; i < M*K; ++i) hA[i] = f32_to_bf16(dist(rng));
  for (int i = 0; i < K*N; ++i) hB[i] = f32_to_bf16(dist(rng));
  for (int i = 0; i < M*N; ++i) hC[i] = 0.0f;

  // Device buffers
  ElementInputA* dA = nullptr;
  ElementInputB* dB = nullptr;
  ElementOutput* dC = nullptr;
  ElementOutput* dD = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&dA, bytesA));
  CHECK_CUDA(cudaMalloc((void**)&dB, bytesB));
  CHECK_CUDA(cudaMalloc((void**)&dC, bytesC));
  CHECK_CUDA(cudaMalloc((void**)&dD, bytesD));
  CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC, hC.data(), bytesC, cudaMemcpyHostToDevice));

  ElementCompute alpha = 1.0f;
  ElementCompute beta  = 0.0f;

  GemmBF16_FP32Out gemm_op;

  typename GemmBF16_FP32Out::Arguments args(
      {M, N, K},
      {dA, lda},
      {dB, ldb},
      {dC, ldc},
      {dD, ldc},
      {alpha, beta},
      split_k_slices
  );

  cutlass::Status status = gemm_op.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "GEMM not supported. Status=" << int(status) << "\n";
    return 1;
  }

  status = gemm_op.initialize(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Initialize failed. Status=" << int(status) << "\n";
    return 1;
  }

  // Warmup
  for (int i = 0; i < 10; ++i) {
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Run failed. Status=" << int(status) << "\n";
      return 1;
    }
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Benchmark
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Run failed at iter " << i << ". Status=" << int(status) << "\n";
      return 1;
    }
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  ms /= float(iters);

  double flops = 2.0 * double(M) * double(N) * double(K);
  double gflops = (flops / (ms * 1.0e-3)) / 1.0e9;

  double bytes_moved = bytesA + bytesB + bytesD;
  double gbps = (bytes_moved / (ms * 1.0e-3)) / 1.0e9;

  std::cout << "BF16 GEMM -> FP32 output (M,N,K)=("
            << M << "," << N << "," << K << ")\n"
            << "splitK=" << split_k_slices << ", iters=" << iters << "\n"
            << "Avg time: " << ms << " ms\n"
            << "Perf: " << gflops << " GFLOP/s\n"
            << "BW:   " << gbps  << " GB/s\n";

  CHECK_CUDA(cudaMemcpy(hD.data(), dD, bytesD, cudaMemcpyDeviceToHost));
  std::cout << "D[0..3]: ";
  for (int i = 0; i < std::min(N, 4); ++i) {
    std::cout << hD[i] << (i+1 < 4 ? ", " : "\n");
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
  return 0;
}
```
</details>

Please build it with `nvcc -std=c++17 -arch=sm_80 -I/root/cutlass/include matmul-cutlass.cu`.

Its outputs on A100:
```
BF16 GEMM -> FP32 output (M,N,K)=(1,4096,4096)
splitK=1, iters=500
Avg time: 0.0925778 ms
Perf: 362.446 GFLOP/s
BW:   362.711 GB/s
D[0..3]: -7.62001, -0.401532, -0.434211, -6.12645
```

It taks `0.1` ms for each round, there is much space to optimize it on A100.

on RTX 3080:
```
BF16 GEMM -> FP32 output (M,N,K)=(1,4096,4096)
splitK=1, iters=500
Avg time: 0.088406 ms
Perf: 379.549 GFLOP/s
BW:   379.827 GB/s
D[0..3]: -7.62001, -0.401532, -0.434211, -6.12645
```

## Mirage code
<details>
  <summary>Mirage code</summary>

```cpp
// Build with: nvcc --expt-relaxed-constexpr -std=c++17 -arch=sm_80 -o mirage matmul-mirage.cu

#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <random>
#include <algorithm>

#define NUM_THREADS 128

constexpr int log2_constexpr(int n, int p = 0) {
  return (n <= 1) ? p : log2_constexpr(n >> 1, p + 1);
}

using bfloat16 = __nv_bfloat16;

template <typename T, size_t ROW, size_t COL, size_t STRIDE>
struct dmem_row {
  T *base_ptr;

  __device__ __forceinline__ dmem_row(T *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T *ptr) {
    base_ptr = ptr;
  }

  __device__ __forceinline__ T *operator()(size_t logical_idx_row,
                                           size_t logical_idx_col) {
    size_t logical_idx = logical_idx_row * STRIDE + logical_idx_col;
    // assert(logical_idx < (ROW*COL));
    return &base_ptr[logical_idx];
  }

  __device__ __forceinline__ T &at(size_t logical_idx) {
    // assert(logical_idx < (ROW*COL));
    return base_ptr[logical_idx];
  }
  __device__ __forceinline__ T &at(size_t logical_idx_row,
                                   size_t logical_idx_col) {

    size_t logical_idx = logical_idx_row * STRIDE + logical_idx_col;
    // assert(logical_idx < (ROW*COL));
    return base_ptr[logical_idx];
  }
};

template <typename T, size_t ROW, size_t COL, size_t STRIDE>
struct dmem_row_const {
  T const *base_ptr;

  __device__ __forceinline__ dmem_row_const(T const *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T const *ptr) {
    base_ptr = ptr;
  }

  __device__ __forceinline__ T const *operator()(size_t logical_idx_row,
                                                 size_t logical_idx_col) const {
    size_t logical_idx = logical_idx_row * STRIDE + logical_idx_col;
    // assert(logical_idx < (ROW*COL));
    return &base_ptr[logical_idx];
  }

  __device__ __forceinline__ T const &at(size_t logical_idx) const {
    // assert(logical_idx < (ROW*COL));
    return base_ptr[logical_idx];
  }
  __device__ __forceinline__ T const &at(size_t logical_idx_row,
                                         size_t logical_idx_col) const {
    size_t logical_idx = logical_idx_row * STRIDE + logical_idx_col;
    // assert(logical_idx < (ROW*COL));
    return base_ptr[logical_idx];
  }
};

template <typename T, size_t ROW, size_t COL, size_t STRIDE>
struct dmem_col_const {
  T const *base_ptr;

  __device__ __forceinline__ dmem_col_const(T const *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T const *ptr) {
    base_ptr = ptr;
  }

  __device__ __forceinline__ T const *operator()(size_t logical_idx_row,
                                                 size_t logical_idx_col) const {
    size_t logical_idx = logical_idx_col * STRIDE + logical_idx_row;
    return &base_ptr[logical_idx];
  }

  __device__ __forceinline__ T const &at(size_t logical_idx) const {
    return base_ptr[logical_idx];
  }
  __device__ __forceinline__ T const &at(size_t logical_idx_row,
                                         size_t logical_idx_col) const {
    size_t logical_idx = logical_idx_col * STRIDE + logical_idx_row;
    return base_ptr[logical_idx];
  }
};


template<typename T, int N>
struct vec_zero_t {
    static __device__ __forceinline__ void fill_zero(T* ptr) {
        // Ensure sizeof(T) * N is a multiple of 16 bytes
        static_assert((sizeof(T) * N) % 16 == 0, "sizeof(T) * N must be a multiple of 16 bytes for proper vectorized operations");

        constexpr int total_bytes = sizeof(T) * N;
        constexpr int num_chunks = total_bytes / sizeof(__uint128_t);
        __uint128_t* vec_ptr = reinterpret_cast<__uint128_t*>(ptr);
        constexpr int max_iters = (num_chunks + NUM_THREADS - 1) / NUM_THREADS;

        #pragma unroll
        for (int i = 0; i < max_iters; ++i) {
            int idx = i * blockDim.x + threadIdx.x;
            if (idx < num_chunks) {
                vec_ptr[idx] = 0ul;
            }
        }
    }
};

template <typename T,
          int B,
          int M,
          int S,
          size_t ROW_,
          size_t COL_,
          size_t STRIDE>
struct smem_row {
  T *__restrict__ base_ptr;
  using value_type = T;
  static constexpr int b = B;
  static constexpr int m = M;
  static constexpr int s = S;

  static constexpr size_t ROW = ROW_;
  static constexpr size_t COL = COL_;
  static constexpr size_t SIZE = ROW * COL;

  // static constexpr size_t Pow2_M = (1 << M);
  // static constexpr size_t Pow2_S = (1 << S);
  // static constexpr size_t Pow2_B = (1 << B);

  __device__ __forceinline__ smem_row(T *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T *ptr) {
    base_ptr = ptr;
  }

  static constexpr size_t size() {
    return ROW * COL;
  }
  // 2D access
  __device__ __forceinline__ T *operator()(size_t logical_idx_row,
                                           size_t logical_idx_col) {
    size_t logical_idx = logical_idx_row * STRIDE + logical_idx_col;
    // assert(logical_idx < SIZE);

    // first get the block -> 3,3,3 means 8*8*8 = 8 rows * 8 cols * 8elements
    // size_t block_idx = logical_idx / (Pow2_M * Pow2_S * Pow2_B);
    // size_t in_block_idx = logical_idx % (Pow2_M * Pow2_S * Pow2_B);

    // //get the irow and icol inside the block
    // size_t irow = in_block_idx / (Pow2_M * Pow2_S);
    // size_t icol = (in_block_idx / (Pow2_M)) % (Pow2_S);
    // icol = irow ^ icol;
    // size_t offset_in_bank = in_block_idx % (Pow2_M);

    // size_t phy_offset = block_idx * (Pow2_M * Pow2_S * Pow2_B) + irow *
    // (Pow2_M * Pow2_S)
    // + icol * (Pow2_M) + offset_in_bank;

    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);
    size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                        (icol << M) + offset_in_bank;
    // printf("phy_offset %d, %d\n", (int)logical_idx, (int)phy_offset);
    return &base_ptr[phy_offset];
  }
  __device__ __forceinline__ T &at(size_t logical_idx_row,
                                   size_t logical_idx_col) {
    size_t logical_idx = logical_idx_row * STRIDE + logical_idx_col;
    // assert(logical_idx < SIZE);

    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);

    size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                        (icol << M) + offset_in_bank;
    return base_ptr[phy_offset];
  }

  // 1D access
  __device__ __forceinline__ T &at(size_t logical_idx) {
    // assert(logical_idx < SIZE);
    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);

    size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                        (icol << M) + offset_in_bank;
    return base_ptr[phy_offset];
  }

  __device__ __forceinline__ T *operator[](size_t logical_idx) const {
    // assert(logical_idx < SIZE);
    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);

    size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                        (icol << M) + offset_in_bank;
    return &base_ptr[phy_offset];
  }
};

template <typename T,
          int B,
          int M,
          int S,
          size_t ROW_,
          size_t COL_,
          size_t STRIDE>
struct smem_col {
  T *base_ptr;

  using value_type = T;

  static constexpr int b = B;
  static constexpr int m = M;
  static constexpr int s = S;

  static constexpr size_t ROW = ROW_;
  static constexpr size_t COL = COL_;

  static constexpr size_t Pow2_M = (1 << M);
  static constexpr size_t Pow2_S = (1 << S);
  static constexpr size_t Pow2_B = (1 << B);

  __device__ __forceinline__ smem_col(T *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T *ptr) {
    base_ptr = ptr;
  }

  __device__ __forceinline__ T *operator()(size_t logical_idx_row,
                                           size_t logical_idx_col) {
    size_t logical_idx = logical_idx_col * STRIDE + logical_idx_row;
    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);

    size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                        (icol << M) + offset_in_bank;
    return &base_ptr[phy_offset];
  }

  __device__ __forceinline__ T &at(size_t logical_idx) {
    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);

    size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                        (icol << M) + offset_in_bank;
    return base_ptr[phy_offset];
  }

  __device__ __forceinline__ T &at(size_t logical_idx_row,
                                   size_t logical_idx_col) {
    size_t logical_idx = logical_idx_col * STRIDE + logical_idx_row;
    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);

    size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                        (icol << M) + offset_in_bank;
    return base_ptr[phy_offset];
  }

  __device__ __forceinline__ T *operator[](size_t logical_idx) const {
    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);

    size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                        (icol << M) + offset_in_bank;
    return &base_ptr[phy_offset];
  }
};

template <typename T>
__device__ __forceinline__ void load_smem(T *smem_ptr, T const *gmem_ptr) {
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
                   smem_int_ptr),
               "l"(gmem_ptr),
               "n"(16),
               "r"(16));
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  if constexpr (N == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
  }
}

__device__ __forceinline__ void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

static __device__ __forceinline__ void clear_8_floats(float *buffer) {
  *((__uint128_t *)(buffer)) = 0ul;
  *((__uint128_t *)(buffer + 4)) = 0ul;
}

template <typename T>
__device__ __forceinline__ void ldsm(T *__restrict__ smem_ptr, uint32_t *R) {
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
      : "r"(smem_int_ptr));
}

__device__ static __forceinline__ void
    mma_m16n16k16_bf16bf16bf32(float *C, uint32_t *A, uint32_t *B, float *D) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
               : "r"(A[0]),
                 "r"(A[1]),
                 "r"(A[2]),
                 "r"(A[3]),
                 "r"(B[0]),
                 "r"(B[1]),
                 "f"(C[0]),
                 "f"(C[1]),
                 "f"(C[2]),
                 "f"(C[3]));

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
               : "r"(A[0]),
                 "r"(A[1]),
                 "r"(A[2]),
                 "r"(A[3]),
                 "r"(B[2]),
                 "r"(B[3]),
                 "f"(C[4]),
                 "f"(C[5]),
                 "f"(C[6]),
                 "f"(C[7]));
}

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE = OUTPUT_SIZE,
          int K_PIPE_MAX = 3>
__device__ __forceinline__ void linear_kernel(void const *input_ptr,
                                              void const *weight_ptr,
                                              void const *residual_ptr,
                                              void *output_ptr,
                                              bool residual = true) {
  constexpr int CHUNK_SIZE = 16 / sizeof(T);
  constexpr int OUTPUT_ATOM_SIZE = OUTPUT_SIZE <= 128 ? OUTPUT_SIZE : 128;
  constexpr int NUM_OUTPUT_ATOMS = OUTPUT_SIZE / OUTPUT_ATOM_SIZE;
  constexpr int TILE_SIZE = 128;
  constexpr int FORLOOP_RANGE = REDUCTION_SIZE / TILE_SIZE;

  constexpr int NUM_CHUNKS_A = BATCH_SIZE * TILE_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_B = TILE_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_C = BATCH_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;

  constexpr int CHUNKS_PER_ROW_A = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_COL_B = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_ROW_C = OUTPUT_ATOM_SIZE / CHUNK_SIZE;

  constexpr int log2_CHUNK_SIZE = log2_constexpr(CHUNK_SIZE);
  constexpr int log2_CHUNKS_PER_ROW_A = log2_constexpr(CHUNKS_PER_ROW_A);
  constexpr int log2_CHUNKS_PER_COL_B = log2_constexpr(CHUNKS_PER_COL_B);
  constexpr int log2_CHUNKS_PER_ROW_C = log2_constexpr(CHUNKS_PER_ROW_C);

  // using SM80_16x8x16_F16F16F16F16_TNX2 = 16X16X16
  constexpr int NUM_WARPS_N =
      OUTPUT_ATOM_SIZE / 16 <= 4 ? OUTPUT_ATOM_SIZE / 16 : 4;
  constexpr int NUM_WARPS_K = 4 / NUM_WARPS_N;

  constexpr int NUM_ITERS_M = 1;
  constexpr int NUM_ITERS_N = OUTPUT_ATOM_SIZE / NUM_WARPS_N / 16;
  constexpr int NUM_ITERS_K = TILE_SIZE / NUM_WARPS_K / 16;

  constexpr int log2_NUM_WARPS_N = log2_constexpr(NUM_WARPS_N);
  constexpr int log2_NUM_ITERS_K = log2_constexpr(NUM_ITERS_K);

  int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  int warp_row = warp_idx >> log2_NUM_WARPS_N;
  int warp_col = warp_idx & (NUM_WARPS_N - 1);
  int lane_idx = threadIdx.x & 0x1f;

  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);
  T const *__restrict__ d_residual =
      residual ? static_cast<T const *>(residual_ptr) : nullptr;
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

  // printf("Kernel 1: weight value = %f\n", __bfloat162float(d_weight[0]));

  int bid = blockIdx.x;
  d_weight += OUTPUT_SIZE * REDUCTION_SIZE * bid;
  d_residual += OUTPUT_SIZE * BATCH_SIZE * bid;
  d_output += OUTPUT_SIZE * BATCH_SIZE * bid;

  // printf("Kernel 2: weight value = %f\n", __bfloat162float(d_weight[0]));

  using InputDmem = dmem_row_const<T, BATCH_SIZE, TILE_SIZE, REDUCTION_SIZE>;
  using WeightDmem =
      dmem_col_const<T, TILE_SIZE, OUTPUT_ATOM_SIZE, REDUCTION_SIZE>;
  using ResidualDmem =
      dmem_row_const<T, BATCH_SIZE, OUTPUT_ATOM_SIZE, O_STRIDE>;
  using OutputDmem = dmem_row<T, BATCH_SIZE, OUTPUT_ATOM_SIZE, O_STRIDE>;

  InputDmem input_dmem(d_input);
  WeightDmem weight_dmem(d_weight);
  ResidualDmem residual_dmem(d_residual);
  OutputDmem output_dmem(d_output);

  extern __shared__ char smem[];

  // STensors' offsets
  constexpr size_t ZERO_BUFFER_OFFSET = 0;
  // sizeof(T) * 8

  constexpr size_t SHARED_INPUT_BUFFER_OFFSET =
      ZERO_BUFFER_OFFSET + sizeof(T) * 8;
  // sizeof(T) * K_PIPE_MAX * BATCH_SIZE * TILE_SIZE

  constexpr size_t SHARED_WEIGHT_BUFFER_OFFSET =
      SHARED_INPUT_BUFFER_OFFSET +
      sizeof(T) * K_PIPE_MAX * BATCH_SIZE * TILE_SIZE;
  // sizeof(T) * K_PIPE_MAX * TILE_SIZE * OUTPUT_ATOM_SIZE

  constexpr size_t SHARED_RESIDUAL_OFFSET =
      SHARED_WEIGHT_BUFFER_OFFSET +
      sizeof(T) * K_PIPE_MAX * TILE_SIZE * OUTPUT_ATOM_SIZE;
  // sizeof(T) * BATCH_SIZE * OUTPUT_ATOM_SIZE

  constexpr size_t MM_INTERMEDIATE_OFFSET =
      SHARED_RESIDUAL_OFFSET + sizeof(T) * BATCH_SIZE * OUTPUT_ATOM_SIZE;
  // sizeof(T) * NUM_WARPS_K * BATCH_SIZE * OUTPUT_ATOM_SIZE

  constexpr size_t SHARED_OUTPUT_OFFSET =
      MM_INTERMEDIATE_OFFSET +
      sizeof(T) * NUM_WARPS_K * BATCH_SIZE * OUTPUT_ATOM_SIZE;
  // sizeof(T) * BATCH_SIZE * OUTPUT_ATOM_SIZE

  // zero buffer
  T *zero_buf = (T *)(smem + ZERO_BUFFER_OFFSET);
  vec_zero_t<T, 8>::fill_zero(zero_buf);

  // copy
  T *shared_input_buffer = (T *)(smem + SHARED_INPUT_BUFFER_OFFSET);
  T *shared_weight_buffer = (T *)(smem + SHARED_WEIGHT_BUFFER_OFFSET);

  // residual
  T *shared_residual =
      residual ? (T *)(smem + SHARED_RESIDUAL_OFFSET) : nullptr;

  // intermediate
  T *mm_intermediate = (T *)(smem + MM_INTERMEDIATE_OFFSET);

  // output
  T *shared_output = (T *)(smem + SHARED_OUTPUT_OFFSET);

  // define the swizzle mode
  using ZeroBufferSmem = smem_row<T, 0, 0, 0, 1, 8, 8>;
  using InputSmem = smem_row<T, 0, 0, 0, BATCH_SIZE, TILE_SIZE, TILE_SIZE>;
  using InputBufferSmem =
      smem_row<T, 0, 0, 0, K_PIPE_MAX * BATCH_SIZE, TILE_SIZE, TILE_SIZE>;
  using WeightSmem =
      smem_col<T, 3, 3, 3, TILE_SIZE, OUTPUT_ATOM_SIZE, TILE_SIZE>;
  using WeightBufferSmem =
      smem_col<T, 3, 3, 3, TILE_SIZE, K_PIPE_MAX * OUTPUT_ATOM_SIZE, TILE_SIZE>;
  using OutputSmem =
      smem_row<T, 0, 0, 0, BATCH_SIZE, OUTPUT_ATOM_SIZE, OUTPUT_ATOM_SIZE>;
  using MatMulIntermediateSmem = smem_row<T,
                                          0,
                                          0,
                                          0,
                                          NUM_WARPS_K * BATCH_SIZE,
                                          OUTPUT_ATOM_SIZE,
                                          OUTPUT_ATOM_SIZE>;

  ZeroBufferSmem zero_buffer(zero_buf);

  InputSmem input_smem(shared_input_buffer);
  WeightSmem weight_smem(shared_weight_buffer);

  OutputSmem residual_smem(shared_residual);

  MatMulIntermediateSmem mm_intermediate_smem(mm_intermediate);

  OutputSmem output_smem(shared_output);

  for (int output_atom_idx = 0; output_atom_idx < NUM_OUTPUT_ATOMS;
       output_atom_idx++,
           d_weight += OUTPUT_ATOM_SIZE * REDUCTION_SIZE,
           d_residual = residual ? d_residual + OUTPUT_ATOM_SIZE : nullptr,
           d_output += OUTPUT_ATOM_SIZE) {
    weight_dmem.set_ptr(d_weight);
    residual_dmem.set_ptr(d_residual);
    output_dmem.set_ptr(d_output);

    InputBufferSmem input_buffer_smem(shared_input_buffer);
    WeightBufferSmem weight_buffer_smem(shared_weight_buffer);

    if (residual) {
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_C; i += NUM_THREADS) {
        int row = i >> log2_CHUNKS_PER_ROW_C;
        int col = (i & (CHUNKS_PER_ROW_C - 1)) << log2_CHUNK_SIZE;
        load_smem(residual_smem(row, col), residual_dmem(row, col));
      }
    }

#pragma unroll
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; k_pipe++) {
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
        int src_row = i >> log2_CHUNKS_PER_ROW_A;
        int dst_row = src_row + ((k_pipe + 1) * BATCH_SIZE);
        int dst_col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
        int src_col = dst_col + (k_pipe << log2_constexpr(TILE_SIZE));
        load_smem(input_buffer_smem(dst_row, dst_col),
                  input_dmem(src_row, src_col));
      }
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_B; i += NUM_THREADS) {
        int dst_row = (i & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
        int src_row = dst_row + (k_pipe << log2_constexpr(TILE_SIZE));
        int src_col = i >> log2_CHUNKS_PER_COL_B;
        int dst_col =
            src_col + ((k_pipe + 1) << log2_constexpr(OUTPUT_ATOM_SIZE));
        // if(__bfloat162float(*weight_dmem(src_row, src_col)) != float(1)) {
        //   printf("Kernel gmem: weight value = %f\n", __bfloat162float(*weight_dmem(src_row, src_col)));
        // }
        load_smem(weight_buffer_smem(dst_row, dst_col),
                  weight_dmem(src_row, src_col));
      }
      cp_async_fence();
    }

    // accumulator
    float s_frag[NUM_ITERS_M][NUM_ITERS_N][8];
    for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
#pragma unroll
      for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
        clear_8_floats(s_frag[m][n]);
      }
    }

    for (int for_idx = 0; for_idx < FORLOOP_RANGE; for_idx++) {
      // copy
      if (for_idx + K_PIPE_MAX - 1 < FORLOOP_RANGE) {
#pragma unroll
        for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
          int row = i >> log2_CHUNKS_PER_ROW_A;
          int dst_col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
          int src_col = dst_col + ((for_idx + K_PIPE_MAX - 1)
                                   << log2_constexpr(TILE_SIZE));
          load_smem(input_buffer_smem(row, dst_col), input_dmem(row, src_col));
        }
#pragma unroll
        for (int i = threadIdx.x; i < NUM_CHUNKS_B; i += NUM_THREADS) {
          int dst_row = (i & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
          int src_row = dst_row + ((for_idx + K_PIPE_MAX - 1)
                                   << log2_constexpr(TILE_SIZE));
          int col = i >> log2_CHUNKS_PER_COL_B;
          load_smem(weight_buffer_smem(dst_row, col),
                    weight_dmem(src_row, col));
        }
        cp_async_fence();
        cp_async_wait<K_PIPE_MAX - 1>();
      } else if (for_idx + K_PIPE_MAX - 1 == FORLOOP_RANGE) {
        cp_async_wait<0>();
      }

      // rotate the buffers
      input_buffer_smem.set_ptr(shared_input_buffer +
                                BATCH_SIZE * TILE_SIZE *
                                    ((for_idx + 1) % K_PIPE_MAX));
      input_smem.set_ptr(shared_input_buffer +
                         BATCH_SIZE * TILE_SIZE * ((for_idx + 1) % K_PIPE_MAX));
      weight_buffer_smem.set_ptr(shared_weight_buffer +
                                 TILE_SIZE * OUTPUT_ATOM_SIZE *
                                     ((for_idx + 1) % K_PIPE_MAX));
      weight_smem.set_ptr(shared_weight_buffer +
                          TILE_SIZE * OUTPUT_ATOM_SIZE *
                              ((for_idx + 1) % K_PIPE_MAX));
      __syncthreads();

      uint32_t a_frag[4], b_frag[4];
      for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
        int m_row = (lane_idx & 0xF);
        bool is_valid = (m_row < BATCH_SIZE);
#pragma unroll
        for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
          int n_col = (n << (4 + log2_NUM_WARPS_N)) + (warp_col << 4) +
                      ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
#pragma unroll
          for (uint32_t k = 0; k < NUM_ITERS_K; k++) {
            int m_col = (warp_row << (4 + log2_NUM_ITERS_K)) + (k << 4) +
                        ((lane_idx >> 4) << 3);
            int n_row = (warp_row << (4 + log2_NUM_ITERS_K)) + (k << 4) +
                        (((lane_idx & 0xF) >> 3) << 3);
            T *src_ptr =
                is_valid ? input_smem(m_row, m_col) : zero_buffer(0, 0);
            ldsm(src_ptr, a_frag);
            ldsm(weight_smem(n_row, n_col), b_frag);
            // if(__bfloat162float(b_frag[0]) == 1) {
            //   printf("b_frag[0]: %f\n", __bfloat162float(*(reinterpret_cast<bfloat16*>(&b_frag[0]))));
            // }
            mma_m16n16k16_bf16bf16bf32(
                s_frag[m][n], a_frag, b_frag, s_frag[m][n]);
          }
        }
      }
      __syncthreads();
    }

    // write back to shared memory
    for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
#pragma unroll
      for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
          int row_in_warp = (lane_idx >> 2) + ((i & 0x1) << 3);
          if (row_in_warp < BATCH_SIZE) {
            int col = (n << (4 + log2_NUM_WARPS_N)) + (warp_col << 4) +
                      ((lane_idx & 0x3) << 1) + ((i >> 1) << 3);
            mm_intermediate_smem.at(warp_row + row_in_warp, col) =
                bfloat16(s_frag[m][n][(i << 1)]);
            mm_intermediate_smem.at(warp_row + row_in_warp, col + 1) =
                bfloat16(s_frag[m][n][(i << 1) | 0x1]);
          }
        }
      }
    }
    __syncthreads();

    // if (NUM_WARPS_K > 1) {
    //   reduction_sum_row<decltype(output_smem), decltype(mm_intermediate_smem)>(
    //       output_smem, mm_intermediate_smem);
    //   __syncthreads();
    // }

#pragma unroll
    for (int row = 0; row < BATCH_SIZE; row++) {
#pragma unroll
      for (int i = threadIdx.x; i < OUTPUT_ATOM_SIZE; i += NUM_THREADS) {
        T val = NUM_WARPS_K > 1 ? output_smem.at(row, i)
                                : mm_intermediate_smem.at(row, i);
        output_dmem.at(row, i) =
            residual ? val + residual_smem.at(row, i) : val;
      }
    }
    if (output_atom_idx + 1 < NUM_OUTPUT_ATOMS) {
      __syncthreads();
    }
  }
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
__global__ void linear_kernel_wrapper(void const *input_ptr,
                                      void const *weight_ptr,
                                      void const *residual_ptr,
                                      void *output_ptr) {
  linear_kernel<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, 4096>(
      input_ptr, weight_ptr, residual_ptr, output_ptr);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_baseline_linear(void const *input_ptr,
                            void const *weight_ptr,
                            void const *residual_ptr,
                            void *output_ptr) {
  dim3 grid_dim(64, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 96 * 1024;

  constexpr int output_size = 64;
  cudaFuncSetAttribute(
      linear_kernel_wrapper<T, BATCH_SIZE, output_size, REDUCTION_SIZE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);

  linear_kernel_wrapper<T, BATCH_SIZE, output_size, REDUCTION_SIZE>
      <<<grid_dim, block_dim, smem_size>>>(
          input_ptr, weight_ptr, residual_ptr, output_ptr);
}

// Matrix dimensions
const int m = 1;
const int k = 4096;
const int n = 4096;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while (0)

int main(int argc, char** argv) {
    int iters = 500;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        }
    }

    // Host memory allocation
    std::vector<bfloat16> h_matrix_A(m * k);
    std::vector<bfloat16> h_matrix_B(k * n);
    std::vector<bfloat16> h_residual(m * n);
    std::vector<bfloat16> h_output_matrix(m * n);

    // Initialize host data (Example with dummy values)
    for (size_t i = 0; i < h_matrix_A.size(); ++i) {
        h_matrix_A[i] = __float2bfloat16(1.0f); 
    }
    for (size_t i = 0; i < h_matrix_B.size(); ++i) {
        h_matrix_B[i] = __float2bfloat16(1.0f);
    }
    for (size_t i = 0; i < h_residual.size(); ++i) {
        h_residual[i] = __float2bfloat16(0.0f);
    }

    // Device memory allocation
    void* d_matrix_A = nullptr;
    void* d_matrix_B = nullptr;
    void* d_residual = nullptr;
    void* d_output_matrix = nullptr;

    CUDA_CHECK(cudaMalloc(&d_matrix_A, h_matrix_A.size() * sizeof(bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_matrix_B, h_matrix_B.size() * sizeof(bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_residual, h_residual.size() * sizeof(bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_output_matrix, h_output_matrix.size() * sizeof(bfloat16)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_matrix_A, h_matrix_A.data(), h_matrix_A.size() * sizeof(bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_matrix_B, h_matrix_B.data(), h_matrix_B.size() * sizeof(bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_residual, h_residual.data(), h_residual.size() * sizeof(bfloat16), cudaMemcpyHostToDevice));

    // Launch the kernel
    launch_baseline_linear<bfloat16, 1, 4096, 4096>(d_matrix_A, d_matrix_B, d_residual, d_output_matrix);
    CUDA_CHECK(cudaDeviceSynchronize());
    // Warmup
    for (int i = 0; i < 10; ++i) {
        launch_baseline_linear<bfloat16, 1, 4096, 4096>(d_matrix_A, d_matrix_B, d_residual, d_output_matrix);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        launch_baseline_linear<bfloat16, 1, 4096, 4096>(d_matrix_A, d_matrix_B, d_residual, d_output_matrix);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iters;

    // Compute GFLOPs + bandwidth
    double flops = 2.0 * double(m) * double(n) * double(k);
    double gflops = (flops / (avg_ms * 1e-3)) / 1e9;

    double bytesA = double(m) * k * sizeof(bfloat16);
    double bytesB = double(k) * n * sizeof(bfloat16);
    double bytesD = double(m) * n * sizeof(bfloat16);
    double gbps = (bytesA + bytesB + bytesD) / (avg_ms * 1e-3) / 1e9;

    // Copy back result
    CUDA_CHECK(cudaMemcpy(h_output_matrix.data(), d_output_matrix, m*n*sizeof(bfloat16), cudaMemcpyDeviceToHost));

    // Log (same as CUTLASS version)
    std::cout << "BF16 GEMM -> FP32 output (M,N,K)=("
              << m << "," << n << "," << k << ")\n"
              << "iters=" << iters << "\n"
              << "Avg time: " << avg_ms << " ms\n"
              << "Perf: " << gflops << " GFLOP/s\n"
              << "BW:   " << gbps  << " GB/s\n";

    std::cout << "D[0..3]: ";
    for (int i = 0; i < std::min(n, 4); ++i) {
        std::cout << __bfloat162float(h_output_matrix[i]) << (i+1<4 ? ", " : "\n");
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_matrix_A));
    CUDA_CHECK(cudaFree(d_matrix_B));
    CUDA_CHECK(cudaFree(d_residual));
    CUDA_CHECK(cudaFree(d_output_matrix));

    std::cout << "CUDA memory allocation and data transfer complete." << std::endl;
    return 0;
}
```
</details>

Please build it with `nvcc --expt-relaxed-constexpr -std=c++17 -arch=sm_80 -o mirage matmul-mirage.cu`.

Its outputs on A100:
```
BF16 GEMM -> FP32 output (M,N,K)=(1,4096,4096)
iters=500
Avg time: 0.0500531 ms
Perf: 670.376 GFLOP/s
BW:   670.704 GB/s
D[0..3]: 4096, 4096, 4096, 4096
```

Its output on RTX3080:
```
BF16 GEMM -> FP32 output (M,N,K)=(1,4096,4096)
iters=500
Avg time: 0.0525944 ms
Perf: 637.984 GFLOP/s
BW:   638.296 GB/s
D[0..3]: 4096, 4096, 4096, 4096
```