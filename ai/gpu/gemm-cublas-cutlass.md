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

Please build it with `nvcc matmul-cublas.cu -lcublas`.

Its outputs on A100:
```
Warming up the GPU with 10 runs...
Starting benchmark with 1000 runs...
----------------------------------------
BF16 GEMM (1x4096) @ (4096x4096) on A100
Total Execution Time: 32.4342 ms
Average Time per Run: 0.0324342 ms
----------------------------------------
```

It's close to `0.02` ms therotical max output.

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
GEMM BF16 (M,N,K)=(1,4096,4096)
splitK = 1, iters = 500
Avg time: 0.101104 ms
Perf: 331.882 GFLOP/s
BW:   332.044 GB/s (A+B+D only)
D[0..3]: -7.625, -0.402344, -0.433594, -6.125
```

It taks `0.1` ms for each round, there is much space to optimize it on A100.