
#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <random>
#include <algorithm>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <type_traits>

#include "cuda-cutlass.h"

#define NUM_THREADS 128

#define DEBUG_LOG 0

#define WARM_UP 1

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

// cutlass has already defined this, it's same as below, just comment out.
// template <int N>
// __device__ __forceinline__ void cp_async_wait() {
//   if constexpr (N == 0) {
//     asm volatile("cp.async.wait_all;\n" ::);
//   } else {
//     asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
//   }
// }


// __device__ __forceinline__ void cp_async_fence() {
//   asm volatile("cp.async.commit_group;\n" ::);
// }

// static __device__ __forceinline__ void clear_8_floats(float *buffer) {
//   *((__uint128_t *)(buffer)) = 0ul;
//   *((__uint128_t *)(buffer + 4)) = 0ul;
// }

template <typename T>
__device__ __forceinline__ void ldsm(T *__restrict__ smem_ptr, uint32_t *R) {
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
      : "r"(smem_int_ptr));
}

// __device__ static __forceinline__ void
//     mma_m16n16k16_bf16bf16bf32(float *C, uint32_t *A, uint32_t *B, float *D) {
//   asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
//                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
//                : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
//                : "r"(A[0]),
//                  "r"(A[1]),
//                  "r"(A[2]),
//                  "r"(A[3]),
//                  "r"(B[0]),
//                  "r"(B[1]),
//                  "f"(C[0]),
//                  "f"(C[1]),
//                  "f"(C[2]),
//                  "f"(C[3]));

//   asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
//                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
//                : "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
//                : "r"(A[0]),
//                  "r"(A[1]),
//                  "r"(A[2]),
//                  "r"(A[3]),
//                  "r"(B[2]),
//                  "r"(B[3]),
//                  "f"(C[4]),
//                  "f"(C[5]),
//                  "f"(C[6]),
//                  "f"(C[7]));
// }

template <int B, int M, int S>
struct SwizzleOffsetCalculator {
    static constexpr size_t B_BITS = B;
    static constexpr size_t M_BITS = M;
    static constexpr size_t S_BITS = S;
    static constexpr size_t BLOCK_BITS = B + M + S;
    static constexpr size_t M_AND_S_BITS = M + S;

    static constexpr size_t MASK_IN_BLOCK = (1 << BLOCK_BITS) - 1;
    static constexpr size_t MASK_S = (1 << S_BITS) - 1;
    static constexpr size_t MASK_M = (1 << M_BITS) - 1;

    __device__ __forceinline__ static size_t get_phy_offset(size_t logical_idx) {
        const size_t block_idx = logical_idx >> BLOCK_BITS;
        const size_t in_block_idx = logical_idx & MASK_IN_BLOCK;

        const size_t irow = in_block_idx >> M_AND_S_BITS;
        size_t icol = (in_block_idx >> M_BITS) & MASK_S;
        icol ^= irow;

        const size_t offset_in_bank = in_block_idx & MASK_M;

        return (block_idx << BLOCK_BITS) + (irow << M_AND_S_BITS) +
               (icol << M_BITS) + offset_in_bank;
    }
};

template <typename T,
          int B,
          int M,
          int S,
          size_t INNER_ROW_,
          size_t OUTER_ROW_,
          size_t COL_>
struct smem_col_2drow {
  T *__restrict__ base_ptr;

  using value_type = T;
  using OffsetCalculator = SwizzleOffsetCalculator<B, M, S>;

  static constexpr size_t INNER_ROW = INNER_ROW_;
  static constexpr size_t log2_INNER_ROW = log2_constexpr(INNER_ROW_);
  static constexpr size_t OUTER_ROW = OUTER_ROW_;
  static constexpr size_t ROW = OUTER_ROW_ * INNER_ROW_;
  static constexpr size_t COL = COL_;
  static constexpr size_t SIZE = ROW * COL;
  static constexpr size_t STRIDE_OUTER_ROW = COL * INNER_ROW_;
  static constexpr size_t STRIDE = INNER_ROW_;

  static constexpr size_t INNER_ROW_MASK = (1 << log2_INNER_ROW) - 1;

  __device__ __forceinline__ smem_col_2drow(T *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T *ptr) { base_ptr = ptr; }

  static constexpr size_t size() { return ROW * COL; }

  __device__ __forceinline__ size_t get_swizzled_icol(size_t logical_idx) {
    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);
    
    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    return icol;
  }

  __device__ __forceinline__ size_t get_logical_idx(size_t logical_idx_row,
                                                    size_t logical_idx_col) {
    size_t inner_row = logical_idx_row & INNER_ROW_MASK;
    size_t outer_row = logical_idx_row >> log2_INNER_ROW;
    size_t logical_idx = outer_row * STRIDE_OUTER_ROW + logical_idx_col * STRIDE + inner_row;
    return logical_idx;
  }

  __device__ __forceinline__ size_t get_bank_idx(size_t logical_idx_row,
                                                 size_t logical_idx_col) {
    size_t logical_idx = get_logical_idx(logical_idx_row, logical_idx_col);
    return get_swizzled_icol(logical_idx);
  }

  __device__ __forceinline__ T *operator()(size_t logical_idx_row,
                                           size_t logical_idx_col) {
    size_t logical_idx = get_logical_idx(logical_idx_row, logical_idx_col);
    return &base_ptr[OffsetCalculator::get_phy_offset(logical_idx)];
  }

};

// Row-major layout with split column dimension: OUTER_COL x INNER_COL
template <typename T,
          int B,
          int M,
          int S,
          size_t ROW_,
          size_t INNER_COL_,
          size_t OUTER_COL_>
struct smem_row_2dcol {
  T *__restrict__ base_ptr;

  using value_type = T;
  using OffsetCalculator = SwizzleOffsetCalculator<B, M, S>;

  static constexpr size_t ROW = ROW_;
  static constexpr size_t INNER_COL = INNER_COL_;
  static constexpr size_t log2_INNER_COL = log2_constexpr(INNER_COL_);
  static constexpr size_t OUTER_COL = OUTER_COL_;
  static constexpr size_t COL = INNER_COL * OUTER_COL;
  static constexpr size_t SIZE = ROW * COL;
  static constexpr size_t STRIDE_OUTER_COL = ROW * INNER_COL;
  static constexpr size_t STRIDE = INNER_COL;

  __device__ __forceinline__ smem_row_2dcol(T *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T *ptr) { base_ptr = ptr; }

  static constexpr size_t size() { return ROW * COL; }

  __device__ __forceinline__ int get_offset_in_bank(size_t logical_idx_row,
                                                    size_t logical_idx_col) {
    size_t inner_col = logical_idx_col & ((1 << log2_INNER_COL) - 1);
    size_t outer_col = logical_idx_col >> log2_INNER_COL;
    size_t logical_idx = outer_col * STRIDE_OUTER_COL + logical_idx_row * STRIDE + inner_col;
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);
    return offset_in_bank;
  }

  __device__ __forceinline__ T *operator()(size_t logical_idx_row,
                                           size_t logical_idx_col) {
    size_t inner_col = logical_idx_col & ((1 << log2_INNER_COL) - 1);
    size_t outer_col = logical_idx_col >> log2_INNER_COL;
    size_t logical_idx = outer_col * STRIDE_OUTER_COL + logical_idx_row * STRIDE + inner_col;
    // return &base_ptr[get_swizzled_offset(logical_idx)];
    return &base_ptr[OffsetCalculator::get_phy_offset(logical_idx)];
  }

};
namespace config {
template <typename T_,
          int BATCH_SIZE_,
          int OUTPUT_SIZE_,
          int REDUCTION_SIZE_,
          int kTileM_,
          int kTileN_,
          int kTileK_,
          int kStage_,
          int kSmemLayoutCBatch_,
          typename ComputeType>
struct GemmConfig;
}

// Modified from
// https://github.com/reed-lau/cute-gemm/blob/main/gemm-multi-stage.cu
template <typename T_,
          int BATCH_SIZE, // 8
          int OUTPUT_SIZE, // 64
          int REDUCTION_SIZE, // 4096
          int O_STRIDE = OUTPUT_SIZE, // 64
          int PIPE_MAX = 3> // 3
__device__ __noinline__ void linear_kernel(void const *input_ptr,
                                           void const *weight_ptr,
                                           void const *residual_ptr,
                                           void *output_ptr,
                                           int num_active_tokens,
                                           bool residual) {
  // template <typename Config>
  // __global__ void /* __launch_bounds__(128, 1) */
  // gemm_multi_stage(void *Dptr, const void *Aptr, const void *Bptr, const void
  // *Rptr, int m, int n, int k) {
#if DEBUG_LOG
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("Entering linear_kernel with BATCH_SIZE: %d, OUTPUT_SIZE: %d, REDUCTION_SIZE: %d, O_STRIDE: %d, PIPE_MAX: %d, residual: %d\n", BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, O_STRIDE, PIPE_MAX, residual);
  }
#endif
  using T = std::conditional_t<std::is_same_v<T_, bfloat16>,
                               cute::bfloat16_t,
                               float>; // A temporary hack
  constexpr int TILE_SIZE = 128;
  constexpr int kSmemLayoutCBatch = 1;
  // TODO: Verify this is efficient
  // constexpr int PIPE_DEPTH = OUTPUT_SIZE < 256 ? 5 : PIPE_MAX;
  using Config = config::GemmConfig<T,
                                    BATCH_SIZE,
                                    OUTPUT_SIZE,
                                    REDUCTION_SIZE,
                                    16,
                                    128,
                                    TILE_SIZE,
                                    PIPE_MAX,
                                    kSmemLayoutCBatch,
                                    float>;
  using namespace cute;
  // if (threadIdx.x == 0) {
  //   printf("SmemLayoutAtom: \n"); print(typename Config::SmemLayoutAtom{});
  //   printf("\n"); printf("SmemLayoutA: \n"); print(typename
  //   Config::SmemLayoutA{}); printf("\n"); printf("SmemLayoutB: \n");
  //   print(typename Config::SmemLayoutB{}); printf("\n");
  //   printf("SmemLayoutAtomC: \n"); print(typename Config::SmemLayoutAtomC{});
  //   printf("\n"); printf("SmemLayoutC: \n"); print(typename
  //   Config::SmemLayoutC{}); printf("\n");
  // }
  // return;

  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using SmemLayoutC = typename Config::SmemLayoutC;
  using TiledMMA = typename Config::MMA;

  using S2RCopyAtomA = typename Config::S2RCopyAtomA;
  using S2RCopyAtomB = typename Config::S2RCopyAtomB;
  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;
  using R2SCopyAtomC = typename Config::R2SCopyAtomC;
  using S2GCopyAtomC = typename Config::S2GCopyAtomC;
  using S2GCopyC = typename Config::S2GCopyC;

  constexpr int kTileM = Config::kTileM; // 16
  constexpr int kTileN = Config::kTileN; // 64
  constexpr int LoopM = Config::LoopM; // 1
  constexpr int LoopN = Config::LoopN; // 1
  constexpr int kTileK = Config::kTileK; // 128
  constexpr int kStage = Config::kStage; // 3
  // constexpr int m = Config::BATCH_SIZE;
  // constexpr int n = Config::OUTPUT_SIZE;
  // constexpr int k = Config::REDUCTION_SIZE;

  extern __shared__ char smem[];
  // Align the shared memory to 128 bytes
  T *shm_data = (T *)((reinterpret_cast<uintptr_t>(smem) + 127) / 128 * 128);

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;

#if DEBUG_LOG
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("kTileM: %d, ", kTileM); 
    printf("kTileN: %d, ", kTileN);
    printf("LoopM: %d, ", LoopM);
    printf("LoopN: %d, ", LoopN);
    printf("kTileK: %d, ", kTileK);
    printf("kStage: %d, ", kStage);
  }
#endif

  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);
  T const *__restrict__ d_residual = static_cast<T const *>(residual_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);
  // CANNOT perform residual when redisual_ptr is nullptr
  // if (residual_ptr == nullptr) {
  //   assert(!residual);
  // }

  int bid = blockIdx.x;
  d_weight += OUTPUT_SIZE * REDUCTION_SIZE * bid;
  d_residual += OUTPUT_SIZE * bid;
  d_output += OUTPUT_SIZE * bid;

  Tensor A = make_tensor(make_gmem_ptr(d_input),
                         make_shape(BATCH_SIZE, REDUCTION_SIZE),
                         make_stride(REDUCTION_SIZE, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(d_weight),
                         make_shape(OUTPUT_SIZE, REDUCTION_SIZE),
                         make_stride(REDUCTION_SIZE, Int<1>{}));
  Tensor D = make_tensor(make_gmem_ptr(d_output),
                         make_shape(BATCH_SIZE, OUTPUT_SIZE),
                         make_stride(O_STRIDE, Int<1>{}));
  Tensor R = make_tensor(make_gmem_ptr(d_residual),
                         make_shape(BATCH_SIZE, OUTPUT_SIZE),
                         make_stride(O_STRIDE, Int<1>{}));

  // create identity tensors for predicate
  auto cA = make_identity_tensor(shape(A)); // (m,k) -> (m,k)
  auto cB = make_identity_tensor(shape(B)); // (n,k) -> (n,k)
  auto cC = make_identity_tensor(shape(D)); // (m,n) -> (m,n)

#pragma unroll
  for (int m_iter = 0; m_iter < LoopM; ++m_iter) {
    // int m_iter = 0;
    // int n_iter = 0;
    // // slice the tensor to small one which is used for current thread block.
    Tensor gA = local_tile(
        A,
        make_tile(Int<kTileM>{}, Int<kTileK>{}),
        make_coord(
            m_iter,
            _)); // (kTileM, kTileK, m, k) (_16,_128,32):(4096,_1,_128)
    auto cta_cA = local_tile(
        cA, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(m_iter, _));
#pragma unroll
    for (int n_iter = 0; n_iter < LoopN; ++n_iter) {
      Tensor gB = local_tile(
          B,
          make_tile(Int<kTileN>{}, Int<kTileK>{}),
          make_coord(n_iter, _)); // (kTileN, kTileK, n, k)
                                  // (_64,_128,32):(4096,_1,_128)
      Tensor gD = local_tile(
          D,
          make_tile(Int<kTileM>{}, Int<kTileN>{}),
          make_coord(m_iter, n_iter)); // (kTileM, kTileN, m, n)
                                       // (_16,_64):(64,_1)
      Tensor gR = local_tile(
          R,
          make_tile(Int<kTileM>{}, Int<kTileN>{}),
          make_coord(m_iter, n_iter)); // (kTileM, kTileN, m, n)
                                       // (_16,_64):(64,_1)
#if DEBUG_LOG
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("gA: \n"); print(gA); printf("\n");
        printf("gB: \n"); print(gB); printf("\n");
        printf("gD: \n"); print(gD); printf("\n");
        printf("gR: \n"); print(gR); printf("\n");
      }
#endif

      // shared memory
      // ((_8,_2),(_64,_2),(_1,_3)):((_64,_512),(_1,_1024),(_0,_2048))
      auto sA =
          make_tensor(make_smem_ptr(Ashm),
                      SmemLayoutA{}); // (kTileM, kTileK, kStage) (_16,_128,_3)
      // ((_8,_8),(_64,_2),(_1,_3)):((_64,_512),(_1,_4096),(_0,_8192)) 
      auto sB =
          make_tensor(make_smem_ptr(Bshm),
                      SmemLayoutB{}); // (kTileN, kTileK, kStage) (_64,_128,_3)

      auto cta_coord = make_coord(
          n_iter, m_iter, _); // make_coord(m_iter,_) / make_coord(n_iter,_)
      auto cta_cB = local_tile(cB,
                               make_tile(Int<kTileN>{}, Int<kTileK>{}),
                               make_coord(n_iter, _)); // same as gB
      auto cta_cC = local_tile(cC,
                               make_tile(Int<kTileM>{}, Int<kTileN>{}),
                               make_coord(m_iter, n_iter)); // same as gD

      TiledMMA tiled_mma;
      auto thr_mma = tiled_mma.get_slice(idx);
      // ((_2,_2,_2),_1,_8):((_1,_2,_4),_0,_8)
      auto tCrA = thr_mma.partition_fragment_A(
          gA(_, _, 0)); // (MMA, MMA_M, MMA_K)  ((_2,_2,_2),_1,_8)
      // ((_2,_2),_2,_8):((_1,_2),_32,_4)
      auto tCrB = thr_mma.partition_fragment_B(
          gB(_, _, 0)); // (MMA, MMA_N, MMA_K) ((_2,_2),_2,_8)
      // ((_2,_2),_1,_2):((_1,_2),_0,_4)
      auto tCrD = thr_mma.partition_fragment_C(
          gD); // (MMA, MMA_M, MMA_N) ((_2,_2),_1,_2):((_1,_2),_0,_4)

#if DEBUG_LOG
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("cta_cA: \n"); print(cta_cA); printf("\n");
        printf("cta_cB: \n"); print(cta_cB); printf("\n");
        printf("cta_cC: \n"); print(cta_cC); printf("\n");
        printf("tCrA: \n"); print(tCrA); printf("\n");
        printf("tCrB: \n"); print(tCrB); printf("\n");
        printf("tCrD: \n"); print(tCrD); printf("\n");
      }
#endif

      // Load residual from global memory to shared memory
      auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
      auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);

      using SmemLayoutAtomC = typename Config::SmemLayoutAtomC;
      auto sR_init = make_tensor(sA.data(), SmemLayoutAtomC{});

      S2GCopyC s2g_tiled_copy_c;
      auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
      auto tCgR_s2g = s2g_thr_copy_c.partition_D(gR); // gR in global memory
      auto tCsR_s2g =
          s2g_thr_copy_c.partition_S(sR_init); // sR_init in shared memory
      auto tCcC = s2g_thr_copy_c.partition_S(cta_cC);
      auto tCpC = make_tensor<bool>(make_shape(size<0>(tCcC), Int<1>{}),
                                    make_stride(Int<1>{}, Int<0>{}));
      CUTE_UNROLL
      for (int i = 0; i < size<0>(tCpC); ++i) {
        tCpC(i, 0) =
            elem_less(tCcC(i, 0, 0), shape(D)); // (m,n) 与 shape(m,n) 比
      }

      if (residual) {
        // cute::copy(s2g_tiled_copy_c, tCgR_s2g, tCsR_s2g);
        cute::copy_if(s2g_tiled_copy_c, tCpC, tCgR_s2g, tCsR_s2g);
        __syncthreads();
        // load residual to accumulator registers
        auto tCrD_r2s_view = r2s_thr_copy_c.retile_D(tCrD); // view of tCrD
        auto tCsR_r2s_view =
            r2s_thr_copy_c.partition_S(sR_init); // view of sR_init
        cute::copy(tCsR_r2s_view, tCrD_r2s_view);
        __syncthreads();
      } else {
        clear(tCrD);
      }

      auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
      auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
      // ((_8,_1),_1,((_2,_2),_2),(_1,_3)):((_1,_0),_0,((16,32),_1024),(_0,_2048))
      auto tAsA = s2r_thr_copy_a.partition_S(
          sA); // ? (CPY, CPY_M, CPY_K, kStage) ((_8,_1),_1,(_2,_2,_2),_3)
      // ((_8,_1),_1,_8):((_1,_0),_0,_8)
      auto tCrA_view = s2r_thr_copy_a.retile_D(
          tCrA); // ? (CPY, CPY_M, CPY_K) ((_8,_1),_1,_8)

      auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
      auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
      // ((_8,_1),_1,((_2,_2),_2),(_1,_3)):((_1,_0),_0,((16,32),_4096),(_0,_8192))
      auto tBsB = s2r_thr_copy_b.partition_S(
          sB); // ? (CPY, CPY_N, CPY_K, kStage) ((_8,_1),_1,(_2,_2,_2),_3)
      // (((_4,_2),_1),_1,_8):(((_1,_32),_0),_0,_4)
      auto tCrB_view = s2r_thr_copy_b.retile_D(
          tCrB); // ? (CPY, CPY_N, CPY_K) ((_4,_2),_1,_8)

#if DEBUG_LOG
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("sA: \n"); print(sA); printf("\n");
        printf("tAsA: \n"); print(tAsA); printf("\n");
        printf("tCrA_view: \n"); print(tCrA_view); printf("\n");
        printf("sB: \n"); print(sB); printf("\n");
        printf("tBsB: \n"); print(tBsB); printf("\n");
        printf("tCrB_view: \n"); print(tCrB_view); printf("\n");
      }
#endif

      G2SCopyA g2s_tiled_copy_a;
      auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
      // ((_8,_1),_2,_1,32):((_1,_0),32768,_0,_128)
      auto tAgA_copy = g2s_thr_copy_a.partition_S(
          gA); // (CPY, CPY_M, CPY_K, k) ((_8,_1),_2,_1,32)
      // ((_8,_1),_2,_1,(_1,_3)):((_1,_0),_512,_0,(_0,_2048))
      auto tAsA_copy = g2s_thr_copy_a.partition_D(
          sA); // (CPY, CPY_M, CPY_K, kStage) ((_8,_1),_2,_1,_3)
      auto tAcA = g2s_thr_copy_a.partition_S(cta_cA);

      G2SCopyB g2s_tiled_copy_b;
      auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
      // ((_8,_1),_8,_1,32):((_1,_0),32768,_0,_128)
      auto tBgB_copy = g2s_thr_copy_b.partition_S(
          gB); // (CPY, CPY_N, CPY_K, k) ((_8,_1),_16,_1,32)
      // ((_8,_1),_8,_1,(_1,_3)):((_1,_0),_512,_0,(_0,_8192))
      auto tBsB_copy = g2s_thr_copy_b.partition_D(
          sB); // (CPY, CPY_N, CPY_K, kStage) ((_8,_1),_16,_1,_3)
      auto tBcB = g2s_thr_copy_b.partition_S(cta_cB);
      // only do predicate on m / n dimension, k dimension use stride-0
      // broadcast
      auto tApA = make_tensor<bool>(
          make_shape(size<1>(tAcA),
                     Int<1>{}), // size: M sub-dimension per thread, K dimension
                                // use 1, placeholder
          make_stride(Int<1>{}, Int<0>{}) // broadcast to K
      );
      auto tBpB = make_tensor<bool>(make_shape(size<1>(tBcB), Int<1>{}),
                                    make_stride(Int<1>{}, Int<0>{}));

      // fill predicate: compare if the coordinate is in shape(A)/shape(B)
      CUTE_UNROLL
      for (int im = 0; im < size<0>(tApA); ++im) {
        // tAcA's coordinate is (CPY, CPY_M, CPY_K, tile_k) -> (m,k)
        tApA(im, 0) =
            elem_less(get<0>(tAcA(0, im, 0, 0)), shape<0>(A)); // m < M ?
      }
      CUTE_UNROLL
      for (int in = 0; in < size<0>(tBpB); ++in) {
        tBpB(in, 0) =
            elem_less(get<0>(tBcB(0, in, 0, 0)), shape<0>(B)); // n < N ?
      }

#if DEBUG_LOG
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("tAgA_copy: \n"); print(tAgA_copy); printf("\n");
        printf("tAsA_copy: \n"); print(tAsA_copy); printf("\n");
        printf("tBgB_copy: \n"); print(tBgB_copy); printf("\n");
        printf("tBsB_copy: \n"); print(tBsB_copy); printf("\n");
      }
#endif

      int itile_to_read = 0;
      int ismem_read_stage = 0;
      int ismem_write_stage = 0;

      // warm up
#pragma unroll
      for (int istage = 0; istage < kStage - 1; ++istage) {
        // cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
        //           tAsA_copy(_, _, _, istage));
        // cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
        //           tBsB_copy(_, _, _, istage));
        cute::copy_if(g2s_tiled_copy_a,
                      tApA,
                      tAgA_copy(_, _, _, istage),
                      tAsA_copy(_, _, _, istage));
        cute::copy_if(g2s_tiled_copy_b,
                      tBpB,
                      tBgB_copy(_, _, _, istage),
                      tBsB_copy(_, _, _, istage));
        cp_async_fence();

        ++itile_to_read;
        ++ismem_write_stage;
      }

      // TODO: cp_async_wait later?
      cp_async_wait<kStage - 2>();
      __syncthreads();

      int ik = 0;
      cute::copy(s2r_tiled_copy_a,
                 tAsA(_, _, ik, ismem_read_stage),
                 tCrA_view(_, _, ik));
      cute::copy(s2r_tiled_copy_b,
                 tBsB(_, _, ik, ismem_read_stage),
                 tCrB_view(_, _, ik));

      int ntile = REDUCTION_SIZE / kTileK; // 4096 / 128 = 32 tiles for the whole K dim reduction
      int nk = size<2>(tCrA);              // 8 iteartions inside a tile

#pragma unroll 1
      for (int itile = 0; itile < ntile; ++itile) {
#pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
          int ik_next = (ik + 1) % nk;

          if (ik == 0) {
            if (itile_to_read < ntile) {
              // cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
              //           tAsA_copy(_, _, _, ismem_write_stage));
              // cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
              //           tBsB_copy(_, _, _, ismem_write_stage));
              cute::copy_if(g2s_tiled_copy_a,
                            tApA,
                            tAgA_copy(_, _, _, itile_to_read),
                            tAsA_copy(_, _, _, ismem_write_stage));
              cute::copy_if(g2s_tiled_copy_b,
                            tBpB,
                            tBgB_copy(_, _, _, itile_to_read),
                            tBsB_copy(_, _, _, ismem_write_stage));

              ++itile_to_read;
              ismem_write_stage = (ismem_write_stage + 1) % kStage;
            }

            cp_async_fence();
          }

          if (ik == nk - 1) {
            if(ntile - itile > 2) {
              cp_async_wait<kStage - 2>();
            } else {
              cp_async_wait<0>();
            }
            __syncthreads();

            ismem_read_stage = (ismem_read_stage + 1) % kStage;
          }

          cute::copy(s2r_tiled_copy_a,
                     tAsA(_, _, ik_next, ismem_read_stage),
                     tCrA_view(_, _, ik_next));
          cute::copy(s2r_tiled_copy_b,
                     tBsB(_, _, ik_next, ismem_read_stage),
                     tCrB_view(_, _, ik_next));

          // put it before above cute::copy didn't affect the perf
          // if (ik == 0) {
          //   if (itile_to_read < ntile) {
          //     // cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
          //     //           tAsA_copy(_, _, _, ismem_write_stage));
          //     // cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
          //     //           tBsB_copy(_, _, _, ismem_write_stage));
          //     cute::copy_if(g2s_tiled_copy_a,
          //                   tApA,
          //                   tAgA_copy(_, _, _, itile_to_read),
          //                   tAsA_copy(_, _, _, ismem_write_stage));
          //     cute::copy_if(g2s_tiled_copy_b,
          //                   tBpB,
          //                   tBgB_copy(_, _, _, itile_to_read),
          //                   tBsB_copy(_, _, _, ismem_write_stage));

          //     ++itile_to_read;
          //     ismem_write_stage = (ismem_write_stage + 1) % kStage;
          //   }

          //   cp_async_fence();
          // }

          cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        } // ik
      }   // itile

      // Epilogue: convert float accumulator result back to bfloat16_t and write
      // back use less shared memory as a scratchpad tile to use large wide
      // instuction Dreg -> shm -> reg -> global Reuse sB
      auto sC = make_tensor(sA(_, _, ismem_read_stage).data(),
                            SmemLayoutC{}); // (_16,_64,_2):(_64,_1,_1024)

      auto tCrC_r2s = r2s_thr_copy_c.retile_S(
          tCrD); // (CPY, CPY_M, CPY_N) ((_2,_8),_1,_1):((_1,_2),_0,_0)
      auto tCsC_r2s = r2s_thr_copy_c.partition_D(
          sC); // (CPY, _1, _1, pipe)
               // ((_2,(_2,(_2,_2))),_1,_1,_2):((_1,(_1024,(_32,72))),_0,_0,_2048)

      auto tCsC_s2g = s2g_thr_copy_c.partition_S(
          sC); // (CPY, _1, _1, pipe) ((_8,_1),_1,_2,_2):((_1,_0),_0,72,_2048)
      auto tCgC_s2g = s2g_thr_copy_c.partition_D(
          gD); // (CPY, CPY_M, CPY_N) ((_8,_1),_1,_2):((_1,_0),_0,_64)

      auto tCpC_ep = make_tensor<bool>(make_shape(size<0>(tCcC), Int<1>{}),
                                       make_stride(Int<1>{}, Int<0>{}));
      CUTE_UNROLL
      for (int i = 0; i < size<0>(tCpC_ep); ++i) {
        tCpC_ep(i, 0) = elem_less(tCcC(i, 0, 0), shape(D));
      }

      auto tC_tmp =
          make_tensor_like<T>(tCrC_r2s); // ((_2,_8),_1,_1):((_1,_2),_0,_0)
      cute::copy(tCrC_r2s, tC_tmp);

#if DEBUG_LOG
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("tCrC_r2s: \n"); print(tCrC_r2s); printf("\n");
        printf("tCsC_r2s: \n"); print(tCsC_r2s); printf("\n");
        printf("tCsC_s2g: \n"); print(tCsC_s2g); printf("\n");
        printf("tCgC_s2g: \n"); print(tCgC_s2g); printf("\n");
        // printf("tCgC_s2gx: \n"); print(tCgC_s2gx); printf("\n");
        // printf("tCrC_r2sx: \n"); print(tCrC_r2sx); printf("\n");
        printf("tC_tmp: \n"); print(tC_tmp); printf("\n");
        // printf("tC_tmp(_, 0): \n"); print(tC_tmp(_, 0)); printf("\n");
        printf("tCsC_r2s(_, 0, 0, 0): \n"); print(tCsC_r2s(_, 0, 0, 0)); printf("\n");
        printf("tCsC_r2s(_, 0, 0, 1): \n"); print(tCsC_r2s(_, 0, 0, 1)); printf("\n");
        printf("tCsC_r2s(_, 0, 0, _): \n"); print(tCsC_r2s(_, 0, 0, _)); printf("\n");
        printf("tCsC_r2s(_, _, _, 0): \n"); print(tCsC_r2s(_, _, _, 0)); printf("\n");
        printf("r2s_tiled_copy_c: \n"); print(r2s_tiled_copy_c); printf("\n");
        printf("s2g_tiled_copy_c: \n"); print(s2g_tiled_copy_c); printf("\n");
      }
#endif
      cute::copy(r2s_tiled_copy_c, tC_tmp, tCsC_r2s(_, _, _, 0));
      // ((_2,_4),_1,_1) -> ((_2,(_2,_2)),_1,_1)
      __syncthreads();

      // cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, _, _, 0), tCgC_s2g);
      cute::copy_if(s2g_tiled_copy_c, tCpC_ep, tCsC_s2g(_, _, _, 0), tCgC_s2g);
    } // n_iter < kLoopN 2
  }   // m_iter < LoopM 1
}

namespace config {

using namespace cute;

template <typename T_,
          int BATCH_SIZE_,
          int OUTPUT_SIZE_,
          int REDUCTION_SIZE_,
          int kTileM_,
          int kTileN_,
          int kTileK_,
          int kStage_,
          int kSmemLayoutCBatch_ = 1,
          typename ComputeType = float>
struct GemmConfig {
  using T = T_;

  static constexpr int BATCH_SIZE = BATCH_SIZE_;
  static constexpr int OUTPUT_SIZE = OUTPUT_SIZE_;
  static constexpr int REDUCTION_SIZE = REDUCTION_SIZE_;
  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_ < OUTPUT_SIZE_ ? kTileN_ : OUTPUT_SIZE_;
  static constexpr int LoopN = (OUTPUT_SIZE_ + kTileN - 1) / kTileN;
  static constexpr int LoopM = (BATCH_SIZE_ + kTileM - 1) / kTileM;
  static constexpr int kTileK = kTileK_;
  static constexpr int kStage = kStage_;
  // TODO: add better way to determine PIPE_DEPTH
  // static constexpr int kStage = OUTPUT_SIZE_ < 128 ? 5 : 3;
  static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;
  static constexpr int BankMaxElemNum = 128 / sizeof(T);

  static constexpr int kShmLoadSwizzleM = 3;
  static constexpr int kShmLoadSwizzleS = 3;
  static constexpr int kShmLoadSwizzleB = 3;

  using SmemLayoutAtom = decltype(composition(
      Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
      make_layout(make_shape(Int<8>{}, Int<BankMaxElemNum>{}),
                  make_stride(Int<BankMaxElemNum>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

  using mma_op = SM80_16x8x16_F32BF16BF16F32_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  static constexpr int kMmaEURepeatM = 1;
  static constexpr int kMmaEURepeatN = 4;
  static constexpr int kMmaEURepeatK = 1;
  static constexpr int N_REG_REPEAT = kTileN / (kMmaEURepeatN * 8);
  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  static constexpr int kMmaPN =
      N_REG_REPEAT * kMmaEURepeatN * get<1>(mma_atom_shape{});
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});
  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
  using G2SCopyA =
      decltype(make_tiled_copy(g2s_copy_atom{},
                               make_layout(make_shape(Int<8>{}, Int<16>{}),
                                           make_stride(Int<16>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;

  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
  using S2RCopyAtomA = s2r_copy_atom;
  using S2RCopyAtomB = s2r_copy_atom;

  using SmemLayoutAtomC = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<kMmaPM>{}, Int<BankMaxElemNum>{}),
                  make_stride(Int<BankMaxElemNum>{}, Int<1>{}))));
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));
  // static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >=
  //                   size(SmemLayoutC{}),
  //               "C shared memory request is large than A's one pipe");
  static_assert(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) >=
                    size(SmemLayoutC{}),
                "C shared memory request is large than B's one pipe");

  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
  using S2GCopyC =
      decltype(make_tiled_copy(S2GCopyAtomC{},
                               make_layout(make_shape(Int<16>{}, Int<8>{}),
                                           make_stride(Int<8>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));

  static constexpr int kThreadNum = size(MMA{});
  static_assert(kThreadNum == 128,
                "This config should use 4 warps (128 threads)");

  static constexpr int shm_size_AB =
      cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
  static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
  static constexpr int kShmSize =
      cute::max(shm_size_AB, shm_size_C) * sizeof(T);
};

} // namespace config

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
__global__ void batching_linear_kernel_wrapper(void const *input_ptr,
                                      void const *weight_ptr,
                                      void const *residual_ptr,
                                      void *output_ptr) {
  linear_kernel<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, 4096>(
      input_ptr, weight_ptr, residual_ptr, output_ptr, BATCH_SIZE, false);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_batching_linear(void const *input_ptr,
                            void const *weight_ptr,
                            void const *residual_ptr,
                            void *output_ptr) {
  constexpr int grid_x = 64;
  dim3 grid_dim(grid_x, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 96 * 1024;

  constexpr int output_size = OUTPUT_SIZE / grid_x;
  cudaFuncSetAttribute(
      batching_linear_kernel_wrapper<T, BATCH_SIZE, output_size, REDUCTION_SIZE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);

  batching_linear_kernel_wrapper<T, BATCH_SIZE, output_size, REDUCTION_SIZE>
      <<<grid_dim, block_dim, smem_size>>>(
          input_ptr, weight_ptr, residual_ptr, output_ptr);
}

// Specific the template for link.
template void launch_batching_linear<__nv_bfloat16, m, n, k>(
    const void*, const void*, const void*, void*);