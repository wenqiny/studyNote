
#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <random>
#include <algorithm>

#include "cuda-cutlass.h"

#define NUM_THREADS 128

#define DEBUG 0

#if DEBUG
#define DCHECK(condition) \
  if((condition) == 0) {\
    printf("Dcheck failed at %s:%d\n", __FILE__, __LINE__);\
  }

#define DCHECK_IMPL(condtion1, condition2) \
  if((condtion1) != 0) {\
    if((condition2) == 0) {\
      printf("Dcheck failed at %s:%d\n", __FILE__, __LINE__);\
    }\
  }
#else
#define DCHECK(condition)
#define DCHECK_IMPL(condtion1, condition2)
#endif // DEBUG

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
__device__ __forceinline__ void cp_async_wait_base() {
  if constexpr (N == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
  }
}

__device__ __forceinline__ void cp_async_fence_base() {
  asm volatile("cp.async.commit_group;\n" ::);
}

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

template <int B, int M, int S>
struct SwizzleOffsetCalculator {
  static constexpr size_t B_BITS = B;
  static constexpr size_t M_BITS = M;
  static constexpr size_t S_BITS = S;

  static constexpr size_t MASK_B = (1 << B_BITS) - 1;
  static constexpr size_t MASK_YYY = MASK_B << (M_BITS + S_BITS);

  __device__ __forceinline__ static size_t get_phy_offset(size_t logical_idx) {
    // refer to
    // https://github.com/NVIDIA/cutlass/blob/e6e2cc29f5e7611dfc6af0ed6409209df0068cf2/include/cute/swizzle.hpp#L76-L79.
    return logical_idx ^ ((logical_idx & MASK_YYY) >> S_BITS);
  }
};

// Row-major layout with split column dimension: OUTER_COL x INNER_COL
template <typename T,
          int B,
          int M,
          int S,
          size_t ROW_,
          size_t COL_,
          size_t STAGE_>
struct smem_row_2dcol {
  T *__restrict__ base_ptr;

  using value_type = T;
  using OffsetCalculator = SwizzleOffsetCalculator<B, M, S>;

  static constexpr size_t ROW = ROW_;
  static constexpr size_t INNER_COL = 128 / sizeof(T);
  static constexpr size_t log2_INNER_COL = log2_constexpr(INNER_COL);
  static constexpr size_t OUTER_COL = COL_ / INNER_COL;
  static constexpr size_t COL = COL_;
  static constexpr size_t SIZE = ROW * COL;
  static constexpr size_t STAGE = STAGE_;
  static constexpr size_t STRIDE_OUTER_COL = ROW * INNER_COL;
  static constexpr size_t STRIDE_ROW =
      INNER_COL; // a row just contains inner col elements
  static constexpr size_t STRIDE_STAGE = SIZE;

  __device__ __forceinline__ smem_row_2dcol(T *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T *ptr) {
    base_ptr = ptr;
  }

  __device__ __forceinline__ T *
      operator()(size_t logical_idx_row, size_t logical_idx_col, size_t stage) {
    // stage was in row dim
    size_t inner_col = logical_idx_col & ((1 << log2_INNER_COL) - 1);
    size_t outer_col = logical_idx_col >> log2_INNER_COL;
    size_t logical_idx = stage * STRIDE_STAGE + outer_col * STRIDE_OUTER_COL +
                         logical_idx_row * STRIDE_ROW + inner_col;
    return &base_ptr[OffsetCalculator::get_phy_offset(logical_idx)];
  }
};

// Column-major layout with split row dimension: OUTER_ROW x INNER_ROW
template <typename T,
          int B,
          int M,
          int S,
          size_t ROW_,
          size_t COL_,
          size_t STAGE_>
struct smem_col_2drow {
  T *__restrict__ base_ptr;

  using value_type = T;
  using OffsetCalculator = SwizzleOffsetCalculator<B, M, S>;

  static constexpr size_t COL = COL_;
  static constexpr size_t INNER_ROW = 128 / sizeof(T);
  static constexpr size_t log2_INNER_ROW = log2_constexpr(INNER_ROW);
  static constexpr size_t OUTER_ROW = ROW_ / INNER_ROW;
  static constexpr size_t ROW = ROW_;
  static constexpr size_t SIZE = ROW * COL;
  static constexpr size_t STAGE = STAGE_;
  static constexpr size_t STRIDE_OUTER_ROW = COL * INNER_ROW;
  static constexpr size_t STRIDE_COL =
      INNER_ROW; // a col just contains inner row elements
  static constexpr size_t STRIDE_STAGE = SIZE;

  static constexpr size_t INNER_ROW_MASK = (1 << log2_INNER_ROW) - 1;

  __device__ __forceinline__ smem_col_2drow(T *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T *ptr) {
    base_ptr = ptr;
  }

  __device__ __forceinline__ size_t get_logical_idx(size_t logical_idx_row,
                                                    size_t logical_idx_col,
                                                    size_t stage) {
    size_t inner_row = logical_idx_row & INNER_ROW_MASK;
    size_t outer_row = logical_idx_row >> log2_INNER_ROW;
    size_t logical_idx = stage * STRIDE_STAGE + outer_row * STRIDE_OUTER_ROW +
                         logical_idx_col * STRIDE_COL + inner_row;
    return logical_idx;
  }

  __device__ __forceinline__ T *
      operator()(size_t logical_idx_row, size_t logical_idx_col, size_t stage) {
    // stage was in col dim
    size_t logical_idx =
        get_logical_idx(logical_idx_row, logical_idx_col, stage);
    return &base_ptr[OffsetCalculator::get_phy_offset(logical_idx)];
  }
};


template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE = OUTPUT_SIZE,
          int PIPE_MAX = 3>
__device__ __forceinline__ void cutlass_linear_kernel(void const *input_ptr,
                                              void const *weight_ptr,
                                              void const *residual_ptr,
                                              void *output_ptr,
                                              int num_active_tokens,
                                              bool residual) {
  constexpr int CHUNK_SIZE = 16 / sizeof(T);
  constexpr int OUTPUT_ATOM_SIZE = OUTPUT_SIZE <= 64 ? OUTPUT_SIZE : 64;
  constexpr int log2_OUTPUT_ATOM_SIZE = log2_constexpr(OUTPUT_ATOM_SIZE);

  constexpr int TILE_SIZE = 128;
  constexpr int log2_TILE_SIZE = log2_constexpr(TILE_SIZE);
  constexpr int FORLOOP_RANGE = REDUCTION_SIZE / TILE_SIZE;

  constexpr int ADJUSTED_PIPE_MAX =
      PIPE_MAX < FORLOOP_RANGE ? PIPE_MAX : FORLOOP_RANGE;

  constexpr int NUM_CHUNKS_A = BATCH_SIZE * TILE_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_B = TILE_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_OUTPUT = BATCH_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;

  constexpr int CHUNKS_PER_ROW_A = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_COL_B = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_ROW_C = OUTPUT_ATOM_SIZE / CHUNK_SIZE;

  constexpr int log2_CHUNK_SIZE = log2_constexpr(CHUNK_SIZE);
  constexpr int log2_CHUNKS_PER_ROW_A = log2_constexpr(CHUNKS_PER_ROW_A);
  constexpr int log2_CHUNKS_PER_COL_B = log2_constexpr(CHUNKS_PER_COL_B);
  constexpr int log2_CHUNKS_PER_ROW_C = log2_constexpr(CHUNKS_PER_ROW_C);

  // using SM80_16x8x16_F16F16F16F16_TNX2 = 16X16X16
  constexpr int NUM_WARPS_N =
      4; // We always use NUM_WARPS_K = 1 and NUM_WARPS_N = 4
  constexpr int NUM_WARPS_K = 4 / NUM_WARPS_N;
  // Do not support split K for now
  static_assert(NUM_WARPS_K == 1);

  // TODO: support NUM_ITERS_M > 1, i.e., BATCH_SIZE > 16
  constexpr int NUM_ITERS_M = 1;
  constexpr int NUM_ITERS_N =
      (OUTPUT_SIZE + OUTPUT_ATOM_SIZE - 1) / OUTPUT_ATOM_SIZE;
  constexpr int NUM_ITERS_K =
      (TILE_SIZE + NUM_WARPS_K * 16 - 1) / (NUM_WARPS_K * 16);
  // constexpr int NUM_ITERS_K = 8;

  constexpr int log2_NUM_WARPS_N = log2_constexpr(NUM_WARPS_N);
  constexpr int log2_NUM_ITERS_K = log2_constexpr(NUM_ITERS_K);

  int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  int warp_row = warp_idx >> log2_NUM_WARPS_N;
  int warp_col = warp_idx & (NUM_WARPS_N - 1);
  int lane_idx = threadIdx.x & 0x1f;

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

  using InputDmem = dmem_row_const<T, BATCH_SIZE, TILE_SIZE, REDUCTION_SIZE>;
  using WeightDmem =
      dmem_col_const<T, TILE_SIZE, OUTPUT_ATOM_SIZE, REDUCTION_SIZE>;
  using ResidualDmem = dmem_row_const<T, BATCH_SIZE, OUTPUT_SIZE, O_STRIDE>;
  using OutputDmem = dmem_row<T, BATCH_SIZE, OUTPUT_SIZE, O_STRIDE>;

  InputDmem input_dmem(d_input);
  WeightDmem weight_dmem(d_weight);
  ResidualDmem residual_dmem(d_residual);
  OutputDmem output_dmem(d_output);

  extern __shared__ char smem[];

  // STensors' offsets
  constexpr size_t ZERO_BUFFER_OFFSET = 0;
  // sizeof(T) * 8

  constexpr size_t SHARED_INPUT_BUFFER_OFFSET =
      ZERO_BUFFER_OFFSET + sizeof(T) * 64;
  // sizeof(T) * BATCH_SIZE * TILE_SIZE

  constexpr size_t SHARED_WEIGHT_BUFFER_OFFSET =
      SHARED_INPUT_BUFFER_OFFSET +
      sizeof(T) * BATCH_SIZE * ADJUSTED_PIPE_MAX * TILE_SIZE;

  constexpr size_t SHARED_OUTPUT_OFFSET =
      // MM_INTERMEDIATE_OFFSET +
      SHARED_WEIGHT_BUFFER_OFFSET +
      sizeof(T) * TILE_SIZE * ADJUSTED_PIPE_MAX * OUTPUT_ATOM_SIZE;

  // zero buffer
  T *zero_buf = (T *)(smem + ZERO_BUFFER_OFFSET);
  vec_zero_t<T, 8>::fill_zero(zero_buf);

  // copy
  T *shared_input_buffer = (T *)(smem + SHARED_INPUT_BUFFER_OFFSET);
  T *shared_weight_buffer = (T *)(smem + SHARED_WEIGHT_BUFFER_OFFSET);

  // output
  T *shared_output = (T *)(smem + SHARED_OUTPUT_OFFSET);

  // define the swizzle mode
  using ZeroBufferSmem = smem_row<T, 0, 0, 0, 1, 8, 8>;
  using InputSmem =
      smem_row_2dcol<T, 3, 3, 3, BATCH_SIZE, TILE_SIZE, ADJUSTED_PIPE_MAX>;
  using WeightSmem = smem_col_2drow<T,
                                    3,
                                    3,
                                    3,
                                    TILE_SIZE,
                                    OUTPUT_ATOM_SIZE,
                                    ADJUSTED_PIPE_MAX>;
  using OutputFullSmem =
      smem_row<T, 3, 3, 3, BATCH_SIZE, OUTPUT_ATOM_SIZE, OUTPUT_ATOM_SIZE>;

  // we no longger need zero buffer, but we could keep it to make sure shared
  // memory was aligned.
  ZeroBufferSmem zero_buffer(zero_buf);

  InputSmem input_smem(shared_input_buffer);
  WeightSmem weight_smem(shared_weight_buffer);

  OutputFullSmem output_smem(shared_output);

#pragma unroll
  for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
    // If we use NUM_ITERS_M and NUM_ITERS_N inside NUM_ITERS_K, the
    // loop for NUM_ITERS_K couldn't be unrolled in nvcc which hurts
    // performance.
#pragma unroll
    for (uint32_t nn = 0; nn < NUM_ITERS_N; nn++) {
      float s_frag[8];

      // should we sync here? if NUM_ITERS_N > 1, I suppose we should do it,
      // because we will write output_smem later, but it may be still used in
      // some warp which are still write to gmem.
      if (NUM_ITERS_N > 1) {
        __syncthreads();
      }
      // Initialize output_smem: if residual is provided, preload it; otherwise
      // zero
#pragma unroll
      for (int i = threadIdx.x; i < BATCH_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;
           i += NUM_THREADS) {
        int row = i / (OUTPUT_ATOM_SIZE / CHUNK_SIZE);
        int dst_col = (i % (OUTPUT_ATOM_SIZE / CHUNK_SIZE)) << log2_CHUNK_SIZE;
        int src_col = dst_col + (nn << log2_OUTPUT_ATOM_SIZE);
        // TODO: use ignore-src in load_smem to avoid if-else
        if (residual) {
          load_smem(output_smem(row, dst_col), residual_dmem(row, src_col));
        } else {
          *((__uint128_t *)((void *)&output_smem.at(row, dst_col))) = 0ul;
        }
      }

      // initialize registers
#pragma unroll
      for (uint32_t r = 0; r < 8; r++) {
        s_frag[r] = 0;
      }

      int ismem_read_stage = 0;
      int ismem_write_stage = 0;

      // Warm up weight and input tiles for the first ADJUSTED_PIPE_MAX - 1
      // tile.
#pragma unroll
      for (int istage = 0; istage < ADJUSTED_PIPE_MAX - 1; ++istage) {
        // we don't need module for ADJUSTED_PIPE_MAX here, because we just load
        // ADJUSTED_PIPE_MAX - 1 pipe.
        int src_stage_offset = istage << log2_TILE_SIZE;

#pragma unroll
        for (int chunk = 0; chunk < NUM_CHUNKS_A / NUM_THREADS; chunk++) {
          int tid = threadIdx.x;
          int threadCol = (tid & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
          int threadRow = tid >> log2_CHUNKS_PER_ROW_A;
          constexpr int ROWS_PER_ITERATION = NUM_THREADS / CHUNKS_PER_ROW_A;

          int dst_col = threadCol;
          int src_col = dst_col + src_stage_offset;

          int row_within = threadRow + chunk * ROWS_PER_ITERATION;
          int src_row = row_within;
          int dst_row = row_within;

          load_smem(input_smem(dst_row, dst_col, istage),
                    input_dmem(src_row, src_col));
        }
#pragma unroll
        for (int chunk = 0; chunk < NUM_CHUNKS_B / NUM_THREADS; chunk++) {
          int tid = threadIdx.x;
          int threadRow = (tid & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
          int threadCol = tid >> log2_CHUNKS_PER_COL_B;
          constexpr int COLS_PER_ITERATION = NUM_THREADS / CHUNKS_PER_COL_B;

          int dst_row = threadRow;
          int src_row = dst_row + src_stage_offset;

          int col_within = threadCol + chunk * COLS_PER_ITERATION;
          int src_col = (nn << log2_OUTPUT_ATOM_SIZE) + col_within;
          int dst_col = col_within;

          load_smem(weight_smem(dst_row, dst_col, istage),
                    weight_dmem(src_row, src_col));
        }
        cp_async_fence_base();

        ++ismem_write_stage;
      } // warm up for ADJUSTED_PIPE_MAX - 1

      constexpr int PIPE_INSIDE_TILE = 2;
      uint32_t a_frag[PIPE_INSIDE_TILE][4], b_frag[PIPE_INSIDE_TILE][4];
      // wait for first warm up pipeline cp.async finished
      cp_async_wait_base<ADJUSTED_PIPE_MAX - 2>();
      __syncthreads();

      int warmup_m_col =
          (warp_row << (4 + log2_NUM_ITERS_K)) + ((lane_idx >> 4) << 3);
      int warmup_n_row =
          (warp_row << (4 + log2_NUM_ITERS_K)) + (((lane_idx & 0xF) >> 3) << 3);
      int warmup_smem_row = (lane_idx & 0xF);
      int warmup_n_col =
          (warp_col << 4) + ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
      T *warmup_input_ptr = input_smem(warmup_smem_row, warmup_m_col, 0);
      DCHECK(warmup_n_col < OUTPUT_ATOM_SIZE);
      T *warmup_weight_ptr = weight_smem(warmup_n_row, warmup_n_col, 0);

      ldsm(warmup_input_ptr, a_frag[0]);
      ldsm(warmup_weight_ptr, b_frag[0]);

#pragma unroll 1
      for (int for_idx = 0; for_idx < FORLOOP_RANGE; for_idx++) {
#pragma unroll
        for (int k = 0; k < NUM_ITERS_K; k++) {
          // TODO(Wenqin): use pointer advance for the pointer for input and
          // weight shared memory instead of address calculation for
          // input_smem and weight_smem, because in each iteration in the K
          // dim for the OUTER_ROW/COL, they just advanced a compile-time know
          // offset, and it seems the CUTLASS version just use some ADD inst to
          // do it.
          int k_next = (k + 1) % NUM_ITERS_K;

          if (k == 0) {
            // loading next tile (for_idx + ADJUSTED_PIPE_MAX - 1) when k is 0.
            if (for_idx + ADJUSTED_PIPE_MAX - 1 < FORLOOP_RANGE) {
              int src_stage_offset = (for_idx + ADJUSTED_PIPE_MAX - 1)
                                     << log2_TILE_SIZE;
              // Prefetch next weight tile into ring buffer stage_write
              // Load input tile at the first output tile
#pragma unroll
              for (int chunk = 0; chunk < NUM_CHUNKS_A / NUM_THREADS; chunk++) {
                // we don't need to hoist the threadCol and threadRow,,
                // accorrding to experiment, the nvcc could hoist these const.
                int tid = threadIdx.x;
                int threadCol = (tid & (CHUNKS_PER_ROW_A - 1))
                                << log2_CHUNK_SIZE;
                int threadRow = tid >> log2_CHUNKS_PER_ROW_A;
                constexpr int ROWS_PER_ITERATION =
                    NUM_THREADS / CHUNKS_PER_ROW_A; // 8

                int dst_col = threadCol;
                int src_col = dst_col + src_stage_offset;

                int row_within = threadRow + chunk * ROWS_PER_ITERATION;
                int src_row = row_within;
                int dst_row = row_within;

                load_smem(input_smem(dst_row, dst_col, ismem_write_stage),
                          input_dmem(src_row, src_col));
              }
#pragma unroll
              for (int chunk = 0; chunk < NUM_CHUNKS_B / NUM_THREADS; chunk++) {
                int tid = threadIdx.x;
                int threadRow = (tid & (CHUNKS_PER_COL_B - 1))
                                << log2_CHUNK_SIZE;
                int threadCol = tid >> log2_CHUNKS_PER_COL_B;
                constexpr int COLS_PER_ITERATION =
                    NUM_THREADS / CHUNKS_PER_COL_B; // 8

                int dst_row = threadRow;
                int src_row = dst_row + src_stage_offset;

                int col_within = threadCol + chunk * COLS_PER_ITERATION;
                int src_col = (nn << log2_OUTPUT_ATOM_SIZE) + col_within;
                int dst_col = col_within;

                load_smem(weight_smem(dst_row, dst_col, ismem_write_stage),
                          weight_dmem(src_row, src_col));
              }
              ismem_write_stage = (ismem_write_stage + 1) % ADJUSTED_PIPE_MAX;
            }
            cp_async_fence_base();
          } // k == 0 for load next tile

          if (k == NUM_ITERS_K - 1) {
            // wait cp.async because we will load next tile data in to regs
            // when k == NUM_ITERS_K - 1.
            if (FORLOOP_RANGE - for_idx > 2) {
              cp_async_wait_base<ADJUSTED_PIPE_MAX - 2>();
            } else {
              cp_async_wait_base<0>();
            }
            __syncthreads();

            // TODO(Wenqin): The comment out code below here is what we could
            // do for just use ADD for input and weight shared memory pointer.
            // int tmp_ismem_read_stage = ismem_read_stage;
            ismem_read_stage = (ismem_read_stage + 1) % ADJUSTED_PIPE_MAX;
            // input_ptr += (ismem_read_stage - tmp_ismem_read_stage) * (8 *
            // 128); weight_ptr += (ismem_read_stage - tmp_ismem_read_stage) *
            // (64 * 128);
          } // k == NUM_ITERS_K - 1

          static_assert(NUM_ITERS_M == 1);

          int m_row = (lane_idx & 0xF) + (m << 4);
          int n_col =
              (warp_col << 4) + ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
          DCHECK(n_col < OUTPUT_ATOM_SIZE);

          int m_col = (warp_row << (4 + log2_NUM_ITERS_K)) + (k_next << 4) +
                      ((lane_idx >> 4) << 3);
          int n_row = (warp_row << (4 + log2_NUM_ITERS_K)) + (k_next << 4) +
                      (((lane_idx & 0xF) >> 3) << 3);

          int smem_row = m_row;
          T *valid_input_ptr = input_smem(smem_row, m_col, ismem_read_stage);
          // we don't need to check for is_input_valid, because we will use
          // num_active_tokens for the output, we will just pick valid output.
          T *input_ptr = valid_input_ptr;

          T *valid_weight_ptr = weight_smem(n_row, n_col, ismem_read_stage);
          T *weight_ptr = valid_weight_ptr;

          ldsm(input_ptr, a_frag[(k + 1) % PIPE_INSIDE_TILE]);
          ldsm(weight_ptr, b_frag[(k + 1) % PIPE_INSIDE_TILE]);
          mma_m16n16k16_bf16bf16bf32(s_frag,
                                     a_frag[k % PIPE_INSIDE_TILE],
                                     b_frag[k % PIPE_INSIDE_TILE],
                                     s_frag);

        } // loop for NUM_ITERS_K
      } // loop for FORLOOP_RANGE

#pragma unroll
      for (uint32_t i = 0; i < 4; i++) {
        int row_in_warp = (lane_idx >> 2) + ((i & 0x1) << 3);
        int col_within =
            (warp_col << 4) + ((lane_idx & 0x3) << 1) + ((i >> 1) << 3);
        int col = col_within;
        DCHECK(col_within < OUTPUT_ATOM_SIZE);
        if (row_in_warp < num_active_tokens) {
          // TODO: try st.matrix here?
          output_smem.at(row_in_warp, col) += bfloat16(s_frag[(i << 1)]);
          output_smem.at(row_in_warp, col + 1) +=
              bfloat16(s_frag[(i << 1) | 0x1]);
        }
      }
      __syncthreads();

      // Final writeback: store accumulated output (residual already included if
      // any)
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_OUTPUT; i += NUM_THREADS) {
        int row = i / CHUNKS_PER_ROW_C;
        int src_col = (i % CHUNKS_PER_ROW_C) << log2_CHUNK_SIZE;
        int dst_col = src_col + (nn << log2_OUTPUT_ATOM_SIZE);
        *((__uint128_t *)((void *)&output_dmem.at(row, dst_col))) =
            *((__uint128_t *)((void *)&output_smem.at(row, src_col)));
      }
    } // loop for NUM_ITERS_N, it may not be 1
  } // loop for NUM_ITERS_M, it should always be 1, no sense loop
}


template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
__global__ void linear_kernel_wrapper(void const *input_ptr,
                                      void const *weight_ptr,
                                      void const *residual_ptr,
                                      void *output_ptr) {
  cutlass_linear_kernel<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, 4096>(
      input_ptr, weight_ptr, residual_ptr, output_ptr, BATCH_SIZE, false);
}



template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_baseline_linear(void const *input_ptr,
                            void const *weight_ptr,
                            void const *residual_ptr,
                            void *output_ptr) {
  constexpr int grid_x = 64;
  dim3 grid_dim(grid_x, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 96 * 1024;

  constexpr int output_size = OUTPUT_SIZE / grid_x;
  cudaFuncSetAttribute(
      linear_kernel_wrapper<T, BATCH_SIZE, output_size, REDUCTION_SIZE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);

  linear_kernel_wrapper<T, BATCH_SIZE, output_size, REDUCTION_SIZE>
      <<<grid_dim, block_dim, smem_size>>>(
          input_ptr, weight_ptr, residual_ptr, output_ptr);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
extern void launch_batching_linear(void const *input_ptr,
                            void const *weight_ptr,
                            void const *residual_ptr,
                            void *output_ptr);

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

    CUDA_CHECK(cudaDeviceSynchronize());
    // Warmup
    for (int i = 0; i < 10; ++i) {
        launch_baseline_linear<bfloat16, batch_size, n, k>(d_matrix_A, d_matrix_B, d_residual, d_output_matrix);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        launch_baseline_linear<bfloat16, batch_size, n, k>(d_matrix_A, d_matrix_B, d_residual, d_output_matrix);
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
    std::cout << "============== CUDA with clang =================" << std::endl;
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

    // benchmark for batching
    CUDA_CHECK(cudaDeviceSynchronize());
    // Warmup
    for (int i = 0; i < 10; ++i) {
        launch_batching_linear<bfloat16, batch_size, n, k>(d_matrix_A, d_matrix_B, d_residual, d_output_matrix);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start1, stop1;
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));
    CUDA_CHECK(cudaEventRecord(start1));
    for (int i = 0; i < iters; ++i) {
        launch_batching_linear<bfloat16, batch_size, n, k>(d_matrix_A, d_matrix_B, d_residual, d_output_matrix);
    }
    CUDA_CHECK(cudaEventRecord(stop1));
    CUDA_CHECK(cudaEventSynchronize(stop1));

    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start1, stop1));
    
    avg_ms = total_ms / iters;
    gflops = (flops / (avg_ms * 1e-3)) / 1e9;

    // Compute GFLOPs + bandwidth
    gbps = (bytesA + bytesB + bytesD) / (avg_ms * 1e-3) / 1e9;

    // Copy back result
    CUDA_CHECK(cudaMemcpy(h_output_matrix.data(), d_output_matrix, m*n*sizeof(bfloat16), cudaMemcpyDeviceToHost));

    // Log (same as CUTLASS version)
    std::cout << "\n============== CUTLASS/CUTE with nvcc =================" << std::endl;
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