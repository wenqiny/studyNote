# LD.MATRIX
`ld.matrix` is a GPU inst which used to load data from shared memory to register before execute a `mma` inst.

## Case study
Let's take `ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4]` as an example, it will loads data from `[%4]` in smem to register `%0, %1, %2, %3`, each threads will use their own registers and some specific address.

It's very straightforward that `%0, %1, %2, %3` is an array for registers, but what's `[%4]` in each thread?

From NV doc, it said "**Each address corresponds to the start of a matrix row. Addresses addr0–addr7 correspond to the rows of the first matrix, addresses addr8–addr15 correspond to the rows of the second matrix, and so on.**".

| .num | Threads 0–7 | Threads 8-15 | Threads 16–23 | Threads 24–31 |
| ---- | ----------- | ------------ | ------------- | ------------- |
| .x1  | addr0–addr7 | – | – | – |
| .x2  | addr0–addr7 | addr8–addr15 | – | – |
| .x4  | addr0–addr7 | addr8–addr15 | addr16–addr23 | addr24–addr31 |

For a `.x4` case, which means there is 4 `8x8` matrix, the address argument for each thread looks like (try to draw a diagram from excalidraw later):

```
-------------------------------------------------------
| t0  | ... 7 elements ... | t8  | ... 7 elements ... |
| t1  | ... 7 elements ... | t9  | ... 7 elements ... |
| ................................................... |
| t7  | ... 7 elements ... | t15 | ... 7 elements ... |
------------------------ half -------------------------
| t16 | ... 7 elements ... | t24 | ... 7 elements ... |
| t17 | ... 7 elements ... | t25 | ... 7 elements ... |
| ................................................... |
| t23 | ... 7 elements ... | t31 | ... 7 elements ... |
-------------------------------------------------------
```


## Code snippet
There is a case it use **1** thread block and **32** threads in this block to load a `16*16` matrix with `m8n8.x4`, please compile it with `nvcc -arch=sm_80 ldmatrix.cu`:

<details>
<summary>code</summary>

```cpp
// Build with `nvcc -arch=sm_80 ldmatrix.cu`

#include <iostream>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <random>
#include <algorithm>

using bfloat16 = __nv_bfloat16;

const int input_row = 16;
const int input_col = 16;
static_assert(input_row == input_col);
const int input_size = input_row * input_col;

__device__ __forceinline__ void ldsm(bfloat16 *__restrict__ smem_ptr, uint32_t *R) {
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
      : "r"(smem_int_ptr));
}

__global__ void ld_matrix(const void* input_ptr) {
    const bfloat16* input = reinterpret_cast<const bfloat16*>(input_ptr);
    extern __shared__ bfloat16 smem[];
    if(threadIdx.x == 0) {
        // initialize the shared memory
        for(size_t i = 0; i < input_size; i ++) {
            smem[i] = input[i];
        }
    }

    // waitting for shared memory ready.
    __syncthreads();
    const int lane_id = threadIdx.x % 32;

    uint32_t regs[4];
    // Please refer to https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix.
    // The 16 * 16 matrix and the pointer to each lane in a warp looks like:

    // -------------------------------------------------------
    // | t0  | ... 7 elements ... | t8  | ... 7 elements ... |
    // | t1  | ... 7 elements ... | t9  | ... 7 elements ... |
    // | ................................................... |
    // | t7  | ... 7 elements ... | t15 | ... 7 elements ... |
    // ------------------------ half -------------------------
    // | t16 | ... 7 elements ... | t24 | ... 7 elements ... |
    // | t17 | ... 7 elements ... | t25 | ... 7 elements ... |
    // | ................................................... |
    // | t23 | ... 7 elements ... | t31 | ... 7 elements ... |
    // -------------------------------------------------------

    bfloat16* thread_ptr = smem;
    // if lane id < 16, we will take it as the upper half
    const int row_offset = ((lane_id & 0x10) >> 4) * 8;
    // if lane id & 0x8, we think they're left half
    const int col_offset = lane_id & 0x8;

    const int row = row_offset + (lane_id & 0x7);
    const int col = col_offset;
    thread_ptr += row * input_row + col;
    ldsm(thread_ptr, regs);

    {
        // load finished and check here.
        if(lane_id == 0) {
            // each thread load 4 32 bits reg, so there is 8 elements.
            for(int i = 0; i < 8; i ++) {
                float tmp = (reinterpret_cast<bfloat16*>(regs))[i];
                printf("At lane 0, the %dth reg is %f\n", i, tmp);
            }
        }
        // Other threads wait here.
        __syncthreads();
    }
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while (0)



int main() {
    std::vector<bfloat16> host(input_size);

    // Initialize host data (Example with dummy values)
    for (size_t i = 0; i < input_size; ++i) {
        host[i] = __float2bfloat16(i);
    }

    void* device = nullptr;

    CUDA_CHECK(cudaMalloc(&device, input_size * sizeof(bfloat16)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(device, host.data(), input_size * sizeof(bfloat16), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());

    ld_matrix<<<1, 32, input_size * sizeof(bfloat16)>>>(device);

    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}
```

</details>

The output looks like:
```
At lane 0, the 0th reg is 0.000000
At lane 0, the 1th reg is 1.000000
At lane 0, the 2th reg is 8.000000
At lane 0, the 3th reg is 9.000000
At lane 0, the 4th reg is 128.000000
At lane 0, the 5th reg is 129.000000
At lane 0, the 6th reg is 136.000000
At lane 0, the 7th reg is 137.000000
```


## Reference
[Official doc](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix)