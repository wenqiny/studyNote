## What's Bank conflict
Bank conflict means some threads in a same warp to access **elements** which were not in **different address (4 bytes range)**, but in same **bank**.

### Bank
Bank means a series of non-continuous slot of share memory, there are total ***32 banks***, it usually was ***4 bytes*** as a slots in modern GPU, which means for the address in `range(0, 4)` bytes are in `bank 0` (**(range(0, 4) % 4) % 32 = 0**), as the same reason, for address in `range(4, 8)` bytes are in `bank 1`, so the address in `range(128, 132)` are also in `bank 0` (**(range(128, 132) % 4) * 32 = 0**).
Therefore, we say the address in `range(0, 4)` and `range(128, 132)` in smem are in same bank.

### Bank conflict case
In the above, we have know address in `range(0, 4)` and `range(128, 132)` in smem are in same bank.

Therefore if we ask threads in one warp to access memory in one bank, like **all 32 thraeds** in a warp to access **bank 0** (includes `range(0, 4)`, `range(128, 132)` and etc.) there should be a **sequencial smem access**, but **not parallel access**.


### Code
For the below code snippet:
<details>
<summary>Code</summary>

```cpp
#include <stdio.h>
#include <string>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <curand_kernel.h>

#define CUDA_CHECK(call)                                                          \
    {                                                                             \
        const cudaError_t error = call;                                           \
        if (error != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error: %s in %s at line %d\n",                  \
                    cudaGetErrorString(error), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    }

#define CURAND_CHECK(call)                                                        \
    {                                                                             \
        const curandStatus_t status = call;                                       \
        if (status != CURAND_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuRAND Error: %d in %s at line %d\n",                \
                    status, __FILE__, __LINE__);                                  \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    }

#define USE_HALF 0

#if USE_HALF
#define data_type half
#else
#define data_type float
#endif

const int read_write_round = 1000000;
const int thread_num = 128;

__global__ void full_conflict(int t, data_type* d_mem, int round){
    __shared__ data_type smem[thread_num * 32];
    const int tid = threadIdx.x;

    curandState_t localState;
    unsigned long long seed = clock(); // Provides some time-based randomness
    unsigned long long sequence = tid; // Ensures each thread has a different sequence

    curand_init(seed, sequence, 0, &localState);

    // Generate a uniform float in [0.0, 1.0)
    smem[tid * 32] = t;
    float rand = curand_uniform(&localState);
    for(int i = 0; i < round; i ++) {
#if USE_HALF
        smem[tid * 32] = __hadd(smem[tid * 32], __float2half(rand));
#else
        smem[tid * 32] = smem[tid * 32] + rand;
#endif
        // NOTE: __syncthreads ask the compiler not to do optimization, and force
        // it do SRAM read and write.
        __syncthreads();
    }
    d_mem[tid] = smem[tid * 32];
}

__global__ void no_conflict(int t, data_type* d_mem, int round){
    __shared__ data_type smem[thread_num];
    const int tid = threadIdx.x;
    curandState_t localState;
    unsigned long long seed = clock(); // Provides some time-based randomness
    unsigned long long sequence = tid; // Ensures each thread has a different sequence

    curand_init(seed, sequence, 0, &localState);

    // Generate a uniform float in [0.0, 1.0)
    
    smem[tid] = t;
    float rand = curand_uniform(&localState);
    for(int i = 0; i < round; i ++) {
#if USE_HALF
        smem[tid] = __hadd(smem[tid], __float2half(rand));
#else
        smem[tid] = smem[tid] + (rand);
#endif
        // NOTE: __syncthreads ask the compiler not to do optimization, and force
        // it do SRAM read and write.
        __syncthreads();
    }
    d_mem[tid] = smem[tid];
}


__global__ void same_address(int t, data_type* d_mem, int round){
    __shared__ data_type smem[thread_num];
    const int tid = threadIdx.x;
    curandState_t localState;
    unsigned long long seed = clock(); // Provides some time-based randomness
    unsigned long long sequence = threadIdx.x; // Ensures each thread has a different sequence

    curand_init(seed, sequence, 0, &localState);

    // Generate a uniform float in [0.0, 1.0)
    
    smem[0] = t;
    float rand = curand_uniform(&localState);
    for(int i = 0; i < read_write_round; i ++) {
#if USE_HALF
        smem[0] = __hadd(smem[0], __float2half(rand));
#else
        smem[0] = smem[0] + rand;
#endif
        // NOTE: __syncthreads ask the compiler not to do optimization, and force
        // it do SRAM read and write.
        __syncthreads();
    }
    d_mem[tid] = smem[tid];
}

void invoke_kernel(auto kernel, std::string kernel_name){
    dim3 grid(1);
    dim3 block(thread_num);
    cudaEvent_t start, stop;
    float elapsedTime;
    const int test_round = 1;

    data_type* d_mem;
    cudaMalloc(&d_mem, thread_num * sizeof(data_type));
    // Create events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // warm up
    for(int i = 0; i < test_round; i ++){
        kernel<<<grid, block>>>(1, d_mem, read_write_round);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEventRecord(start, 0);
    for(int i = 0; i < test_round; i ++){
        kernel<<<grid, block>>>(1, d_mem, read_write_round);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Time elapsed for %s: %.3f ms\n", kernel_name.c_str(), elapsedTime);
}

int main(){
    invoke_kernel(&full_conflict, "full_conflict");
    invoke_kernel(&no_conflict, "no_conflict");
    invoke_kernel(&same_address, "same_address");
}
```

</details>

### Output for float
its output for `float` is:
```
Time elapsed for full_conflict: 136.737 ms
Time elapsed for no_conflict: 34.483 ms
Time elapsed for same_address: 32.837 ms
```

### Output for half
its output for `half` is:
```
Time elapsed for full_conflict: 83.670 ms
Time elapsed for no_conflict: 43.161 ms
Time elapsed for same_address: 40.520 ms
```

### Summary
#### Number of bank conflict
We could see for the full_conflict take slowest speed, it takes about **4x** regression when we test for `float`, the reason why `half` get a relatively low regression, it's because the size of `half` is smaller than `float`, for a **warp (32 threads)**, please see below table:
| data type | size of warp access | bank conflict per inst |
| --------- | ------------------- | ---------------------- |
| float | $$32 \times 4 = 128\ \text{bytes}$$ | $$\frac{128\ \text{bytes}}{128\ \text{bytes}} \times 32 - 1 = 31$$ |
| half | $$32 \times 2 = 64\ \text{bytes}$$ | $$\frac{64\ \text{bytes}}{128\ \text{bytes}} \times 32 - 1 = 15$$ |

We could see there is just $\frac{15}{31}=0.48$ of bank conflict compare **half** with **float**.

#### Access same 4 bytes but for two half
Given **half** type takes 2 bytes, so if we ask two threads in a same warp to access two different **half** elements inside a same **4 bytes** bank, there is also no bank confilt, we could see the output for `half`, it didn't show much regression compare the `no_conflict` with `same_address`, and we could also see there is **0** bank conflict on nsigh compute.

### Register bank conflict
According to this [paper](https://arxiv.org/pdf/1804.06826), it said there is a concept called register bank conflict in GPU.

GPU will divide all registers into two banks, like `(R0, R2 ...)` is in a bank, and `(R1, R3, ...)` is in another bank.

For a **FFMA** inst, it utilize 3 register as its input like `FFMA R16, R12, R80, R16`, all of `R16`, `R12` and `R80` is in a same bank, each thread could load **64 bits** at most in a cycle (only **2 registers**, each register is **32 bits**), so this instruction may suffer from bank conflict.

The above paper proposed to modify the instruction as `FFMA R17, R12, R80, R17`, we could see it loads data from two banks, hence it will improve performance.

The papaer is based on **Volta** GPU architecture, I didn't know the latest GPU's behaviors.