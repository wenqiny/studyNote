## What's Bank conflict
Bank conflict means some threads in a same warp to access elements which were not in **different address**, but in same **bank**.

### Bank
Bank means a series of non-continuous slot of share memory, there are total ***32 banks***, it usually was ***4 bytes*** as a slots in modern GPU, which means for the address in `range(0, 4)` bytes are in `bank 0` (**(range(0, 4) % 4) % 32 = 0**), as the same reason, for address in `range(4, 8)` bytes are in `bank 1`, so the address in `range(128, 132)` are also in `bank 0` (**(range(128, 132) % 4) * 32 = 0**).
Therefore, we say the address in `range(0, 4)` and `range(128, 132)` in smem are in same bank.

### Bank conflict case
In the above, we have know address in `range(0, 4)` and `range(128, 132)` in smem are in same bank.

Therefore if we ask threads in one warp to access memory in one bank, like **all 32 thraeds** in a warp to access **bank 0** (includes `range(0, 4)`, `range(128, 132)` and etc.) there should be a **sequencial smem access**, but **not parallel access**.


### Code
For the below code snippet:
```cpp
#include <stdio.h>
#include <string>

__global__ void full_conflict(int t){
    __shared__ int smem[1024];
    int tid = threadIdx.x;
    smem[tid * 32] = t;
    smem[tid * 32] = smem[tid * 32] + 1;
}

__global__ void no_conflict(int t){
    __shared__ int smem[1024];
    int tid = threadIdx.x;
    smem[tid] = t;
    smem[tid] = smem[tid] + 1;
}


__global__ void same_address(int t){
    __shared__ int smem[1024];
    int tid = threadIdx.x;
    smem[0] = t;
    smem[0] = smem[0] + 1;
}

void invoke_kernel(auto kernel, std::string kernel_name){
    dim3 grid(1);
    dim3 block(32);
    cudaEvent_t start, stop;
    float elapsedTime;

    // Create events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // warm up
    for(int i = 0; i < 100; i ++){
        kernel<<<grid, block>>>(1);
    }

    cudaEventRecord(start, 0);
    for(int i = 0; i < 100; i ++){
        kernel<<<grid, block>>>(1);
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

its output is:
```
Time elapsed for full_conflict: 0.137 ms
Time elapsed for no_conflict: 0.121 ms
Time elapsed for same_address: 0.121 ms
```

We could see for the full_conflict take slowest speed.


### Register bank conflict
According to this [paper](https://arxiv.org/pdf/1804.06826), it said there is a concept called register bank conflict in GPU.

GPU will divide all registers into two banks, like `(R0, R2 ...)` is in a bank, and `(R1, R3, ...)` is in another bank.

For a **FFMA** inst, it utilize 3 register as its input like `FFMA R16, R12, R80, R16`, all of `R16`, `R12` and `R80` is in a same bank, each thread could load **64 bits** at most in a cycle (only **2 registers**, each register is **32 bits**), so this instruction may suffer from bank conflict.

The above paper proposed to modify the instruction as `FFMA R17, R12, R80, R17`, we could see it loads data from two banks, hence it will improve performance.

The papaer is based on **Volta** GPU architecture, I didn't know the latest GPU's behaviors.