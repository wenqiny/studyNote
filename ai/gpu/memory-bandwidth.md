# Memory bandwidth
Memory bandwidth is a crucial metrics for a GPU, espicially in DL workload.

## GPU spec
We could get the maximum GPU bandwidth from GPU spec, like for RTX3080, it's about `760 GB/s`.

The data may came from the **GPU memory perspective**, which means it's calculated by a formula like:

$$
\begin{align*}
\text{Effective Data Rate (Gbps)} &= \text{Memory Clock (GHz)} \times \text{Multiplier}\\
\text{Memory Bandwidth (GB/s)} &= \frac{\text{Memory Bus Width (bits)} \times \text{Effective Data Rate (Gbps)​}}{\text{8 (bits/byte)}}
\end{align*}
$$

For RTX3080, it's GDDR6X, the $\text{Multiplier}$ is 16:

$$
\begin{align*}
\text{Effective Data Rate} &= 1.188 \times 16 \approx 19\ \text{(Gbps)}\\
\text{Memory Bandwidth} &= \frac{320 \times 19}{8} = 760\ \text{(GB/s)}
\end{align*}
$$

## Measure microbench
In the real world workload, the memory bandwidth is not very easy to achive the max bandwidth, we may have to ask more SMs and threads in GPU to issue load/store inst, which could help us achieve the max bandwidth, let's illustrate this in a microbench.

There is an const in the below code snippet `const int blocks_per_grid = 68;`, it control how many SMs we use in the memory bandwidth test, in RTX3080, there are 68 SMs, let's see the bandwidth for different bandwidth:

| Launch grid | Achieved Bandwidth (GB/s) |
| ----------- | ------------------------- |
| 34 | 211 |
| 68 | 369 |
| 136 | 520 |
| 272 | 560 |
| 544 | 563 |
| 1088 | 616 |
| 100000 | 670 |

The microbench:

```cpp
#include <iostream>
#include <cuda_runtime.h>

__global__ void bandwidth_kernel(float* d_out, const float* d_in, size_t num_elements) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Each thread processes multiple elements
    // The stride is the total number of threads in the grid (68 * 256)
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < num_elements; i += stride) {
        d_out[i] = d_in[i];
    }
}

int main() {
    // A large number of elements to guarantee it doesn't fit in the L2 cache
    const size_t num_elements = 1024 * 1024 * 1024; // 1B elements (~4 GB)
    const size_t size_bytes = num_elements * sizeof(float);
    const int num_runs = 10; // Run multiple times for stable average

    float *d_in, *d_out;
    cudaMalloc(&d_in, size_bytes);
    cudaMalloc(&d_out, size_bytes);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Set launch parameters to 68 blocks, one for each SM
    const int threads_per_block = 256;
    const int blocks_per_grid = 136; 

    cudaEventRecord(start);
    for (int i = 0; i < num_runs; ++i) {
        bandwidth_kernel<<<blocks_per_grid, threads_per_block>>>(d_out, d_in, num_elements);
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    double total_bytes_transferred = 2.0 * size_bytes * num_runs;
    double bandwidth_gb_s = (total_bytes_transferred / (milliseconds / 1000.0)) / 1e9;
    
    std::cout << "Average kernel execution time (over " << num_runs << " runs): " << milliseconds / num_runs << " ms" << std::endl;
    std::cout << "Achieved Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
```
