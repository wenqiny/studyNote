## LDGSTS

`ldgsts` is a new feature on `Ampere` and later Nvidia GPU, which means **load global and store shared**, investigate it later.

### Code
```
#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

// For RTX3080
constexpr int blockNum = 68;
constexpr int threadNum = 512;
constexpr int M = threadNum;
constexpr bool is_same_address_all_blocks = true;

// The kernel load 2 * M elements into SRAM, and calculate the sum of them.
__global__ void ldgsts_kernel(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C) {
    __shared__ float sA[M];
    __shared__ float sB[M];
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    const int offset = is_same_address_all_blocks ? 0 : M * bid;

    // issue the ldgsts.
    __pipeline_memcpy_async(&sA[0], &A[offset + tid], sizeof(float));
    __pipeline_memcpy_async(&sB[0], &B[offset + tid], sizeof(float));

    // Commit the pipeline (initiate the async copies)
    __pipeline_commit();
    
    // Wait for previous pipeline operations to complete
    __pipeline_wait_prior(0);
    
    // Synchronize to ensure tiles are loaded
    __syncthreads();
    
    float tmp = sA[tid] + sB[tid];
    
    C[offset + tid] = tmp;
}


int main() {
    
    // Allocate host memory
    float *h_A = new float[blockNum * M];
    float *h_B = new float[blockNum * M];
    float *h_C = new float[blockNum * M];

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, blockNum * M * sizeof(float));
    cudaMalloc(&d_B, blockNum * M * sizeof(float));
    cudaMalloc(&d_C, blockNum * M * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, blockNum * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, blockNum * M * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    dim3 grid(blockNum);
    dim3 block(threadNum);
    
    ldgsts_kernel<<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaError_t error1 = cudaGetLastError();
    if (error1 != cudaSuccess) {
        std::cerr << "ldgsts_kernel Launch Failed: " << cudaGetErrorString(error1) << std::endl;
    } else {
        std::cout << "ldgsts_kernel Launch Successful" << std::endl;
    }
    
    // Copy results back to host
    cudaMemcpy(h_C, d_C, blockNum * M * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
```