## CUDA graph

CUDA graph was used for improve CUDA performance by reducing the cost for **frequent kernel launch**.

For example, if we would like to implement an attention op for transformer with **SDPA**, there are usually 3 steps:
1. A matmul ($Q K^T$) to calculate attention: `(B, H, T, D) @ (B, H, T, D) -> (B, H, T, T)`
2. A softmax to get the attention score ($\text{softmax}(Q K^T)$): `(B, H, T, T) -> (B, H, T, T)`
3. A matmul ($\text{softmax}(QK^T) V$): `(B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)`

### Two implement
There are two implements:
1. Write a function to call the 3 different kernels for each step.
2. Use CUDA graph to record what the attention did, then we just need launch the graph.

### Time cost compare
```
Standard Kernel Launch Time (avg per iteration): 0.286205 ms
CUDA Graph Execution Time (avg per iteration): 0.284861 ms
CUDA Graph Speedup: 1.00472x
```
There is slightly improvement, but the case is very easy, in the real world train/inference, the improvement should be different.



### Code
``` cuda
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Define dimensions
const int B = 1, H = 32, D = 128, T = 1024;
const int M = B * H, K = D, N = T;

float *d_Q, *d_K, *d_V, *d_Scores, *d_Softmax, *d_Out;
cudaStream_t stream;

// Memory initialization
void initializeMemory() {
    CHECK_CUDA(cudaMalloc(&d_Q, M * N * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, M * N * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, M * N * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Scores, M * N * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Softmax, M * N * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Out, M * N * K * sizeof(float)));
}

// Matrix multiplication kernel
__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}

// Softmax kernel
__global__ void softmax(float* A, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float max_val = -1e9;
    for (int j = 0; j < N; j++) {
        max_val = fmaxf(max_val, A[row * N + j]);
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < N; j++) {
        A[row * N + j] = expf(A[row * N + j] - max_val);
        sum_exp += A[row * N + j];
    }

    for (int j = 0; j < N; j++) {
        A[row * N + j] /= sum_exp;
    }
}

// Standard kernel execution (without cudaDeviceSynchronize)
void standardAttention(cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    
    matmul<<<gridSize, blockSize, 0, stream>>>(d_Q, d_K, d_Scores, M, N, K);
    softmax<<<(M + 255) / 256, 256, 0, stream>>>(d_Scores, M, N);
    matmul<<<gridSize, blockSize, 0, stream>>>(d_Scores, d_V, d_Out, M, N, K);
}

// CUDA Graph setup
cudaGraph_t graph;
cudaGraphExec_t instance;

void captureCudaGraph() {
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Begin capture
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    standardAttention(stream);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    // Instantiate the graph
    CHECK_CUDA(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
}

void executeCudaGraph() {
    CHECK_CUDA(cudaGraphLaunch(instance, stream));
}

// Timing function with warm-up
float timeKernelExecution(void (*func)(), int iterations, int warmup = 10) {
    // Warm-up phase (discard results)
    for (int i = 0; i < warmup; i++) {
        func();
    }
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; i++) {
        func();
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return milliseconds / iterations;
}


// Cleanup CUDA resources
void cleanup() {
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaGraphExecDestroy(instance));
    CHECK_CUDA(cudaStreamDestroy(stream));

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_Scores));
    CHECK_CUDA(cudaFree(d_Softmax));
    CHECK_CUDA(cudaFree(d_Out));
}

// Main benchmarking function
int main() {
    initializeMemory();

    int iterations = 10000;

    // Benchmark standard kernel launch
    float standardTime = timeKernelExecution([](){ standardAttention(stream); }, iterations);
    std::cout << "Standard Kernel Launch Time (avg per iteration): " << standardTime << " ms" << std::endl;

    // Capture CUDA Graph and Benchmark
    captureCudaGraph();
    float graphTime = timeKernelExecution(executeCudaGraph, iterations);
    std::cout << "CUDA Graph Execution Time (avg per iteration): " << graphTime << " ms" << std::endl;

    // Performance improvement
    float speedup = standardTime / graphTime;
    std::cout << "CUDA Graph Speedup: " << speedup << "x" << std::endl;

    cleanup();
    return 0;
}
```