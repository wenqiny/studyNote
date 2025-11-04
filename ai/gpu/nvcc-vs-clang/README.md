## NVCC vs CLANG
NVCC is **not always the best compiler** for `.cu` files.

### Case
I tried to compare a matmul kernel for shape `M, N, K: (8, 4096, 4096)` in two ways:
1. **CUTLASS/CUTE** lib with nvcc as compiler.
2. **Hand written CUDA** with clang as compiler.

#### Compare two kernels
The two kernel shows **same performance** on A100:
```
============== CUDA with clang =================
BF16 GEMM -> FP32 output (M,N,K)=(8,4096,4096)
iters=500
Avg time: 0.0329236 ms
Perf: 8153.27 GFLOP/s
BW:   1023.14 GB/s
D[0..3]: 4096, 4096, 4096, 4096

============== CUTLASS/CUTE with nvcc =================
BF16 GEMM -> FP32 output (M,N,K)=(8,4096,4096)
iters=500
Avg time: 0.0318628 ms
Perf: 8424.73 GFLOP/s
BW:   1057.21 GB/s
D[0..3]: 4096, 4096, 4096, 4096
CUDA memory allocation and data transfer complete.
```

#### Hand written code with nvcc
If we use **nvcc** to compile the **hand written CUDA** kernel, we see there is about **10%** regression compared with **clang** compiled one and the **CUTLASS/CUTE** version.

And from NCU, we could see there is about **40% more instructions** (including LOP3, IMAD, they're used for calculating the address/offset for the memory).

#### How to reproduce
In the folder, there are serveral source files could reproduce this experiment, you could build them with below commands:
```
nvcc -dc --expt-relaxed-constexpr -O3 -std=c++17 -arch=sm_80 -I/root/cutlass/include --ptxas-options=-v -o cutlass-obj.o ./cutlass.cu

nvcc -dlink -arch=sm_80 cutlass-obj.o -o cutlass-obj-dlink.o

clang++-20 cuda-and-compare.cu -o cuda-and-compare-obj.o --cuda-gpu-arch=sm_80 -O3 -std=c++17 -L/usr/local/cuda/lib64 -lcudart  -isystem /usr/include/c++/11 -isystem /usr/include/x86_64-linux-gnu/c++/11 -L/usr/lib/x86_64-linux-gnu/libstdc++.so -c

clang++-20 -std=c++17 -O3 cuda-and-compare-obj.o cutlass-obj.o cutlass-obj-dlink.o  -L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread -o main
```


### Summary
1. NVCC is **not the always best compiler** for CUDA, espicially for some hand-written case, it may not have the optimization phase for them, but CLANG/LLVM could do that.
2. We should use **CUTLASS/CUTE** as much as possiable, because NVCC is highly optimized for their code pattern, they could help us remove redundant instructions.