# syncthreads

`__syncthreads` must be called in two branches if there is  divergence. and we couldn't **allow divergence inside a warp**:

the below case is ok:
```
#include <stdio.h>


__global__ void kernel() {
    int tid = threadIdx.x;
    if(tid < 32) {
        printf("partition 1\n");
        __syncthreads();
    } else {
        printf("partition 2\n");
        __syncthreads();
    }
    printf("finished!\n");
}

int main(){
    dim3 grid(1);
    dim3 block(128);
    kernel<<<grid, block>>>();

    cudaDeviceSynchronize();

    return 0;
}
```

but if we set the `32` to `16` it **will hang**:

```
#include <stdio.h>


__global__ void kernel() {
    int tid = threadIdx.x;
    if(tid < 16) {
        printf("partition 1\n");
        __syncthreads();
    } else {
        printf("partition 2\n");
        __syncthreads();
    }
    printf("finished!\n");
}

int main(){
    dim3 grid(1);
    dim3 block(128);
    kernel<<<grid, block>>>();

    cudaDeviceSynchronize();

    return 0;
}
```