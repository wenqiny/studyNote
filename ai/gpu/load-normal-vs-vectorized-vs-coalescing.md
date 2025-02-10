## Normal vs vectorized load
In our understand if we load elements in vectorized, it will bring benefits much, is it correct?

Let's see a case, which is inspired from `layer_norm` op, in this op, we should load all elements in one vector, then add them up together.

### naive normal vs vectorized
For naive normal, we will load one element then add it to the sum in PTX code, its CUDA code looks like:
```cpp
for(int i = 0; i < length; i ++){
    int offset = i + length * tid;
    if(offset < N){
        total += blk_mem[offset];
    }
}
```

it we compare it with vectorized load:
```
Time elapsed for normal_load: 59.303 ms
Time elapsed for vetorized_load: 18.758 ms
```

### new normal vs vectorized
For new nromal way, we could try to load all elements once then **store them in registers**, then we sum them up, in PTX code, it means we execute 32 load inst firstly, then do 31 inst to add them up, its CUDA looks like:
```cpp
float regs[length];
for(int i = 0; i < length; i ++){
    int offset = i + length * tid;
    if(offset < N){
        // total += blk_mem[offset];
        regs[i] = blk_mem[offset];
    }
}
for(int i = 0; i < length; i ++){
    total += regs[i];
}
```

its output is:
```
Time elapsed for normal_load: 27.381 ms
Time elapsed for vetorized_load: 18.773 ms
```

### Coalescing normal vs vectorized
Given we know the meory load issue from thread to Cache/memory are in warp level, for the previous 2 kenels, each load global issed will access for **32 different addresses** with gap as **32 * 4 = 128 bytes**, so it needs **32 sectors**, but if we make **all threads in a warp** to access contiguous/consecutive memory, which could **coalescing memory access**, its CUDA code loosk like:
```cpp
const int threads = 256; // it's thread number in a block.
for(int i = 0; i < length; i ++){
    int offset = i * threads + tid;
    if(offset < N){
        total += blk_mem[offset];
    }
}
```

its output is:
```
Time elapsed for normal_load: 2.219 ms
Time elapsed for vetorized_load: 2.379 ms
```
We could see the non-vectorized version even get close performance with vectorized kernel!!!


### Analysis
We could see if we utilize the new normal way, it bring **~50%** perf gain, why?
Some of my guess:
1. it could issue much load inst together, which allow shcedule to switch the warp out then execute another warp, which helps to **hidden the load stall latency**. If we load one element then add it, it may not be very easy to be switched out, because the stall is small (just 1 element load).
2. If the 32 elements are contiguous and aligned, loading them all together in a batch makes better use of memory bandwidth.
3. If we could make the `sectors/inst` number lower by **coalescing**, which could help use improve memory coalescing, and benefit performance **much**!!!

The above bullet 1 and 2 are **guess**, we may verify it through some tool like nsigh compute.

### Code:
please see cuda and triton in this [gist](https://gist.github.com/ywq880611/d19d1a3d839011fa3f542216d8399a02).