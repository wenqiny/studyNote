## Nsight
This md will contain some thing or experience I met when I was using nsight tools.

### Nsight Compute
This part is about nsight compute
#### Details-Memory workload analysis
In this part, there is two perspective, one is how much inst was executed, another one is how much memory/cache was transmited.
1. Inst
In the inst part, one inst means **one inst was issued in a warp**, which means if **at least one thread** in a warp executed an inst which access global memory, this performance event (inst) will be added by 1, so if we would like to get the whole memory access inst count, we should use this number **multiply by 32** in generally.

2. Memory/cache transmission
It means how much memory/cache was transmission between device memory and L2 cache, or L2 cache between L1 cache, it's a **global number**, which means we didn't need to multiply it by any other number.
And if there is a cache hit, like some data already in L2 cache, we didn't need to access from device memory again, so there is no transmission data between them, in generally, the total memory transmission between cache and memory may be **less than** the total memory access inst number multiply by the data size.