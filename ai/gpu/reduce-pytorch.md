## How to reduce on a block/warp in CUDA
### What reduce means?
Usually we would like to know the **sum** or **max** of some variable in a kernel among **a block** or **a warp**, like in **softmax** op, we have to get the max to do safe softmax, and use sum to calculate the final result.

### How to do it in warp
In pytorch, there is a warp reduce method:
```cpp
template <typename T, class ReduceOp>
__inline__ __device__ T WarpReduce(T val, const ReduceOp& op) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val = op.combine(val, op.warp_shfl_down(val, offset));
  }
  return val;
}
```
`val` is the variable in each thread, `op` is the reduce operator.
In the `warp_shfl_down`: 
1. it will call `__shfl_down_sync` API to shuffle the result to the previous `offset`th thread.
2. it will iterate among all warp by offset with `(16, 8, 4, 2, 1)`, so we could image first it will do reduce between `(0, 16)`, `(1, 17)` ... then `(0, 8)`, `(1, 9)` ... finally `(0,1)`
3. Finally, the `val` in the **1st thread in the warp** will be the real result, we shouldn't access the `val` in other thread, it's a UB(undefined behavior).

### How to do it in block
In pytorch, there is a block reduce method:
```cpp
template <typename T, class ReduceOp, typename B = Block1D>
__inline__ __device__ T
BlockReduce(T val, const ReduceOp& op, const T& identity_element, T* shared) {
  const int tid = B::Tid();
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  val = WarpReduce(val, op);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < B::Warps()) ? shared[lid] : identity_element;
  if (wid == 0) {
    val = WarpReduce(val, op);
  }
  return val;
}
```
We could see it will:
1. invoke the `WarpReduce` for each thread to store the reduced value in `val`, so it's the **reduced value among a warp**.
2. it will call `WarpReduce` again for **first warp**, each **thread** in the warp will process the **reduced value among a warp** in step 1, then we will return the finally in **1st there in the block**.

***NOTE!!!!***: We could see there is just **2 round** of `WarpReduce` invocation, so we couldn't use the `BlockReduce` for a block with more than `32 * 32 = 1024` thread!!!