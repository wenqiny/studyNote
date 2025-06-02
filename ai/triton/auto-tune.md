## Auto tune
Auto tune was used for getting a better layout of the triton kernel, by auto tunning some hyper parameter like `BLOCK_SIZE_M` and `BLOCK_SIZE_N` when changing the **key** parameter.

### Case
```
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
         # Add more configs or let autotuner explore if needed
    ],
    key=['M', 'N', 'K_dim'],
)
@triton.jit
def _infonce_matmul_fwd_kernel(
    Q, K, Z,  # Pointers to matrices
    M, N, K_dim,  # Matrix dimensions
    stride_qm, stride_qk,  # Strides for Q
    stride_kn, stride_kk,  # Strides for K (transposed view conceptually)
    stride_zm, stride_zn,  # Strides for Z
    temperature: tl.constexpr,  # Temperature scaling factor
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  # Tile sizes
    GROUP_SIZE_M: tl.constexpr, # Used for grouping blocks for potentially better L2 cache performance
):
```

For the above kernel, when we encounter any different number/tensor of **'M', 'N' or 'K_dim'**, we will try all the configuration we have pre-defined to benchmark which one could bring a better performance.

### Code
Please see triton code at [here](https://github.com/triton-lang/triton/blob/cda4229558c5dca7f7c4734bedd3e596ebcae0b8/python/triton/runtime/autotuner.py#L211-L257) for how to benchmark these configurations.
The most important code snippet looks like:
```
key = [_args[key] for key in self.keys if key in _args]
for _, arg in _args.items():
    if hasattr(arg, "dtype"):
        key.append(str(arg.dtype))
key = tuple(key)
if key not in self.cache:
    ## Benchmark each configuration and store the best one to cache at here
config = self.cache[key]
```