## GMEM load order
**Note**: should verify later, it's just my **guess**!

It seems the perf is related to the load-use (**producer-cosumer**) in the CUDA, if we tweak the order, we may got a better performance.

In other words, it means we could **hidden the GMEM load latency** by execution some other computation instructions!

### Case
1. baseline:
```
Load  A
Use   A
Load  B
Use   B
```
In this case, we may have to **wait** until we finished the `Use A`, then we could issue the load instruction of `Load  B`.

2. opt:
```
Load  A
Load  B
Use   A
Use   B
```
In this case, we may be able to issue the `Load A` and `Load B` simutaneously, then if A is ready, we could immediately execute `Use A`, after it we could execute `Use B` immediately.

The time pipeline may looks like:
1. baseline
```
==============================
|<--------Load A(30)-------->|
                              ====================
                              |<---Use  A(20)--->|
                                                  ==============================
                                                  |<--------Load A(30)-------->|
                                                                                ====================
                                                                                |<---Use  A(20)--->|
```
The total time cost should be **100**.

2. opt:
```
==============================
|<--------Load A(30)-------->|
          ==============================
|<-(10)->||<--------Load A(30)-------->|
                              ====================
                              |<---Use  A(20)--->|
                                                  
                                                  ====================
                                                  |<---Use  A(20)--->|
```
The total time cost should be **70**.

### Code:

We could take a kernel for merge attention output as an example, we just please the code snippet between `Another load` into two place, it brings **>10%** perf gain:

1. baseline
```cpp
template <typename scalar_t, int HEAD_SIZE,
          bool SHOULD_OUTPUT_LSE>
__global__ void merge_attn_states_kernel(
    scalar_t* __restrict__ output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    float* __restrict__ output_lse,  // [NUM_HEADS, NUM_TOKENS]
    const scalar_t* __restrict__ prefix_output, // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    const float* __restrict__ prefix_lse,  // [NUM_HEADS, NUM_TOKENS]
    const scalar_t* __restrict__ suffix_output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    const float* __restrict__ suffix_lse  // [NUM_HEADS, NUM_TOKENS]
    ) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int num_tokens = gridDim.x;
    const int num_heads = gridDim.y;

    // Thread count of a block is HEAD_SIZE
    const int tid = threadIdx.x;

    // Load p_lse and s_lse, the both lse is same for each thread in a block.
    const int lse_offset = head_idx * num_tokens + token_idx;
    float p_lse = prefix_lse[lse_offset];
    float s_lse = suffix_lse[lse_offset];
    
    // Get the maximum for safe exp.
    p_lse = p_lse == FLT_MAX ? -FLT_MAX : p_lse;
    s_lse = s_lse == FLT_MAX ? -FLT_MAX : s_lse;
    float max_lse = fmaxf(p_lse, s_lse);
    p_lse -= max_lse;
    s_lse -= max_lse;

    // Get the output sum of exp.
    float p_exp_lse = __expf(p_lse);
    float s_exp_lse = __expf(s_lse);
    float out_se = p_exp_lse + s_exp_lse;

    if(SHOULD_OUTPUT_LSE) {
      // Write lse back if necessary.
      float out_lse = __logf(out_se) + max_lse;
      output_lse[lse_offset] = out_lse;
    }
    
    // Compute scale firstly in case of overflow.
    float p_scale = p_exp_lse / out_se;
    float s_scale = s_exp_lse / out_se;

==================================== Another load ====================================
    const int output_offset = token_idx * HEAD_SIZE * num_heads + head_idx * HEAD_SIZE;
    const scalar_t* p_out_ptr = prefix_output + output_offset;
    const scalar_t* s_out_ptr = suffix_output + output_offset;

    // Each thread write back an scalar in a head.
    scalar_t* output_ptr = output + output_offset;

    float p_out_f = to_float(p_out_ptr[tid]);
    float s_out_f = to_float(s_out_ptr[tid]);
==================================== Another load ====================================

    float res = p_out_f * p_scale + s_out_f * s_scale;
    from_float(output_ptr[tid], res);
}
```

2. opt
```cpp
template <typename scalar_t, int HEAD_SIZE,
          bool SHOULD_OUTPUT_LSE>
__global__ void merge_attn_states_kernel(
    scalar_t* __restrict__ output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    float* __restrict__ output_lse,  // [NUM_HEADS, NUM_TOKENS]
    const scalar_t* __restrict__ prefix_output, // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    const float* __restrict__ prefix_lse,  // [NUM_HEADS, NUM_TOKENS]
    const scalar_t* __restrict__ suffix_output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    const float* __restrict__ suffix_lse  // [NUM_HEADS, NUM_TOKENS]
    ) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int num_tokens = gridDim.x;
    const int num_heads = gridDim.y;

    // Thread count of a block is HEAD_SIZE
    const int tid = threadIdx.x;

    // Load p_lse and s_lse, the both lse is same for each thread in a block.
    const int lse_offset = head_idx * num_tokens + token_idx;
    float p_lse = prefix_lse[lse_offset];
    float s_lse = suffix_lse[lse_offset];

==================================== Another load ====================================
    const int output_offset = token_idx * HEAD_SIZE * num_heads + head_idx * HEAD_SIZE;
    const scalar_t* p_out_ptr = prefix_output + output_offset;
    const scalar_t* s_out_ptr = suffix_output + output_offset;

    // Each thread write back an scalar in a head.
    scalar_t* output_ptr = output + output_offset;

    float p_out_f = to_float(p_out_ptr[tid]);
    float s_out_f = to_float(s_out_ptr[tid]);
==================================== Another load ====================================
    
    // Get the maximum for safe exp.
    p_lse = p_lse == FLT_MAX ? -FLT_MAX : p_lse;
    s_lse = s_lse == FLT_MAX ? -FLT_MAX : s_lse;
    float max_lse = fmaxf(p_lse, s_lse);
    p_lse -= max_lse;
    s_lse -= max_lse;

    // Get the output sum of exp.
    float p_exp_lse = __expf(p_lse);
    float s_exp_lse = __expf(s_lse);
    float out_se = p_exp_lse + s_exp_lse;

    if(SHOULD_OUTPUT_LSE) {
      // Write lse back if necessary.
      float out_lse = __logf(out_se) + max_lse;
      output_lse[lse_offset] = out_lse;
    }
    
    // Compute scale firstly in case of overflow.
    float p_scale = p_exp_lse / out_se;
    float s_scale = s_exp_lse / out_se;

    float res = p_out_f * p_scale + s_out_f * s_scale;
    from_float(output_ptr[tid], res);
}
```
