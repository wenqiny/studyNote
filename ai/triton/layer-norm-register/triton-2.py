import torch

import triton
import triton.language as tl

try:
    # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
    # should not be added to extras_require in setup.py.
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False

DEVICE = torch.device("cuda:0")


@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    
    # # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)

# %%
# Benchmark
# ---------
#
# We can now compare the performance of our kernel against that of PyTorch.
# Here we focus on inputs that have Less than 64KB per feature.
# Specifically, one can set :code:`'mode': 'backward'` to benchmark the backward pass.


class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        # print(f"N is {N}, BLOCK_SIZE is {BLOCK_SIZE}, x.element_size() is {x.element_size()}")
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # num_warps = 32
        # enqueue kernel
        compiled_kernel = _layer_norm_fwd_fused[(M, )](  #
            x_arg, y, weight, bias, mean, rstd,  #
            x_arg.stride(0), N, eps,  #
            # BLOCK_SIZE=BLOCK_SIZE)
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        # print(compiled_kernel.asm.keys())
        # print(compiled_kernel.asm["mlir"])
        # print(rstd)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y


layer_norm = LayerNorm.apply

for cnt in range(8192, 8193):
# for cnt in range(8193, 8191, -1):
    device = torch.device("cuda:0")
    dtype= torch.float32
    x_shape = (4096, cnt)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    eps=1e-5
    # x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    x = torch.ones(x_shape, dtype=dtype, device=device)

    #warm up
    need_warmup = True
    round = 10
    # need_warmup = False
    # round = 1
    if need_warmup:
        for _ in range(round):
            output = layer_norm(x, w_shape, weight, bias, eps)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    # Start time
    start_time.record()

    # Apply layernorm
    for _ in range(round):
        output = layer_norm(x, w_shape, weight, bias, eps)

    # End time
    end_time.record()

    torch.cuda.synchronize()

    # Calculate elapsed time
    elapsed_time_ms = start_time.elapsed_time(end_time)
    # print(f"CUDA Time: {elapsed_time_ms:.6f} ms")
    gbps = lambda ms: round * 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    print(f"n as {cnt} of softmax: {gbps(elapsed_time_ms):.6f} gb/s")