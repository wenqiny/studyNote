## Lecture 01
### PyTorch profile:
1. torch.profiler.profile 
```
with torch.profiler.profile() as prof:
  ... pytorch code here ...
print(prof)
```
to profile pytorch code

2. Same as above one, but extract it to chrome tracing file.
```
print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")
```

### Add CPP/CUDA extension in pytorch
```
from torch.utils.cpp_extension import load_inline
my_module = load_inline(...)
```

It will write .cc/.cu file in a tmp folder and try to compile and integreta into pytorch.

### Visualize torch.compile
```
import torch

torch._logging.set_logs(output_code=True)

@torch.compile()
def fn(x, y):
    z = x + y
    return z + 2

inputs = (torch.ones(2, 2, device="cuda"), torch.zeros(2, 2, device="cuda"))
fn(*inputs)
```

It seems torch will use triton to compile this `fn` into a cuda kernel.

### NCU
```
ncu --set full -o ncu.log python pytorch_square.py 
```
Use `ncu` to profile cuad, but it runs failed in my cloud GPU, maybe should setup some other environment, like privilege.