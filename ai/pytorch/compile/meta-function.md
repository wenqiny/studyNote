## Meta function
### Functionality
meta function was used for simulate how a op works.
### How it works
For example, `torch._fft_r2c` will return a tensor with different **shape** and **stride** as its first input in real `cpu/cuda` kernel.
So, the `meta_fft_r2c` in `torch/_meta_registrations.py` will implement the logic for calculate the correct **shape** and **stride** wrt to the fake tensor input when compiling this model/function in `inductor`
### How it dispatch for each op
In `torch/_meta_registrations.py`, it will invoke `def activate_meta():` when we `import torch`, in this `activate_meta` method, it will registe the meta function for each op.
**Note!!! Not all the ops have its own meta function, some one may be call to `empty_like` or something else(please see details in the method).**