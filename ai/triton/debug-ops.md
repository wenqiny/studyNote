## Debug Ops
There are some ops we could use to debug triton code, when we're running the triton code.

### static print
Something looks like:
```
tl.static_print()
```
It will print the log when the triton code **was being compiling**.

### device print
something looks like:
```
tl.device_print("perfix ", somthing...)
```
Or
```
print("perfix ", somthing...)
```
It will print the log when the triton code **was being executing**.
Therefore we usually do some filter when we're running triton code, like:
```
if pid == 1:
    tl.device_print("perfix ", somthing...)
```