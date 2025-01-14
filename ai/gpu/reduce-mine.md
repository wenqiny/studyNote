## Try for reduce by my self

### Warp reduce
here is a reduce for a warp:
```cpp
__global__ void printtid(){
    int tid =  threadIdx.x;
    int value = threadIdx.x;
    for(int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        int shfl_value = __shfl_down_sync(0xffffffff, value, offset);
        if(threadIdx.x == 31){
            printf("for thread 31, the shfl_value is %d\n", shfl_value);
        }
        if(threadIdx.x == 0){
            printf("for thread 0, the shfl_value is %d\n", shfl_value);
        }
        value = min(value, shfl_value);
    }
    printf("for thread %d, its value is %d\n", tid, value);
}
```

its output is on my local:
```
for thread 31, the shfl_value is 31
for thread 0, the shfl_value is 16
for thread 31, the shfl_value is 31
for thread 0, the shfl_value is 8
for thread 31, the shfl_value is 31
for thread 0, the shfl_value is 4
for thread 31, the shfl_value is 31
for thread 0, the shfl_value is 2
for thread 31, the shfl_value is 31
for thread 0, the shfl_value is 1
for thread 0, its value is 0
for thread 1, its value is 1
for thread 2, its value is 2
for thread 3, its value is 3
for thread 4, its value is 4
for thread 5, its value is 5
for thread 6, its value is 6
for thread 7, its value is 7
for thread 8, its value is 8
for thread 9, its value is 9
for thread 10, its value is 10
for thread 11, its value is 11
for thread 12, its value is 12
for thread 13, its value is 13
for thread 14, its value is 14
for thread 15, its value is 15
for thread 16, its value is 16
for thread 17, its value is 17
for thread 18, its value is 18
for thread 19, its value is 19
for thread 20, its value is 20
for thread 21, its value is 21
for thread 22, its value is 22
for thread 23, its value is 23
for thread 24, its value is 24
for thread 25, its value is 25
for thread 26, its value is 26
for thread 27, its value is 27
for thread 28, its value is 28
for thread 29, its value is 29
for thread 30, its value is 30
for thread 31, its value is 31
```

We could see for `thread 31`, it always get `31` from `__shfl_down_sync` method, the `31` is a **UB**, we shouldn't use this data to do any computation, **but we do it!!!! In any thread, we will iterate 5 times**.

So in 5 iteration:
1. range(16, 32) threads were **UB**, because they take **UB** number as input for `max` or `min`.
2. range(8, 16) threads were **UB**, because they take **UB** number from range(16, 24) threads as input.
3. range(4, 8) threads were **UB**, ditto.
4. range(2, 4) threads were **UB**, ditto.
5. range(1, 2) threads were **UB**, ditto.

Therefore, we take all the above range together, the number in range(1, 32) is **UB**, we could **just truse only the number in thread 0!!!**