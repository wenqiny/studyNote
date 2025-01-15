## How to chose grid and block size
I understand this is a `tremendous` (I don't know whether this word is proper here, but trump seems really like this word, I just show respect to him, hhhhhhh) question we could discuss with, I will list some points what I have thought make sense.

### A sensitive number for block size: ***32*** 
If we just take **a warp (32 threads)** to process a vector of elements in a matrix, we could avoid using share mem to reduce between block, but use `__shfl_sync_xx` to do the warp reduce! It helps on performance.

### A big number for block size: ***1024***
If we decide to use **not just one warp** to process a vector (may suffer from the register pressure), we could use a block size as much as more, because we have already have to use some **share memory** to reduce block, but in such a case we could just use **1024** threads at most, because if we allocate **1025** threads in a block, we may have to do 3 rounds of reduce, please see details in [this](reduce-pytorch.md).