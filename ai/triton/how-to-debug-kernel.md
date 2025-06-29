## How to debug triton kernel
There are serverial ways to debug triton kernel.

### cuda-gdb
```
(tritonbuild) wenqin@wenqin-System-Product-Name:~/study/triton-code$ cp /home/wenqin/miniconda3/envs/tritonbuild/bin/python3.10 /home/wenqin/miniconda3/envs/tritonbuild/bin/python3.10.bkp
(tritonbuild) wenqin@wenqin-System-Product-Name:~/study/triton-code$ strip --strip-debug /home/wenqin/miniconda3/envs/tritonbuild/bin/python3.10
(tritonbuild) wenqin@wenqin-System-Product-Name:~/study/triton-code$ cuda-gdb --symbols=none --args python issue-6647.py
```

We should stripe the the `python` binary firstly, otherwise it will load lots of symbol for python, which make us hard to debug **kernel code**.

Then we could set break point for **kernel name** (it's the name which was defined in `.py` file for the triton kernel) or **line of code file** , like:

```
(cuda-gdb) b repro_kernel
Function "repro_kernel" not defined.
Make breakpoint pending on future shared library load? (y or [n]) y
```

Or

```
(cuda-gdb) b /home/wenqin/study/triton-code/issue-6647.py:27
No symbol table is loaded.  Use the "file" command.
Make breakpoint pending on future shared library load? (y or [n]) y
Breakpoint 1 (/home/wenqin/study/triton-code/issue-6647.py:27) pending.
```

### Note
Didn't set breakpoint on some SASS inst like:
```
SHFL.BFLY PT, R14, R13, 0x10, 0x1f;
BAR.SYNC.DEFER_BLOCKING 0x0;
```

It seems they will stuch the gdb, maybe related to sync???