## How to debug triton cpp
Sometimes we would like to debug triton cpp code, like for each optimizer cpp pass.

### Build
We should build triton with:
```
TRITON_BUILD_WITH_CCACHE=true \
  LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
  LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
  LLVM_SYSPATH=$LLVM_BUILD_DIR \
  MAX_JOBS=10 \
  DEBUG=1 \
  CMAKE_BUILD_TYPE=Debug \
  pip install -e . --no-build-isolation
```

### Run
There is two way to debug: `gdb` and `vscode`.

#### GDB
We should run gdb with: `gdb --args env TRITON_ALWAYS_COMPILE=1 python ../../triton-code/issue-6647.py`, we should make sure each time we force to compile with `TRITON_ALWAYS_COMPILE=1`.

#### VS code
The template for `launch.json` on `vscode` looks like:
```
{
    "version": "0.2.0",
    "configurations": [
      {
        "name": "debug triton",
        "type": "cppdbg",
        "request": "launch",
        "program": "/home/wenqin/miniconda3/envs/tritonbuild-debug/bin/python", // path to conda env python
        "args": [
            "/home/wenqin/study/triton-code/issue-6647.py"
        ],
        "stopAtEntry": false,
        "cwd": "${workspaceFolder}",
        "environment": [
            {
                "name": "TRITON_ALWAYS_COMPILE", // force to compile even cache hit
                "value": "1"
            },
        ],
        "externalConsole": false,
        "MIMode": "gdb",
        "miDebuggerPath": "/usr/bin/gdb",
        // "setupCommands": [
        //   {
        //     "description": "Enable pretty-printing for gdb",
        //     "text": "-enable-pretty-printing",
        //     "ignoreFailures": true
        //   }
        // ]
      }
    ]
  }
```