# Compilation pipeline
The markdown contains some notes about how a triton code convert to SASS code.

## Python -> AST

## AST -> TTIR
in this call chain: `def ast_to_ttir(` -> `generator.visit(fn.parse())` -> `CodeGenerator::visit` -> `def visit_Module(`, then it will call serveral function like `visit_FunctionDef`, `visit_Return` and `visit_List` for convert AST to TTIR.

The above `visit_x` function will call some other functions in: `python/triton/compiler/code_generator.py`, `python/triton/language/core.py`, `python/triton/language/standard.py` and `python/triton/language/semantic.py`.

The other functions in `python/triton/language/semantic.py`, like `permute` will call something like: `self.builder.create_trans` to create the `TTIR`.

How python call to cpp to create `TTIR`:
We could see this file `python/src/ir.cc`, it contains some method like: `.def("create_for_op",` to create `TTIR` op.