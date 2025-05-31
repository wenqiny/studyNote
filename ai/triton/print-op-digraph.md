## Print IR digraph for triton
We would like to visulize the IR digraph for analysis, here is how to do it.

### Patch
```
diff --git a/python/src/passes.cc b/python/src/passes.cc
index a1b03acf0..60a496eaa 100644
--- a/python/src/passes.cc
+++ b/python/src/passes.cc
@@ -32,6 +32,8 @@ void init_triton_passes_common(py::module &&m) {
   ADD_PASS_WRAPPER_0("add_cse", createCSEPass);
   ADD_PASS_WRAPPER_0("add_licm", createLoopInvariantCodeMotionPass);
   ADD_PASS_WRAPPER_0("print_ir", createPrintIRPass);
+  ADD_PASS_WRAPPER_0("add_print_op_graph", createPrintOpGraphPass);
+  ADD_PASS_WRAPPER_0("add_print_op_stats", createPrintOpStatsPass);
 }
 
 void init_triton_passes_ttir(py::module &&m) {
diff --git a/third_party/nvidia/backend/compiler.py b/third_party/nvidia/backend/compiler.py
index f67e03c04..353e9dc42 100644
--- a/third_party/nvidia/backend/compiler.py
+++ b/third_party/nvidia/backend/compiler.py
@@ -218,6 +218,7 @@ class CUDABackend(BaseBackend):
         passes.common.add_cse(pm)
         passes.common.add_symbol_dce(pm)
         passes.ttir.add_loop_unroll(pm)
+        passes.common.add_print_op_graph(pm)
         pm.run(mod)
         return mod
 
@@ -288,6 +289,7 @@ class CUDABackend(BaseBackend):
             nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
             nvidia.passes.ttnvgpuir.add_fence_insertion(pm)
         passes.common.add_canonicalizer(pm)
+        passes.common.add_print_op_graph(pm)
         pm.run(mod)
         metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)
         tensordesc_meta = mod.get_tensordesc_metadata()
@@ -317,6 +319,8 @@ class CUDABackend(BaseBackend):
         passes.common.add_canonicalizer(pm)
         passes.common.add_cse(pm)
         passes.common.add_symbol_dce(pm)
+        # passes.common.add_print_op_stats(pm)
+        passes.common.add_print_op_graph(pm)
         if not knobs.compilation.disable_line_info:
             passes.llvmir.add_di_scope(pm)
         pm.run(mod)
```

### How to get IR graph
1. Apply the patch above, and we should recompile trition, because there is some change to `cpp` part.
2. Run a triton kernel, then we could see the digraph was dump in console.
3. Then we could use a command like `dot -Tpng dump.dot -o dump.png` to generate the IR digraph.