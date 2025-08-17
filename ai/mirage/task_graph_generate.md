# Task graph
Task graph in MPK was used in assign each task to each scheduler/worker and then run on each SM.

## Source file folder
There are two major file folders:
1. `src/kernel/`: it contains some files about the kernel level op, which means it's at the abstruction level of a layer in a neural network, so it contains some data like `DTensor`, which is the GMEM tensor.
2. `src/threadblock`: it contains some files about thread block level, which means it's at the abstruction level of a thread block in a GPU, so it take `DTensor` as its input and output, but it just process a partiton of it at once, the partition is called `STensor`, which is in SMEM.
   


## Data structure
* KNGraph
  It's the kernel graph
* TBGraph
  It contain some feilds like:
  ```
  dim3 grid_dim, block_dim, cluster_dim{4, 4, 1}; -> dims
  int forloop_range; -> for loop map
  int reduction_dimx;
  std::vector<mirage::threadblock::TBOperator *> operators; -> both input and output ops
  // memory allocator
  off_t smem_offset;
  std::vector<std::pair<off_t, size_t>> allocated_tensors;
  ```

How we build different OP in different graph?
For kernel OP:
```
self.kn_graph.customized([input, weight_norm, weight_linear, output], tb_graph)
```
it build a customized kernel op and push it into the kernel graph.

For thread graph graph:
```
self.kn_graph.register_task(tb_graph, "rmsnorm_linear") -> it build a thread block op and its tb graph.
```


## How we generate the task graph

It will use some method like: `self.kn_graph.customized([input, weight_norm, weight_linear, output], tb_graph)` to add each layer/op into the tbgraph (thread block graph), the `customized` was defined as:
```
int Graph::customized(std::vector<DTensor const *> _inputs,
                      DTensor **outputs,
                      mirage::threadblock::Graph const *bgraph) {
  std::vector<DTensor> inputs;
  for (auto const &t : _inputs) {
    inputs.push_back(t == nullptr ? DTensor::EMPTY_TENSOR : *t);
  }
  KNOperator *op = create_customized_op(inputs, *bgraph);
  assert(op != nullptr);
  operators.push_back(op);
  for (size_t i = 0; i < op->output_tensors.size(); i++) {
    outputs[i] = &op->output_tensors[i];
  }
  return op->output_tensors.size();
}
```

Then, it will use `register_mugraph` to generates the task graph for each thread block, which means it will divide each layer into different task, each task will be assigned to a dedicated SM/block, meanwhile it will handle the **address offset** for each input/output tensor in the block.

And it will do some dependence analysis for the tasks and events in `dfs_create_events_add_tasks` to build the graph and make it effiencient.