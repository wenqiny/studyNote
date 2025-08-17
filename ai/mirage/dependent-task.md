# Dependent task
In the mirage, there are serveral tasks, but between tasks in two adjancent layers, there is some dependence to keep the order for these tasks to run successfully, but how to control them to make it run effienciently is an interesting question.

Let's image a case for **an linear layer (for query) followed by a attention layer**, some configurations for the model in below table:

| configurations | number |
| --------- | ------------- |
| linear weights | (2048, 4096) |
| linear GPU blocks (linear tasks) | 32 |
| columns in each GPU blocks | 2048/32=128 |
| head size | 128 |
| query head | 32 |
| kv head (attn tasks) | 8 |
| query tensor size | 32*128=4096 |

We assume each block of linear layer will process **32** columns, and each block will process a kv head in attention layer, which means it will generate **4 (32/8=4) query head (its size is 128*4=512)** as output, and .

So there are **8** attention tasks, and **32** linear task.

## Brute force solution
To simplify the design, we could ask all the attention layer tasks hold on until all linear task was finished.

In this implementation, we could ask all **8** attention tasks to be launched after all **32** linear tasks was finished, but it's **unefficient**!

## Dependence check solution
We could see in the **8** attention tasks and **32** linear tasks, the **first** attention task needs only **4*128=512** elements in query, all its dependent scalar came from the previous **4** task for linear layer, so it looks like:

$$
\begin{aligned}
l_0, l_1, l_2, l_3 &\to a_0 \\
l_4, l_5, l_6, l_7 &\to a_1 \\
&\cdots \\
l_{28}, l_{29}, l_{30}, l_{31} &\to a_7 \\
\end{aligned}
$$

That's means we could do a more fine granularity control over all the tasks based on the dependence analysis, that's help we make these task run fastly.

## Source code
Some code snippet in MPK in `src/kernel/runtime.cc` for the above looks like:
```
// Step 2.1: analyze dependencies between thread blocks of the two ops
std::vector<int> producer_partition(mirage::config::MAX_TENSOR_DIMS, 1);
std::vector<int> consumer_partition(mirage::config::MAX_TENSOR_DIMS, 1);
int num_shared_tensors = 0;
int3 input_map, output_map;
for (auto const &input : input_ops) {
for (auto const &output : pre_output_ops) {
    if (input->dtensor.guid == output->dtensor.guid) {
    input_map = input->input_map;
    output_map = output->input_map;
    num_shared_tensors++;
    }
}
}
// assert that their is at least a single tensor shared between ops
assert(num_shared_tensors >= 1);
for (int d = 0; d < mirage::config::MAX_TENSOR_DIMS; d++) {
if (d == input_map.x) {
    consumer_partition[d] = bgraph.grid_dim.x;
}
if (d == input_map.y) {
    consumer_partition[d] = bgraph.grid_dim.y;
}
if (d == input_map.z) {
    consumer_partition[d] = bgraph.grid_dim.z;
}
if (d == output_map.x) {
    producer_partition[d] = pre_op->bgraph.grid_dim.x;
}
if (d == output_map.y) {
    producer_partition[d] = pre_op->bgraph.grid_dim.y;
}
if (d == output_map.z) {
    producer_partition[d] = pre_op->bgraph.grid_dim.z;
}
}
// Step 2.2: create events and add tasks
// number of events is the product of gcd of producer/consumer
std::vector<int> event_dims(mirage::config::MAX_TENSOR_DIMS, 1);
for (int d = 0; d < mirage::config::MAX_TENSOR_DIMS; d++) {
event_dims[d] = std::gcd(producer_partition[d], consumer_partition[d]);
}
dfs_create_events_add_tasks(0,                       /*depth*/
                            my_gpu_id,               /*my_gpu_id*/
                            event_dims,              /*event_dims*/
                            input_map,               /*input_map*/
                            output_map,              /*output_map*/
                            bgraph.grid_dim,         /*consumer_grid_dim*/
                            pre_op->bgraph.grid_dim, /*producer_grid_dim*/
                            dim3(0, 0, 0),           /*consumer_lo_bid*/
                            bgraph.grid_dim,         /*consumer_hi_bid*/
                            dim3(0, 0, 0),           /*producer_lo_bid*/
                            pre_op->bgraph.grid_dim, /*producer_hi_bid*/
                            all_events,
                            all_tasks,
                            tasks,        /*cur_op_tasks*/
                            pre_task_map, /*pre_task_map*/
                            cur_task_map /*cur_task_map)*/);
```