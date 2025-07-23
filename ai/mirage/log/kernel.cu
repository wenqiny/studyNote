#include "persistent_kernel.cuh"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
using json = nlohmann::json;
using namespace mirage::runtime;
size_t get_event_id(int my_gpu_id, size_t event_pos, bool nvshmem_event) {
  size_t event_id = ((static_cast<size_t>(my_gpu_id) << 32) | event_pos);
  if (nvshmem_event) {
    event_id = event_id | EVENT_NVSHMEM_TAG;
  }
  return event_id;
}

void construct_task_graph(int num_gpus,
                          int my_gpu_id,
                          std::vector<TaskDesc> &all_tasks,
                          std::vector<EventDesc> &all_events,
                          std::vector<TaskId> &first_tasks,
                          std::map<std::string, void*> const &all_tensors) {
  std::filesystem::path file_path(__FILE__);
  std::ifstream json_file(file_path.parent_path().string()+"/task_graph.json");
  nlohmann::json json_task_graph;
  json_file >> json_task_graph;
  for (json const &task : json_task_graph["all_tasks"]) {
    TaskDesc task_desc(static_cast<TaskType>(task.at("task_type")),
                task.at("variant_id"));
    if (task.at("trigger_event").is_number_integer()) {
      task_desc.trigger_event = task.at("trigger_event").get<unsigned long long int>();
    }
    else {
      assert(false);
    }
    if (task.at("dependent_event").is_number_integer()) {
      task_desc.dependent_event = task.at("dependent_event").get<unsigned long long int>();
    }
    else {
      assert(false);
    }
    task_desc.num_inputs = 0;
    for (json const &tensor : task["inputs"]) {
      TensorDesc input;
      std::string name = tensor.at("base_ptr").get<std::string>();
      assert(all_tensors.find(name) != all_tensors.end());
      off_t offset = tensor.at("offset").get<off_t>();
      input.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;
      assert(tensor.at("dims").size() == tensor.at("strides").size());
      input.num_dims = tensor.at("dims").size();
      input.data_type = tensor.at("data_type").get<int>();
      for (int i = 0; i < input.num_dims; i++) {
        input.dim[i] = tensor["dims"][i].get<int>();
        input.stride[i] = tensor["strides"][i].get<int>();
      }
      task_desc.inputs[task_desc.num_inputs++] = input;
    }
    task_desc.num_outputs = 0;
    for (json const &tensor : task["outputs"]) {
      TensorDesc output;
      std::string name = tensor.at("base_ptr").get<std::string>();
      assert(all_tensors.find(name) != all_tensors.end());
      off_t offset = tensor.at("offset").get<off_t>();
      output.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;
      assert(tensor.at("dims").size() == tensor.at("strides").size());
      output.num_dims = tensor.at("dims").size();
      output.data_type = tensor.at("data_type").get<int>();
      for (int i = 0; i < output.num_dims; i++) {
        output.dim[i] = tensor["dims"][i];
        output.stride[i] = tensor["strides"][i];
      }
      task_desc.outputs[task_desc.num_outputs++] = output;
    }
    all_tasks.push_back(task_desc);
  }
  for (json const &e : json_task_graph["all_events"]) {
    EventType event_type = static_cast<EventType>(e.at("event_type").get<int>());
    int num_triggers = e.at("num_triggers").get<int>();
    int first_task_id = e.at("first_task_id").get<int>();
    int last_task_id = e.at("last_task_id").get<int>();
    all_events.push_back(EventDesc(event_type, num_triggers, first_task_id, last_task_id));
  }
  for (json const &t : json_task_graph["first_tasks"]) {
    first_tasks.push_back(t.get<int>());
  }
}

static void _init_persistent_kernel(std::vector<TaskDesc> &all_tasks,
                                    std::vector<EventDesc> &all_events,
                                  std::vector<TaskId> &first_tasks,
                                  int num_gpus,
                                  int my_gpu_id) {
  assert(num_gpus = 1);
  std::map<std::string, void*> all_tensors;
  char *input_token = (char*)(0x7f0626aa8400);
  all_tensors["input_token"] = input_token;
  char *cos_position_embedding = (char*)(0x7f0613400000);
  all_tensors["cos_position_embedding"] = cos_position_embedding;
  char *sin_position_embedding = (char*)(0x7f0616820000);
  all_tensors["sin_position_embedding"] = sin_position_embedding;
  void *embed_out;
  cudaMalloc(&embed_out, 2048);
  all_tensors["embed_out"] = embed_out;
  void *attn_in;
  cudaMalloc(&attn_in, 8192);
  all_tensors["attn_in"] = attn_in;
  void *attn_out;
  cudaMalloc(&attn_out, 4096);
  all_tensors["attn_out"] = attn_out;
  void *attn_proj_out;
  cudaMalloc(&attn_proj_out, 2048);
  all_tensors["attn_proj_out"] = attn_proj_out;
  void *all_reduce_buf;
  cudaMalloc(&all_reduce_buf, 2048);
  all_tensors["all_reduce_buf"] = all_reduce_buf;
  void *attn_allreduce_out;
  cudaMalloc(&attn_allreduce_out, 2048);
  all_tensors["attn_allreduce_out"] = attn_allreduce_out;
  void *mlp_mid;
  cudaMalloc(&mlp_mid, 12288);
  all_tensors["mlp_mid"] = mlp_mid;
  void *mlp_out;
  cudaMalloc(&mlp_out, 2048);
  all_tensors["mlp_out"] = mlp_out;
  void *mlp_final;
  cudaMalloc(&mlp_final, 2048);
  all_tensors["mlp_final"] = mlp_final;
  void *argmax_in;
  cudaMalloc(&argmax_in, 307200);
  all_tensors["argmax_in"] = argmax_in;
  void *argmax_part_value;
  cudaMalloc(&argmax_part_value, 192);
  all_tensors["argmax_part_value"] = argmax_part_value;
  void *argmax_part_index;
  cudaMalloc(&argmax_part_index, 768);
  all_tensors["argmax_part_index"] = argmax_part_index;
  void *argmax_out;
  cudaMalloc(&argmax_out, 8);
  all_tensors["argmax_out"] = argmax_out;
  char *embed_tokens = (char*)(0x7f05be000000);
  all_tensors["embed_tokens"] = embed_tokens;
  char *layer_0_input_layernorm = (char*)(0x7f0626a03e00);
  all_tensors["layer_0_input_layernorm"] = layer_0_input_layernorm;
  char *layer_0_q_proj = (char*)(0x7f05d20c0000);
  all_tensors["layer_0_q_proj"] = layer_0_q_proj;
  char *layer_0_k_proj = (char*)(0x7f05d1ac0000);
  all_tensors["layer_0_k_proj"] = layer_0_k_proj;
  char *layer_0_v_proj = (char*)(0x7f05d24c0000);
  all_tensors["layer_0_v_proj"] = layer_0_v_proj;
  char *layer_0_q_norm = (char*)(0x7f0626a04e00);
  all_tensors["layer_0_q_norm"] = layer_0_q_norm;
  char *layer_0_k_norm = (char*)(0x7f0626a00200);
  all_tensors["layer_0_k_norm"] = layer_0_k_norm;
  char *layer_0_k_cache = (char*)(0x7f0636000000);
  all_tensors["layer_0_k_cache"] = layer_0_k_cache;
  char *layer_0_v_cache = (char*)(0x7f0628000000);
  all_tensors["layer_0_v_cache"] = layer_0_v_cache;
  char *layer_0_o_proj = (char*)(0x7f05d1cc0000);
  all_tensors["layer_0_o_proj"] = layer_0_o_proj;
  char *layer_0_post_attn_layernorm = (char*)(0x7f0626a04600);
  all_tensors["layer_0_post_attn_layernorm"] = layer_0_post_attn_layernorm;
  char *layer_0_gate_proj = (char*)(0x7f05d0ec0000);
  all_tensors["layer_0_gate_proj"] = layer_0_gate_proj;
  char *layer_0_up_proj = (char*)(0x7f05d14c0000);
  all_tensors["layer_0_up_proj"] = layer_0_up_proj;
  char *layer_0_down_proj = (char*)(0x7f05d08c0000);
  all_tensors["layer_0_down_proj"] = layer_0_down_proj;
  char *layer_1_input_layernorm = (char*)(0x7f0626a05000);
  all_tensors["layer_1_input_layernorm"] = layer_1_input_layernorm;
  char *layer_1_q_proj = (char*)(0x7f05d3ec0000);
  all_tensors["layer_1_q_proj"] = layer_1_q_proj;
  char *layer_1_k_proj = (char*)(0x7f05d38c0000);
  all_tensors["layer_1_k_proj"] = layer_1_k_proj;
  char *layer_1_v_proj = (char*)(0x7f05d42c0000);
  all_tensors["layer_1_v_proj"] = layer_1_v_proj;
  char *layer_1_q_norm = (char*)(0x7f0626a06200);
  all_tensors["layer_1_q_norm"] = layer_1_q_norm;
  char *layer_1_k_norm = (char*)(0x7f0626a06000);
  all_tensors["layer_1_k_norm"] = layer_1_k_norm;
  char *layer_1_k_cache = (char*)(0x7f0636800000);
  all_tensors["layer_1_k_cache"] = layer_1_k_cache;
  char *layer_1_v_cache = (char*)(0x7f0628800000);
  all_tensors["layer_1_v_cache"] = layer_1_v_cache;
  char *layer_1_o_proj = (char*)(0x7f05d3ac0000);
  all_tensors["layer_1_o_proj"] = layer_1_o_proj;
  char *layer_1_post_attn_layernorm = (char*)(0x7f0626a05800);
  all_tensors["layer_1_post_attn_layernorm"] = layer_1_post_attn_layernorm;
  char *layer_1_gate_proj = (char*)(0x7f05d2cc0000);
  all_tensors["layer_1_gate_proj"] = layer_1_gate_proj;
  char *layer_1_up_proj = (char*)(0x7f05d32c0000);
  all_tensors["layer_1_up_proj"] = layer_1_up_proj;
  char *layer_1_down_proj = (char*)(0x7f05d26c0000);
  all_tensors["layer_1_down_proj"] = layer_1_down_proj;
  char *layer_2_input_layernorm = (char*)(0x7f0626a12c00);
  all_tensors["layer_2_input_layernorm"] = layer_2_input_layernorm;
  char *layer_2_q_proj = (char*)(0x7f05e88c0000);
  all_tensors["layer_2_q_proj"] = layer_2_q_proj;
  char *layer_2_k_proj = (char*)(0x7f05e82c0000);
  all_tensors["layer_2_k_proj"] = layer_2_k_proj;
  char *layer_2_v_proj = (char*)(0x7f05e8cc0000);
  all_tensors["layer_2_v_proj"] = layer_2_v_proj;
  char *layer_2_q_norm = (char*)(0x7f0626a13e00);
  all_tensors["layer_2_q_norm"] = layer_2_q_norm;
  char *layer_2_k_norm = (char*)(0x7f0626a13c00);
  all_tensors["layer_2_k_norm"] = layer_2_k_norm;
  char *layer_2_k_cache = (char*)(0x7f0637000000);
  all_tensors["layer_2_k_cache"] = layer_2_k_cache;
  char *layer_2_v_cache = (char*)(0x7f0629000000);
  all_tensors["layer_2_v_cache"] = layer_2_v_cache;
  char *layer_2_o_proj = (char*)(0x7f05e84c0000);
  all_tensors["layer_2_o_proj"] = layer_2_o_proj;
  char *layer_2_post_attn_layernorm = (char*)(0x7f0626a13400);
  all_tensors["layer_2_post_attn_layernorm"] = layer_2_post_attn_layernorm;
  char *layer_2_gate_proj = (char*)(0x7f05e76c0000);
  all_tensors["layer_2_gate_proj"] = layer_2_gate_proj;
  char *layer_2_up_proj = (char*)(0x7f05e7cc0000);
  all_tensors["layer_2_up_proj"] = layer_2_up_proj;
  char *layer_2_down_proj = (char*)(0x7f05e70c0000);
  all_tensors["layer_2_down_proj"] = layer_2_down_proj;
  char *layer_3_input_layernorm = (char*)(0x7f0626a1e000);
  all_tensors["layer_3_input_layernorm"] = layer_3_input_layernorm;
  char *layer_3_q_proj = (char*)(0x7f05f96c0000);
  all_tensors["layer_3_q_proj"] = layer_3_q_proj;
  char *layer_3_k_proj = (char*)(0x7f05f90c0000);
  all_tensors["layer_3_k_proj"] = layer_3_k_proj;
  char *layer_3_v_proj = (char*)(0x7f05f9ac0000);
  all_tensors["layer_3_v_proj"] = layer_3_v_proj;
  char *layer_3_q_norm = (char*)(0x7f0626a1f200);
  all_tensors["layer_3_q_norm"] = layer_3_q_norm;
  char *layer_3_k_norm = (char*)(0x7f0626a1f000);
  all_tensors["layer_3_k_norm"] = layer_3_k_norm;
  char *layer_3_k_cache = (char*)(0x7f0637800000);
  all_tensors["layer_3_k_cache"] = layer_3_k_cache;
  char *layer_3_v_cache = (char*)(0x7f0629800000);
  all_tensors["layer_3_v_cache"] = layer_3_v_cache;
  char *layer_3_o_proj = (char*)(0x7f05f92c0000);
  all_tensors["layer_3_o_proj"] = layer_3_o_proj;
  char *layer_3_post_attn_layernorm = (char*)(0x7f0626a1e800);
  all_tensors["layer_3_post_attn_layernorm"] = layer_3_post_attn_layernorm;
  char *layer_3_gate_proj = (char*)(0x7f05f84c0000);
  all_tensors["layer_3_gate_proj"] = layer_3_gate_proj;
  char *layer_3_up_proj = (char*)(0x7f05f8ac0000);
  all_tensors["layer_3_up_proj"] = layer_3_up_proj;
  char *layer_3_down_proj = (char*)(0x7f05f7ec0000);
  all_tensors["layer_3_down_proj"] = layer_3_down_proj;
  char *layer_4_input_layernorm = (char*)(0x7f0626a1f400);
  all_tensors["layer_4_input_layernorm"] = layer_4_input_layernorm;
  char *layer_4_q_proj = (char*)(0x7f05fb4c0000);
  all_tensors["layer_4_q_proj"] = layer_4_q_proj;
  char *layer_4_k_proj = (char*)(0x7f05faec0000);
  all_tensors["layer_4_k_proj"] = layer_4_k_proj;
  char *layer_4_v_proj = (char*)(0x7f05fb8c0000);
  all_tensors["layer_4_v_proj"] = layer_4_v_proj;
  char *layer_4_q_norm = (char*)(0x7f0626a20600);
  all_tensors["layer_4_q_norm"] = layer_4_q_norm;
  char *layer_4_k_norm = (char*)(0x7f0626a20400);
  all_tensors["layer_4_k_norm"] = layer_4_k_norm;
  char *layer_4_k_cache = (char*)(0x7f0638000000);
  all_tensors["layer_4_k_cache"] = layer_4_k_cache;
  char *layer_4_v_cache = (char*)(0x7f062a000000);
  all_tensors["layer_4_v_cache"] = layer_4_v_cache;
  char *layer_4_o_proj = (char*)(0x7f05fb0c0000);
  all_tensors["layer_4_o_proj"] = layer_4_o_proj;
  char *layer_4_post_attn_layernorm = (char*)(0x7f0626a1fc00);
  all_tensors["layer_4_post_attn_layernorm"] = layer_4_post_attn_layernorm;
  char *layer_4_gate_proj = (char*)(0x7f05fa2c0000);
  all_tensors["layer_4_gate_proj"] = layer_4_gate_proj;
  char *layer_4_up_proj = (char*)(0x7f05fa8c0000);
  all_tensors["layer_4_up_proj"] = layer_4_up_proj;
  char *layer_4_down_proj = (char*)(0x7f05f9cc0000);
  all_tensors["layer_4_down_proj"] = layer_4_down_proj;
  char *layer_5_input_layernorm = (char*)(0x7f0626a20800);
  all_tensors["layer_5_input_layernorm"] = layer_5_input_layernorm;
  char *layer_5_q_proj = (char*)(0x7f05fd2c0000);
  all_tensors["layer_5_q_proj"] = layer_5_q_proj;
  char *layer_5_k_proj = (char*)(0x7f05fccc0000);
  all_tensors["layer_5_k_proj"] = layer_5_k_proj;
  char *layer_5_v_proj = (char*)(0x7f05fd6c0000);
  all_tensors["layer_5_v_proj"] = layer_5_v_proj;
  char *layer_5_q_norm = (char*)(0x7f0626a21a00);
  all_tensors["layer_5_q_norm"] = layer_5_q_norm;
  char *layer_5_k_norm = (char*)(0x7f0626a21800);
  all_tensors["layer_5_k_norm"] = layer_5_k_norm;
  char *layer_5_k_cache = (char*)(0x7f0638800000);
  all_tensors["layer_5_k_cache"] = layer_5_k_cache;
  char *layer_5_v_cache = (char*)(0x7f062a800000);
  all_tensors["layer_5_v_cache"] = layer_5_v_cache;
  char *layer_5_o_proj = (char*)(0x7f05fcec0000);
  all_tensors["layer_5_o_proj"] = layer_5_o_proj;
  char *layer_5_post_attn_layernorm = (char*)(0x7f0626a21000);
  all_tensors["layer_5_post_attn_layernorm"] = layer_5_post_attn_layernorm;
  char *layer_5_gate_proj = (char*)(0x7f05fc0c0000);
  all_tensors["layer_5_gate_proj"] = layer_5_gate_proj;
  char *layer_5_up_proj = (char*)(0x7f05fc6c0000);
  all_tensors["layer_5_up_proj"] = layer_5_up_proj;
  char *layer_5_down_proj = (char*)(0x7f05fbac0000);
  all_tensors["layer_5_down_proj"] = layer_5_down_proj;
  char *layer_6_input_layernorm = (char*)(0x7f0626a21c00);
  all_tensors["layer_6_input_layernorm"] = layer_6_input_layernorm;
  char *layer_6_q_proj = (char*)(0x7f05ff0c0000);
  all_tensors["layer_6_q_proj"] = layer_6_q_proj;
  char *layer_6_k_proj = (char*)(0x7f05feac0000);
  all_tensors["layer_6_k_proj"] = layer_6_k_proj;
  char *layer_6_v_proj = (char*)(0x7f05ff4c0000);
  all_tensors["layer_6_v_proj"] = layer_6_v_proj;
  char *layer_6_q_norm = (char*)(0x7f0626a22e00);
  all_tensors["layer_6_q_norm"] = layer_6_q_norm;
  char *layer_6_k_norm = (char*)(0x7f0626a22c00);
  all_tensors["layer_6_k_norm"] = layer_6_k_norm;
  char *layer_6_k_cache = (char*)(0x7f0639000000);
  all_tensors["layer_6_k_cache"] = layer_6_k_cache;
  char *layer_6_v_cache = (char*)(0x7f062b000000);
  all_tensors["layer_6_v_cache"] = layer_6_v_cache;
  char *layer_6_o_proj = (char*)(0x7f05fecc0000);
  all_tensors["layer_6_o_proj"] = layer_6_o_proj;
  char *layer_6_post_attn_layernorm = (char*)(0x7f0626a22400);
  all_tensors["layer_6_post_attn_layernorm"] = layer_6_post_attn_layernorm;
  char *layer_6_gate_proj = (char*)(0x7f05fdec0000);
  all_tensors["layer_6_gate_proj"] = layer_6_gate_proj;
  char *layer_6_up_proj = (char*)(0x7f05fe4c0000);
  all_tensors["layer_6_up_proj"] = layer_6_up_proj;
  char *layer_6_down_proj = (char*)(0x7f05fd8c0000);
  all_tensors["layer_6_down_proj"] = layer_6_down_proj;
  char *layer_7_input_layernorm = (char*)(0x7f0626a23000);
  all_tensors["layer_7_input_layernorm"] = layer_7_input_layernorm;
  char *layer_7_q_proj = (char*)(0x7f0600ec0000);
  all_tensors["layer_7_q_proj"] = layer_7_q_proj;
  char *layer_7_k_proj = (char*)(0x7f06008c0000);
  all_tensors["layer_7_k_proj"] = layer_7_k_proj;
  char *layer_7_v_proj = (char*)(0x7f06012c0000);
  all_tensors["layer_7_v_proj"] = layer_7_v_proj;
  char *layer_7_q_norm = (char*)(0x7f0626a24200);
  all_tensors["layer_7_q_norm"] = layer_7_q_norm;
  char *layer_7_k_norm = (char*)(0x7f0626a24000);
  all_tensors["layer_7_k_norm"] = layer_7_k_norm;
  char *layer_7_k_cache = (char*)(0x7f0639800000);
  all_tensors["layer_7_k_cache"] = layer_7_k_cache;
  char *layer_7_v_cache = (char*)(0x7f062b800000);
  all_tensors["layer_7_v_cache"] = layer_7_v_cache;
  char *layer_7_o_proj = (char*)(0x7f0600ac0000);
  all_tensors["layer_7_o_proj"] = layer_7_o_proj;
  char *layer_7_post_attn_layernorm = (char*)(0x7f0626a23800);
  all_tensors["layer_7_post_attn_layernorm"] = layer_7_post_attn_layernorm;
  char *layer_7_gate_proj = (char*)(0x7f05ffcc0000);
  all_tensors["layer_7_gate_proj"] = layer_7_gate_proj;
  char *layer_7_up_proj = (char*)(0x7f06002c0000);
  all_tensors["layer_7_up_proj"] = layer_7_up_proj;
  char *layer_7_down_proj = (char*)(0x7f05ff6c0000);
  all_tensors["layer_7_down_proj"] = layer_7_down_proj;
  char *layer_8_input_layernorm = (char*)(0x7f0626a24400);
  all_tensors["layer_8_input_layernorm"] = layer_8_input_layernorm;
  char *layer_8_q_proj = (char*)(0x7f0602cc0000);
  all_tensors["layer_8_q_proj"] = layer_8_q_proj;
  char *layer_8_k_proj = (char*)(0x7f06026c0000);
  all_tensors["layer_8_k_proj"] = layer_8_k_proj;
  char *layer_8_v_proj = (char*)(0x7f06030c0000);
  all_tensors["layer_8_v_proj"] = layer_8_v_proj;
  char *layer_8_q_norm = (char*)(0x7f0626a25600);
  all_tensors["layer_8_q_norm"] = layer_8_q_norm;
  char *layer_8_k_norm = (char*)(0x7f0626a25400);
  all_tensors["layer_8_k_norm"] = layer_8_k_norm;
  char *layer_8_k_cache = (char*)(0x7f063a000000);
  all_tensors["layer_8_k_cache"] = layer_8_k_cache;
  char *layer_8_v_cache = (char*)(0x7f062c000000);
  all_tensors["layer_8_v_cache"] = layer_8_v_cache;
  char *layer_8_o_proj = (char*)(0x7f06028c0000);
  all_tensors["layer_8_o_proj"] = layer_8_o_proj;
  char *layer_8_post_attn_layernorm = (char*)(0x7f0626a24c00);
  all_tensors["layer_8_post_attn_layernorm"] = layer_8_post_attn_layernorm;
  char *layer_8_gate_proj = (char*)(0x7f0601ac0000);
  all_tensors["layer_8_gate_proj"] = layer_8_gate_proj;
  char *layer_8_up_proj = (char*)(0x7f06020c0000);
  all_tensors["layer_8_up_proj"] = layer_8_up_proj;
  char *layer_8_down_proj = (char*)(0x7f06014c0000);
  all_tensors["layer_8_down_proj"] = layer_8_down_proj;
  char *layer_9_input_layernorm = (char*)(0x7f0626a25800);
  all_tensors["layer_9_input_layernorm"] = layer_9_input_layernorm;
  char *layer_9_q_proj = (char*)(0x7f0612800000);
  all_tensors["layer_9_q_proj"] = layer_9_q_proj;
  char *layer_9_k_proj = (char*)(0x7f06044c0000);
  all_tensors["layer_9_k_proj"] = layer_9_k_proj;
  char *layer_9_v_proj = (char*)(0x7f0604ac0000);
  all_tensors["layer_9_v_proj"] = layer_9_v_proj;
  char *layer_9_q_norm = (char*)(0x7f0626a26a00);
  all_tensors["layer_9_q_norm"] = layer_9_q_norm;
  char *layer_9_k_norm = (char*)(0x7f0626a26800);
  all_tensors["layer_9_k_norm"] = layer_9_k_norm;
  char *layer_9_k_cache = (char*)(0x7f063a800000);
  all_tensors["layer_9_k_cache"] = layer_9_k_cache;
  char *layer_9_v_cache = (char*)(0x7f062c800000);
  all_tensors["layer_9_v_cache"] = layer_9_v_cache;
  char *layer_9_o_proj = (char*)(0x7f06046c0000);
  all_tensors["layer_9_o_proj"] = layer_9_o_proj;
  char *layer_9_post_attn_layernorm = (char*)(0x7f0626a26000);
  all_tensors["layer_9_post_attn_layernorm"] = layer_9_post_attn_layernorm;
  char *layer_9_gate_proj = (char*)(0x7f06038c0000);
  all_tensors["layer_9_gate_proj"] = layer_9_gate_proj;
  char *layer_9_up_proj = (char*)(0x7f0603ec0000);
  all_tensors["layer_9_up_proj"] = layer_9_up_proj;
  char *layer_9_down_proj = (char*)(0x7f06032c0000);
  all_tensors["layer_9_down_proj"] = layer_9_down_proj;
  char *layer_10_input_layernorm = (char*)(0x7f0626a06400);
  all_tensors["layer_10_input_layernorm"] = layer_10_input_layernorm;
  char *layer_10_q_proj = (char*)(0x7f05d5cc0000);
  all_tensors["layer_10_q_proj"] = layer_10_q_proj;
  char *layer_10_k_proj = (char*)(0x7f05d56c0000);
  all_tensors["layer_10_k_proj"] = layer_10_k_proj;
  char *layer_10_v_proj = (char*)(0x7f05d60c0000);
  all_tensors["layer_10_v_proj"] = layer_10_v_proj;
  char *layer_10_q_norm = (char*)(0x7f0626a07600);
  all_tensors["layer_10_q_norm"] = layer_10_q_norm;
  char *layer_10_k_norm = (char*)(0x7f0626a07400);
  all_tensors["layer_10_k_norm"] = layer_10_k_norm;
  char *layer_10_k_cache = (char*)(0x7f063b000000);
  all_tensors["layer_10_k_cache"] = layer_10_k_cache;
  char *layer_10_v_cache = (char*)(0x7f062d000000);
  all_tensors["layer_10_v_cache"] = layer_10_v_cache;
  char *layer_10_o_proj = (char*)(0x7f05d58c0000);
  all_tensors["layer_10_o_proj"] = layer_10_o_proj;
  char *layer_10_post_attn_layernorm = (char*)(0x7f0626a06c00);
  all_tensors["layer_10_post_attn_layernorm"] = layer_10_post_attn_layernorm;
  char *layer_10_gate_proj = (char*)(0x7f05d4ac0000);
  all_tensors["layer_10_gate_proj"] = layer_10_gate_proj;
  char *layer_10_up_proj = (char*)(0x7f05d50c0000);
  all_tensors["layer_10_up_proj"] = layer_10_up_proj;
  char *layer_10_down_proj = (char*)(0x7f05d44c0000);
  all_tensors["layer_10_down_proj"] = layer_10_down_proj;
  char *layer_11_input_layernorm = (char*)(0x7f0626a07800);
  all_tensors["layer_11_input_layernorm"] = layer_11_input_layernorm;
  char *layer_11_q_proj = (char*)(0x7f05d7ac0000);
  all_tensors["layer_11_q_proj"] = layer_11_q_proj;
  char *layer_11_k_proj = (char*)(0x7f05d74c0000);
  all_tensors["layer_11_k_proj"] = layer_11_k_proj;
  char *layer_11_v_proj = (char*)(0x7f05d7ec0000);
  all_tensors["layer_11_v_proj"] = layer_11_v_proj;
  char *layer_11_q_norm = (char*)(0x7f0626a08a00);
  all_tensors["layer_11_q_norm"] = layer_11_q_norm;
  char *layer_11_k_norm = (char*)(0x7f0626a08800);
  all_tensors["layer_11_k_norm"] = layer_11_k_norm;
  char *layer_11_k_cache = (char*)(0x7f063b800000);
  all_tensors["layer_11_k_cache"] = layer_11_k_cache;
  char *layer_11_v_cache = (char*)(0x7f062d800000);
  all_tensors["layer_11_v_cache"] = layer_11_v_cache;
  char *layer_11_o_proj = (char*)(0x7f05d76c0000);
  all_tensors["layer_11_o_proj"] = layer_11_o_proj;
  char *layer_11_post_attn_layernorm = (char*)(0x7f0626a08000);
  all_tensors["layer_11_post_attn_layernorm"] = layer_11_post_attn_layernorm;
  char *layer_11_gate_proj = (char*)(0x7f05d68c0000);
  all_tensors["layer_11_gate_proj"] = layer_11_gate_proj;
  char *layer_11_up_proj = (char*)(0x7f05d6ec0000);
  all_tensors["layer_11_up_proj"] = layer_11_up_proj;
  char *layer_11_down_proj = (char*)(0x7f05d62c0000);
  all_tensors["layer_11_down_proj"] = layer_11_down_proj;
  char *layer_12_input_layernorm = (char*)(0x7f0626a08c00);
  all_tensors["layer_12_input_layernorm"] = layer_12_input_layernorm;
  char *layer_12_q_proj = (char*)(0x7f05d98c0000);
  all_tensors["layer_12_q_proj"] = layer_12_q_proj;
  char *layer_12_k_proj = (char*)(0x7f05d92c0000);
  all_tensors["layer_12_k_proj"] = layer_12_k_proj;
  char *layer_12_v_proj = (char*)(0x7f05d9cc0000);
  all_tensors["layer_12_v_proj"] = layer_12_v_proj;
  char *layer_12_q_norm = (char*)(0x7f0626a09e00);
  all_tensors["layer_12_q_norm"] = layer_12_q_norm;
  char *layer_12_k_norm = (char*)(0x7f0626a09c00);
  all_tensors["layer_12_k_norm"] = layer_12_k_norm;
  char *layer_12_k_cache = (char*)(0x7f063c000000);
  all_tensors["layer_12_k_cache"] = layer_12_k_cache;
  char *layer_12_v_cache = (char*)(0x7f062e000000);
  all_tensors["layer_12_v_cache"] = layer_12_v_cache;
  char *layer_12_o_proj = (char*)(0x7f05d94c0000);
  all_tensors["layer_12_o_proj"] = layer_12_o_proj;
  char *layer_12_post_attn_layernorm = (char*)(0x7f0626a09400);
  all_tensors["layer_12_post_attn_layernorm"] = layer_12_post_attn_layernorm;
  char *layer_12_gate_proj = (char*)(0x7f05d86c0000);
  all_tensors["layer_12_gate_proj"] = layer_12_gate_proj;
  char *layer_12_up_proj = (char*)(0x7f05d8cc0000);
  all_tensors["layer_12_up_proj"] = layer_12_up_proj;
  char *layer_12_down_proj = (char*)(0x7f05d80c0000);
  all_tensors["layer_12_down_proj"] = layer_12_down_proj;
  char *layer_13_input_layernorm = (char*)(0x7f0626a0a000);
  all_tensors["layer_13_input_layernorm"] = layer_13_input_layernorm;
  char *layer_13_q_proj = (char*)(0x7f05db6c0000);
  all_tensors["layer_13_q_proj"] = layer_13_q_proj;
  char *layer_13_k_proj = (char*)(0x7f05db0c0000);
  all_tensors["layer_13_k_proj"] = layer_13_k_proj;
  char *layer_13_v_proj = (char*)(0x7f05dbac0000);
  all_tensors["layer_13_v_proj"] = layer_13_v_proj;
  char *layer_13_q_norm = (char*)(0x7f0626a0b200);
  all_tensors["layer_13_q_norm"] = layer_13_q_norm;
  char *layer_13_k_norm = (char*)(0x7f0626a0b000);
  all_tensors["layer_13_k_norm"] = layer_13_k_norm;
  char *layer_13_k_cache = (char*)(0x7f063c800000);
  all_tensors["layer_13_k_cache"] = layer_13_k_cache;
  char *layer_13_v_cache = (char*)(0x7f062e800000);
  all_tensors["layer_13_v_cache"] = layer_13_v_cache;
  char *layer_13_o_proj = (char*)(0x7f05db2c0000);
  all_tensors["layer_13_o_proj"] = layer_13_o_proj;
  char *layer_13_post_attn_layernorm = (char*)(0x7f0626a0a800);
  all_tensors["layer_13_post_attn_layernorm"] = layer_13_post_attn_layernorm;
  char *layer_13_gate_proj = (char*)(0x7f05da4c0000);
  all_tensors["layer_13_gate_proj"] = layer_13_gate_proj;
  char *layer_13_up_proj = (char*)(0x7f05daac0000);
  all_tensors["layer_13_up_proj"] = layer_13_up_proj;
  char *layer_13_down_proj = (char*)(0x7f05d9ec0000);
  all_tensors["layer_13_down_proj"] = layer_13_down_proj;
  char *layer_14_input_layernorm = (char*)(0x7f0626a0b400);
  all_tensors["layer_14_input_layernorm"] = layer_14_input_layernorm;
  char *layer_14_q_proj = (char*)(0x7f05dd4c0000);
  all_tensors["layer_14_q_proj"] = layer_14_q_proj;
  char *layer_14_k_proj = (char*)(0x7f05dcec0000);
  all_tensors["layer_14_k_proj"] = layer_14_k_proj;
  char *layer_14_v_proj = (char*)(0x7f05dd8c0000);
  all_tensors["layer_14_v_proj"] = layer_14_v_proj;
  char *layer_14_q_norm = (char*)(0x7f0626a0c600);
  all_tensors["layer_14_q_norm"] = layer_14_q_norm;
  char *layer_14_k_norm = (char*)(0x7f0626a0c400);
  all_tensors["layer_14_k_norm"] = layer_14_k_norm;
  char *layer_14_k_cache = (char*)(0x7f063d000000);
  all_tensors["layer_14_k_cache"] = layer_14_k_cache;
  char *layer_14_v_cache = (char*)(0x7f062f000000);
  all_tensors["layer_14_v_cache"] = layer_14_v_cache;
  char *layer_14_o_proj = (char*)(0x7f05dd0c0000);
  all_tensors["layer_14_o_proj"] = layer_14_o_proj;
  char *layer_14_post_attn_layernorm = (char*)(0x7f0626a0bc00);
  all_tensors["layer_14_post_attn_layernorm"] = layer_14_post_attn_layernorm;
  char *layer_14_gate_proj = (char*)(0x7f05dc2c0000);
  all_tensors["layer_14_gate_proj"] = layer_14_gate_proj;
  char *layer_14_up_proj = (char*)(0x7f05dc8c0000);
  all_tensors["layer_14_up_proj"] = layer_14_up_proj;
  char *layer_14_down_proj = (char*)(0x7f05dbcc0000);
  all_tensors["layer_14_down_proj"] = layer_14_down_proj;
  char *layer_15_input_layernorm = (char*)(0x7f0626a0c800);
  all_tensors["layer_15_input_layernorm"] = layer_15_input_layernorm;
  char *layer_15_q_proj = (char*)(0x7f05df2c0000);
  all_tensors["layer_15_q_proj"] = layer_15_q_proj;
  char *layer_15_k_proj = (char*)(0x7f05decc0000);
  all_tensors["layer_15_k_proj"] = layer_15_k_proj;
  char *layer_15_v_proj = (char*)(0x7f05df6c0000);
  all_tensors["layer_15_v_proj"] = layer_15_v_proj;
  char *layer_15_q_norm = (char*)(0x7f0626a0da00);
  all_tensors["layer_15_q_norm"] = layer_15_q_norm;
  char *layer_15_k_norm = (char*)(0x7f0626a0d800);
  all_tensors["layer_15_k_norm"] = layer_15_k_norm;
  char *layer_15_k_cache = (char*)(0x7f063d800000);
  all_tensors["layer_15_k_cache"] = layer_15_k_cache;
  char *layer_15_v_cache = (char*)(0x7f062f800000);
  all_tensors["layer_15_v_cache"] = layer_15_v_cache;
  char *layer_15_o_proj = (char*)(0x7f05deec0000);
  all_tensors["layer_15_o_proj"] = layer_15_o_proj;
  char *layer_15_post_attn_layernorm = (char*)(0x7f0626a0d000);
  all_tensors["layer_15_post_attn_layernorm"] = layer_15_post_attn_layernorm;
  char *layer_15_gate_proj = (char*)(0x7f05de0c0000);
  all_tensors["layer_15_gate_proj"] = layer_15_gate_proj;
  char *layer_15_up_proj = (char*)(0x7f05de6c0000);
  all_tensors["layer_15_up_proj"] = layer_15_up_proj;
  char *layer_15_down_proj = (char*)(0x7f05ddac0000);
  all_tensors["layer_15_down_proj"] = layer_15_down_proj;
  char *layer_16_input_layernorm = (char*)(0x7f0626a0dc00);
  all_tensors["layer_16_input_layernorm"] = layer_16_input_layernorm;
  char *layer_16_q_proj = (char*)(0x7f05e10c0000);
  all_tensors["layer_16_q_proj"] = layer_16_q_proj;
  char *layer_16_k_proj = (char*)(0x7f05e0ac0000);
  all_tensors["layer_16_k_proj"] = layer_16_k_proj;
  char *layer_16_v_proj = (char*)(0x7f05e14c0000);
  all_tensors["layer_16_v_proj"] = layer_16_v_proj;
  char *layer_16_q_norm = (char*)(0x7f0626a0ee00);
  all_tensors["layer_16_q_norm"] = layer_16_q_norm;
  char *layer_16_k_norm = (char*)(0x7f0626a0ec00);
  all_tensors["layer_16_k_norm"] = layer_16_k_norm;
  char *layer_16_k_cache = (char*)(0x7f063e000000);
  all_tensors["layer_16_k_cache"] = layer_16_k_cache;
  char *layer_16_v_cache = (char*)(0x7f0630000000);
  all_tensors["layer_16_v_cache"] = layer_16_v_cache;
  char *layer_16_o_proj = (char*)(0x7f05e0cc0000);
  all_tensors["layer_16_o_proj"] = layer_16_o_proj;
  char *layer_16_post_attn_layernorm = (char*)(0x7f0626a0e400);
  all_tensors["layer_16_post_attn_layernorm"] = layer_16_post_attn_layernorm;
  char *layer_16_gate_proj = (char*)(0x7f05dfec0000);
  all_tensors["layer_16_gate_proj"] = layer_16_gate_proj;
  char *layer_16_up_proj = (char*)(0x7f05e04c0000);
  all_tensors["layer_16_up_proj"] = layer_16_up_proj;
  char *layer_16_down_proj = (char*)(0x7f05df8c0000);
  all_tensors["layer_16_down_proj"] = layer_16_down_proj;
  char *layer_17_input_layernorm = (char*)(0x7f0626a0f000);
  all_tensors["layer_17_input_layernorm"] = layer_17_input_layernorm;
  char *layer_17_q_proj = (char*)(0x7f05e2ec0000);
  all_tensors["layer_17_q_proj"] = layer_17_q_proj;
  char *layer_17_k_proj = (char*)(0x7f05e28c0000);
  all_tensors["layer_17_k_proj"] = layer_17_k_proj;
  char *layer_17_v_proj = (char*)(0x7f05e32c0000);
  all_tensors["layer_17_v_proj"] = layer_17_v_proj;
  char *layer_17_q_norm = (char*)(0x7f0626a10200);
  all_tensors["layer_17_q_norm"] = layer_17_q_norm;
  char *layer_17_k_norm = (char*)(0x7f0626a10000);
  all_tensors["layer_17_k_norm"] = layer_17_k_norm;
  char *layer_17_k_cache = (char*)(0x7f063e800000);
  all_tensors["layer_17_k_cache"] = layer_17_k_cache;
  char *layer_17_v_cache = (char*)(0x7f0630800000);
  all_tensors["layer_17_v_cache"] = layer_17_v_cache;
  char *layer_17_o_proj = (char*)(0x7f05e2ac0000);
  all_tensors["layer_17_o_proj"] = layer_17_o_proj;
  char *layer_17_post_attn_layernorm = (char*)(0x7f0626a0f800);
  all_tensors["layer_17_post_attn_layernorm"] = layer_17_post_attn_layernorm;
  char *layer_17_gate_proj = (char*)(0x7f05e1cc0000);
  all_tensors["layer_17_gate_proj"] = layer_17_gate_proj;
  char *layer_17_up_proj = (char*)(0x7f05e22c0000);
  all_tensors["layer_17_up_proj"] = layer_17_up_proj;
  char *layer_17_down_proj = (char*)(0x7f05e16c0000);
  all_tensors["layer_17_down_proj"] = layer_17_down_proj;
  char *layer_18_input_layernorm = (char*)(0x7f0626a10400);
  all_tensors["layer_18_input_layernorm"] = layer_18_input_layernorm;
  char *layer_18_q_proj = (char*)(0x7f05e4cc0000);
  all_tensors["layer_18_q_proj"] = layer_18_q_proj;
  char *layer_18_k_proj = (char*)(0x7f05e46c0000);
  all_tensors["layer_18_k_proj"] = layer_18_k_proj;
  char *layer_18_v_proj = (char*)(0x7f05e50c0000);
  all_tensors["layer_18_v_proj"] = layer_18_v_proj;
  char *layer_18_q_norm = (char*)(0x7f0626a11600);
  all_tensors["layer_18_q_norm"] = layer_18_q_norm;
  char *layer_18_k_norm = (char*)(0x7f0626a11400);
  all_tensors["layer_18_k_norm"] = layer_18_k_norm;
  char *layer_18_k_cache = (char*)(0x7f063f000000);
  all_tensors["layer_18_k_cache"] = layer_18_k_cache;
  char *layer_18_v_cache = (char*)(0x7f0631000000);
  all_tensors["layer_18_v_cache"] = layer_18_v_cache;
  char *layer_18_o_proj = (char*)(0x7f05e48c0000);
  all_tensors["layer_18_o_proj"] = layer_18_o_proj;
  char *layer_18_post_attn_layernorm = (char*)(0x7f0626a10c00);
  all_tensors["layer_18_post_attn_layernorm"] = layer_18_post_attn_layernorm;
  char *layer_18_gate_proj = (char*)(0x7f05e3ac0000);
  all_tensors["layer_18_gate_proj"] = layer_18_gate_proj;
  char *layer_18_up_proj = (char*)(0x7f05e40c0000);
  all_tensors["layer_18_up_proj"] = layer_18_up_proj;
  char *layer_18_down_proj = (char*)(0x7f05e34c0000);
  all_tensors["layer_18_down_proj"] = layer_18_down_proj;
  char *layer_19_input_layernorm = (char*)(0x7f0626a11800);
  all_tensors["layer_19_input_layernorm"] = layer_19_input_layernorm;
  char *layer_19_q_proj = (char*)(0x7f05e6ac0000);
  all_tensors["layer_19_q_proj"] = layer_19_q_proj;
  char *layer_19_k_proj = (char*)(0x7f05e64c0000);
  all_tensors["layer_19_k_proj"] = layer_19_k_proj;
  char *layer_19_v_proj = (char*)(0x7f05e6ec0000);
  all_tensors["layer_19_v_proj"] = layer_19_v_proj;
  char *layer_19_q_norm = (char*)(0x7f0626a12a00);
  all_tensors["layer_19_q_norm"] = layer_19_q_norm;
  char *layer_19_k_norm = (char*)(0x7f0626a12800);
  all_tensors["layer_19_k_norm"] = layer_19_k_norm;
  char *layer_19_k_cache = (char*)(0x7f063f800000);
  all_tensors["layer_19_k_cache"] = layer_19_k_cache;
  char *layer_19_v_cache = (char*)(0x7f0631800000);
  all_tensors["layer_19_v_cache"] = layer_19_v_cache;
  char *layer_19_o_proj = (char*)(0x7f05e66c0000);
  all_tensors["layer_19_o_proj"] = layer_19_o_proj;
  char *layer_19_post_attn_layernorm = (char*)(0x7f0626a12000);
  all_tensors["layer_19_post_attn_layernorm"] = layer_19_post_attn_layernorm;
  char *layer_19_gate_proj = (char*)(0x7f05e58c0000);
  all_tensors["layer_19_gate_proj"] = layer_19_gate_proj;
  char *layer_19_up_proj = (char*)(0x7f05e5ec0000);
  all_tensors["layer_19_up_proj"] = layer_19_up_proj;
  char *layer_19_down_proj = (char*)(0x7f05e52c0000);
  all_tensors["layer_19_down_proj"] = layer_19_down_proj;
  char *layer_20_input_layernorm = (char*)(0x7f0626a14000);
  all_tensors["layer_20_input_layernorm"] = layer_20_input_layernorm;
  char *layer_20_q_proj = (char*)(0x7f05ea6c0000);
  all_tensors["layer_20_q_proj"] = layer_20_q_proj;
  char *layer_20_k_proj = (char*)(0x7f05ea0c0000);
  all_tensors["layer_20_k_proj"] = layer_20_k_proj;
  char *layer_20_v_proj = (char*)(0x7f05eaac0000);
  all_tensors["layer_20_v_proj"] = layer_20_v_proj;
  char *layer_20_q_norm = (char*)(0x7f0626a15200);
  all_tensors["layer_20_q_norm"] = layer_20_q_norm;
  char *layer_20_k_norm = (char*)(0x7f0626a15000);
  all_tensors["layer_20_k_norm"] = layer_20_k_norm;
  char *layer_20_k_cache = (char*)(0x7f0640000000);
  all_tensors["layer_20_k_cache"] = layer_20_k_cache;
  char *layer_20_v_cache = (char*)(0x7f0632000000);
  all_tensors["layer_20_v_cache"] = layer_20_v_cache;
  char *layer_20_o_proj = (char*)(0x7f05ea2c0000);
  all_tensors["layer_20_o_proj"] = layer_20_o_proj;
  char *layer_20_post_attn_layernorm = (char*)(0x7f0626a14800);
  all_tensors["layer_20_post_attn_layernorm"] = layer_20_post_attn_layernorm;
  char *layer_20_gate_proj = (char*)(0x7f05e94c0000);
  all_tensors["layer_20_gate_proj"] = layer_20_gate_proj;
  char *layer_20_up_proj = (char*)(0x7f05e9ac0000);
  all_tensors["layer_20_up_proj"] = layer_20_up_proj;
  char *layer_20_down_proj = (char*)(0x7f05e8ec0000);
  all_tensors["layer_20_down_proj"] = layer_20_down_proj;
  char *layer_21_input_layernorm = (char*)(0x7f0626a15400);
  all_tensors["layer_21_input_layernorm"] = layer_21_input_layernorm;
  char *layer_21_q_proj = (char*)(0x7f05ec4c0000);
  all_tensors["layer_21_q_proj"] = layer_21_q_proj;
  char *layer_21_k_proj = (char*)(0x7f05ebec0000);
  all_tensors["layer_21_k_proj"] = layer_21_k_proj;
  char *layer_21_v_proj = (char*)(0x7f05ec8c0000);
  all_tensors["layer_21_v_proj"] = layer_21_v_proj;
  char *layer_21_q_norm = (char*)(0x7f0626a16600);
  all_tensors["layer_21_q_norm"] = layer_21_q_norm;
  char *layer_21_k_norm = (char*)(0x7f0626a16400);
  all_tensors["layer_21_k_norm"] = layer_21_k_norm;
  char *layer_21_k_cache = (char*)(0x7f0640800000);
  all_tensors["layer_21_k_cache"] = layer_21_k_cache;
  char *layer_21_v_cache = (char*)(0x7f0632800000);
  all_tensors["layer_21_v_cache"] = layer_21_v_cache;
  char *layer_21_o_proj = (char*)(0x7f05ec0c0000);
  all_tensors["layer_21_o_proj"] = layer_21_o_proj;
  char *layer_21_post_attn_layernorm = (char*)(0x7f0626a15c00);
  all_tensors["layer_21_post_attn_layernorm"] = layer_21_post_attn_layernorm;
  char *layer_21_gate_proj = (char*)(0x7f05eb2c0000);
  all_tensors["layer_21_gate_proj"] = layer_21_gate_proj;
  char *layer_21_up_proj = (char*)(0x7f05eb8c0000);
  all_tensors["layer_21_up_proj"] = layer_21_up_proj;
  char *layer_21_down_proj = (char*)(0x7f05eacc0000);
  all_tensors["layer_21_down_proj"] = layer_21_down_proj;
  char *layer_22_input_layernorm = (char*)(0x7f0626a16800);
  all_tensors["layer_22_input_layernorm"] = layer_22_input_layernorm;
  char *layer_22_q_proj = (char*)(0x7f05ee2c0000);
  all_tensors["layer_22_q_proj"] = layer_22_q_proj;
  char *layer_22_k_proj = (char*)(0x7f05edcc0000);
  all_tensors["layer_22_k_proj"] = layer_22_k_proj;
  char *layer_22_v_proj = (char*)(0x7f05ee6c0000);
  all_tensors["layer_22_v_proj"] = layer_22_v_proj;
  char *layer_22_q_norm = (char*)(0x7f0626a17a00);
  all_tensors["layer_22_q_norm"] = layer_22_q_norm;
  char *layer_22_k_norm = (char*)(0x7f0626a17800);
  all_tensors["layer_22_k_norm"] = layer_22_k_norm;
  char *layer_22_k_cache = (char*)(0x7f0641000000);
  all_tensors["layer_22_k_cache"] = layer_22_k_cache;
  char *layer_22_v_cache = (char*)(0x7f0633000000);
  all_tensors["layer_22_v_cache"] = layer_22_v_cache;
  char *layer_22_o_proj = (char*)(0x7f05edec0000);
  all_tensors["layer_22_o_proj"] = layer_22_o_proj;
  char *layer_22_post_attn_layernorm = (char*)(0x7f0626a17000);
  all_tensors["layer_22_post_attn_layernorm"] = layer_22_post_attn_layernorm;
  char *layer_22_gate_proj = (char*)(0x7f05ed0c0000);
  all_tensors["layer_22_gate_proj"] = layer_22_gate_proj;
  char *layer_22_up_proj = (char*)(0x7f05ed6c0000);
  all_tensors["layer_22_up_proj"] = layer_22_up_proj;
  char *layer_22_down_proj = (char*)(0x7f05ecac0000);
  all_tensors["layer_22_down_proj"] = layer_22_down_proj;
  char *layer_23_input_layernorm = (char*)(0x7f0626a17c00);
  all_tensors["layer_23_input_layernorm"] = layer_23_input_layernorm;
  char *layer_23_q_proj = (char*)(0x7f05f00c0000);
  all_tensors["layer_23_q_proj"] = layer_23_q_proj;
  char *layer_23_k_proj = (char*)(0x7f05efac0000);
  all_tensors["layer_23_k_proj"] = layer_23_k_proj;
  char *layer_23_v_proj = (char*)(0x7f05f04c0000);
  all_tensors["layer_23_v_proj"] = layer_23_v_proj;
  char *layer_23_q_norm = (char*)(0x7f0626a18e00);
  all_tensors["layer_23_q_norm"] = layer_23_q_norm;
  char *layer_23_k_norm = (char*)(0x7f0626a18c00);
  all_tensors["layer_23_k_norm"] = layer_23_k_norm;
  char *layer_23_k_cache = (char*)(0x7f0641800000);
  all_tensors["layer_23_k_cache"] = layer_23_k_cache;
  char *layer_23_v_cache = (char*)(0x7f0633800000);
  all_tensors["layer_23_v_cache"] = layer_23_v_cache;
  char *layer_23_o_proj = (char*)(0x7f05efcc0000);
  all_tensors["layer_23_o_proj"] = layer_23_o_proj;
  char *layer_23_post_attn_layernorm = (char*)(0x7f0626a18400);
  all_tensors["layer_23_post_attn_layernorm"] = layer_23_post_attn_layernorm;
  char *layer_23_gate_proj = (char*)(0x7f05eeec0000);
  all_tensors["layer_23_gate_proj"] = layer_23_gate_proj;
  char *layer_23_up_proj = (char*)(0x7f05ef4c0000);
  all_tensors["layer_23_up_proj"] = layer_23_up_proj;
  char *layer_23_down_proj = (char*)(0x7f05ee8c0000);
  all_tensors["layer_23_down_proj"] = layer_23_down_proj;
  char *layer_24_input_layernorm = (char*)(0x7f0626a19000);
  all_tensors["layer_24_input_layernorm"] = layer_24_input_layernorm;
  char *layer_24_q_proj = (char*)(0x7f05f1ec0000);
  all_tensors["layer_24_q_proj"] = layer_24_q_proj;
  char *layer_24_k_proj = (char*)(0x7f05f18c0000);
  all_tensors["layer_24_k_proj"] = layer_24_k_proj;
  char *layer_24_v_proj = (char*)(0x7f05f22c0000);
  all_tensors["layer_24_v_proj"] = layer_24_v_proj;
  char *layer_24_q_norm = (char*)(0x7f0626a1a200);
  all_tensors["layer_24_q_norm"] = layer_24_q_norm;
  char *layer_24_k_norm = (char*)(0x7f0626a1a000);
  all_tensors["layer_24_k_norm"] = layer_24_k_norm;
  char *layer_24_k_cache = (char*)(0x7f0642000000);
  all_tensors["layer_24_k_cache"] = layer_24_k_cache;
  char *layer_24_v_cache = (char*)(0x7f0634000000);
  all_tensors["layer_24_v_cache"] = layer_24_v_cache;
  char *layer_24_o_proj = (char*)(0x7f05f1ac0000);
  all_tensors["layer_24_o_proj"] = layer_24_o_proj;
  char *layer_24_post_attn_layernorm = (char*)(0x7f0626a19800);
  all_tensors["layer_24_post_attn_layernorm"] = layer_24_post_attn_layernorm;
  char *layer_24_gate_proj = (char*)(0x7f05f0cc0000);
  all_tensors["layer_24_gate_proj"] = layer_24_gate_proj;
  char *layer_24_up_proj = (char*)(0x7f05f12c0000);
  all_tensors["layer_24_up_proj"] = layer_24_up_proj;
  char *layer_24_down_proj = (char*)(0x7f05f06c0000);
  all_tensors["layer_24_down_proj"] = layer_24_down_proj;
  char *layer_25_input_layernorm = (char*)(0x7f0626a1a400);
  all_tensors["layer_25_input_layernorm"] = layer_25_input_layernorm;
  char *layer_25_q_proj = (char*)(0x7f05f3cc0000);
  all_tensors["layer_25_q_proj"] = layer_25_q_proj;
  char *layer_25_k_proj = (char*)(0x7f05f36c0000);
  all_tensors["layer_25_k_proj"] = layer_25_k_proj;
  char *layer_25_v_proj = (char*)(0x7f05f40c0000);
  all_tensors["layer_25_v_proj"] = layer_25_v_proj;
  char *layer_25_q_norm = (char*)(0x7f0626a1b600);
  all_tensors["layer_25_q_norm"] = layer_25_q_norm;
  char *layer_25_k_norm = (char*)(0x7f0626a1b400);
  all_tensors["layer_25_k_norm"] = layer_25_k_norm;
  char *layer_25_k_cache = (char*)(0x7f0642800000);
  all_tensors["layer_25_k_cache"] = layer_25_k_cache;
  char *layer_25_v_cache = (char*)(0x7f0634800000);
  all_tensors["layer_25_v_cache"] = layer_25_v_cache;
  char *layer_25_o_proj = (char*)(0x7f05f38c0000);
  all_tensors["layer_25_o_proj"] = layer_25_o_proj;
  char *layer_25_post_attn_layernorm = (char*)(0x7f0626a1ac00);
  all_tensors["layer_25_post_attn_layernorm"] = layer_25_post_attn_layernorm;
  char *layer_25_gate_proj = (char*)(0x7f05f2ac0000);
  all_tensors["layer_25_gate_proj"] = layer_25_gate_proj;
  char *layer_25_up_proj = (char*)(0x7f05f30c0000);
  all_tensors["layer_25_up_proj"] = layer_25_up_proj;
  char *layer_25_down_proj = (char*)(0x7f05f24c0000);
  all_tensors["layer_25_down_proj"] = layer_25_down_proj;
  char *layer_26_input_layernorm = (char*)(0x7f0626a1b800);
  all_tensors["layer_26_input_layernorm"] = layer_26_input_layernorm;
  char *layer_26_q_proj = (char*)(0x7f05f5ac0000);
  all_tensors["layer_26_q_proj"] = layer_26_q_proj;
  char *layer_26_k_proj = (char*)(0x7f05f54c0000);
  all_tensors["layer_26_k_proj"] = layer_26_k_proj;
  char *layer_26_v_proj = (char*)(0x7f05f5ec0000);
  all_tensors["layer_26_v_proj"] = layer_26_v_proj;
  char *layer_26_q_norm = (char*)(0x7f0626a1ca00);
  all_tensors["layer_26_q_norm"] = layer_26_q_norm;
  char *layer_26_k_norm = (char*)(0x7f0626a1c800);
  all_tensors["layer_26_k_norm"] = layer_26_k_norm;
  char *layer_26_k_cache = (char*)(0x7f0643000000);
  all_tensors["layer_26_k_cache"] = layer_26_k_cache;
  char *layer_26_v_cache = (char*)(0x7f0635000000);
  all_tensors["layer_26_v_cache"] = layer_26_v_cache;
  char *layer_26_o_proj = (char*)(0x7f05f56c0000);
  all_tensors["layer_26_o_proj"] = layer_26_o_proj;
  char *layer_26_post_attn_layernorm = (char*)(0x7f0626a1c000);
  all_tensors["layer_26_post_attn_layernorm"] = layer_26_post_attn_layernorm;
  char *layer_26_gate_proj = (char*)(0x7f05f48c0000);
  all_tensors["layer_26_gate_proj"] = layer_26_gate_proj;
  char *layer_26_up_proj = (char*)(0x7f05f4ec0000);
  all_tensors["layer_26_up_proj"] = layer_26_up_proj;
  char *layer_26_down_proj = (char*)(0x7f05f42c0000);
  all_tensors["layer_26_down_proj"] = layer_26_down_proj;
  char *layer_27_input_layernorm = (char*)(0x7f0626a1cc00);
  all_tensors["layer_27_input_layernorm"] = layer_27_input_layernorm;
  char *layer_27_q_proj = (char*)(0x7f05f78c0000);
  all_tensors["layer_27_q_proj"] = layer_27_q_proj;
  char *layer_27_k_proj = (char*)(0x7f05f72c0000);
  all_tensors["layer_27_k_proj"] = layer_27_k_proj;
  char *layer_27_v_proj = (char*)(0x7f05f7cc0000);
  all_tensors["layer_27_v_proj"] = layer_27_v_proj;
  char *layer_27_q_norm = (char*)(0x7f0626a1de00);
  all_tensors["layer_27_q_norm"] = layer_27_q_norm;
  char *layer_27_k_norm = (char*)(0x7f0626a1dc00);
  all_tensors["layer_27_k_norm"] = layer_27_k_norm;
  char *layer_27_k_cache = (char*)(0x7f0643800000);
  all_tensors["layer_27_k_cache"] = layer_27_k_cache;
  char *layer_27_v_cache = (char*)(0x7f0635800000);
  all_tensors["layer_27_v_cache"] = layer_27_v_cache;
  char *layer_27_o_proj = (char*)(0x7f05f74c0000);
  all_tensors["layer_27_o_proj"] = layer_27_o_proj;
  char *layer_27_post_attn_layernorm = (char*)(0x7f0626a1d400);
  all_tensors["layer_27_post_attn_layernorm"] = layer_27_post_attn_layernorm;
  char *layer_27_gate_proj = (char*)(0x7f05f66c0000);
  all_tensors["layer_27_gate_proj"] = layer_27_gate_proj;
  char *layer_27_up_proj = (char*)(0x7f05f6cc0000);
  all_tensors["layer_27_up_proj"] = layer_27_up_proj;
  char *layer_27_down_proj = (char*)(0x7f05f60c0000);
  all_tensors["layer_27_down_proj"] = layer_27_down_proj;
  char *model_norm_weight = (char*)(0x7f0626a26c00);
  all_tensors["model_norm_weight"] = model_norm_weight;
  char *lm_head = (char*)(0x7f0544000000);
  all_tensors["lm_head"] = lm_head;
  all_tensors["nullptr"] = nullptr;
  construct_task_graph(num_gpus, my_gpu_id, all_tasks, all_events, first_tasks, all_tensors);
}

__device__ __forceinline__
void _execute_task(TaskDesc const& task_desc,
                   RuntimeConfig const &runtime_config) {
  if (task_desc.task_type == TASK_EMBEDDING && task_desc.variant_id == 0) {
      kernel::embedding_kernel<bfloat16, 1, 16, 1024>(
      runtime_config.tokens + runtime_config.step[0], 
      task_desc.inputs[1].base_ptr,
      task_desc.outputs[0].base_ptr);

  }
  else if (task_desc.task_type == TASK_RMS_NORM_LINEAR && task_desc.variant_id == 0) {
      kernel::norm_linear_task_impl<bfloat16, 1, 64, 1024, 4096>(
      task_desc.inputs[0].base_ptr,
      task_desc.inputs[1].base_ptr,
      task_desc.inputs[2].base_ptr,
      1e-6f,
      task_desc.outputs[0].base_ptr);

  }
  else if (task_desc.task_type == TASK_RMS_NORM_LINEAR && task_desc.variant_id == 1) {
      kernel::norm_linear_task_impl<bfloat16, 1, 64, 1024, 6144>(
      task_desc.inputs[0].base_ptr,
      task_desc.inputs[1].base_ptr,
      task_desc.inputs[2].base_ptr,
      1e-6f,
      task_desc.outputs[0].base_ptr);

  }
  else if (task_desc.task_type == TASK_RMS_NORM_LINEAR && task_desc.variant_id == 2) {
      kernel::norm_linear_task_impl<bfloat16, 1, 1600, 1024, 153600>(
      task_desc.inputs[0].base_ptr,
      task_desc.inputs[1].base_ptr,
      task_desc.inputs[2].base_ptr,
      1e-6f,
      task_desc.outputs[0].base_ptr);

  }
  else if (task_desc.task_type == TASK_ATTENTION_1 && task_desc.variant_id == 0) {
      kernel::single_batch_decoding_kernel<bfloat16, 2, 1, 128, 1024>(
      task_desc.inputs[0].base_ptr,
      task_desc.inputs[1].base_ptr,
      task_desc.inputs[2].base_ptr,
      task_desc.outputs[0].base_ptr,
      runtime_config.step[0] + 1,
      true,
      true,
      task_desc.inputs[3].base_ptr,
      task_desc.inputs[4].base_ptr,
      task_desc.inputs[5].base_ptr,
      task_desc.inputs[6].base_ptr,
      1e-6f,
      1e-6f);

  }
  else if (task_desc.task_type == TASK_SILU_MUL_LINEAR_WITH_RESIDUAL && task_desc.variant_id == 0) {
      kernel::silu_mul_linear_task_impl<bfloat16, 1, 64, 3072, 1024>(
      task_desc.inputs[0].base_ptr,
      task_desc.inputs[1].base_ptr,
      task_desc.inputs[2].base_ptr,
      task_desc.outputs[0].base_ptr,
      runtime_config.my_gpu_id == 0);

  }
  else if (task_desc.task_type == TASK_LINEAR_WITH_RESIDUAL && task_desc.variant_id == 0) {
      kernel::linear_kernel<bfloat16, 1, 64, 2048, 1024>(
      task_desc.inputs[0].base_ptr,
      task_desc.inputs[1].base_ptr,
      task_desc.inputs[2].base_ptr,
      task_desc.outputs[0].base_ptr,
      runtime_config.my_gpu_id == 0);

  }
  else if (task_desc.task_type == TASK_ARGMAX_PARTIAL && task_desc.variant_id == 0) {
      kernel::argmax_partial_kernel<bfloat16, 1, 1600, 1>(
      task_desc.inputs[0].base_ptr,
      task_desc.outputs[0].base_ptr,
      task_desc.outputs[1].base_ptr);

  }
  else if (task_desc.task_type == TASK_ARGMAX_REDUCE && task_desc.variant_id == 0) {
      kernel::argmax_reduce_kernel<bfloat16, 1600, 96>(
      task_desc.inputs[0].base_ptr,
      task_desc.inputs[1].base_ptr,
      task_desc.outputs[0].base_ptr,
      runtime_config.step[0],
      runtime_config.tokens);

  }
}
