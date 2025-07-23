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
      json j = task.at("trigger_event");
      int gpu_offset = j.at("gpu_offset").get<int>();
      size_t event_pos = j.at("event_pos").get<size_t>();
      bool is_nvshmem = j.at("is_nvshmem").get<bool>();
      task_desc.trigger_event = get_event_id((my_gpu_id + gpu_offset) % num_gpus, event_pos, is_nvshmem);
    }
    if (task.at("dependent_event").is_number_integer()) {
      task_desc.dependent_event = task.at("dependent_event").get<unsigned long long int>();
    }
    else {
      json j = task.at("dependent_event");
      int gpu_offset = j.at("gpu_offset").get<int>();
      size_t event_pos = j.at("event_pos").get<size_t>();
      bool is_nvshmem = j.at("is_nvshmem").get<bool>();
      task_desc.dependent_event = get_event_id((my_gpu_id + gpu_offset) % num_gpus, event_pos, is_nvshmem);
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
  char *input_token = (char*)(0x7bddc0aa7800);
  all_tensors["input_token"] = input_token;
  char *cos_position_embedding = (char*)(0x7bddad400000);
  all_tensors["cos_position_embedding"] = cos_position_embedding;
  char *sin_position_embedding = (char*)(0x7bddb0820000);
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
  char *embed_tokens = (char*)(0x7bdd58000000);
  all_tensors["embed_tokens"] = embed_tokens;
  char *layer_0_input_layernorm = (char*)(0x7bddc0a03e00);
  all_tensors["layer_0_input_layernorm"] = layer_0_input_layernorm;
  char *layer_0_q_proj = (char*)(0x7bdd6c0c0000);
  all_tensors["layer_0_q_proj"] = layer_0_q_proj;
  char *layer_0_k_proj = (char*)(0x7bdd6bac0000);
  all_tensors["layer_0_k_proj"] = layer_0_k_proj;
  char *layer_0_v_proj = (char*)(0x7bdd6c4c0000);
  all_tensors["layer_0_v_proj"] = layer_0_v_proj;
  char *layer_0_q_norm = (char*)(0x7bddc0a04e00);
  all_tensors["layer_0_q_norm"] = layer_0_q_norm;
  char *layer_0_k_norm = (char*)(0x7bddc0a00200);
  all_tensors["layer_0_k_norm"] = layer_0_k_norm;
  char *layer_0_k_cache = (char*)(0x7bddd0000000);
  all_tensors["layer_0_k_cache"] = layer_0_k_cache;
  char *layer_0_v_cache = (char*)(0x7bddc2000000);
  all_tensors["layer_0_v_cache"] = layer_0_v_cache;
  char *layer_0_o_proj = (char*)(0x7bdd6bcc0000);
  all_tensors["layer_0_o_proj"] = layer_0_o_proj;
  char *layer_0_post_attn_layernorm = (char*)(0x7bddc0a04600);
  all_tensors["layer_0_post_attn_layernorm"] = layer_0_post_attn_layernorm;
  char *layer_0_gate_proj = (char*)(0x7bdd6aec0000);
  all_tensors["layer_0_gate_proj"] = layer_0_gate_proj;
  char *layer_0_up_proj = (char*)(0x7bdd6b4c0000);
  all_tensors["layer_0_up_proj"] = layer_0_up_proj;
  char *layer_0_down_proj = (char*)(0x7bdd6a8c0000);
  all_tensors["layer_0_down_proj"] = layer_0_down_proj;
  char *layer_1_input_layernorm = (char*)(0x7bddc0a05000);
  all_tensors["layer_1_input_layernorm"] = layer_1_input_layernorm;
  char *layer_1_q_proj = (char*)(0x7bdd6dec0000);
  all_tensors["layer_1_q_proj"] = layer_1_q_proj;
  char *layer_1_k_proj = (char*)(0x7bdd6d8c0000);
  all_tensors["layer_1_k_proj"] = layer_1_k_proj;
  char *layer_1_v_proj = (char*)(0x7bdd6e2c0000);
  all_tensors["layer_1_v_proj"] = layer_1_v_proj;
  char *layer_1_q_norm = (char*)(0x7bddc0a06200);
  all_tensors["layer_1_q_norm"] = layer_1_q_norm;
  char *layer_1_k_norm = (char*)(0x7bddc0a06000);
  all_tensors["layer_1_k_norm"] = layer_1_k_norm;
  char *layer_1_k_cache = (char*)(0x7bddd0800000);
  all_tensors["layer_1_k_cache"] = layer_1_k_cache;
  char *layer_1_v_cache = (char*)(0x7bddc2800000);
  all_tensors["layer_1_v_cache"] = layer_1_v_cache;
  char *layer_1_o_proj = (char*)(0x7bdd6dac0000);
  all_tensors["layer_1_o_proj"] = layer_1_o_proj;
  char *layer_1_post_attn_layernorm = (char*)(0x7bddc0a05800);
  all_tensors["layer_1_post_attn_layernorm"] = layer_1_post_attn_layernorm;
  char *layer_1_gate_proj = (char*)(0x7bdd6ccc0000);
  all_tensors["layer_1_gate_proj"] = layer_1_gate_proj;
  char *layer_1_up_proj = (char*)(0x7bdd6d2c0000);
  all_tensors["layer_1_up_proj"] = layer_1_up_proj;
  char *layer_1_down_proj = (char*)(0x7bdd6c6c0000);
  all_tensors["layer_1_down_proj"] = layer_1_down_proj;
  char *layer_2_input_layernorm = (char*)(0x7bddc0a12c00);
  all_tensors["layer_2_input_layernorm"] = layer_2_input_layernorm;
  char *layer_2_q_proj = (char*)(0x7bdd828c0000);
  all_tensors["layer_2_q_proj"] = layer_2_q_proj;
  char *layer_2_k_proj = (char*)(0x7bdd822c0000);
  all_tensors["layer_2_k_proj"] = layer_2_k_proj;
  char *layer_2_v_proj = (char*)(0x7bdd82cc0000);
  all_tensors["layer_2_v_proj"] = layer_2_v_proj;
  char *layer_2_q_norm = (char*)(0x7bddc0a13e00);
  all_tensors["layer_2_q_norm"] = layer_2_q_norm;
  char *layer_2_k_norm = (char*)(0x7bddc0a13c00);
  all_tensors["layer_2_k_norm"] = layer_2_k_norm;
  char *layer_2_k_cache = (char*)(0x7bddd1000000);
  all_tensors["layer_2_k_cache"] = layer_2_k_cache;
  char *layer_2_v_cache = (char*)(0x7bddc3000000);
  all_tensors["layer_2_v_cache"] = layer_2_v_cache;
  char *layer_2_o_proj = (char*)(0x7bdd824c0000);
  all_tensors["layer_2_o_proj"] = layer_2_o_proj;
  char *layer_2_post_attn_layernorm = (char*)(0x7bddc0a13400);
  all_tensors["layer_2_post_attn_layernorm"] = layer_2_post_attn_layernorm;
  char *layer_2_gate_proj = (char*)(0x7bdd816c0000);
  all_tensors["layer_2_gate_proj"] = layer_2_gate_proj;
  char *layer_2_up_proj = (char*)(0x7bdd81cc0000);
  all_tensors["layer_2_up_proj"] = layer_2_up_proj;
  char *layer_2_down_proj = (char*)(0x7bdd810c0000);
  all_tensors["layer_2_down_proj"] = layer_2_down_proj;
  char *layer_3_input_layernorm = (char*)(0x7bddc0a1e000);
  all_tensors["layer_3_input_layernorm"] = layer_3_input_layernorm;
  char *layer_3_q_proj = (char*)(0x7bdd936c0000);
  all_tensors["layer_3_q_proj"] = layer_3_q_proj;
  char *layer_3_k_proj = (char*)(0x7bdd930c0000);
  all_tensors["layer_3_k_proj"] = layer_3_k_proj;
  char *layer_3_v_proj = (char*)(0x7bdd93ac0000);
  all_tensors["layer_3_v_proj"] = layer_3_v_proj;
  char *layer_3_q_norm = (char*)(0x7bddc0a1f200);
  all_tensors["layer_3_q_norm"] = layer_3_q_norm;
  char *layer_3_k_norm = (char*)(0x7bddc0a1f000);
  all_tensors["layer_3_k_norm"] = layer_3_k_norm;
  char *layer_3_k_cache = (char*)(0x7bddd1800000);
  all_tensors["layer_3_k_cache"] = layer_3_k_cache;
  char *layer_3_v_cache = (char*)(0x7bddc3800000);
  all_tensors["layer_3_v_cache"] = layer_3_v_cache;
  char *layer_3_o_proj = (char*)(0x7bdd932c0000);
  all_tensors["layer_3_o_proj"] = layer_3_o_proj;
  char *layer_3_post_attn_layernorm = (char*)(0x7bddc0a1e800);
  all_tensors["layer_3_post_attn_layernorm"] = layer_3_post_attn_layernorm;
  char *layer_3_gate_proj = (char*)(0x7bdd924c0000);
  all_tensors["layer_3_gate_proj"] = layer_3_gate_proj;
  char *layer_3_up_proj = (char*)(0x7bdd92ac0000);
  all_tensors["layer_3_up_proj"] = layer_3_up_proj;
  char *layer_3_down_proj = (char*)(0x7bdd91ec0000);
  all_tensors["layer_3_down_proj"] = layer_3_down_proj;
  char *layer_4_input_layernorm = (char*)(0x7bddc0a1f400);
  all_tensors["layer_4_input_layernorm"] = layer_4_input_layernorm;
  char *layer_4_q_proj = (char*)(0x7bdd954c0000);
  all_tensors["layer_4_q_proj"] = layer_4_q_proj;
  char *layer_4_k_proj = (char*)(0x7bdd94ec0000);
  all_tensors["layer_4_k_proj"] = layer_4_k_proj;
  char *layer_4_v_proj = (char*)(0x7bdd958c0000);
  all_tensors["layer_4_v_proj"] = layer_4_v_proj;
  char *layer_4_q_norm = (char*)(0x7bddc0a20600);
  all_tensors["layer_4_q_norm"] = layer_4_q_norm;
  char *layer_4_k_norm = (char*)(0x7bddc0a20400);
  all_tensors["layer_4_k_norm"] = layer_4_k_norm;
  char *layer_4_k_cache = (char*)(0x7bddd2000000);
  all_tensors["layer_4_k_cache"] = layer_4_k_cache;
  char *layer_4_v_cache = (char*)(0x7bddc4000000);
  all_tensors["layer_4_v_cache"] = layer_4_v_cache;
  char *layer_4_o_proj = (char*)(0x7bdd950c0000);
  all_tensors["layer_4_o_proj"] = layer_4_o_proj;
  char *layer_4_post_attn_layernorm = (char*)(0x7bddc0a1fc00);
  all_tensors["layer_4_post_attn_layernorm"] = layer_4_post_attn_layernorm;
  char *layer_4_gate_proj = (char*)(0x7bdd942c0000);
  all_tensors["layer_4_gate_proj"] = layer_4_gate_proj;
  char *layer_4_up_proj = (char*)(0x7bdd948c0000);
  all_tensors["layer_4_up_proj"] = layer_4_up_proj;
  char *layer_4_down_proj = (char*)(0x7bdd93cc0000);
  all_tensors["layer_4_down_proj"] = layer_4_down_proj;
  char *layer_5_input_layernorm = (char*)(0x7bddc0a20800);
  all_tensors["layer_5_input_layernorm"] = layer_5_input_layernorm;
  char *layer_5_q_proj = (char*)(0x7bdd972c0000);
  all_tensors["layer_5_q_proj"] = layer_5_q_proj;
  char *layer_5_k_proj = (char*)(0x7bdd96cc0000);
  all_tensors["layer_5_k_proj"] = layer_5_k_proj;
  char *layer_5_v_proj = (char*)(0x7bdd976c0000);
  all_tensors["layer_5_v_proj"] = layer_5_v_proj;
  char *layer_5_q_norm = (char*)(0x7bddc0a21a00);
  all_tensors["layer_5_q_norm"] = layer_5_q_norm;
  char *layer_5_k_norm = (char*)(0x7bddc0a21800);
  all_tensors["layer_5_k_norm"] = layer_5_k_norm;
  char *layer_5_k_cache = (char*)(0x7bddd2800000);
  all_tensors["layer_5_k_cache"] = layer_5_k_cache;
  char *layer_5_v_cache = (char*)(0x7bddc4800000);
  all_tensors["layer_5_v_cache"] = layer_5_v_cache;
  char *layer_5_o_proj = (char*)(0x7bdd96ec0000);
  all_tensors["layer_5_o_proj"] = layer_5_o_proj;
  char *layer_5_post_attn_layernorm = (char*)(0x7bddc0a21000);
  all_tensors["layer_5_post_attn_layernorm"] = layer_5_post_attn_layernorm;
  char *layer_5_gate_proj = (char*)(0x7bdd960c0000);
  all_tensors["layer_5_gate_proj"] = layer_5_gate_proj;
  char *layer_5_up_proj = (char*)(0x7bdd966c0000);
  all_tensors["layer_5_up_proj"] = layer_5_up_proj;
  char *layer_5_down_proj = (char*)(0x7bdd95ac0000);
  all_tensors["layer_5_down_proj"] = layer_5_down_proj;
  char *layer_6_input_layernorm = (char*)(0x7bddc0a21c00);
  all_tensors["layer_6_input_layernorm"] = layer_6_input_layernorm;
  char *layer_6_q_proj = (char*)(0x7bdd990c0000);
  all_tensors["layer_6_q_proj"] = layer_6_q_proj;
  char *layer_6_k_proj = (char*)(0x7bdd98ac0000);
  all_tensors["layer_6_k_proj"] = layer_6_k_proj;
  char *layer_6_v_proj = (char*)(0x7bdd994c0000);
  all_tensors["layer_6_v_proj"] = layer_6_v_proj;
  char *layer_6_q_norm = (char*)(0x7bddc0a22e00);
  all_tensors["layer_6_q_norm"] = layer_6_q_norm;
  char *layer_6_k_norm = (char*)(0x7bddc0a22c00);
  all_tensors["layer_6_k_norm"] = layer_6_k_norm;
  char *layer_6_k_cache = (char*)(0x7bddd3000000);
  all_tensors["layer_6_k_cache"] = layer_6_k_cache;
  char *layer_6_v_cache = (char*)(0x7bddc5000000);
  all_tensors["layer_6_v_cache"] = layer_6_v_cache;
  char *layer_6_o_proj = (char*)(0x7bdd98cc0000);
  all_tensors["layer_6_o_proj"] = layer_6_o_proj;
  char *layer_6_post_attn_layernorm = (char*)(0x7bddc0a22400);
  all_tensors["layer_6_post_attn_layernorm"] = layer_6_post_attn_layernorm;
  char *layer_6_gate_proj = (char*)(0x7bdd97ec0000);
  all_tensors["layer_6_gate_proj"] = layer_6_gate_proj;
  char *layer_6_up_proj = (char*)(0x7bdd984c0000);
  all_tensors["layer_6_up_proj"] = layer_6_up_proj;
  char *layer_6_down_proj = (char*)(0x7bdd978c0000);
  all_tensors["layer_6_down_proj"] = layer_6_down_proj;
  char *layer_7_input_layernorm = (char*)(0x7bddc0a23000);
  all_tensors["layer_7_input_layernorm"] = layer_7_input_layernorm;
  char *layer_7_q_proj = (char*)(0x7bdd9aec0000);
  all_tensors["layer_7_q_proj"] = layer_7_q_proj;
  char *layer_7_k_proj = (char*)(0x7bdd9a8c0000);
  all_tensors["layer_7_k_proj"] = layer_7_k_proj;
  char *layer_7_v_proj = (char*)(0x7bdd9b2c0000);
  all_tensors["layer_7_v_proj"] = layer_7_v_proj;
  char *layer_7_q_norm = (char*)(0x7bddc0a24200);
  all_tensors["layer_7_q_norm"] = layer_7_q_norm;
  char *layer_7_k_norm = (char*)(0x7bddc0a24000);
  all_tensors["layer_7_k_norm"] = layer_7_k_norm;
  char *layer_7_k_cache = (char*)(0x7bddd3800000);
  all_tensors["layer_7_k_cache"] = layer_7_k_cache;
  char *layer_7_v_cache = (char*)(0x7bddc5800000);
  all_tensors["layer_7_v_cache"] = layer_7_v_cache;
  char *layer_7_o_proj = (char*)(0x7bdd9aac0000);
  all_tensors["layer_7_o_proj"] = layer_7_o_proj;
  char *layer_7_post_attn_layernorm = (char*)(0x7bddc0a23800);
  all_tensors["layer_7_post_attn_layernorm"] = layer_7_post_attn_layernorm;
  char *layer_7_gate_proj = (char*)(0x7bdd99cc0000);
  all_tensors["layer_7_gate_proj"] = layer_7_gate_proj;
  char *layer_7_up_proj = (char*)(0x7bdd9a2c0000);
  all_tensors["layer_7_up_proj"] = layer_7_up_proj;
  char *layer_7_down_proj = (char*)(0x7bdd996c0000);
  all_tensors["layer_7_down_proj"] = layer_7_down_proj;
  char *layer_8_input_layernorm = (char*)(0x7bddc0a24400);
  all_tensors["layer_8_input_layernorm"] = layer_8_input_layernorm;
  char *layer_8_q_proj = (char*)(0x7bdd9ccc0000);
  all_tensors["layer_8_q_proj"] = layer_8_q_proj;
  char *layer_8_k_proj = (char*)(0x7bdd9c6c0000);
  all_tensors["layer_8_k_proj"] = layer_8_k_proj;
  char *layer_8_v_proj = (char*)(0x7bdd9d0c0000);
  all_tensors["layer_8_v_proj"] = layer_8_v_proj;
  char *layer_8_q_norm = (char*)(0x7bddc0a25600);
  all_tensors["layer_8_q_norm"] = layer_8_q_norm;
  char *layer_8_k_norm = (char*)(0x7bddc0a25400);
  all_tensors["layer_8_k_norm"] = layer_8_k_norm;
  char *layer_8_k_cache = (char*)(0x7bddd4000000);
  all_tensors["layer_8_k_cache"] = layer_8_k_cache;
  char *layer_8_v_cache = (char*)(0x7bddc6000000);
  all_tensors["layer_8_v_cache"] = layer_8_v_cache;
  char *layer_8_o_proj = (char*)(0x7bdd9c8c0000);
  all_tensors["layer_8_o_proj"] = layer_8_o_proj;
  char *layer_8_post_attn_layernorm = (char*)(0x7bddc0a24c00);
  all_tensors["layer_8_post_attn_layernorm"] = layer_8_post_attn_layernorm;
  char *layer_8_gate_proj = (char*)(0x7bdd9bac0000);
  all_tensors["layer_8_gate_proj"] = layer_8_gate_proj;
  char *layer_8_up_proj = (char*)(0x7bdd9c0c0000);
  all_tensors["layer_8_up_proj"] = layer_8_up_proj;
  char *layer_8_down_proj = (char*)(0x7bdd9b4c0000);
  all_tensors["layer_8_down_proj"] = layer_8_down_proj;
  char *layer_9_input_layernorm = (char*)(0x7bddc0a25800);
  all_tensors["layer_9_input_layernorm"] = layer_9_input_layernorm;
  char *layer_9_q_proj = (char*)(0x7bddac800000);
  all_tensors["layer_9_q_proj"] = layer_9_q_proj;
  char *layer_9_k_proj = (char*)(0x7bdd9e4c0000);
  all_tensors["layer_9_k_proj"] = layer_9_k_proj;
  char *layer_9_v_proj = (char*)(0x7bdd9eac0000);
  all_tensors["layer_9_v_proj"] = layer_9_v_proj;
  char *layer_9_q_norm = (char*)(0x7bddc0a26a00);
  all_tensors["layer_9_q_norm"] = layer_9_q_norm;
  char *layer_9_k_norm = (char*)(0x7bddc0a26800);
  all_tensors["layer_9_k_norm"] = layer_9_k_norm;
  char *layer_9_k_cache = (char*)(0x7bddd4800000);
  all_tensors["layer_9_k_cache"] = layer_9_k_cache;
  char *layer_9_v_cache = (char*)(0x7bddc6800000);
  all_tensors["layer_9_v_cache"] = layer_9_v_cache;
  char *layer_9_o_proj = (char*)(0x7bdd9e6c0000);
  all_tensors["layer_9_o_proj"] = layer_9_o_proj;
  char *layer_9_post_attn_layernorm = (char*)(0x7bddc0a26000);
  all_tensors["layer_9_post_attn_layernorm"] = layer_9_post_attn_layernorm;
  char *layer_9_gate_proj = (char*)(0x7bdd9d8c0000);
  all_tensors["layer_9_gate_proj"] = layer_9_gate_proj;
  char *layer_9_up_proj = (char*)(0x7bdd9dec0000);
  all_tensors["layer_9_up_proj"] = layer_9_up_proj;
  char *layer_9_down_proj = (char*)(0x7bdd9d2c0000);
  all_tensors["layer_9_down_proj"] = layer_9_down_proj;
  char *layer_10_input_layernorm = (char*)(0x7bddc0a06400);
  all_tensors["layer_10_input_layernorm"] = layer_10_input_layernorm;
  char *layer_10_q_proj = (char*)(0x7bdd6fcc0000);
  all_tensors["layer_10_q_proj"] = layer_10_q_proj;
  char *layer_10_k_proj = (char*)(0x7bdd6f6c0000);
  all_tensors["layer_10_k_proj"] = layer_10_k_proj;
  char *layer_10_v_proj = (char*)(0x7bdd700c0000);
  all_tensors["layer_10_v_proj"] = layer_10_v_proj;
  char *layer_10_q_norm = (char*)(0x7bddc0a07600);
  all_tensors["layer_10_q_norm"] = layer_10_q_norm;
  char *layer_10_k_norm = (char*)(0x7bddc0a07400);
  all_tensors["layer_10_k_norm"] = layer_10_k_norm;
  char *layer_10_k_cache = (char*)(0x7bddd5000000);
  all_tensors["layer_10_k_cache"] = layer_10_k_cache;
  char *layer_10_v_cache = (char*)(0x7bddc7000000);
  all_tensors["layer_10_v_cache"] = layer_10_v_cache;
  char *layer_10_o_proj = (char*)(0x7bdd6f8c0000);
  all_tensors["layer_10_o_proj"] = layer_10_o_proj;
  char *layer_10_post_attn_layernorm = (char*)(0x7bddc0a06c00);
  all_tensors["layer_10_post_attn_layernorm"] = layer_10_post_attn_layernorm;
  char *layer_10_gate_proj = (char*)(0x7bdd6eac0000);
  all_tensors["layer_10_gate_proj"] = layer_10_gate_proj;
  char *layer_10_up_proj = (char*)(0x7bdd6f0c0000);
  all_tensors["layer_10_up_proj"] = layer_10_up_proj;
  char *layer_10_down_proj = (char*)(0x7bdd6e4c0000);
  all_tensors["layer_10_down_proj"] = layer_10_down_proj;
  char *layer_11_input_layernorm = (char*)(0x7bddc0a07800);
  all_tensors["layer_11_input_layernorm"] = layer_11_input_layernorm;
  char *layer_11_q_proj = (char*)(0x7bdd71ac0000);
  all_tensors["layer_11_q_proj"] = layer_11_q_proj;
  char *layer_11_k_proj = (char*)(0x7bdd714c0000);
  all_tensors["layer_11_k_proj"] = layer_11_k_proj;
  char *layer_11_v_proj = (char*)(0x7bdd71ec0000);
  all_tensors["layer_11_v_proj"] = layer_11_v_proj;
  char *layer_11_q_norm = (char*)(0x7bddc0a08a00);
  all_tensors["layer_11_q_norm"] = layer_11_q_norm;
  char *layer_11_k_norm = (char*)(0x7bddc0a08800);
  all_tensors["layer_11_k_norm"] = layer_11_k_norm;
  char *layer_11_k_cache = (char*)(0x7bddd5800000);
  all_tensors["layer_11_k_cache"] = layer_11_k_cache;
  char *layer_11_v_cache = (char*)(0x7bddc7800000);
  all_tensors["layer_11_v_cache"] = layer_11_v_cache;
  char *layer_11_o_proj = (char*)(0x7bdd716c0000);
  all_tensors["layer_11_o_proj"] = layer_11_o_proj;
  char *layer_11_post_attn_layernorm = (char*)(0x7bddc0a08000);
  all_tensors["layer_11_post_attn_layernorm"] = layer_11_post_attn_layernorm;
  char *layer_11_gate_proj = (char*)(0x7bdd708c0000);
  all_tensors["layer_11_gate_proj"] = layer_11_gate_proj;
  char *layer_11_up_proj = (char*)(0x7bdd70ec0000);
  all_tensors["layer_11_up_proj"] = layer_11_up_proj;
  char *layer_11_down_proj = (char*)(0x7bdd702c0000);
  all_tensors["layer_11_down_proj"] = layer_11_down_proj;
  char *layer_12_input_layernorm = (char*)(0x7bddc0a08c00);
  all_tensors["layer_12_input_layernorm"] = layer_12_input_layernorm;
  char *layer_12_q_proj = (char*)(0x7bdd738c0000);
  all_tensors["layer_12_q_proj"] = layer_12_q_proj;
  char *layer_12_k_proj = (char*)(0x7bdd732c0000);
  all_tensors["layer_12_k_proj"] = layer_12_k_proj;
  char *layer_12_v_proj = (char*)(0x7bdd73cc0000);
  all_tensors["layer_12_v_proj"] = layer_12_v_proj;
  char *layer_12_q_norm = (char*)(0x7bddc0a09e00);
  all_tensors["layer_12_q_norm"] = layer_12_q_norm;
  char *layer_12_k_norm = (char*)(0x7bddc0a09c00);
  all_tensors["layer_12_k_norm"] = layer_12_k_norm;
  char *layer_12_k_cache = (char*)(0x7bddd6000000);
  all_tensors["layer_12_k_cache"] = layer_12_k_cache;
  char *layer_12_v_cache = (char*)(0x7bddc8000000);
  all_tensors["layer_12_v_cache"] = layer_12_v_cache;
  char *layer_12_o_proj = (char*)(0x7bdd734c0000);
  all_tensors["layer_12_o_proj"] = layer_12_o_proj;
  char *layer_12_post_attn_layernorm = (char*)(0x7bddc0a09400);
  all_tensors["layer_12_post_attn_layernorm"] = layer_12_post_attn_layernorm;
  char *layer_12_gate_proj = (char*)(0x7bdd726c0000);
  all_tensors["layer_12_gate_proj"] = layer_12_gate_proj;
  char *layer_12_up_proj = (char*)(0x7bdd72cc0000);
  all_tensors["layer_12_up_proj"] = layer_12_up_proj;
  char *layer_12_down_proj = (char*)(0x7bdd720c0000);
  all_tensors["layer_12_down_proj"] = layer_12_down_proj;
  char *layer_13_input_layernorm = (char*)(0x7bddc0a0a000);
  all_tensors["layer_13_input_layernorm"] = layer_13_input_layernorm;
  char *layer_13_q_proj = (char*)(0x7bdd756c0000);
  all_tensors["layer_13_q_proj"] = layer_13_q_proj;
  char *layer_13_k_proj = (char*)(0x7bdd750c0000);
  all_tensors["layer_13_k_proj"] = layer_13_k_proj;
  char *layer_13_v_proj = (char*)(0x7bdd75ac0000);
  all_tensors["layer_13_v_proj"] = layer_13_v_proj;
  char *layer_13_q_norm = (char*)(0x7bddc0a0b200);
  all_tensors["layer_13_q_norm"] = layer_13_q_norm;
  char *layer_13_k_norm = (char*)(0x7bddc0a0b000);
  all_tensors["layer_13_k_norm"] = layer_13_k_norm;
  char *layer_13_k_cache = (char*)(0x7bddd6800000);
  all_tensors["layer_13_k_cache"] = layer_13_k_cache;
  char *layer_13_v_cache = (char*)(0x7bddc8800000);
  all_tensors["layer_13_v_cache"] = layer_13_v_cache;
  char *layer_13_o_proj = (char*)(0x7bdd752c0000);
  all_tensors["layer_13_o_proj"] = layer_13_o_proj;
  char *layer_13_post_attn_layernorm = (char*)(0x7bddc0a0a800);
  all_tensors["layer_13_post_attn_layernorm"] = layer_13_post_attn_layernorm;
  char *layer_13_gate_proj = (char*)(0x7bdd744c0000);
  all_tensors["layer_13_gate_proj"] = layer_13_gate_proj;
  char *layer_13_up_proj = (char*)(0x7bdd74ac0000);
  all_tensors["layer_13_up_proj"] = layer_13_up_proj;
  char *layer_13_down_proj = (char*)(0x7bdd73ec0000);
  all_tensors["layer_13_down_proj"] = layer_13_down_proj;
  char *layer_14_input_layernorm = (char*)(0x7bddc0a0b400);
  all_tensors["layer_14_input_layernorm"] = layer_14_input_layernorm;
  char *layer_14_q_proj = (char*)(0x7bdd774c0000);
  all_tensors["layer_14_q_proj"] = layer_14_q_proj;
  char *layer_14_k_proj = (char*)(0x7bdd76ec0000);
  all_tensors["layer_14_k_proj"] = layer_14_k_proj;
  char *layer_14_v_proj = (char*)(0x7bdd778c0000);
  all_tensors["layer_14_v_proj"] = layer_14_v_proj;
  char *layer_14_q_norm = (char*)(0x7bddc0a0c600);
  all_tensors["layer_14_q_norm"] = layer_14_q_norm;
  char *layer_14_k_norm = (char*)(0x7bddc0a0c400);
  all_tensors["layer_14_k_norm"] = layer_14_k_norm;
  char *layer_14_k_cache = (char*)(0x7bddd7000000);
  all_tensors["layer_14_k_cache"] = layer_14_k_cache;
  char *layer_14_v_cache = (char*)(0x7bddc9000000);
  all_tensors["layer_14_v_cache"] = layer_14_v_cache;
  char *layer_14_o_proj = (char*)(0x7bdd770c0000);
  all_tensors["layer_14_o_proj"] = layer_14_o_proj;
  char *layer_14_post_attn_layernorm = (char*)(0x7bddc0a0bc00);
  all_tensors["layer_14_post_attn_layernorm"] = layer_14_post_attn_layernorm;
  char *layer_14_gate_proj = (char*)(0x7bdd762c0000);
  all_tensors["layer_14_gate_proj"] = layer_14_gate_proj;
  char *layer_14_up_proj = (char*)(0x7bdd768c0000);
  all_tensors["layer_14_up_proj"] = layer_14_up_proj;
  char *layer_14_down_proj = (char*)(0x7bdd75cc0000);
  all_tensors["layer_14_down_proj"] = layer_14_down_proj;
  char *layer_15_input_layernorm = (char*)(0x7bddc0a0c800);
  all_tensors["layer_15_input_layernorm"] = layer_15_input_layernorm;
  char *layer_15_q_proj = (char*)(0x7bdd792c0000);
  all_tensors["layer_15_q_proj"] = layer_15_q_proj;
  char *layer_15_k_proj = (char*)(0x7bdd78cc0000);
  all_tensors["layer_15_k_proj"] = layer_15_k_proj;
  char *layer_15_v_proj = (char*)(0x7bdd796c0000);
  all_tensors["layer_15_v_proj"] = layer_15_v_proj;
  char *layer_15_q_norm = (char*)(0x7bddc0a0da00);
  all_tensors["layer_15_q_norm"] = layer_15_q_norm;
  char *layer_15_k_norm = (char*)(0x7bddc0a0d800);
  all_tensors["layer_15_k_norm"] = layer_15_k_norm;
  char *layer_15_k_cache = (char*)(0x7bddd7800000);
  all_tensors["layer_15_k_cache"] = layer_15_k_cache;
  char *layer_15_v_cache = (char*)(0x7bddc9800000);
  all_tensors["layer_15_v_cache"] = layer_15_v_cache;
  char *layer_15_o_proj = (char*)(0x7bdd78ec0000);
  all_tensors["layer_15_o_proj"] = layer_15_o_proj;
  char *layer_15_post_attn_layernorm = (char*)(0x7bddc0a0d000);
  all_tensors["layer_15_post_attn_layernorm"] = layer_15_post_attn_layernorm;
  char *layer_15_gate_proj = (char*)(0x7bdd780c0000);
  all_tensors["layer_15_gate_proj"] = layer_15_gate_proj;
  char *layer_15_up_proj = (char*)(0x7bdd786c0000);
  all_tensors["layer_15_up_proj"] = layer_15_up_proj;
  char *layer_15_down_proj = (char*)(0x7bdd77ac0000);
  all_tensors["layer_15_down_proj"] = layer_15_down_proj;
  char *layer_16_input_layernorm = (char*)(0x7bddc0a0dc00);
  all_tensors["layer_16_input_layernorm"] = layer_16_input_layernorm;
  char *layer_16_q_proj = (char*)(0x7bdd7b0c0000);
  all_tensors["layer_16_q_proj"] = layer_16_q_proj;
  char *layer_16_k_proj = (char*)(0x7bdd7aac0000);
  all_tensors["layer_16_k_proj"] = layer_16_k_proj;
  char *layer_16_v_proj = (char*)(0x7bdd7b4c0000);
  all_tensors["layer_16_v_proj"] = layer_16_v_proj;
  char *layer_16_q_norm = (char*)(0x7bddc0a0ee00);
  all_tensors["layer_16_q_norm"] = layer_16_q_norm;
  char *layer_16_k_norm = (char*)(0x7bddc0a0ec00);
  all_tensors["layer_16_k_norm"] = layer_16_k_norm;
  char *layer_16_k_cache = (char*)(0x7bddd8000000);
  all_tensors["layer_16_k_cache"] = layer_16_k_cache;
  char *layer_16_v_cache = (char*)(0x7bddca000000);
  all_tensors["layer_16_v_cache"] = layer_16_v_cache;
  char *layer_16_o_proj = (char*)(0x7bdd7acc0000);
  all_tensors["layer_16_o_proj"] = layer_16_o_proj;
  char *layer_16_post_attn_layernorm = (char*)(0x7bddc0a0e400);
  all_tensors["layer_16_post_attn_layernorm"] = layer_16_post_attn_layernorm;
  char *layer_16_gate_proj = (char*)(0x7bdd79ec0000);
  all_tensors["layer_16_gate_proj"] = layer_16_gate_proj;
  char *layer_16_up_proj = (char*)(0x7bdd7a4c0000);
  all_tensors["layer_16_up_proj"] = layer_16_up_proj;
  char *layer_16_down_proj = (char*)(0x7bdd798c0000);
  all_tensors["layer_16_down_proj"] = layer_16_down_proj;
  char *layer_17_input_layernorm = (char*)(0x7bddc0a0f000);
  all_tensors["layer_17_input_layernorm"] = layer_17_input_layernorm;
  char *layer_17_q_proj = (char*)(0x7bdd7cec0000);
  all_tensors["layer_17_q_proj"] = layer_17_q_proj;
  char *layer_17_k_proj = (char*)(0x7bdd7c8c0000);
  all_tensors["layer_17_k_proj"] = layer_17_k_proj;
  char *layer_17_v_proj = (char*)(0x7bdd7d2c0000);
  all_tensors["layer_17_v_proj"] = layer_17_v_proj;
  char *layer_17_q_norm = (char*)(0x7bddc0a10200);
  all_tensors["layer_17_q_norm"] = layer_17_q_norm;
  char *layer_17_k_norm = (char*)(0x7bddc0a10000);
  all_tensors["layer_17_k_norm"] = layer_17_k_norm;
  char *layer_17_k_cache = (char*)(0x7bddd8800000);
  all_tensors["layer_17_k_cache"] = layer_17_k_cache;
  char *layer_17_v_cache = (char*)(0x7bddca800000);
  all_tensors["layer_17_v_cache"] = layer_17_v_cache;
  char *layer_17_o_proj = (char*)(0x7bdd7cac0000);
  all_tensors["layer_17_o_proj"] = layer_17_o_proj;
  char *layer_17_post_attn_layernorm = (char*)(0x7bddc0a0f800);
  all_tensors["layer_17_post_attn_layernorm"] = layer_17_post_attn_layernorm;
  char *layer_17_gate_proj = (char*)(0x7bdd7bcc0000);
  all_tensors["layer_17_gate_proj"] = layer_17_gate_proj;
  char *layer_17_up_proj = (char*)(0x7bdd7c2c0000);
  all_tensors["layer_17_up_proj"] = layer_17_up_proj;
  char *layer_17_down_proj = (char*)(0x7bdd7b6c0000);
  all_tensors["layer_17_down_proj"] = layer_17_down_proj;
  char *layer_18_input_layernorm = (char*)(0x7bddc0a10400);
  all_tensors["layer_18_input_layernorm"] = layer_18_input_layernorm;
  char *layer_18_q_proj = (char*)(0x7bdd7ecc0000);
  all_tensors["layer_18_q_proj"] = layer_18_q_proj;
  char *layer_18_k_proj = (char*)(0x7bdd7e6c0000);
  all_tensors["layer_18_k_proj"] = layer_18_k_proj;
  char *layer_18_v_proj = (char*)(0x7bdd7f0c0000);
  all_tensors["layer_18_v_proj"] = layer_18_v_proj;
  char *layer_18_q_norm = (char*)(0x7bddc0a11600);
  all_tensors["layer_18_q_norm"] = layer_18_q_norm;
  char *layer_18_k_norm = (char*)(0x7bddc0a11400);
  all_tensors["layer_18_k_norm"] = layer_18_k_norm;
  char *layer_18_k_cache = (char*)(0x7bddd9000000);
  all_tensors["layer_18_k_cache"] = layer_18_k_cache;
  char *layer_18_v_cache = (char*)(0x7bddcb000000);
  all_tensors["layer_18_v_cache"] = layer_18_v_cache;
  char *layer_18_o_proj = (char*)(0x7bdd7e8c0000);
  all_tensors["layer_18_o_proj"] = layer_18_o_proj;
  char *layer_18_post_attn_layernorm = (char*)(0x7bddc0a10c00);
  all_tensors["layer_18_post_attn_layernorm"] = layer_18_post_attn_layernorm;
  char *layer_18_gate_proj = (char*)(0x7bdd7dac0000);
  all_tensors["layer_18_gate_proj"] = layer_18_gate_proj;
  char *layer_18_up_proj = (char*)(0x7bdd7e0c0000);
  all_tensors["layer_18_up_proj"] = layer_18_up_proj;
  char *layer_18_down_proj = (char*)(0x7bdd7d4c0000);
  all_tensors["layer_18_down_proj"] = layer_18_down_proj;
  char *layer_19_input_layernorm = (char*)(0x7bddc0a11800);
  all_tensors["layer_19_input_layernorm"] = layer_19_input_layernorm;
  char *layer_19_q_proj = (char*)(0x7bdd80ac0000);
  all_tensors["layer_19_q_proj"] = layer_19_q_proj;
  char *layer_19_k_proj = (char*)(0x7bdd804c0000);
  all_tensors["layer_19_k_proj"] = layer_19_k_proj;
  char *layer_19_v_proj = (char*)(0x7bdd80ec0000);
  all_tensors["layer_19_v_proj"] = layer_19_v_proj;
  char *layer_19_q_norm = (char*)(0x7bddc0a12a00);
  all_tensors["layer_19_q_norm"] = layer_19_q_norm;
  char *layer_19_k_norm = (char*)(0x7bddc0a12800);
  all_tensors["layer_19_k_norm"] = layer_19_k_norm;
  char *layer_19_k_cache = (char*)(0x7bddd9800000);
  all_tensors["layer_19_k_cache"] = layer_19_k_cache;
  char *layer_19_v_cache = (char*)(0x7bddcb800000);
  all_tensors["layer_19_v_cache"] = layer_19_v_cache;
  char *layer_19_o_proj = (char*)(0x7bdd806c0000);
  all_tensors["layer_19_o_proj"] = layer_19_o_proj;
  char *layer_19_post_attn_layernorm = (char*)(0x7bddc0a12000);
  all_tensors["layer_19_post_attn_layernorm"] = layer_19_post_attn_layernorm;
  char *layer_19_gate_proj = (char*)(0x7bdd7f8c0000);
  all_tensors["layer_19_gate_proj"] = layer_19_gate_proj;
  char *layer_19_up_proj = (char*)(0x7bdd7fec0000);
  all_tensors["layer_19_up_proj"] = layer_19_up_proj;
  char *layer_19_down_proj = (char*)(0x7bdd7f2c0000);
  all_tensors["layer_19_down_proj"] = layer_19_down_proj;
  char *layer_20_input_layernorm = (char*)(0x7bddc0a14000);
  all_tensors["layer_20_input_layernorm"] = layer_20_input_layernorm;
  char *layer_20_q_proj = (char*)(0x7bdd846c0000);
  all_tensors["layer_20_q_proj"] = layer_20_q_proj;
  char *layer_20_k_proj = (char*)(0x7bdd840c0000);
  all_tensors["layer_20_k_proj"] = layer_20_k_proj;
  char *layer_20_v_proj = (char*)(0x7bdd84ac0000);
  all_tensors["layer_20_v_proj"] = layer_20_v_proj;
  char *layer_20_q_norm = (char*)(0x7bddc0a15200);
  all_tensors["layer_20_q_norm"] = layer_20_q_norm;
  char *layer_20_k_norm = (char*)(0x7bddc0a15000);
  all_tensors["layer_20_k_norm"] = layer_20_k_norm;
  char *layer_20_k_cache = (char*)(0x7bddda000000);
  all_tensors["layer_20_k_cache"] = layer_20_k_cache;
  char *layer_20_v_cache = (char*)(0x7bddcc000000);
  all_tensors["layer_20_v_cache"] = layer_20_v_cache;
  char *layer_20_o_proj = (char*)(0x7bdd842c0000);
  all_tensors["layer_20_o_proj"] = layer_20_o_proj;
  char *layer_20_post_attn_layernorm = (char*)(0x7bddc0a14800);
  all_tensors["layer_20_post_attn_layernorm"] = layer_20_post_attn_layernorm;
  char *layer_20_gate_proj = (char*)(0x7bdd834c0000);
  all_tensors["layer_20_gate_proj"] = layer_20_gate_proj;
  char *layer_20_up_proj = (char*)(0x7bdd83ac0000);
  all_tensors["layer_20_up_proj"] = layer_20_up_proj;
  char *layer_20_down_proj = (char*)(0x7bdd82ec0000);
  all_tensors["layer_20_down_proj"] = layer_20_down_proj;
  char *layer_21_input_layernorm = (char*)(0x7bddc0a15400);
  all_tensors["layer_21_input_layernorm"] = layer_21_input_layernorm;
  char *layer_21_q_proj = (char*)(0x7bdd864c0000);
  all_tensors["layer_21_q_proj"] = layer_21_q_proj;
  char *layer_21_k_proj = (char*)(0x7bdd85ec0000);
  all_tensors["layer_21_k_proj"] = layer_21_k_proj;
  char *layer_21_v_proj = (char*)(0x7bdd868c0000);
  all_tensors["layer_21_v_proj"] = layer_21_v_proj;
  char *layer_21_q_norm = (char*)(0x7bddc0a16600);
  all_tensors["layer_21_q_norm"] = layer_21_q_norm;
  char *layer_21_k_norm = (char*)(0x7bddc0a16400);
  all_tensors["layer_21_k_norm"] = layer_21_k_norm;
  char *layer_21_k_cache = (char*)(0x7bddda800000);
  all_tensors["layer_21_k_cache"] = layer_21_k_cache;
  char *layer_21_v_cache = (char*)(0x7bddcc800000);
  all_tensors["layer_21_v_cache"] = layer_21_v_cache;
  char *layer_21_o_proj = (char*)(0x7bdd860c0000);
  all_tensors["layer_21_o_proj"] = layer_21_o_proj;
  char *layer_21_post_attn_layernorm = (char*)(0x7bddc0a15c00);
  all_tensors["layer_21_post_attn_layernorm"] = layer_21_post_attn_layernorm;
  char *layer_21_gate_proj = (char*)(0x7bdd852c0000);
  all_tensors["layer_21_gate_proj"] = layer_21_gate_proj;
  char *layer_21_up_proj = (char*)(0x7bdd858c0000);
  all_tensors["layer_21_up_proj"] = layer_21_up_proj;
  char *layer_21_down_proj = (char*)(0x7bdd84cc0000);
  all_tensors["layer_21_down_proj"] = layer_21_down_proj;
  char *layer_22_input_layernorm = (char*)(0x7bddc0a16800);
  all_tensors["layer_22_input_layernorm"] = layer_22_input_layernorm;
  char *layer_22_q_proj = (char*)(0x7bdd882c0000);
  all_tensors["layer_22_q_proj"] = layer_22_q_proj;
  char *layer_22_k_proj = (char*)(0x7bdd87cc0000);
  all_tensors["layer_22_k_proj"] = layer_22_k_proj;
  char *layer_22_v_proj = (char*)(0x7bdd886c0000);
  all_tensors["layer_22_v_proj"] = layer_22_v_proj;
  char *layer_22_q_norm = (char*)(0x7bddc0a17a00);
  all_tensors["layer_22_q_norm"] = layer_22_q_norm;
  char *layer_22_k_norm = (char*)(0x7bddc0a17800);
  all_tensors["layer_22_k_norm"] = layer_22_k_norm;
  char *layer_22_k_cache = (char*)(0x7bdddb000000);
  all_tensors["layer_22_k_cache"] = layer_22_k_cache;
  char *layer_22_v_cache = (char*)(0x7bddcd000000);
  all_tensors["layer_22_v_cache"] = layer_22_v_cache;
  char *layer_22_o_proj = (char*)(0x7bdd87ec0000);
  all_tensors["layer_22_o_proj"] = layer_22_o_proj;
  char *layer_22_post_attn_layernorm = (char*)(0x7bddc0a17000);
  all_tensors["layer_22_post_attn_layernorm"] = layer_22_post_attn_layernorm;
  char *layer_22_gate_proj = (char*)(0x7bdd870c0000);
  all_tensors["layer_22_gate_proj"] = layer_22_gate_proj;
  char *layer_22_up_proj = (char*)(0x7bdd876c0000);
  all_tensors["layer_22_up_proj"] = layer_22_up_proj;
  char *layer_22_down_proj = (char*)(0x7bdd86ac0000);
  all_tensors["layer_22_down_proj"] = layer_22_down_proj;
  char *layer_23_input_layernorm = (char*)(0x7bddc0a17c00);
  all_tensors["layer_23_input_layernorm"] = layer_23_input_layernorm;
  char *layer_23_q_proj = (char*)(0x7bdd8a0c0000);
  all_tensors["layer_23_q_proj"] = layer_23_q_proj;
  char *layer_23_k_proj = (char*)(0x7bdd89ac0000);
  all_tensors["layer_23_k_proj"] = layer_23_k_proj;
  char *layer_23_v_proj = (char*)(0x7bdd8a4c0000);
  all_tensors["layer_23_v_proj"] = layer_23_v_proj;
  char *layer_23_q_norm = (char*)(0x7bddc0a18e00);
  all_tensors["layer_23_q_norm"] = layer_23_q_norm;
  char *layer_23_k_norm = (char*)(0x7bddc0a18c00);
  all_tensors["layer_23_k_norm"] = layer_23_k_norm;
  char *layer_23_k_cache = (char*)(0x7bdddb800000);
  all_tensors["layer_23_k_cache"] = layer_23_k_cache;
  char *layer_23_v_cache = (char*)(0x7bddcd800000);
  all_tensors["layer_23_v_cache"] = layer_23_v_cache;
  char *layer_23_o_proj = (char*)(0x7bdd89cc0000);
  all_tensors["layer_23_o_proj"] = layer_23_o_proj;
  char *layer_23_post_attn_layernorm = (char*)(0x7bddc0a18400);
  all_tensors["layer_23_post_attn_layernorm"] = layer_23_post_attn_layernorm;
  char *layer_23_gate_proj = (char*)(0x7bdd88ec0000);
  all_tensors["layer_23_gate_proj"] = layer_23_gate_proj;
  char *layer_23_up_proj = (char*)(0x7bdd894c0000);
  all_tensors["layer_23_up_proj"] = layer_23_up_proj;
  char *layer_23_down_proj = (char*)(0x7bdd888c0000);
  all_tensors["layer_23_down_proj"] = layer_23_down_proj;
  char *layer_24_input_layernorm = (char*)(0x7bddc0a19000);
  all_tensors["layer_24_input_layernorm"] = layer_24_input_layernorm;
  char *layer_24_q_proj = (char*)(0x7bdd8bec0000);
  all_tensors["layer_24_q_proj"] = layer_24_q_proj;
  char *layer_24_k_proj = (char*)(0x7bdd8b8c0000);
  all_tensors["layer_24_k_proj"] = layer_24_k_proj;
  char *layer_24_v_proj = (char*)(0x7bdd8c2c0000);
  all_tensors["layer_24_v_proj"] = layer_24_v_proj;
  char *layer_24_q_norm = (char*)(0x7bddc0a1a200);
  all_tensors["layer_24_q_norm"] = layer_24_q_norm;
  char *layer_24_k_norm = (char*)(0x7bddc0a1a000);
  all_tensors["layer_24_k_norm"] = layer_24_k_norm;
  char *layer_24_k_cache = (char*)(0x7bdddc000000);
  all_tensors["layer_24_k_cache"] = layer_24_k_cache;
  char *layer_24_v_cache = (char*)(0x7bddce000000);
  all_tensors["layer_24_v_cache"] = layer_24_v_cache;
  char *layer_24_o_proj = (char*)(0x7bdd8bac0000);
  all_tensors["layer_24_o_proj"] = layer_24_o_proj;
  char *layer_24_post_attn_layernorm = (char*)(0x7bddc0a19800);
  all_tensors["layer_24_post_attn_layernorm"] = layer_24_post_attn_layernorm;
  char *layer_24_gate_proj = (char*)(0x7bdd8acc0000);
  all_tensors["layer_24_gate_proj"] = layer_24_gate_proj;
  char *layer_24_up_proj = (char*)(0x7bdd8b2c0000);
  all_tensors["layer_24_up_proj"] = layer_24_up_proj;
  char *layer_24_down_proj = (char*)(0x7bdd8a6c0000);
  all_tensors["layer_24_down_proj"] = layer_24_down_proj;
  char *layer_25_input_layernorm = (char*)(0x7bddc0a1a400);
  all_tensors["layer_25_input_layernorm"] = layer_25_input_layernorm;
  char *layer_25_q_proj = (char*)(0x7bdd8dcc0000);
  all_tensors["layer_25_q_proj"] = layer_25_q_proj;
  char *layer_25_k_proj = (char*)(0x7bdd8d6c0000);
  all_tensors["layer_25_k_proj"] = layer_25_k_proj;
  char *layer_25_v_proj = (char*)(0x7bdd8e0c0000);
  all_tensors["layer_25_v_proj"] = layer_25_v_proj;
  char *layer_25_q_norm = (char*)(0x7bddc0a1b600);
  all_tensors["layer_25_q_norm"] = layer_25_q_norm;
  char *layer_25_k_norm = (char*)(0x7bddc0a1b400);
  all_tensors["layer_25_k_norm"] = layer_25_k_norm;
  char *layer_25_k_cache = (char*)(0x7bdddc800000);
  all_tensors["layer_25_k_cache"] = layer_25_k_cache;
  char *layer_25_v_cache = (char*)(0x7bddce800000);
  all_tensors["layer_25_v_cache"] = layer_25_v_cache;
  char *layer_25_o_proj = (char*)(0x7bdd8d8c0000);
  all_tensors["layer_25_o_proj"] = layer_25_o_proj;
  char *layer_25_post_attn_layernorm = (char*)(0x7bddc0a1ac00);
  all_tensors["layer_25_post_attn_layernorm"] = layer_25_post_attn_layernorm;
  char *layer_25_gate_proj = (char*)(0x7bdd8cac0000);
  all_tensors["layer_25_gate_proj"] = layer_25_gate_proj;
  char *layer_25_up_proj = (char*)(0x7bdd8d0c0000);
  all_tensors["layer_25_up_proj"] = layer_25_up_proj;
  char *layer_25_down_proj = (char*)(0x7bdd8c4c0000);
  all_tensors["layer_25_down_proj"] = layer_25_down_proj;
  char *layer_26_input_layernorm = (char*)(0x7bddc0a1b800);
  all_tensors["layer_26_input_layernorm"] = layer_26_input_layernorm;
  char *layer_26_q_proj = (char*)(0x7bdd8fac0000);
  all_tensors["layer_26_q_proj"] = layer_26_q_proj;
  char *layer_26_k_proj = (char*)(0x7bdd8f4c0000);
  all_tensors["layer_26_k_proj"] = layer_26_k_proj;
  char *layer_26_v_proj = (char*)(0x7bdd8fec0000);
  all_tensors["layer_26_v_proj"] = layer_26_v_proj;
  char *layer_26_q_norm = (char*)(0x7bddc0a1ca00);
  all_tensors["layer_26_q_norm"] = layer_26_q_norm;
  char *layer_26_k_norm = (char*)(0x7bddc0a1c800);
  all_tensors["layer_26_k_norm"] = layer_26_k_norm;
  char *layer_26_k_cache = (char*)(0x7bdddd000000);
  all_tensors["layer_26_k_cache"] = layer_26_k_cache;
  char *layer_26_v_cache = (char*)(0x7bddcf000000);
  all_tensors["layer_26_v_cache"] = layer_26_v_cache;
  char *layer_26_o_proj = (char*)(0x7bdd8f6c0000);
  all_tensors["layer_26_o_proj"] = layer_26_o_proj;
  char *layer_26_post_attn_layernorm = (char*)(0x7bddc0a1c000);
  all_tensors["layer_26_post_attn_layernorm"] = layer_26_post_attn_layernorm;
  char *layer_26_gate_proj = (char*)(0x7bdd8e8c0000);
  all_tensors["layer_26_gate_proj"] = layer_26_gate_proj;
  char *layer_26_up_proj = (char*)(0x7bdd8eec0000);
  all_tensors["layer_26_up_proj"] = layer_26_up_proj;
  char *layer_26_down_proj = (char*)(0x7bdd8e2c0000);
  all_tensors["layer_26_down_proj"] = layer_26_down_proj;
  char *layer_27_input_layernorm = (char*)(0x7bddc0a1cc00);
  all_tensors["layer_27_input_layernorm"] = layer_27_input_layernorm;
  char *layer_27_q_proj = (char*)(0x7bdd918c0000);
  all_tensors["layer_27_q_proj"] = layer_27_q_proj;
  char *layer_27_k_proj = (char*)(0x7bdd912c0000);
  all_tensors["layer_27_k_proj"] = layer_27_k_proj;
  char *layer_27_v_proj = (char*)(0x7bdd91cc0000);
  all_tensors["layer_27_v_proj"] = layer_27_v_proj;
  char *layer_27_q_norm = (char*)(0x7bddc0a1de00);
  all_tensors["layer_27_q_norm"] = layer_27_q_norm;
  char *layer_27_k_norm = (char*)(0x7bddc0a1dc00);
  all_tensors["layer_27_k_norm"] = layer_27_k_norm;
  char *layer_27_k_cache = (char*)(0x7bdddd800000);
  all_tensors["layer_27_k_cache"] = layer_27_k_cache;
  char *layer_27_v_cache = (char*)(0x7bddcf800000);
  all_tensors["layer_27_v_cache"] = layer_27_v_cache;
  char *layer_27_o_proj = (char*)(0x7bdd914c0000);
  all_tensors["layer_27_o_proj"] = layer_27_o_proj;
  char *layer_27_post_attn_layernorm = (char*)(0x7bddc0a1d400);
  all_tensors["layer_27_post_attn_layernorm"] = layer_27_post_attn_layernorm;
  char *layer_27_gate_proj = (char*)(0x7bdd906c0000);
  all_tensors["layer_27_gate_proj"] = layer_27_gate_proj;
  char *layer_27_up_proj = (char*)(0x7bdd90cc0000);
  all_tensors["layer_27_up_proj"] = layer_27_up_proj;
  char *layer_27_down_proj = (char*)(0x7bdd900c0000);
  all_tensors["layer_27_down_proj"] = layer_27_down_proj;
  char *model_norm_weight = (char*)(0x7bddc0a26c00);
  all_tensors["model_norm_weight"] = model_norm_weight;
  char *lm_head = (char*)(0x7bdcec000000);
  all_tensors["lm_head"] = lm_head;
  construct_task_graph(num_gpus, my_gpu_id, all_tasks, all_events, first_tasks, all_tensors);
}

__device__ __forceinline__
void _execute_task(TaskDesc const& task_desc,
                   int *step,
                   long long *tokens) {
  if (task_desc.task_type == TASK_EMBEDDING && task_desc.variant_id == 0) {
      kernel::embedding_kernel<bfloat16, 1024>(
      task_desc.inputs[0].base_ptr,
      task_desc.inputs[1].base_ptr,
      task_desc.outputs[0].base_ptr,
      step[0],
      tokens);

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
      step[0],
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
      task_desc.outputs[0].base_ptr);

  }
  else if (task_desc.task_type == TASK_LINEAR_WITH_RESIDUAL && task_desc.variant_id == 0) {
      kernel::linear_kernel<bfloat16, 1, 64, 2048, 1024>(
      task_desc.inputs[0].base_ptr,
      task_desc.inputs[1].base_ptr,
      task_desc.inputs[2].base_ptr,
      task_desc.outputs[0].base_ptr);

  }
  else if (task_desc.task_type == TASK_ARGMAX_PARTIAL && task_desc.variant_id == 0) {
      kernel::argmax_partial_kernel<bfloat16, 1600>(
      task_desc.inputs[0].base_ptr,
      task_desc.outputs[0].base_ptr,
      task_desc.outputs[1].base_ptr);

  }
  else if (task_desc.task_type == TASK_ARGMAX_REDUCE && task_desc.variant_id == 0) {
      kernel::argmax_reduce_kernel<bfloat16, 1600, 96>(
      task_desc.inputs[0].base_ptr,
      task_desc.inputs[1].base_ptr,
      task_desc.outputs[0].base_ptr,
      step[0],
      tokens);

  }
}
