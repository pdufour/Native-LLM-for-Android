
#include <jni.h>
#include <iostream>
#include <fstream>
#include <android/asset_manager_jni.h>
#include "onnxruntime_cxx_api.h"
#include "nnapi_provider_factory.h"
#include "tokenizer.hpp"
#include "half.hpp"

using half_float::half;
const OrtApi *ort_runtime_A;
OrtSession *session_model_A;
std::vector<const char *> input_names_A;
std::vector<const char *> output_names_A;
std::vector<std::vector<std::int64_t>> input_dims_A;
std::vector<std::vector<std::int64_t>> output_dims_A;
std::vector<ONNXTensorElementDataType> input_types_A;
std::vector<ONNXTensorElementDataType> output_types_A;
std::vector<OrtValue *> input_tensors_A;
std::vector<OrtValue *> output_tensors_A;
const OrtApi *ort_runtime_B;
OrtSession *session_model_B;
std::vector<const char *> input_names_B;
std::vector<const char *> output_names_B;
std::vector<std::vector<std::int64_t>> input_dims_B;
std::vector<std::vector<std::int64_t>> output_dims_B;
std::vector<ONNXTensorElementDataType> input_types_B;
std::vector<ONNXTensorElementDataType> output_types_B;
std::vector<OrtValue *> input_tensors_B;
std::vector<OrtValue *> output_tensors_B;
std::unique_ptr<Tokenizer> tokenizer;
int response_count = 0;
int save_index = 0;
int64_t history_len = 0;
int64_t ids_len = 0;
int64_t control_factor = 0;
half attention_mask = half(-65504.f);
const std::string file_name_A = "Yuan_2B_1024_A.ort";
const std::string file_name_B = "Yuan_2B_1024_B.ort";
const int max_token_history = 1024; // Please set this value to match the model name flag.
const int hidden_size = 2048;
const int start_id = 77187;
const int end_id_0 = 77185;
const int past_key_value_size = 49152 * max_token_history; // 24 * 32 * 64
const int single_chat_limit = 1024;                        // It is recommended to set it to max_token_history/3, and use phrases like 'go ahead', 'go on', or 'and then?' to continue answering."
const int next_chat_buffer = max_token_history - single_chat_limit;
const int input_ids_buffer_size = max_token_history * sizeof(int32_t);
const int past_key_values_buffer_size = past_key_value_size * sizeof(half);
const int last_hidden_states_buffer_size = hidden_size * sizeof(float);
const int before_hidden_states_size = 98304; // 24 * 2 * hidden_size
const int before_hidden_states_buffer_size = before_hidden_states_size * sizeof(half);
std::vector<int32_t> input_ids(max_token_history, 0);
std::vector<int> accumulate_num_ids(30, 0); // Just make sure the size is enough before reaching max_token_history.
std::vector<int> num_ids_per_chat(30, 0);   // Same size with accumulate_num_ids.
std::vector<int> save_max_logit_position(max_token_history, 0);
const std::string storage_path = "/storage/emulated/0/Android/data/com.example.myapplication/files/";
const std::string vocab_file = "/data/user/0/com.example.myapplication/cache/vocab_Yuan2.txt"; // We have moved the vocab.txt from assets to the cache folder in Java process.
const char *qnn_htp_so = "/data/user/0/com.example.myapplication/cache/libQnnHtp.so";          //  If use (std::string + "libQnnHtp.so").c_str() instead, it will open failed.
const char *qnn_cpu_so = "/data/user/0/com.example.myapplication/cache/libQnnCpu.so";          //  If use (std::string + "libQnnCpu.so").c_str() instead, it will open failed.
void *key_states;
void *value_states;
void *before_hidden_states;
