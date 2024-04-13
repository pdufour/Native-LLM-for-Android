#include "project.h"

inline static std::vector<int> get_input_ids(const std::string& query) {
    std::vector<int> ids = tokenizer->encode(query);
    ids.insert(ids.begin(),{1, 1786, 4194, 95388});  // Chat prompt head
    ids.insert(ids.end(),{95396, 10850, 95388});  // Chat prompt tail
    return ids;
}

inline static std::string get_output_words(const int& id) {
    std::string words = tokenizer->decode(id);
    if (words.length() == 6 && words[0] == '<' && words[words.length() - 1] == '>' && words[1] == '0' && words[2] == 'x') {
        words = static_cast<char>(std::stoi(words.substr(3, 2), nullptr, 16));
    }
    return words;
}

inline static void clear_history() {
    save_index = 0;
    history_len = 0;
    attention_mask = -999999999999.f;
    accumulate_num_ids[0] = 0;
    num_ids_per_chat[0] = 0;
    std::fill(input_ids.begin(),input_ids.end(),0);
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Pre_1Process(JNIEnv *env, jobject clazz) {
    tokenizer.reset(new Sentencepiece);
    tokenizer->load( vocab_file);
    for (int i = 1; i < theta.size(); i++) {
        theta[i] = theta[i - 1] + 2;  // even sequence
    }
    for (int i = 1; i < theta.size(); i++) {
        theta[i] = std::powf(10000.f, -theta[i] * 0.015625f);  // 1/(10000^(x/64))
    }
    theta[0] = 1.f;
    std::vector<float> idx_theta(max_token_history * theta.size(), 0.f);
    std::move(theta.begin(), theta.end(),idx_theta.begin() + theta.size());
    int index = theta.size() + theta.size();
    for (float i = 2.f; i < static_cast<float> (max_token_history); i += 1.f) {
        for (int j = 0; j < theta.size(); j++) {
            idx_theta[index] = i * theta[j];
            index++;
        }
    }
    std::vector<float> temp_cos(idx_theta.size(), 0.f);
    std::vector<float> temp_sin(idx_theta.size(), 0.f);
    for (int i = 0; i < idx_theta.size(); i++) {
        temp_cos[i] = std::cosf(idx_theta[i]);
        temp_sin[i] = std::sinf(idx_theta[i]);
    }
    index = 0;
    int index2 = 0;
    for (int i = 0; i < max_token_history; i++) {
        std::copy(temp_cos.begin() + index, temp_cos.begin() + index + theta.size(), cos_rotary_pos_emb.begin() + index2);
        std::move(temp_cos.begin() + index, temp_cos.begin() + index + theta.size(), cos_rotary_pos_emb.begin() + index2 + theta.size());
        std::copy(temp_sin.begin() + index, temp_sin.begin() + index + theta.size(), sin_rotary_pos_emb.begin() + index2);
        std::move(temp_sin.begin() + index, temp_sin.begin() + index + theta.size(), sin_rotary_pos_emb.begin() + index2 + theta.size());
        index += theta.size();
        index2 = index + index;
    }
    return JNI_TRUE;
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_Run_1LLM(JNIEnv *env, jclass clazz, jstring jquery,
                                                     jboolean add_prompt,
                                                     jboolean clear) {
    if (add_prompt) {
        if (clear) {
            clear_history();
        }
        const char *query = env->GetStringUTFChars(jquery, nullptr);
        std::vector<int32_t> get_ids = get_input_ids(query);
        ids_len = get_ids.size();
        num_ids_per_chat[save_index] = ids_len;
        if (save_index > 0) {
            accumulate_num_ids[save_index] = num_ids_per_chat[save_index] + accumulate_num_ids[save_index - 1];
            if (accumulate_num_ids[save_index] > next_chat_buffer) {
                bool over_inputs = true;
                for (int i = 0; i < save_index; i++) {
                    if (accumulate_num_ids[save_index] - accumulate_num_ids[i] <= next_chat_buffer) {
                        std::move(input_ids.begin() + accumulate_num_ids[i], input_ids.end(), input_ids.begin());
                        int k = i + 1;
                        for (int j = k; j <= save_index; j++) {
                            accumulate_num_ids[j] -= accumulate_num_ids[i];
                        }
                        ids_len = accumulate_num_ids[save_index];
                        std::move(get_ids.begin(), get_ids.end(), input_ids.begin() + accumulate_num_ids[save_index - 1]);
                        std::move(num_ids_per_chat.begin() + k, num_ids_per_chat.end(), num_ids_per_chat.begin());
                        std::move(accumulate_num_ids.begin() + k, accumulate_num_ids.end(), accumulate_num_ids.begin());
                        save_index -= k;
                        over_inputs = false;
                        break;
                    }
                }
                if (over_inputs) {
                    clear_history();
                    return env->NewStringUTF("Over_Inputs");
                }
            } else {
                std::move(get_ids.begin(), get_ids.end(),input_ids.begin() + accumulate_num_ids[save_index - 1]);
                ids_len = accumulate_num_ids[save_index];
            }
        } else {
            if (num_ids_per_chat[0] >= max_token_history) {
                clear_history();
                return env->NewStringUTF("Over_Inputs");
            } else {
                accumulate_num_ids[0] = num_ids_per_chat[0];
                std::move(get_ids.begin(), get_ids.end(),input_ids.begin());
            }
        }
    }
    std::vector<float> past_key_values;
    {
        OrtMemoryInfo *memory_info;
        ort_runtime_A->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
        ort_runtime_A->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(input_ids.data()), input_ids_buffer_size,
                input_dims_A[0].data(), input_dims_A[0].size(), input_types_A[0],
                &input_tensors_A[0]);
        ort_runtime_A->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(&attention_mask), sizeof(float),
                input_dims_A[1].data(), input_dims_A[1].size(), input_types_A[1],
                &input_tensors_A[1]);
        ort_runtime_A->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(cos_rotary_pos_emb.data()), rotary_pos_emb_buffer_size,
                input_dims_A[2].data(), input_dims_A[2].size(), input_types_A[2],
                &input_tensors_A[2]);
        ort_runtime_A->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(sin_rotary_pos_emb.data()), rotary_pos_emb_buffer_size,
                input_dims_A[3].data(), input_dims_A[3].size(), input_types_A[3],
                &input_tensors_A[3]);
        if (add_prompt) {
            past_key_values.resize(past_key_value_size, 0.f);
            ort_runtime_A->CreateTensorWithDataAsOrtValue(
                    memory_info,
                    reinterpret_cast<void*>(past_key_values.data()), past_key_values_buffer_size,
                    input_dims_A[4].data(), input_dims_A[4].size(), input_types_A[4],
                    &input_tensors_A[4]);
            ort_runtime_A->CreateTensorWithDataAsOrtValue(
                    memory_info,
                    reinterpret_cast<void*>(past_key_values.data()), past_key_values_buffer_size,
                    input_dims_A[5].data(), input_dims_A[5].size(), input_types_A[5],
                    &input_tensors_A[5]);
        } else {
            ort_runtime_A->CreateTensorWithDataAsOrtValue(
                    memory_info,
                    reinterpret_cast<void*>(reinterpret_cast<float*> (key_states_A)),
                    past_key_values_buffer_size,
                    input_dims_A[4].data(), input_dims_A[4].size(), input_types_A[4],
                    &input_tensors_A[4]);
            ort_runtime_A->CreateTensorWithDataAsOrtValue(
                    memory_info,
                    reinterpret_cast<void*>(reinterpret_cast<float*> (value_states_A)),
                    past_key_values_buffer_size,
                    input_dims_A[5].data(), input_dims_A[5].size(), input_types_A[5],
                    &input_tensors_A[5]);
        }
        ort_runtime_A->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(&history_len), sizeof(int64_t),
                input_dims_A[6].data(), input_dims_A[6].size(), input_types_A[6],
                &input_tensors_A[6]);
        ort_runtime_A->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(&ids_len), sizeof(int64_t),
                input_dims_A[7].data(), input_dims_A[7].size(), input_types_A[7],
                &input_tensors_A[7]);
        ort_runtime_A->ReleaseMemoryInfo(memory_info);
        ort_runtime_A->Run(session_model_A, nullptr, input_names_A.data(),
                           (const OrtValue *const *) input_tensors_A.data(),
                           input_tensors_A.size(), output_names_A.data(), output_names_A.size(),
                           output_tensors_A.data());
    }
    {
        void* last_hidden_state;
        ort_runtime_A->GetTensorMutableData(output_tensors_A[0], &last_hidden_state);
        ort_runtime_A->GetTensorMutableData(output_tensors_A[1], &key_states_A);
        ort_runtime_A->GetTensorMutableData(output_tensors_A[2], &value_states_A);
        OrtMemoryInfo *memory_info;
        ort_runtime_B->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
        ort_runtime_B->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(reinterpret_cast<float*> (last_hidden_state)), hidden_state_buffer_size,
                input_dims_B[0].data(), input_dims_B[0].size(), input_types_B[0],
                &input_tensors_B[0]);
        ort_runtime_B->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(&attention_mask), sizeof(float),
                input_dims_B[1].data(), input_dims_B[1].size(), input_types_B[1],
                &input_tensors_B[1]);
        ort_runtime_B->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(cos_rotary_pos_emb.data()), rotary_pos_emb_buffer_size,
                input_dims_B[2].data(), input_dims_B[2].size(), input_types_B[2],
                &input_tensors_B[2]);
        ort_runtime_B->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(sin_rotary_pos_emb.data()), rotary_pos_emb_buffer_size,
                input_dims_B[3].data(), input_dims_B[3].size(), input_types_B[3],
                &input_tensors_B[3]);
        if (add_prompt) {
            ort_runtime_B->CreateTensorWithDataAsOrtValue(
                    memory_info,
                    reinterpret_cast<void*>(past_key_values.data()), past_key_values_buffer_size,
                    input_dims_B[4].data(), input_dims_B[4].size(), input_types_B[4],
                    &input_tensors_B[4]);
            ort_runtime_B->CreateTensorWithDataAsOrtValue(
                    memory_info,
                    reinterpret_cast<void*>(past_key_values.data()), past_key_values_buffer_size,
                    input_dims_B[5].data(), input_dims_B[5].size(), input_types_B[5],
                    &input_tensors_B[5]);
        } else {
            ort_runtime_B->CreateTensorWithDataAsOrtValue(
                    memory_info,
                    reinterpret_cast<void*>(reinterpret_cast<float*> (key_states_B)),
                    past_key_values_buffer_size,
                    input_dims_B[4].data(), input_dims_B[4].size(), input_types_B[4],
                    &input_tensors_B[4]);
            ort_runtime_B->CreateTensorWithDataAsOrtValue(
                    memory_info,
                    reinterpret_cast<void*>(reinterpret_cast<float*> (value_states_B)),
                    past_key_values_buffer_size,
                    input_dims_B[5].data(), input_dims_B[5].size(), input_types_B[5],
                    &input_tensors_B[5]);
        }
        ort_runtime_B->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(&history_len), sizeof(int64_t),
                input_dims_B[6].data(), input_dims_B[6].size(), input_types_B[6],
                &input_tensors_B[6]);
        ort_runtime_B->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(&ids_len), sizeof(int64_t),
                input_dims_B[7].data(), input_dims_B[7].size(), input_types_B[7],
                &input_tensors_B[7]);
        ort_runtime_B->ReleaseMemoryInfo(memory_info);
        ort_runtime_B->Run(session_model_B, nullptr, input_names_B.data(),
                           (const OrtValue *const *) input_tensors_B.data(),
                           input_tensors_B.size(), output_names_B.data(), output_names_B.size(),
                           output_tensors_B.data());
    }
    {
        void* max_logit_id;
        ort_runtime_B->GetTensorMutableData(output_tensors_B[0], &max_logit_id);
        ort_runtime_B->GetTensorMutableData(output_tensors_B[1], &key_states_B);
        ort_runtime_B->GetTensorMutableData(output_tensors_B[2], &value_states_B);
        input_ids[0] = static_cast<int32_t> (reinterpret_cast<int64_t*> (max_logit_id)[0]);
        history_len += ids_len;
        if (add_prompt) {
            ids_len = 1;
            response_count = 0;
            attention_mask = 0.f;
        }
    }
    if ((input_ids[0] != end_id_0) && (response_count < single_chat_limit) && (history_len < max_token_history)) {
        save_max_logit_position[response_count] = input_ids[0];
        response_count += 1;
        return env->NewStringUTF(get_output_words(input_ids[0]).c_str());
    } else {
        save_max_logit_position[response_count] = end_id_0;
        response_count += 1;
        num_ids_per_chat[save_index] += response_count;
        attention_mask = -999999999.f;
        history_len = 0;
        input_ids[0] = start_id;
        if (save_index > 0) {
            accumulate_num_ids[save_index] = num_ids_per_chat[save_index] + accumulate_num_ids[save_index - 1];
            if (accumulate_num_ids[save_index] > next_chat_buffer) {
                for (int i = 0; i < save_index; i++) {
                    if (accumulate_num_ids[save_index] - accumulate_num_ids[i] <= next_chat_buffer) {
                        std::move(input_ids.begin() + accumulate_num_ids[i],input_ids.end(),input_ids.begin());
                        int k = i + 1;
                        for (int j = k; j <= save_index; j++) {
                            accumulate_num_ids[j] -= accumulate_num_ids[i];
                        }
                        std::move(save_max_logit_position.begin(),save_max_logit_position.begin() + response_count,input_ids.begin() + accumulate_num_ids[save_index] - response_count);
                        std::move(num_ids_per_chat.begin() + k,num_ids_per_chat.end(),num_ids_per_chat.begin());
                        std::move(accumulate_num_ids.begin() + k,accumulate_num_ids.end(),accumulate_num_ids.begin());
                        save_index -= i;
                        return env->NewStringUTF("END");
                    }
                }
                clear_history();
            } else {
                std::move(save_max_logit_position.begin(),save_max_logit_position.begin() + response_count,input_ids.begin() + accumulate_num_ids[save_index] - response_count);
                save_index += 1;
            }
        } else {
            std::move(save_max_logit_position.begin(),save_max_logit_position.begin() + response_count,input_ids.begin() + accumulate_num_ids[0]);
            accumulate_num_ids[0] = num_ids_per_chat[0];
            save_index += 1;
        }
        return env->NewStringUTF("END");
    }
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1A(JNIEnv *env, jobject clazz,
                                                            jobject asset_manager,
                                                            jboolean use_gpu,
                                                            jboolean use_fp16,
                                                            jboolean use_nnapi,
                                                            jboolean use_xnnpack,
                                                            jboolean use_qnn,
                                                            jboolean use_dsp_npu) {
    OrtStatus *status;
    OrtAllocator *allocator;
    OrtEnv *ort_env_A;
    OrtSessionOptions *session_options_A;
    {
        std::vector<char> fileBuffer;
        size_t fileSize;
        if (asset_manager != nullptr) {
            AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
            AAsset* asset = AAssetManager_open(mgr,file_name_A.c_str(), AASSET_MODE_BUFFER);
            fileSize = AAsset_getLength(asset);
            fileBuffer.resize(fileSize);
            AAsset_read(asset,fileBuffer.data(),fileSize);
        } else {
            std::ifstream model_file(storage_path + file_name_A, std::ios::binary | std::ios::ate);
            if (!model_file.is_open()) {
                return JNI_FALSE;
            }
            fileSize = model_file.tellg();
            model_file.seekg(0, std::ios::beg);
            fileBuffer.resize(fileSize);
            if (!model_file.read(fileBuffer.data(), fileSize)) {
                return JNI_FALSE;
            }
            model_file.close();
        }
        ort_runtime_A = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort_runtime_A->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &ort_env_A);
        ort_runtime_A->CreateSessionOptions(&session_options_A);
        ort_runtime_A->DisableProfiling(session_options_A);
        ort_runtime_A->EnableCpuMemArena(session_options_A);
        ort_runtime_A->EnableMemPattern(session_options_A);
        ort_runtime_A->SetSessionExecutionMode(session_options_A, ORT_SEQUENTIAL);
        ort_runtime_A->SetInterOpNumThreads(session_options_A, 2);
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.dynamic_block_base", "2");  // One block can contain 1 or more cores, and sharing 1 job.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, // Binding the #cpu to run the model. 'A;B;' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "session.intra_op_thread_affinities",
                                             "1;2");  // It is the best cost/performance (C/P) value setting for running the MiniCPM 2B LLM on the Kirin 990 5G, due to limitations imposed by the RAM bandwidth.
        ort_runtime_A->SetIntraOpNumThreads(session_options_A, 3); // dynamic_block_base + 1
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.inter_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.force_spinning_stop",
                                             "0");  // 1 for low power
        ort_runtime_A->SetSessionGraphOptimizationLevel(session_options_A, ORT_ENABLE_ALL);
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "optimization.minimal_build_optimizations",
                                             "");   // Keep empty for full optimization
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_prepacking",
                                             "0");  // 0 for enable
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "optimization.enable_gelu_approximation",
                                             "0");  // Set 0 is better for this model
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");  //
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.disable_aot_function_inlining",
                                             "0");  // 0 for speed
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.qdqisint8allowed",
                                             "1");  // 1 for Arm
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.enable_quant_qdq_cleanup",
                                             "1");  // 0 for precision, 1 for performance
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.disable_double_qdq_remover",
                                             "0");  // 1 for precision, 0 for performance
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_quant_qdq",
                                             "0");  // 0 for use Int8
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.use_ort_model_bytes_directly",
                                             "1");  // Use this option to lower the peak memory during loading.
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.use_ort_model_bytes_for_initializers",
                                             "0");  // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.set_denormal_as_zero",
                                             "0");  // // Use 0 instead of NaN or Inf.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.use_env_allocators",
                                             "1");  // Use it to lower memory usage.
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.use_device_allocator_for_initializers",
                                             "1");  // Use it to lower memory usage.
        std::vector<const char*> option_keys = {};
        std::vector<const char*> option_values = {};
        if (use_qnn) {  // It needs the permission of HTP hardware, and then follow the onnx document to generate the specific format to run on HTP.
            if (use_dsp_npu) {
                option_keys.push_back("backend_path");
                option_values.push_back(qnn_htp_so);
                option_keys.push_back("htp_performance_mode");
                option_values.push_back("burst");
                option_keys.push_back("htp_graph_finalization_optimization_mode");
                option_values.push_back("3");
                option_keys.push_back("soc_model");
                option_values.push_back("0");  // 0 for unknown
                option_keys.push_back("htp_arch");
                option_values.push_back("73");  // 0 for unknown
                option_keys.push_back("device_id");
                option_values.push_back("0");  // 0 for single device
                option_keys.push_back("vtcm_mb");
                option_values.push_back("8");  // 0 for auto
                option_keys.push_back("qnn_context_priority");
                option_values.push_back("high");
                if (use_fp16) {  // Enable it to run a float model on HTP. (Both fp16 & fp32 format)
                    option_keys.push_back("enable_htp_fp16_precision");
                    option_values.push_back("1");
                } else {
                    option_keys.push_back("enable_htp_fp16_precision");
                    option_values.push_back("0");
                }
                ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                                     "ep.context_enable", "1");
                ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                                     "ep.context_embed_mode", "1");
                ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                                     "ep.context_file_path", storage_path.c_str());  // Default to original_file_name_ctx.onnx if not specified
                ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                                     "session.use_ort_model_bytes_directly",
                                                     "0");  // Cancel this option.
            } else {
                option_keys.push_back("backend_path");
                option_values.push_back(qnn_cpu_so);
            }
            ort_runtime_A->SessionOptionsAppendExecutionProvider(session_options_A, "QNN", option_keys.data(), option_values.data(), option_keys.size());
        } else if (use_nnapi) {
            uint32_t nnapi_flags = 0;
            if (use_gpu | use_dsp_npu) {
                nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;
            } else {
                nnapi_flags |= NNAPI_FLAG_CPU_ONLY;
            }
            if (use_fp16) {
                nnapi_flags |= NNAPI_FLAG_USE_FP16;
            }
            OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options_A, nnapi_flags);
        } else if (use_xnnpack) {
            option_keys.push_back("intra_op_num_threads");
            option_values.push_back("4");
            ort_runtime_A->SessionOptionsAppendExecutionProvider(session_options_A, "XNNPACK", option_keys.data(), option_values.data(), option_keys.size());
        }
        status = ort_runtime_A->CreateSessionFromArray(ort_env_A, fileBuffer.data(), fileSize,
                                                       session_options_A, &session_model_A);
    }
    if (status != nullptr) {
        return JNI_FALSE;
    }
    std::size_t amount_of_input;
    ort_runtime_A->GetAllocatorWithDefaultOptions(&allocator);
    ort_runtime_A->SessionGetInputCount(session_model_A, &amount_of_input);
    input_names_A.resize(amount_of_input);
    input_dims_A.resize(amount_of_input);
    input_types_A.resize(amount_of_input);
    input_tensors_A.resize(amount_of_input);
    for (size_t i = 0; i < amount_of_input; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_A->SessionGetInputName(session_model_A, i, allocator, &name);
        input_names_A[i] = name;
        ort_runtime_A->SessionGetInputTypeInfo(session_model_A, i, &typeinfo);
        ort_runtime_A->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_A->GetTensorElementType(tensor_info, &type);
        input_types_A[i] = type;
        ort_runtime_A->GetDimensionsCount(tensor_info, &dimensions);
        input_dims_A[i].resize(dimensions);
        ort_runtime_A->GetDimensions(tensor_info, input_dims_A[i].data(), dimensions);
        ort_runtime_A->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) ort_runtime_A->ReleaseTypeInfo(typeinfo);
    }
    std::size_t amount_of_output;
    ort_runtime_A->SessionGetOutputCount(session_model_A, &amount_of_output);
    output_names_A.resize(amount_of_output);
    output_dims_A.resize(amount_of_output);
    output_types_A.resize(amount_of_output);
    output_tensors_A.resize(amount_of_output);
    for (size_t i = 0; i < amount_of_output; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_A->SessionGetOutputName(session_model_A, i, allocator, &name);
        output_names_A[i] = name;
        ort_runtime_A->SessionGetOutputTypeInfo(session_model_A, i, &typeinfo);
        ort_runtime_A->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_A->GetTensorElementType(tensor_info, &type);
        output_types_A[i] = type;
        ort_runtime_A->GetDimensionsCount(tensor_info, &dimensions);
        output_dims_A[i].resize(dimensions);
        ort_runtime_A->GetDimensions(tensor_info, output_dims_A[i].data(), dimensions);
        ort_runtime_A->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) ort_runtime_A->ReleaseTypeInfo(typeinfo);
    }
    return JNI_TRUE;
}


extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1B(JNIEnv *env, jobject clazz,
                                                            jobject asset_manager,
                                                            jboolean use_fp16,
                                                            jboolean use_gpu,
                                                            jboolean use_nnapi,
                                                            jboolean use_xnnpack,
                                                            jboolean use_qnn,
                                                            jboolean use_dsp_npu) {
    OrtStatus *status;
    OrtAllocator *allocator;
    OrtEnv *ort_env_B;
    OrtSessionOptions *session_options_B;
    {
        std::vector<char> fileBuffer;
        size_t fileSize;
        if (asset_manager != nullptr) {
            AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
            AAsset* asset = AAssetManager_open(mgr,file_name_B.c_str(), AASSET_MODE_BUFFER);
            fileSize = AAsset_getLength(asset);
            fileBuffer.resize(fileSize);
            AAsset_read(asset,fileBuffer.data(),fileSize);
        } else {
            std::ifstream model_file(storage_path + file_name_B, std::ios::binary | std::ios::ate);
            if (!model_file.is_open()) {
                return JNI_FALSE;
            }
            fileSize = model_file.tellg();
            model_file.seekg(0, std::ios::beg);
            fileBuffer.resize(fileSize);
            if (!model_file.read(fileBuffer.data(), fileSize)) {
                return JNI_FALSE;
            }
            model_file.close();
        }
        ort_runtime_B = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort_runtime_B->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &ort_env_B);
        ort_runtime_B->CreateSessionOptions(&session_options_B);
        ort_runtime_B->DisableProfiling(session_options_B);
        ort_runtime_B->EnableCpuMemArena(session_options_B);
        ort_runtime_B->EnableMemPattern(session_options_B);
        ort_runtime_B->SetSessionExecutionMode(session_options_B, ORT_SEQUENTIAL);
        ort_runtime_B->SetInterOpNumThreads(session_options_B, 2);
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.dynamic_block_base", "2");
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.intra_op_thread_affinities",
                                             "1;2");
        ort_runtime_B->SetIntraOpNumThreads(session_options_B, 3); // dynamic_block_base + 1
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.inter_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.intra_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.force_spinning_stop",
                                             "0");  // 1 for low power
        ort_runtime_B->SetSessionGraphOptimizationLevel(session_options_B, ORT_ENABLE_ALL);
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "optimization.minimal_build_optimizations",
                                             "");   // Keep empty for full optimization
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.disable_prepacking",
                                             "0");  // 0 for enable
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "optimization.enable_gelu_approximation",
                                             "0");  // Set 0 is better for this model
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");  //
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.disable_aot_function_inlining",
                                             "0");  // 0 for speed
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.qdqisint8allowed",
                                             "1");  // 1 for Arm
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.enable_quant_qdq_cleanup",
                                             "1");  // 0 for precision, 1 for performance
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.disable_double_qdq_remover",
                                             "0");  // 1 for precision, 0 for performance
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.disable_quant_qdq",
                                             "0");  // 0 for use Int8
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.use_ort_model_bytes_directly",
                                             "1");  // Use this option to lower the peak memory during loading.
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.use_ort_model_bytes_for_initializers",
                                             "0");  // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.set_denormal_as_zero",
                                             "0");  // // Use 0 instead of NaN or Inf.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.use_env_allocators",
                                             "1");  // Use it to lower memory usage.
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.use_device_allocator_for_initializers",
                                             "1");  // Use it to lower memory usage.
        std::vector<const char*> option_keys = {};
        std::vector<const char*> option_values = {};
        if (use_qnn) {  // It needs the permission of HTP hardware, and then follow the onnx document to generate the specific format to run on HTP.
            if (use_dsp_npu) {
                option_keys.push_back("backend_path");
                option_values.push_back(qnn_htp_so);
                option_keys.push_back("htp_performance_mode");
                option_values.push_back("burst");
                option_keys.push_back("htp_graph_finalization_optimization_mode");
                option_values.push_back("3");
                option_keys.push_back("soc_model");
                option_values.push_back("0");  // 0 for unknown
                option_keys.push_back("htp_arch");
                option_values.push_back("73");  // 0 for unknown
                option_keys.push_back("device_id");
                option_values.push_back("0");  // 0 for single device
                option_keys.push_back("vtcm_mb");
                option_values.push_back("8");  // 0 for auto
                option_keys.push_back("qnn_context_priority");
                option_values.push_back("high");
                if (use_fp16) {  // Enable it to run a float model on HTP. (Both fp16 & fp32 format)
                    option_keys.push_back("enable_htp_fp16_precision");
                    option_values.push_back("1");
                } else {
                    option_keys.push_back("enable_htp_fp16_precision");
                    option_values.push_back("0");
                }
                ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                                     "ep.context_enable", "1");
                ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                                     "ep.context_embed_mode", "1");
                ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                                     "ep.context_file_path", storage_path.c_str());  // Default to original_file_name_ctx.onnx if not specified
                ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                                     "session.use_ort_model_bytes_directly",
                                                     "0");  // Cancel this option.
            } else {
                option_keys.push_back("backend_path");
                option_values.push_back(qnn_cpu_so);
            }
            ort_runtime_B->SessionOptionsAppendExecutionProvider(session_options_B, "QNN", option_keys.data(), option_values.data(), option_keys.size());
        } else if (use_nnapi) {
            uint32_t nnapi_flags = 0;
            if (use_gpu | use_dsp_npu) {
                nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;
            } else {
                nnapi_flags |= NNAPI_FLAG_CPU_ONLY;
            }
            if (use_fp16) {
                nnapi_flags |= NNAPI_FLAG_USE_FP16;
            }
            OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options_B, nnapi_flags);
        } else if (use_xnnpack) {
            option_keys.push_back("intra_op_num_threads");
            option_values.push_back("4");
            ort_runtime_B->SessionOptionsAppendExecutionProvider(session_options_B, "XNNPACK", option_keys.data(), option_values.data(), option_keys.size());
        }
        status = ort_runtime_B->CreateSessionFromArray(ort_env_B, fileBuffer.data(), fileSize,
                                                       session_options_B, &session_model_B);
    }
    if (status != nullptr) {
        return JNI_FALSE;
    }
    std::size_t amount_of_input;
    ort_runtime_B->GetAllocatorWithDefaultOptions(&allocator);
    ort_runtime_B->SessionGetInputCount(session_model_B, &amount_of_input);
    input_names_B.resize(amount_of_input);
    input_dims_B.resize(amount_of_input);
    input_types_B.resize(amount_of_input);
    input_tensors_B.resize(amount_of_input);
    for (size_t i = 0; i < amount_of_input; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_B->SessionGetInputName(session_model_B, i, allocator, &name);
        input_names_B[i] = name;
        ort_runtime_B->SessionGetInputTypeInfo(session_model_B, i, &typeinfo);
        ort_runtime_B->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_B->GetTensorElementType(tensor_info, &type);
        input_types_B[i] = type;
        ort_runtime_B->GetDimensionsCount(tensor_info, &dimensions);
        input_dims_B[i].resize(dimensions);
        ort_runtime_B->GetDimensions(tensor_info, input_dims_B[i].data(), dimensions);
        ort_runtime_B->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) ort_runtime_B->ReleaseTypeInfo(typeinfo);
    }
    std::size_t amount_of_output;
    ort_runtime_B->SessionGetOutputCount(session_model_B, &amount_of_output);
    output_names_B.resize(amount_of_output);
    output_dims_B.resize(amount_of_output);
    output_types_B.resize(amount_of_output);
    output_tensors_B.resize(amount_of_output);
    for (size_t i = 0; i < amount_of_output; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_B->SessionGetOutputName(session_model_B, i, allocator, &name);
        output_names_B[i] = name;
        ort_runtime_B->SessionGetOutputTypeInfo(session_model_B, i, &typeinfo);
        ort_runtime_B->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_B->GetTensorElementType(tensor_info, &type);
        output_types_B[i] = type;
        ort_runtime_B->GetDimensionsCount(tensor_info, &dimensions);
        output_dims_B[i].resize(dimensions);
        ort_runtime_B->GetDimensions(tensor_info, output_dims_B[i].data(), dimensions);
        ort_runtime_B->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) ort_runtime_B->ReleaseTypeInfo(typeinfo);
    }
    return JNI_TRUE;
}
