import time
import torch
import numpy as np
import onnxruntime
from PIL import Image
import shutil
import gc
import os
import transformers
import sys
import requests
from io import BytesIO

def log_numpy_array(name, arr):
    """Helper function to log numpy array details"""
    print(f"\n[NUMPY] {name}:")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Min/Max: {np.min(arr):.6f} / {np.max(arr):.6f}")
    if arr.size < 10:  # Only print full values for small arrays
        print(f"  Values: {arr}")
    print(f"  First few values: {arr.flatten()[:5]}")
    print(f"  Last few values: {arr.flatten()[-5:]}")

def log_session_io(session, name):
    """Helper function to log ONNX session inputs/outputs"""
    print(f"\n[SESSION {name}] Input details:")
    for idx, input in enumerate(session.get_inputs()):
        print(f"  Input_{idx}:")
        print(f"    Name: {input.name}")
        print(f"    Shape: {input.shape}")
        print(f"    Type: {input.type}")

    print(f"\n[SESSION {name}] Output details:")
    for idx, output in enumerate(session.get_outputs()):
        print(f"  Output_{idx}:")
        print(f"    Name: {output.name}")
        print(f"    Shape: {output.shape}")
        print(f"    Type: {output.type}")

def is_valid_image_path(image_path):
    """Helper function to validate image path"""
    if not os.path.exists(image_path):
        print(f"\n[ERROR] Image path does not exist: {image_path}")
        return False
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    _, ext = os.path.splitext(image_path)
    valid = ext.lower() in valid_extensions
    if not valid:
        print(f"\n[ERROR] Invalid image extension: {ext}")
    return valid

print("\n[STARTUP] Beginning QwenVL initialization...")

try:
    from export_config import INPUT_IMAGE_SIZE, IMAGE_RESIZE, MAX_SEQ_LENGTH, HEIGHT_FACTOR, WIDTH_FACTOR
    print("\n[CONFIG] Loaded configuration from export_config.py")
except:
    print("\n[CONFIG] Using default values as export_config import failed")
    INPUT_IMAGE_SIZE = [960, 960]
    HEIGHT_FACTOR = 10
    WIDTH_FACTOR = 10
    IMAGE_RESIZE = [HEIGHT_FACTOR * 28, WIDTH_FACTOR * 28]
    MAX_SEQ_LENGTH = 1024

print(f"\n[CONFIG] Settings:")
print(f"  INPUT_IMAGE_SIZE: {INPUT_IMAGE_SIZE}")
print(f"  HEIGHT_FACTOR: {HEIGHT_FACTOR}")
print(f"  WIDTH_FACTOR: {WIDTH_FACTOR}")
print(f"  IMAGE_RESIZE: {IMAGE_RESIZE}")
print(f"  MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")

path = sys.argv[1]
print(f"\n[PATHS] Model path: {path}")

if len(sys.argv) > 2:
    script_dir = sys.argv[2]
else:
    script_dir = os.path.dirname(__file__)
print(f"[PATHS] Script directory: {script_dir}")

# Define model paths
modified_path_E = r'./modeling_modified/part_E/modeling_qwen2_vl.py'
onnx_model_A = os.path.join('/Users/paul.dufour/Nuber/Qwen2-VL-2B-Instruct-onnx/', 'onnx/QwenVL_A.onnx')
onnx_model_B = os.path.join('/Users/paul.dufour/Nuber/Qwen2-VL-2B-Instruct-onnx/', 'onnx/QwenVL_B_q4f16.onnx')
onnx_model_C = os.path.join('/Users/paul.dufour/Nuber/Qwen2-VL-2B-Instruct-onnx/', 'onnx/QwenVL_C_q4f16.onnx')
onnx_model_D = os.path.join('/Users/paul.dufour/Nuber/Qwen2-VL-2B-Instruct-onnx/', 'onnx/QwenVL_D_q4f16.onnx')
onnx_model_E = os.path.join(script_dir, 'onnx/QwenVL_E.onnx')
transformers_qwen2_path = transformers.__file__.replace('__init__.py', 'models/qwen2_vl/modeling_qwen2_vl.py')

print("\n[PATHS] ONNX model paths:")
print(f"  Model A: {onnx_model_A}")
print(f"  Model B: {onnx_model_B}")
print(f"  Model C: {onnx_model_C}")
print(f"  Model D: {onnx_model_D}")
print(f"  Model E: {onnx_model_E}")

# image_path = r"./psyduck.png"
image_url = "http://localhost:3004/car_960.jpg"
# image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
query = "Describe this image."

print("\n[IMPORTS] Loading transformers...")
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer

print("\n[MODEL] Loading base model...")
with torch.inference_mode():
    model = Qwen2VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32, device_map="mps", low_cpu_mem_usage=True)
    max_seq_len = MAX_SEQ_LENGTH
    num_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size


print("\n[MODEL] Configuration:")
print(f"  max_seq_len: {max_seq_len}")
print(f"  num_heads: {num_heads}")
print(f"  num_key_value_heads: {num_key_value_heads}")
print(f"  head_dim: {head_dim}")
print(f"  num_layers: {num_layers}")
print(f"  hidden_size: {hidden_size}")

print('\n[ONNX] Starting QwenVL with ONNXRuntime')
max_single_chat_length = 12
print(f"  max_single_chat_length: {max_single_chat_length}")

print("\n[TOKENIZER] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

print("\n[ONNX] Configuring session options...")
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3
session_opts.inter_op_num_threads = 0
session_opts.intra_op_num_threads = 0
session_opts.enable_cpu_mem_arena = True
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")

print("\n[SESSIONS] Loading and analyzing all ONNX sessions...")
ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts)
log_session_io(ort_session_A, "A")

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts)
log_session_io(ort_session_B, "B")

ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts)
log_session_io(ort_session_C, "C")

ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts)
log_session_io(ort_session_D, "D")

ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts)
log_session_io(ort_session_E, "E")

print("\n[NAMES] Getting input/output names...")
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name
print(f"  Session A - Input: {in_name_A0}, Output: {out_name_A0}")

in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
in_name_B0 = in_name_B[0].name
in_name_B1 = in_name_B[1].name
out_name_B0 = out_name_B[0].name
print(f"  Session B - Inputs: {in_name_B0}, {in_name_B1}, Output: {out_name_B0}")

in_name_C = ort_session_C.get_inputs()
out_name_C = ort_session_C.get_outputs()
in_name_C0 = in_name_C[0].name
out_name_C0 = out_name_C[0].name
print(f"  Session C - Input: {in_name_C0}, Output: {out_name_C0}")

in_name_D = ort_session_D.get_inputs()
out_name_D = ort_session_D.get_outputs()
in_name_D0 = in_name_D[0].name
in_name_D1 = in_name_D[1].name
in_name_D2 = in_name_D[2].name
in_name_D3 = in_name_D[3].name
in_name_D4 = in_name_D[4].name
out_name_D0 = out_name_D[0].name
out_name_D1 = out_name_D[1].name
print(f"  Session D - Inputs: {in_name_D0}, {in_name_D1}, {in_name_D2}, {in_name_D3}, {in_name_D4}")
print(f"           - Outputs: {out_name_D0}, {out_name_D1}")

in_name_E = ort_session_E.get_inputs()
out_name_E = ort_session_E.get_outputs()
in_name_E0 = in_name_E[0].name
in_name_E1 = in_name_E[1].name
in_name_E2 = in_name_E[2].name
in_name_E3 = in_name_E[3].name
in_name_E4 = in_name_E[4].name
in_name_E5 = in_name_E[5].name
in_name_E6 = in_name_E[6].name
in_name_E7 = in_name_E[7].name
out_name_E0 = out_name_E[0].name
out_name_E1 = out_name_E[1].name
out_name_E2 = out_name_E[2].name
print(f"  Session E - Inputs: {in_name_E0}, {in_name_E1}, {in_name_E2}, {in_name_E3}")
print(f"           -         {in_name_E4}, {in_name_E5}, {in_name_E6}, {in_name_E7}")
print(f"           - Outputs: {out_name_E0}, {out_name_E1}, {out_name_E2}")

# if is_valid_image_path(image_path):
print("\n[IMAGE] Processing image...")
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
# image = Image.open(image_path)
print(f"  Original size: {image.size}")
print(f"  Original mode: {image.mode}")

# image = image.resize((INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))
# print(f"  Resized to: {image.size}")

if image.mode != 'RGB':
    print(f"  Converting from {image.mode} to RGB")
    image = image.convert('RGB')

pixel_values = np.transpose(np.array(image).astype(np.float32), (2, 0, 1))
pixel_values = np.expand_dims(pixel_values, axis=0) / 255.0
log_numpy_array("pixel_values", pixel_values)
use_vision = True
# else:
#     print("\n[WARNING] No valid image found")
#     use_vision = False


# use_vision = False

print("\n[TOKENIZATION] Processing prompt...")
prompt = f"\n<|im_start|>user\n<|vision_start|><|vision_end|>{query}<|im_end|>\n<|im_start|>assistant\n"
print(f"Raw prompt: {prompt}")
prompt_head_len = np.array([5], dtype=np.int64)
log_numpy_array("prompt_head_len", prompt_head_len)

image_embed_size = WIDTH_FACTOR * HEIGHT_FACTOR
print(f"image_embed_size: {image_embed_size}")

token = tokenizer(prompt, return_tensors='pt')['input_ids']
print(f"Token shape: {token.shape}")
print(f"Token values: {token}")

print("\n[INITIALIZATION] Creating numpy arrays...")
ids_len = np.array([token.shape[1]], dtype=np.int64)
log_numpy_array("ids_len", ids_len)

input_ids = np.zeros(max_seq_len, dtype=np.int32)
input_ids[:ids_len[0]] = token[0, :]
log_numpy_array("input_ids", input_ids)

history_len = np.zeros(1, dtype=np.int64)
log_numpy_array("history_len", history_len)

past_key_states = np.zeros((num_layers, num_key_value_heads, max_seq_len, head_dim), dtype=np.float16)
log_numpy_array("past_key_states", past_key_states)

past_values_states = past_key_states
log_numpy_array("past_values_states", past_values_states)

attention_mask = np.array([-65504.0], dtype=np.float16)
log_numpy_array("attention_mask", attention_mask)

pos_factor = np.array([0.0], dtype=np.float16)
log_numpy_array("pos_factor", pos_factor)

pos_factor_v = 1 - image_embed_size + WIDTH_FACTOR
print(f"pos_factor_v: {pos_factor_v}")

dummy = np.array(0, dtype=np.int32)
log_numpy_array("dummy", dummy)

# ... (continuing from where we left off)

print("\n[INFERENCE] Running initial inference...")
print("Computing hidden states...")
hidden_states = ort_session_B.run(
    [out_name_B0],
    {
        in_name_B0: input_ids,
        in_name_B1: ids_len
    })[0]
log_numpy_array("hidden_states (initial)", hidden_states)

print("\nComputing position IDs...")
position_ids, = ort_session_C.run(
    [out_name_C0],
    {
        in_name_C0: dummy
    })
log_numpy_array("position_ids (initial)", position_ids)

if use_vision:
    print('\n[VISION] Processing vision inputs...')
    start_time = time.time()

    print("Computing image embeddings...")
    log_numpy_array("pixel_values", pixel_values)
    image_embed = ort_session_A.run(
        [out_name_A0],
        {in_name_A0: pixel_values})[0]
    log_numpy_array("image_embed", image_embed)

    print("\nUpdating sequence lengths...")
    ids_len += image_embed_size
    log_numpy_array("updated ids_len", ids_len)

    split_factor = np.array(max_seq_len - ids_len[0] - image_embed_size, dtype=np.int32)
    log_numpy_array("split_factor", split_factor)

    ids_len_minus = np.array(ids_len[0] - prompt_head_len[0], dtype=np.int32)
    log_numpy_array("ids_len_minus", ids_len_minus)

    print("\nProcessing vision embeddings...")
    print("Session D inputs:")
    log_numpy_array("hidden_states", hidden_states)
    log_numpy_array("image_embed", image_embed)
    log_numpy_array("ids_len", ids_len)
    log_numpy_array("ids_len_minus", ids_len_minus)
    log_numpy_array("split_factor", split_factor)

    hidden_states, position_ids = ort_session_D.run(
        [out_name_D0, out_name_D1],
        {
            in_name_D0: hidden_states,
            in_name_D1: image_embed,
            in_name_D2: ids_len,
            in_name_D3: ids_len_minus,
            in_name_D4: split_factor
        })
    log_numpy_array("updated hidden_states", hidden_states)
    log_numpy_array("updated position_ids", position_ids)

    end_time = time.time()
    print(f'\nVision processing complete. Time: {(end_time - start_time):.2f}s')

print('\n[GENERATION] Starting text generation...')
print('Query:', query)
print("\nGenerated response:")
end_time = time.time()
num_decode = 0

while (num_decode < max_single_chat_length) & (history_len < max_seq_len):
    print(f"\n[GENERATION] Step {num_decode}")
    print("Session E inputs:")

    log_numpy_array("hidden_states", hidden_states)
    log_numpy_array("attention_mask", attention_mask)
    log_numpy_array("past_key_states", past_key_states)
    log_numpy_array("past_value_states", past_values_states)
    log_numpy_array("history_len", history_len)
    log_numpy_array("ids_len", ids_len)
    log_numpy_array("position_ids", position_ids)
    log_numpy_array("pos_factor", pos_factor)

    # print(f"  hidden_states shape: {hidden_states.shape}")
    # print(f"  attention_mask shape: {attention_mask.shape}")
    # print(f"  past_key_states shape: {past_key_states.shape}")
    # print(f"  past_values_states shape: {past_values_states.shape}")
    # print(f"  history_len shape: {history_len.shape}")
    # print(f"  ids_len shape: {ids_len.shape}")
    # print(f"  position_ids shape: {position_ids.shape}")
    # print(f"  pos_factor shape: {pos_factor.shape}")

    token_id, past_key_states, past_values_states = ort_session_E.run(
        [out_name_E0, out_name_E1, out_name_E2],
        {
            in_name_E0: hidden_states,
            in_name_E1: attention_mask,
            in_name_E2: past_key_states,
            in_name_E3: past_values_states,
            in_name_E4: history_len,
            in_name_E5: ids_len,
            in_name_E6: position_ids,
            in_name_E7: pos_factor
        })

    print(f"\nStep {num_decode} outputs:")
    print(f"  token_id: {token_id}")
    log_numpy_array(f"past_key_states (step {num_decode})", past_key_states)
    log_numpy_array(f"past_values_states (step {num_decode})", past_values_states)

    if (token_id == 151643) | (token_id == 151645):
        print("\n[GENERATION] Reached stop token")
        break
    else:
        num_decode += 1
        if num_decode < 2:
            print("\n[GENERATION] First decode step adjustments:")
            history_len += ids_len[0]
            print(f"  Updated history_len: {history_len}")

            ids_len[0] = 1
            print(f"  Updated ids_len: {ids_len}")

            attention_mask = np.array([0.0], dtype=np.float16)
            print(f"  Updated attention_mask: {attention_mask}")

            if use_vision:
                pos_factor = np.array(pos_factor_v + ids_len[0], dtype=np.float16)
            else:
                pos_factor = np.array(history_len[0] + 1, dtype=np.float16)
            print(f"  Updated pos_factor: {pos_factor}")
        else:
            print(f"\n[GENERATION] Regular step {num_decode} adjustments:")
            history_len += 1
            pos_factor += 1
            print(f"  Updated history_len: {history_len}")
            print(f"  Updated pos_factor: {pos_factor}")

        input_ids[0] = token_id
        hidden_states = ort_session_B.run(
            [out_name_B0],
            {
                in_name_B0: input_ids,
                in_name_B1: ids_len
            })[0]
        log_numpy_array(f"hidden_states (step {num_decode})", hidden_states)

        decoded_token = tokenizer.decode(token_id)
        print(f"Decoded token: {decoded_token}")
        print(decoded_token, end="", flush=True)

generation_time = time.time() - end_time
print(f"\n\n[PERFORMANCE] Generation complete:")
print(f"  Total tokens generated: {num_decode}")
print(f"  Total time: {generation_time:.2f}s")
print(f"  Speed: {num_decode / generation_time:.3f} tokens/s")
