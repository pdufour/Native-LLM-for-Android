import time
import torch
import numpy as np
from PIL import Image
import shutil
import gc
import os
import transformers
import sys

try:
    from export_config import INPUT_IMAGE_SIZE, IMAGE_RESIZE, MAX_SEQ_LENGTH, HEIGHT_FACTOR, WIDTH_FACTOR
except:
    # Default Values if import failed
    INPUT_IMAGE_SIZE = [960, 960]                                       # Input image shape. Should be a multiple of GPU group (e.g., 16) for optimal efficiency.
    HEIGHT_FACTOR = 10                                                  # Adjust this value to determine the resize shape and vision resolution.
    WIDTH_FACTOR = 10                                                   # Adjust this value to determine the resize shape and vision resolution.
    IMAGE_RESIZE = [HEIGHT_FACTOR * 28, WIDTH_FACTOR * 28]              # 28 = self.patch_size * self.merge_size
    MAX_SEQ_LENGTH = 1024                                               # The max token length. Note, this value include the 10 tokens for system prompt and (HEIGHT_FACTOR * WIDTH_FACTOR) tokens for image prompt. Hence, only (MAX_SEQ_LENGTH - (HEIGHT_FACTOR * WIDTH_FACTOR) - 10) tokens for query + response.

path = sys.argv[1]  # Set the folder path where the Qwen2-VL whole project downloaded.

if len(sys.argv) > 2:
    script_dir = sys.argv[2]
else:
    script_dir = os.path.dirname(__file__)

# Replace the original "modeling_qwen2_vl.py" with the modified "modeling_qwen2_vl.py", which stored at the folder "modeling_modified".
modified_path_E = r'./modeling_modified/part_E/modeling_qwen2_vl.py'    # The path where the modified modeling_qwen2_vl.py stored.
onnx_model_E = os.path.join(script_dir, 'onnx/QwenVL_E.onnx')
transformers_qwen2_path = transformers.__file__.replace('__init__.py', 'models/qwen2_vl/modeling_qwen2_vl.py')  # Dynamically get the path to the transformers package

shutil.copyfile(modified_path_E, transformers_qwen2_path)
shutil.copyfile("export_config.py", transformers_qwen2_path.replace("modeling_qwen2_vl", "export_config"))

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer

# Load the model
with torch.inference_mode():
    model = Qwen2VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32, low_cpu_mem_usage=False)
    max_seq_len = MAX_SEQ_LENGTH
    num_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    past_key_states = torch.zeros((num_layers, num_key_value_heads, max_seq_len, head_dim), dtype=torch.float16)
    past_value_states = past_key_states
    history_len = torch.tensor([10], dtype=torch.long)  # "10" is just a dummy value.
    ids_len = torch.tensor([10], dtype=torch.long)  # "10" is just a dummy value.
    image_embed_size = WIDTH_FACTOR * HEIGHT_FACTOR
    image_pad_len = torch.tensor([image_embed_size], dtype=torch.long)
    ids_len = ids_len + image_pad_len
    hidden_states = torch.ones((max_seq_len, hidden_size), dtype=torch.float16)
    position_ids = torch.ones((3, 1, max_seq_len), dtype=torch.float16)
    kv_seq_len = ids_len + history_len
    attention_mask = torch.tensor([-65504.0], dtype=torch.float16)
    pos_factor = torch.tensor(0.0, dtype=torch.float16)

    torch._constrain_as_size(ids_len.item(), 0, 5000)

    sqrt_hidden_size = torch.sqrt(torch.tensor(hidden_size, dtype=torch.float32))
    model.model.norm.weight.data *= sqrt_hidden_size
    for i in range(num_layers):
        model.model.layers._modules[f'{i}'].input_layernorm.weight.data *= sqrt_hidden_size
        model.model.layers._modules[f'{i}'].post_attention_layernorm.weight.data *= sqrt_hidden_size

        layer_attn = model.model.layers._modules[f'{i}'].self_attn
        qkv_weight = torch.cat([layer_attn.q_proj.weight.data, layer_attn.k_proj.weight.data, layer_attn.v_proj.weight.data], dim=0)
        qkv_bias = torch.cat([layer_attn.q_proj.bias.data, layer_attn.k_proj.bias.data, layer_attn.v_proj.bias.data], dim=0)
        layer_attn.qkv_proj = torch.nn.Linear(qkv_weight.shape[1], qkv_weight.shape[0], bias=True)
        layer_attn.qkv_proj.weight.data.copy_(qkv_weight)
        layer_attn.qkv_proj.bias.data.copy_(qkv_bias)
        del layer_attn.q_proj
        del layer_attn.k_proj
        del layer_attn.v_proj
        gc.collect()

    print('\nExport Part_E Start...')
    torch.onnx.export(
        model,
        (hidden_states, attention_mask, past_key_states, past_value_states, history_len, ids_len, position_ids, pos_factor),
        onnx_model_E,
        input_names=[
            'hidden_states',
            'attention_mask',
            'past_key_states',
            'past_value_states',
            'history_len',
            'ids_len',
            'position_ids',
            'pos_factor'
        ],
        output_names=['max_logit_ids', 'past_key_states', 'past_value_states'],
        do_constant_folding=True,
        opset_version=20,
        dynamo=True,
        strict=False,
    )
    del model
    del hidden_states
    del attention_mask
    del past_key_states
    del past_value_states
    del history_len
    del ids_len
    del kv_seq_len
    del position_ids
    gc.collect()
    print('\nExport Part_E Done!')
