import sys
import shutil
import gc
import torch
import transformers

try:
    from export_config import INPUT_IMAGE_SIZE, IMAGE_RESIZE, MAX_SEQ_LENGTH, WIDTH_FACTOR, HEIGHT_FACTOR
except:
    # Default Values if import failed
    INPUT_IMAGE_SIZE = [960, 960]                                                                  # Input image shape. Should be a multiple of GPU group (e.g., 16) for optimal efficiency.
    HEIGHT_FACTOR = 10                                                                             # Adjust this value to determine the resize shape and vision resolution.
    WIDTH_FACTOR = 10                                                                              # Adjust this value to determine the resize shape and vision resolution.
    IMAGE_RESIZE = [HEIGHT_FACTOR * 28, WIDTH_FACTOR * 28]                                         # 28 = self.patch_size * self.merge_size
    MAX_SEQ_LENGTH = 1024                                                                          # The max token length. Note, this value include the 10 tokens for system prompt and (HEIGHT_FACTOR * WIDTH_FACTOR) tokens for image prompt. Hence, only (MAX_SEQ_LENGTH - (HEIGHT_FACTOR * WIDTH_FACTOR) - 10) tokens for query + response.


import os

script_dir = os.path.dirname(__file__)
path = sys.argv[1]  # Set the folder path where the Qwen2-VL whole project downloaded.
# Replace the original "modeling_qwen2_vl.py" with the modified "modeling_qwen2_vl.py", which stored at the folder "modeling_modified".
modified_path_A = os.path.join(script_dir, 'modeling_modified/part_ABCD/modeling_qwen2_vl.py')  # The path where the modified modeling_qwen2_vl.py stored.
onnx_model_A = os.path.join(script_dir, 'onnx/QwenVL_A.onnx')                                          # Assign a path where the exported QwenVL model stored.
onnx_model_B = os.path.join(script_dir, 'onnx/QwenVL_B.onnx')
onnx_model_C = os.path.join(script_dir, 'onnx/QwenVL_C.onnx')
onnx_model_D = os.path.join(script_dir, 'onnx/QwenVL_D.onnx')

transformers_qwen2_path = transformers.__file__.replace('__init__.py', 'models/qwen2_vl/modeling_qwen2_vl.py')  # Dynamically get the path to the transformers package

shutil.copyfile(modified_path_A, transformers_qwen2_path)
shutil.copyfile("export_config.py", transformers_qwen2_path.replace("modeling_qwen2_vl", "export_config"))

from transformers import Qwen2VLForConditionalGeneration

# make export directory "onnx" (clear out any existing path)
shutil.rmtree(os.path.join(script_dir, 'onnx'), ignore_errors=True)
os.makedirs(os.path.join(script_dir, 'onnx'))

def quantize_to_uint8(tensor, scale, zero_point):
    return ((tensor - zero_point) * scale).round().clamp(0, 255).to(torch.uint8)

def convert_layernorms_to_float32(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.LayerNorm):
            # Create new LayerNorm with same params but force float32
            new_layer = torch.nn.LayerNorm(
                child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine
            ).float()
            # Copy weights if they exist
            if child.elementwise_affine:
                new_layer.weight.data = child.weight.data.float()
                new_layer.bias.data = child.bias.data.float()
            # Replace the layer
            setattr(module, name, new_layer)
        else:
            # Recursive call for nested modules
            convert_layernorms_to_float32(child)


class QwenVL_PartB(torch.nn.Module):
    def __init__(self, embed_data, scale, zero_point, hidden_size, max_seq_len):
        super(QwenVL_PartB, self).__init__()
        self.embed_data = embed_data
        self.scale = scale.half()
        self.zero_point = zero_point.half()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

    def forward(self, input_ids, ids_len):
        ids = input_ids[:ids_len]
        hidden_states = self.embed_data[ids] * self.scale[ids] + self.zero_point[ids]
        return torch.cat((hidden_states, torch.zeros(self.max_seq_len - ids_len, self.hidden_size, dtype=torch.float16)), dim=0)


class QwenVL_PartC(torch.nn.Module):
    def __init__(self, max_seq_len):
        super(QwenVL_PartC, self).__init__()
        self.position_ids = torch.arange(0, max_seq_len, dtype=torch.float16).repeat(1, 3, 1, 1)

    def forward(self, dummy):
        return self.position_ids[dummy]


class QwenVL_PartD(torch.nn.Module):
    def __init__(self, width_factor, height_factor, prompt_head_len, max_seq_len):
        super(QwenVL_PartD, self).__init__()
        self.image_factor = width_factor * height_factor
        self.prompt_head_len = prompt_head_len
        self.max_seq_len = max_seq_len
        self.image_factor_plus = self.image_factor + self.prompt_head_len
        self.position_ids = torch.arange(0, self.max_seq_len, dtype=torch.float16).repeat(3, 1, 1)
        self.position_ids[0, :, self.prompt_head_len: self.image_factor_plus] = self.prompt_head_len
        j = self.prompt_head_len
        for i in range(self.prompt_head_len, self.image_factor_plus, width_factor):
            self.position_ids[1, :, i: i + width_factor] = j
            j += 1
        self.start_id = self.prompt_head_len + width_factor
        fill_id = torch.arange(self.prompt_head_len, self.start_id, dtype=torch.float16)
        for i in range(self.start_id, self.image_factor_plus, width_factor):
            self.position_ids[2, :, i: i + width_factor] = fill_id
        self.fill_tail_position = torch.arange(self.start_id, self.max_seq_len, dtype=torch.float16).repeat(3, 1, 1)

    def forward(self, hidden_states, image_embed, ids_len, ids_len_minus, split_factor):
        part_1, part_2, _, part_3 = torch.split(hidden_states, [self.prompt_head_len, ids_len_minus, self.image_factor, split_factor], dim=0)
        self.position_ids[:, :, self.image_factor_plus:ids_len] = self.fill_tail_position[:, :, :ids_len - self.image_factor_plus]
        return torch.cat((part_1, image_embed, part_2, part_3), dim=0), self.position_ids

def convert_module_to_float32(module):
    """
    Recursively converts all parameters and buffers in a module to float32.
    """
    for child in module.children():
        convert_module_to_float32(child)

    if hasattr(module, 'weight') and module.weight is not None:
        module.weight.data = module.weight.data.float()
    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data = module.bias.data.float()

    # Convert any other parameters
    for param in module.parameters(recurse=False):
        param.data = param.data.float()

    # Convert buffers
    for buffer_name, buffer in module.named_buffers(recurse=False):
        if buffer is not None:
            setattr(module, buffer_name, buffer.float())

    # Set module to float32
    module.float()

# Load the model
with torch.inference_mode():
    model = Qwen2VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True)
    max_seq_len = MAX_SEQ_LENGTH
    num_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    prompt_head_len = 5  # \n<|im_start|>user\n<|vision_start|>
    ids_len = torch.tensor([10], dtype=torch.long)  # "10" is just a dummy value.
    history_len = torch.tensor([10], dtype=torch.long)  # "10" is just a dummy value.
    image_embed_size = WIDTH_FACTOR * HEIGHT_FACTOR
    image_embed = torch.ones((image_embed_size, hidden_size), dtype=torch.float16)
    pixel_values = torch.ones([1, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]]).to(torch.float32)
    image_pad_len = torch.tensor([image_embed_size], dtype=torch.long)
    ids_len = ids_len + image_pad_len
    kv_seq_len = ids_len + history_len
    input_ids = torch.ones(max_seq_len, dtype=torch.int32)
    hidden_states = torch.ones((max_seq_len, hidden_size), dtype=torch.float16)
    ids_len_minus = torch.tensor(ids_len[0] - prompt_head_len, dtype=torch.int32)
    split_factor = torch.tensor(max_seq_len - ids_len[0] - image_embed_size, dtype=torch.int32)
    dummy = torch.tensor(0, dtype=torch.int32)

    model.model.embed_tokens.weight.requires_grad = False
    data = model.model.embed_tokens.weight.data
    zero_point = (torch.min(data, dim=1)[0]).unsqueeze(1)
    scale = ((torch.max(data, dim=1)[0] - zero_point[:, 0]) / 255.0).unsqueeze(1)
    embed_data = quantize_to_uint8(data, 1.0 / scale, zero_point)

    print('\nExport Part_A Start...')
    # model = model.float()

    # convert_layernorms_to_float32(model)
    # convert_module_to_float32(model)

    torch.onnx.export(
        model,
        (pixel_values),
        onnx_model_A,
        input_names=[
            'pixel_values'
        ],
        output_names=['image_embed'],
        do_constant_folding=True,
        opset_version=20,
        dynamo=True,
        verbose=True,
    )
    del model
    del pixel_values

    gc.collect()
    print('\nExport Part_A Done! \n\nExport Part_B Start...')

    model = QwenVL_PartB(embed_data, scale, zero_point, hidden_size, max_seq_len)
    torch.onnx.export(
        model,
        (input_ids, ids_len),
        onnx_model_B,
        input_names=[
            'input_ids',
            'ids_len'
        ],
        output_names=['hidden_states'],
        do_constant_folding=True,
        opset_version=20,
        dynamo=False,
        verbose=True,
    )
    del model
    del embed_data
    del data
    del scale
    del zero_point
    del input_ids
    gc.collect()
    print('\nExport Part_B Done! \n\nExport Part_C Start...')

    model = QwenVL_PartC(max_seq_len)
    torch.onnx.export(
        model,
        (dummy,),
        onnx_model_C,
        input_names=[
            'dummy'
        ],
        output_names=['position_ids'],
        do_constant_folding=True,
        opset_version=20,
        dynamo=False,
        verbose=True,
    )
    del model
    del dummy
    print('\nExport Part_C Done! \n\nExport Part_C Start...')

    model = QwenVL_PartD(WIDTH_FACTOR, HEIGHT_FACTOR, prompt_head_len, max_seq_len)
    torch.onnx.export(
        model,
        (hidden_states, image_embed, ids_len, ids_len_minus, split_factor),
        onnx_model_D,
        input_names=[
            'hidden_states',
            'image_embed',
            'ids_len',
            'ids_len_minus',
            'split_factor'
        ],
        output_names=['hidden_states', 'position_ids'],
        do_constant_folding=True,
        opset_version=20,
        dynamo=False
    )
    print('\nExport Part_D Done! \n\nNext, please execute the QwenVL_Export_E.py to export the last part and run the test.')
