#!/usr/bin/env python3
import argparse
import os
import sys
import onnx
import gc
import psutil
import onnxslim
import numpy as np
from pathlib import Path
from onnxslim.cli._main import slim
from onnxslim.core import (
    convert_data_format,
    freeze,
    input_modification,
    input_shape_modification,
    optimize,
    output_modification,
    shape_infer,
)
from onnxslim.utils import (
    check_onnx,
    check_point,
    check_result,
    onnxruntime_inference,
    save,
    summarize_model,
    gen_onnxruntime_input_data,
    get_model_size_and_initializer_size,
    get_model_subgraph_size,
    get_max_tensor,
    format_model_info,
    print_model_info_as_table,
    dump_model_info_to_disk,
    get_opset,
    get_ir_version,
    calculate_tensor_size,
)


class CleanContext:
    def __init__(self, name=""):
        self.name = name
        self.initial_modules = set(sys.modules.keys())
        self.initial_mem = psutil.Process().memory_info().rss

    def __enter__(self):
        gc.collect()
        print(f"\n🔄 Starting {self.name}")
        print(f"📊 Initial Memory: {self._format_memory()}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up new modules
        current_modules = set(sys.modules.keys())
        for mod in current_modules - self.initial_modules:
            if not mod.startswith(('onnx', 'numpy', 'os', 'sys')):
                try:
                    del sys.modules[mod]
                except:
                    pass

        # Force GC
        gc.collect()
        gc.collect()
        gc.collect()

        # Clean arrays
        for obj in gc.get_objects():
            if isinstance(obj, np.ndarray):
                try:
                    obj.fill(0)
                except:
                    pass

        print(f"✨ Finished {self.name}")
        print(f"📊 Final Memory: {self._format_memory()}")
        delta = psutil.Process().memory_info().rss - self.initial_mem
        print(f"📈 Memory Delta: {self._format_bytes(delta)}")

    def _format_memory(self):
        process = psutil.Process()
        vm = psutil.virtual_memory()
        mem_gb = process.memory_info().rss / (1024 ** 3)
        return f"{mem_gb:.1f}GB (System: {vm.percent}%)"

    def _format_bytes(self, num_bytes):
        sign = '+' if num_bytes > 0 else ''
        for unit in ['B', 'KB', 'MB', 'GB']:
            if abs(num_bytes) < 1024 or unit == 'GB':
                return f"{sign}{num_bytes/1024**(['B','KB','MB','GB'].index(unit)):.1f} {unit}"
            num_bytes /= 1024

def wrapped_function(func):
    def wrapper(*args, **kwargs):
        with CleanContext(func.__name__) as ctx:
            return func(*args, **kwargs)
    return wrapper


# Define the print function with debug prints
def debug_print_model_info(*args, **kwargs):
    print(f"\n🔍 Debug: print_model_info_as_table called with args: {args[:2]}")
    result = print_model_info_as_table(*args, **kwargs)
    print("🔍 Debug: print_model_info_as_table completed")
    return result


# Wrap all the functions
# Complete list of functions to wrap
original_funcs = {
    # Core functions
    'convert_data_format': (onnxslim.core, convert_data_format),
    'freeze': (onnxslim.core, freeze),
    'input_modification': (onnxslim.core, input_modification),
    'input_shape_modification': (onnxslim.core, input_shape_modification),
    'optimize': (onnxslim.core, optimize),
    'output_modification': (onnxslim.core, output_modification),
    'shape_infer': (onnxslim.core, shape_infer),

    # Main utils functions
    'check_onnx': (onnxslim.utils, check_onnx),
    'check_point': (onnxslim.utils, check_point),
    'check_result': (onnxslim.utils, check_result),
    'onnxruntime_inference': (onnxslim.utils, onnxruntime_inference),
    'save': (onnxslim.utils, save),
    'summarize_model': (onnxslim.utils, summarize_model),

    # Additional utils functions
    'gen_onnxruntime_input_data': (onnxslim.utils, gen_onnxruntime_input_data),
    'get_model_size_and_initializer_size': (onnxslim.utils, get_model_size_and_initializer_size),
    'get_model_subgraph_size': (onnxslim.utils, get_model_subgraph_size),
    'get_max_tensor': (onnxslim.utils, get_max_tensor),
    'format_model_info': (onnxslim.utils, format_model_info),
    'print_model_info_as_table': (onnxslim.utils, print_model_info_as_table),
    'dump_model_info_to_disk': (onnxslim.utils, dump_model_info_to_disk),

    # Helper functions
    'get_opset': (onnxslim.utils, get_opset),
    'get_ir_version': (onnxslim.utils, get_ir_version),
    'calculate_tensor_size': (onnxslim.utils, calculate_tensor_size),
}

wrapped_funcs = {}
for name, (module, func) in original_funcs.items():
    if name == 'print_model_info_as_table':
        # Special wrapper for print function
        wrapped_funcs[name] = (module, wrapped_function(debug_print_model_info))
    else:
        wrapped_funcs[name] = (module, wrapped_function(func))


def process_model(input_path: str, **kwargs) -> bool:
    with CleanContext("Model Processing") as ctx:
        try:
            print("\n📥 Loading model...")
            model = onnx.load(input_path)

            # Replace functions with wrapped versions
            for name, (module, func) in wrapped_funcs.items():
                setattr(module, name, func)

            try:
                print("\n⚙️ Starting optimization...")
                kwargs['verbose'] = True
                model = slim(model, **kwargs)
                return True
            finally:
                # Restore original functions
                for name, (module, func) in original_funcs.items():
                    setattr(module, name, func)

        except Exception as e:
            print(f"\n❌ Error processing model: {e}")
            return False

def main():
    with CleanContext("Main Execution") as ctx:
        parser = argparse.ArgumentParser(description='Optimize ONNX model with memory-efficient functions')
        parser.add_argument('input_model', help='input onnx model')
        parser.add_argument('output_model', help='output onnx model')
        parser.add_argument('--kwargs', nargs=argparse.REMAINDER,
                           help='Additional arguments passed directly to ONNX-SLIM')

        args, unknown = parser.parse_known_args()

        if not os.path.exists(args.input_model):
            print(f"❌ Error: Input model {args.input_model} does not exist")
            exit(1)

        # Convert all unknown args to kwargs
        kwargs = {}
        i = 0
        print(unknown)
        while i < len(unknown):
            if unknown[i].startswith('--'):
                key = unknown[i][2:].replace('-', '_')
                if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                    kwargs[key] = unknown[i + 1]
                    i += 2
                else:
                    kwargs[key] = True
                    i += 1
            else:
                i += 1

        kwargs['output_model'] = args.output_model
        if process_model(args.input_model, **kwargs):
            print(f"\n✅ Model optimization completed successfully")
        else:
            print("\n❌ Failed to optimize model")
            exit(1)

if __name__ == '__main__':
    main()
