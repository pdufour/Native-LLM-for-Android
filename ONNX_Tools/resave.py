#!/usr/bin/env python3
import argparse
import os
import onnx
import onnx_graphsurgeon as gs
import numpy
from onnx import TensorProto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_model', help='Input ONNX model path')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite the input file instead of creating a new one')
    args = parser.parse_args()

    print('Loading model')
    model = onnx.load(args.input_model)
    print('Loading graph')
    graph = gs.import_onnx(model)
    modified_graph = graph

    if args.overwrite:
        # Directly overwrite the original file
        onnx.save(gs.export_onnx(modified_graph), args.input_model)
        print(f"Original file overwritten: {args.input_model}")
    else:
        # Use original naming scheme for new file
        input_dir = os.path.dirname(args.input_model)
        base_name = os.path.splitext(os.path.basename(args.input_model))[0]
        output_path = os.path.join(input_dir, f"{base_name}_fixed.onnx")

        onnx.save(gs.export_onnx(modified_graph), output_path)
        print(f"Modified model saved as: {output_path}")

if __name__ == '__main__':
    main()
