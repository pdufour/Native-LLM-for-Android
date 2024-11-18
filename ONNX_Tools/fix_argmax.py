#!/usr/bin/env python3
import argparse
import os
import onnx
import onnx_graphsurgeon as gs
import numpy
from onnx import TensorProto

def fix_argmax(graph):
    for node in graph.nodes:
        if node.op == "ArgMax":
            # Create new int64 output variable
            int64_var = gs.Variable(f"{node.name}_int64_output", dtype=numpy.int64)

            cast_node = gs.Node("Cast",
                        inputs=[int64_var],
                        outputs=[node.outputs[0]],
                        attrs={"to": TensorProto.INT64})

            node.outputs = [int64_var]
            graph.nodes.append(cast_node)

    graph.cleanup().toposort()
    return graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_model', help='Input ONNX model path')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite the input file instead of creating a new one')
    args = parser.parse_args()

    model = onnx.load(args.input_model)
    graph = gs.import_onnx(model)
    modified_graph = fix_argmax(graph)

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
