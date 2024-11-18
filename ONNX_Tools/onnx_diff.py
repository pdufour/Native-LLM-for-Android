import argparse
import os
import sys

import numpy as np
import onnxruntime as onnxrt


float_dict = {
    'tensor(float16)': 'float16',
    'tensor(float)': 'float32',
    'tensor(double)': 'float64'
}

integer_dict = {
    'tensor(int32)': 'int32',
    'tensor(int8)': 'int8',
    'tensor(uint8)': 'uint8',
    'tensor(int16)': 'int16',
    'tensor(uint16)': 'uint16',
    'tensor(int64)': 'int64',
    'tensor(uint64)': 'uint64'
}


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="onnx input model file")
    parser.add_argument("--node", help="node names to pull")
    args = parser.parse_args()

    args.input = args.input.split(",")
    if args.node:
        args.node = args.node.split(",")
    return args


def make_feed(sess):
    np.random.seed(1)
    feeds = {}
    for input_meta in sess.get_inputs():
        # replace any symbolic dimensions (value is None) with 1
        shape = [dim if dim and not isinstance(dim, str) else 1 for dim in
                 input_meta.shape]
        if input_meta.type in float_dict:
            feeds[input_meta.name] = np.random.rand(*shape).astype(float_dict[input_meta.type])
        elif input_meta.type in integer_dict:
            feeds[input_meta.name] = np.random.uniform(high=1000, size=tuple(shape)).astype(
                integer_dict[input_meta.type])
        elif input_meta.type == 'tensor(bool)':
            feeds[input_meta.name] = np.random.randint(2, size=tuple(shape)).astype('bool')
        else:
            print("unsupported input type {} for input {}".format(input_meta.type, input_meta.name))
            sys.exit(-1)
    return feeds


def main():
    args = get_args()

    sess1 = onnxrt.InferenceSession(args.input[0])
    sess2 = onnxrt.InferenceSession(args.input[1])
    # for meta in sess2.get_outputs():
    #    print(meta)

    feeds = make_feed(sess1)
    res1 = sess1.run([args.node[0]], feeds)  # fetch all outputs
    feeds = make_feed(sess2)
    res2 = sess2.run([args.node[1]], feeds)  # fetch all outputs

    np.testing.assert_allclose(res1, res2, rtol=0.1)
    print(f"ok {args.node[0]}, {args.node[1]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
