#!/usr/bin/env python3
"""Reproducible export: make the Voxtral audio encoder time-axis variable.

Input:  audio_encoder_q4f16.onnx + audio_encoder_q4f16.onnx_data (HF revision,
        as downloaded by the app; verified by SHA-256 below)
Output: audio_encoder_var_q4f16.onnx (+ the SAME .onnx_data file, reused)

The original graph hardcodes the 30 s window in three places:
  1. the input declaration (batch, 128, 3000)
  2. 128 Reshape shape constants containing the conv frame count 1500
  3. a positional-embedding Add broadcasting [1500, 1280]

This script patches exactly those: the time axis becomes symbolic, the frame
count is computed at runtime from the graph (Shape of the transposed conv
output), and the positional embedding is sliced to that count. No weights are
touched; the external data file is byte-identical (only referenced).

KNOWN SEMANTIC DIFFERENCE (verified with onnxruntime 1.31, see
docs/performance-0.5.4.md): the encoder attention is NOT causal. With a short
window, every audio token attends only over the real audio instead of the
full 30 s (which the fixed graph zero-pads). Output shapes are identical
per token; values differ. Speed: 5.5 s -> 0.7 s for 4 s of audio (x86 CI
container, direction only). Because of the context difference this stays an
EXPERIMENT behind a setting, off by default; run an A/B transcription test on
the target device before enabling.

Usage:
  python3 patch_encoder_varlen.py <model-dir>   # patches in place next to the original
"""
import hashlib, os, sys
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto

EXPECTED_ONNX_SHA256 = "156a6291053180757636c0caf5f6be3a4e08b90afd93d77646a6bd609ace206e"

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def patch(src_onnx, dst_onnx, data_name):
    m = onnx.load(src_onnx, load_external_data=False)
    g = m.graph

    inp = [x for x in g.input if x.name == 'audio_values'][0]
    d = inp.type.tensor_type.shape.dim
    assert d[1].dim_value == 128 and d[2].dim_value == 3000, "unexpected input shape"
    d[2].dim_param = 'time'
    d[2].ClearField('dim_value')

    # runtime frame count from the transposed conv output [1, frames, 1280]
    nodes = list(g.node)
    g.ClearField('node')
    g.node.extend([onnx.helper.make_node('Shape', inputs=['/audio_tower/Transpose_output_0'],
                                         outputs=['/var/frames_shape'])])
    g.initializer.append(onnx.helper.make_tensor('/var/idx1', TensorProto.INT64, [1], [1]))
    g.node.extend([onnx.helper.make_node('Gather', inputs=['/var/frames_shape', '/var/idx1'],
                                         outputs=['/var/frames'], axis=0)])
    g.node.extend(nodes)

    # Reshape shape constants with the hardcoded 1500 -> dynamic frames
    inits = {i.name: i for i in g.initializer}
    pre = {}
    uniq = 0
    for n in g.node:
        if n.op_type != 'Reshape':
            continue
        init = inits.get(n.input[1])
        if init is None:
            continue
        a = numpy_helper.to_array(init)
        if 1500 not in list(a):
            continue
        uniq += 1
        parts = []
        for idx, v in enumerate(a):
            if v == 1500:
                parts.append('/var/frames')
            else:
                cn = f'/var/c{uniq}_{idx}'
                g.initializer.append(onnx.helper.make_tensor(cn, TensorProto.INT64, [1], [int(v)]))
                parts.append(cn)
        dyn = f'/var/rs{uniq}'
        pre[n.name] = onnx.helper.make_node('Concat', inputs=parts, outputs=[dyn],
                                            axis=0, name=f'/var/concat{uniq}')
        n.input[1] = dyn
    final = []
    for n in g.node:
        if n.name in pre:
            final.append(pre[n.name])
        final.append(n)
    g.ClearField('node')
    g.node.extend(final)

    # positional embedding sliced to the runtime frame count
    g.initializer.append(onnx.helper.make_tensor('/var/zero', TensorProto.INT64, [1], [0]))
    g.initializer.append(onnx.helper.make_tensor('/var/axis0', TensorProto.INT64, [1], [0]))
    add2 = [x for x in g.node if x.name == '/audio_tower/Add_2'][0]
    W = 'audio_tower.embed_positions.weight'
    slice_node = onnx.helper.make_node('Slice', inputs=[W, '/var/zero', '/var/frames', '/var/axis0'],
                                       outputs=['/var/pos_sliced'], name='/var/posslice')
    final = []
    for n in g.node:
        if n.name == '/audio_tower/Add_2':
            final.append(slice_node)
            for idx, inp_ in enumerate(n.input):
                if inp_ == W:
                    n.input[idx] = '/var/pos_sliced'
        final.append(n)
    g.ClearField('node')
    g.node.extend(final)

    out = [x for x in g.output if x.name == 'audio_features'][0]
    od = out.type.tensor_type.shape.dim
    od[0].dim_param = 'rows'
    od[0].ClearField('dim_value')

    onnx.save(m, dst_onnx)
    m2 = onnx.load(dst_onnx)
    for init in m2.graph.initializer:
        for ref in init.external_data:
            if ref.key == 'location':
                ref.value = data_name
    onnx.save(m2, dst_onnx)
    return uniq

def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    d = sys.argv[1]
    src = os.path.join(d, 'audio_encoder_q4f16.onnx')
    dst = os.path.join(d, 'audio_encoder_var_q4f16.onnx')
    if not os.path.exists(src):
        print(f"missing {src}")
        sys.exit(1)
    digest = sha256(src)
    print(f"source graph sha256: {digest}")
    if digest != EXPECTED_ONNX_SHA256:
        print("WARNING: graph checksum differs from the verified revision -"
              " re-run the parity test before using the patched model")
    n = patch(src, dst, 'audio_encoder_var_q4f16.onnx_data')
    print(f"patched {n} reshape constants -> {dst}")
    print("reuse the original data file:")
    print(f"  cp {os.path.join(d, 'audio_encoder_q4f16.onnx_data')} "
          f"{os.path.join(d, 'audio_encoder_var_q4f16.onnx_data')}")

if __name__ == '__main__':
    main()
