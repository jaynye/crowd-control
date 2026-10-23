"""Does planner_sonic.onnx contain a MotionBricks checkpoint's weights?

    .venv/bin/python -I scripts/smoke/motionbricks_weights_match.py <model-step=*.ckpt> <planner_sonic.onnx>

Run it on a compute node (it loads about 1 GB of tensors). The checkpoints are Git LFS
objects of GR00T-WholeBodyControl (`motionbricks/out/`); fetch them into
`$CROWD_CHECKPOINTS/MotionBricks/` by their LFS oid, not into the clone.

For every tensor of the checkpoint's state_dict, look for an ONNX initializer with the
same values: equal, transposed, or as a contiguous row block (q/k/v slices of a fused
in_proj). Reports matched tensors and parameters, and the largest absolute difference.
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / ".cache/sonic-probe"))  # onnx
import collections
import json

import numpy as np
import onnx
import torch
from onnx import numpy_helper

CKPT = sys.argv[1]
ONNX = sys.argv[2]

try:
  ck = torch.load(CKPT, map_location="cpu", weights_only=True)
except Exception as exc:  # Lightning stores hyper-parameters as objects
  print("weights_only load failed:", type(exc).__name__, str(exc)[:200])
  raise
sd = ck["state_dict"] if "state_dict" in ck else ck
sd = {k: v.float().numpy() for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()}
print("checkpoint tensors", len(sd), "params", sum(v.size for v in sd.values()))
print("prefixes", collections.Counter(k.split(".")[0] for k in sd))

model = onnx.load(ONNX)
inits = [numpy_helper.to_array(t).astype(np.float32) for t in model.graph.initializer
         if t.data_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16)]
consts = []
for node in model.graph.node:
  if node.op_type == "Constant":
    for a in node.attribute:
      if a.name == "value" and a.t.data_type == onnx.TensorProto.FLOAT:
        consts.append(numpy_helper.to_array(a.t).astype(np.float32))
pool = [a for a in inits + consts if a.size >= 8]
print("onnx float initializers/constants", len(pool), "params", sum(a.size for a in pool))

by_size = collections.defaultdict(list)
for a in pool:
  by_size[a.size].append(a.reshape(-1))
rows = collections.defaultdict(list)  # last-dim width -> 2-D arrays, for slice matches
for a in pool:
  if a.ndim == 2:
    rows[a.shape[1]].append(a)
    rows[a.shape[0]].append(a.T)


def find(v):
  flat = v.reshape(-1)
  best = None
  for cand in by_size.get(v.size, []):
    d = float(np.abs(cand - flat).max())
    best = d if best is None else min(best, d)
    if v.ndim == 2:
      d = float(np.abs(cand - v.T.reshape(-1)).max())
      best = min(best, d)
  if best is not None and best < 1e-6:
    return "equal", best
  if v.ndim == 2:
    for a in rows.get(v.shape[1], []):
      if a.shape[0] > v.shape[0]:
        for s in range(0, a.shape[0] - v.shape[0] + 1, v.shape[0]):
          d = float(np.abs(a[s:s + v.shape[0]] - v).max())
          if d < 1e-6:
            return "slice", d
  return "missing", best


stats = collections.Counter()
params = collections.Counter()
examples = collections.defaultdict(list)
for k, v in sd.items():
  how, d = find(v)
  stats[how] += 1
  params[how] += v.size
  if len(examples[how]) < 12:
    examples[how].append((k, list(v.shape), d))
print(json.dumps({"tensors": stats, "params": params}, indent=1))
for how, ex in examples.items():
  print("==", how)
  for e in ex:
    print("  ", e)
