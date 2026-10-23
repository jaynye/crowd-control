"""A batched torch execution of an ONNX graph exported at batch 1.

`OnnxPort(model, example)` evaluates the shape and constant subgraphs once (from the
example inputs) and keeps a plan of the data-dependent operators, each executed with the
ONNX semantics. Every operator keeps the graph's batch-1 logical shapes, so
`torch.vmap(port)` adds an independent leading batch dimension: B rows give the same
numbers as B single-row ONNX Runtime calls (within floating-point batching differences),
and the call can be captured in a CUDA graph. It covers the operators of the GEAR-SONIC
kinematic planner (`planner_sonic.onnx`); others raise NotImplementedError.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper as nh


class OnnxPort:
  """Exact ONNX operators, preserving batch-1 logical shapes inside torch.vmap.

  Shape/constant subgraphs are evaluated once; no output/reference values are frozen.
  The extra vmap dimension is independent of the ONNX model's internal singleton.
  """

  def __init__(self, model, example, device="cuda"):
    self.device = device
    self.names = [v.name for v in model.graph.input]
    self.outputs = [v.name for v in model.graph.output]
    self.static = {v.name: torch.from_numpy(nh.to_array(v).copy()) for v in model.graph.initializer}
    self.gpu = {}
    self.plan = []
    self.flops = Counter()
    env = {k: torch.from_numpy(v).to(device) for k, v in example.items()}
    for n in model.graph.node:
      attr = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
      if n.op_type == "Constant":
        self.static[n.output[0]] = torch.from_numpy(nh.to_array(attr["value"]).copy())
        continue
      # Shape is metadata, even when its operand is a runtime value.
      if n.op_type == "Shape":
        x = env[n.input[0]] if n.input[0] in env else self.static[n.input[0]]
        y = torch.tensor(list(x.shape)[attr.get("start", 0) : attr.get("end", len(x.shape))], dtype=torch.int64)
        self.static[n.output[0]] = y
        continue
      static = all(not k or k in self.static for k in n.input)
      xs = [
        None if not k else self.static[k] if static else self.value(k, env, n.op_type, i) for i, k in enumerate(n.input)
      ]
      if n.op_type == "Slice":
        starts = xs[1].reshape(-1).tolist()
        ends = xs[2].reshape(-1).tolist()
        axes = xs[3].reshape(-1).tolist() if len(xs) > 3 else list(range(len(starts)))
        steps = xs[4].reshape(-1).tolist() if len(xs) > 4 else [1] * len(starts)
        attr["_slice_indices"] = [
          torch.tensor(list(range(xs[0].shape[d]))[slice(b, e, st)], dtype=torch.int64, device=xs[0].device)
          for b, e, d, st in zip(starts, ends, axes, steps)
        ]
      try:
        ys = self.op(n.op_type, xs, attr)
      except Exception as e:
        raise RuntimeError(f"{n.name} ({n.op_type}) {[None if x is None else tuple(x.shape) for x in xs]}: {e}") from e
      if not isinstance(ys, (list, tuple)):
        ys = [ys]
      if not static:
        if n.op_type == "MatMul":
          self.flops["MatMul"] += 2 * ys[0].numel() * xs[0].shape[-1]
        if n.op_type == "Gemm":
          self.flops["Gemm"] += 2 * ys[0].numel() * xs[0].shape[0 if attr.get("transA", 0) else 1]
        if n.op_type == "Conv":
          self.flops["Conv"] += 2 * ys[0].numel() * int(np.prod(xs[1].shape[1:]))
      target = self.static if static else env
      target.update(zip(n.output, ys))
      if not static:
        self.plan.append((n.op_type, list(n.input), list(n.output), attr, n.name))
    self.dead = {}
    last = {}
    for i, (_, ins, _outs, _, _) in enumerate(self.plan):
      for k in ins:
        if k and k not in self.static:
          last[k] = i
    for k in self.outputs:
      last.pop(k, None)
    for k, i in last.items():
      self.dead.setdefault(i, []).append(k)

  def value(self, k, env, op, i):
    if k in env:
      return env[k]
    # Shape/control tensors stay on the CPU and never depend on data values.
    meta = {
      "Reshape": {1},
      "Expand": {1},
      "Tile": {1},
      "Slice": {1, 2, 3, 4},
      "Squeeze": {1},
      "Unsqueeze": {1},
      "ConstantOfShape": {0},
      "Range": {0, 1, 2},
      "TopK": {1},
      "Pad": {1, 2, 3},
      "Resize": {1, 2, 3},
      "ReduceSum": {1},
      "CumSum": {1},
      "Split": {1},
      "OneHot": {1},
      "Clip": {1, 2},
    }
    if i in meta.get(op, set()):
      return self.static[k]
    if k not in self.gpu:
      self.gpu[k] = self.static[k].to(self.device)
    return self.gpu[k]

  def __call__(self, *args):
    env = dict(zip(self.names, args))
    for i, (op, ins, outs, attr, name) in enumerate(self.plan):
      xs = [self.value(k, env, op, j) if k else None for j, k in enumerate(ins)]
      try:
        ys = self.op(op, xs, attr)
      except Exception as e:
        raise RuntimeError(f"{name} ({op}): {e}") from e
      if not isinstance(ys, (list, tuple)):
        ys = [ys]
      env.update(zip(outs, ys))
      for k in self.dead.get(i, []):
        env.pop(k, None)
    return tuple(env[k] if k in env else self.gpu.get(k, self.static[k]) for k in self.outputs)

  @staticmethod
  def op(op, x, a):
    def ints(t):
      return [int(v) for v in t.reshape(-1).tolist()]

    axis = a.get("axis", 0)
    unary = {
      "Identity": lambda z: z,
      "Reciprocal": torch.reciprocal,
      "Atan": torch.atan,
      "Cos": torch.cos,
      "Sin": torch.sin,
      "Neg": torch.neg,
      "Sqrt": torch.sqrt,
      "Abs": torch.abs,
      "Floor": torch.floor,
      "Not": torch.logical_not,
      "IsNaN": torch.isnan,
      "Relu": F.relu,
    }
    binary = {
      "Add": torch.add,
      "Sub": torch.sub,
      "Mul": torch.mul,
      "Pow": torch.pow,
      "Greater": torch.gt,
      "Less": torch.lt,
      "Equal": torch.eq,
      "LessOrEqual": torch.le,
      "GreaterOrEqual": torch.ge,
      "And": torch.logical_and,
      "Or": torch.logical_or,
      "MatMul": torch.matmul,
    }
    if op in unary:
      return unary[op](x[0])
    if op == "MatMul" and x[1].ndim == 2 and x[0].ndim >= 2:
      # One mm for every row: torch.matmul would broadcast the 2-D weight into a batched
      # matmul, which under vmap becomes thousands of matrix-vector products (7x slower for
      # the SONIC planner at batch 4,096; equal results).
      k, n = x[1].shape
      return (x[0].reshape(-1, k) @ x[1]).reshape(*x[0].shape[:-1], n)
    if op in binary:
      return binary[op](x[0], x[1])
    if op == "Cast":
      return x[0].to(
        {1: torch.float32, 6: torch.int32, 7: torch.int64, 9: torch.bool, 10: torch.float16, 11: torch.float64}[a["to"]]
      )
    if op == "Div":
      return torch.div(x[0], x[1], rounding_mode=None if x[0].is_floating_point() else "trunc")
    if op == "Mod":
      return (torch.fmod if a.get("fmod", 0) else torch.remainder)(x[0], x[1])
    if op == "Where":
      return torch.where(*x)
    if op in ("Max", "Min"):
      y = x[0]
      for z in x[1:]:
        y = (torch.maximum if op == "Max" else torch.minimum)(y, z)
      return y
    if op == "Reshape":
      shape = ints(x[1])
      shape = [x[0].shape[i] if v == 0 and not a.get("allowzero", 0) else v for i, v in enumerate(shape)]
      return x[0].reshape(shape)
    if op == "Expand":
      return x[0].expand(torch.broadcast_shapes(x[0].shape, ints(x[1])))
    if op == "Tile":
      return x[0].repeat(ints(x[1]))
    if op == "Unsqueeze":
      y = x[0]
      axes = sorted(v if v >= 0 else v + y.ndim + x[1].numel() for v in ints(x[1]))
      for d in axes:
        y = y.unsqueeze(d)
      return y
    if op == "Squeeze":
      y = x[0]
      for d in sorted([v % y.ndim for v in ints(x[1])], reverse=True):
        y = y.squeeze(d)
      return y
    if op == "Transpose":
      return x[0].permute(a.get("perm", list(reversed(range(x[0].ndim)))))
    if op == "Concat":
      return torch.cat(x, dim=axis)
    if op == "Split":
      return torch.split(x[0], ints(x[1]), dim=axis)
    if op == "Slice":
      y = x[0]
      starts = ints(x[1])
      ints(x[2])
      axes = ints(x[3]) if len(x) > 3 else list(range(len(starts)))
      ints(x[4]) if len(x) > 4 else [1] * len(starts)
      for d, idx in zip(axes, a["_slice_indices"]):
        y = torch.index_select(y, d, idx)
      return y
    if op == "Gather":
      idx = x[1].long()
      idx = torch.remainder(idx, x[0].shape[axis])
      return torch.index_select(x[0], axis, idx.reshape(-1)).reshape(
        x[0].shape[: axis % x[0].ndim] + idx.shape + x[0].shape[axis % x[0].ndim + 1 :]
      )
    if op == "GatherElements":
      return torch.gather(x[0], axis, torch.remainder(x[1].long(), x[0].shape[axis]))
    if op == "ScatterElements":
      return torch.scatter(x[0], axis, x[1].long(), x[2])
    if op == "ScatterND":
      idx = tuple(x[1][..., j].long() for j in range(x[1].shape[-1]))
      return torch.index_put(x[0], idx, x[2], accumulate=a.get("reduction", b"none") == b"add")
    if op.startswith("Reduce"):
      axes = tuple(ints(x[1]) if len(x) > 1 else a.get("axes", list(range(x[0].ndim))))
      keep = bool(a.get("keepdims", 1))
      if op == "ReduceSum":
        return torch.sum(x[0], dim=axes, keepdim=keep)
      if op == "ReduceL2":
        return torch.sqrt(torch.sum(x[0] * x[0], dim=axes, keepdim=keep))
    if op in ("ArgMax", "ArgMin"):
      return (torch.argmax if op == "ArgMax" else torch.argmin)(x[0], dim=axis, keepdim=bool(a.get("keepdims", 1)))
    if op == "OneHot":
      depth = int(x[1])
      idx = torch.remainder(x[0].long(), depth)
      y = idx.unsqueeze(-1) == torch.arange(depth, device=idx.device)
      y = torch.where(y, x[2][1], x[2][0])
      return y.movedim(-1, axis) if axis != -1 else y
    if op == "ConstantOfShape":
      return torch.full(
        ints(x[0]),
        nh.to_array(a["value"]).reshape(-1)[0].item(),
        dtype=torch.from_numpy(nh.to_array(a["value"]).copy()).dtype,
      )
    if op == "Range":
      return torch.arange(x[0].item(), x[1].item(), x[2].item(), dtype=x[0].dtype)
    if op == "Gemm":
      u = x[0].t() if a.get("transA", 0) else x[0]
      v = x[1].t() if a.get("transB", 0) else x[1]
      return a.get("alpha", 1) * torch.matmul(u, v) + (a.get("beta", 1) * x[2] if len(x) > 2 else 0)
    if op == "LeakyRelu":
      return F.leaky_relu(x[0], a.get("alpha", 0.01))
    if op == "Softmax":
      return F.softmax(x[0], dim=a.get("axis", -1))
    if op == "LayerNormalization":
      return F.layer_norm(x[0], x[0].shape[a.get("axis", -1) :], x[1], x[2], a.get("epsilon", 1e-5))
    if op == "Flatten":
      return x[0].reshape(int(np.prod(x[0].shape[:axis])), int(np.prod(x[0].shape[axis:])))
    if op == "Conv":
      pads = a.get("pads", [0, 0])
      assert pads[: len(pads) // 2] == pads[len(pads) // 2 :]
      return F.conv1d(
        x[0],
        x[1],
        x[2] if len(x) > 2 else None,
        stride=tuple(a.get("strides", [1])),
        padding=tuple(pads[: len(pads) // 2]),
        dilation=tuple(a.get("dilations", [1])),
        groups=a.get("group", 1),
      )
    if op == "Pad":
      pads = ints(x[1])
      pairs = list(zip(pads[: len(pads) // 2], pads[len(pads) // 2 :]))
      return F.pad(
        x[0],
        [v for pair in reversed(pairs) for v in pair],
        mode=a.get("mode", b"constant").decode(),
        value=float(x[2]) if len(x) > 2 and x[2] is not None else 0,
      )
    if op == "Resize":
      sizes = ints(x[3])[-(x[0].ndim - 2) :] if len(x) > 3 and x[3] is not None else None
      scales = tuple(float(v) for v in x[2].tolist()[2:]) if sizes is None else None
      mode = a.get("mode", b"nearest").decode()
      return F.interpolate(
        x[0],
        size=sizes,
        scale_factor=scales,
        mode="linear" if mode == "linear" else "nearest",
        align_corners=a.get("coordinate_transformation_mode") == b"align_corners" if mode == "linear" else None,
      )
    if op == "Clip":
      return torch.clamp(
        x[0],
        min=float(x[1]) if len(x) > 1 and x[1] is not None else None,
        max=float(x[2]) if len(x) > 2 and x[2] is not None else None,
      )
    if op == "TopK":
      # ONNX ties favor the lower index, unlike torch.topk.
      idx = torch.argsort(x[0], dim=axis, descending=bool(a.get("largest", 1)), stable=True).narrow(axis, 0, int(x[1]))
      return torch.gather(x[0], axis, idx), idx
    if op == "CumSum":
      return torch.cumsum(x[0], dim=int(x[1]))
    if op == "Einsum":
      return torch.einsum(a["equation"].decode(), *x)
    raise NotImplementedError(op)
