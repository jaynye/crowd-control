"""Throwaway release-inspection and batching probe. Run on a Slurm compute node."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper as nh

MODEL = Path("checkpoints/GEAR-SONIC/planner_sonic.onnx")


def graphs(g, prefix="main"):
  yield prefix, g
  for n in g.node:
    for a in n.attribute:
      if a.type == onnx.AttributeProto.GRAPH:
        yield from graphs(a.g, prefix + "/" + n.name + "/" + a.name)


def inspect(run):
  start = time.perf_counter()
  m = onnx.load(MODEL)
  rec = {
    "load_s": time.perf_counter() - start,
    "sha256": hashlib.file_digest(MODEL.open("rb"), "sha256").hexdigest(),
    "opsets": [(o.domain, o.version) for o in m.opset_import],
    "graphs": [],
  }
  dump = []
  for path, g in graphs(m.graph):
    constants = {x.name: nh.to_array(x) for x in g.initializer if np.prod(x.dims) < 100}
    for n in g.node:
      if n.op_type == "Constant":
        for a in n.attribute:
          if a.name == "value" and np.prod(a.t.dims) < 100:
            constants[n.output[0]] = nh.to_array(a.t)
    d = {
      "path": path,
      "nodes": len(g.node),
      "ops": dict(Counter(n.op_type for n in g.node)),
      "inputs": [str(x) for x in g.input],
      "outputs": [str(x) for x in g.output],
      "initializers": [(x.name, list(x.dims)) for x in g.initializer],
      "shape_constants": {k: v.tolist() for k, v in constants.items()},
      "sensitive": [],
    }
    for n in g.node:
      if n.op_type in {
        "Loop",
        "If",
        "Scan",
        "Reshape",
        "Expand",
        "Tile",
        "Gather",
        "Squeeze",
        "Slice",
        "RandomUniform",
        "RandomUniformLike",
        "Multinomial",
        "TopK",
      }:
        d["sensitive"].append(
          {
            "name": n.name,
            "op": n.op_type,
            "in": list(n.input),
            "out": list(n.output),
            "attrs": [
              (a.name, onnx.helper.get_attribute_value(a) if a.type != onnx.AttributeProto.GRAPH else "<graph>")
              for a in n.attribute
            ],
            "constants": {k: constants[k].tolist() for k in n.input if k in constants},
          }
        )
      dump.append(f"{path} {n.name} {n.op_type} {list(n.input)} -> {list(n.output)}")
    rec["graphs"].append(d)
  run.write_json("inspection.json", rec)
  (run.path / "nodes.txt").write_text("\n".join(dump))
  print(json.dumps({k: v for k, v in rec.items() if k != "graphs"}), flush=True)
  for d in rec["graphs"]:
    print(d["path"], d["nodes"], d["ops"], flush=True)
    print("INPUTS", d["inputs"], "OUTPUTS", d["outputs"], flush=True)


def inputs(case=0):
  q = np.zeros((1, 4, 36), np.float32)
  q[:, :, 2] = 0.788740
  q[:, :, 3] = 1
  # Default robot pose, MuJoCo tree order; supported initial context from deployment.
  q[:, :, 7:] = [
    -0.1,
    0,
    0,
    0.3,
    -0.2,
    0,
    -0.1,
    0,
    0,
    0.3,
    -0.2,
    0,
    0,
    0,
    0,
    0.2,
    0.2,
    0,
    1.28,
    0,
    0,
    0,
    0.2,
    -0.2,
    0,
    1.28,
    0,
    0,
    0,
  ]
  heading = 0.3 * case
  if case:
    q[:, :, 0] += case * 0.25
    q[:, :, 1] -= case * 0.13
    q[:, :, 3] = np.cos(heading / 2)
    q[:, :, 6] = np.sin(heading / 2)
  a = dict(
    context_mujoco_qpos=q,
    target_vel=np.array([0.8 + 0.1 * case], np.float32),
    mode=np.array([2 if case != 3 else 0], np.int64),
    movement_direction=np.array([[np.cos(heading), np.sin(heading), 0]], np.float32),
    facing_direction=np.array([[np.cos(heading + 0.1 * case), np.sin(heading + 0.1 * case), 0]], np.float32),
    random_seed=np.array([1234 + 17 * case], np.int64),
    has_specific_target=np.array([[int(case == 2)]], np.int64),
    specific_target_positions=np.array(
      [[[0.2, 0.1, 0.78874], [0.3, 0.2, 0.78874], [0.4, 0.3, 0.78874], [0.5, 0.4, 0.78874]]], np.float32
    ),
    specific_target_headings=np.full((1, 4), heading, np.float32),
    allowed_pred_num_tokens=np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], np.int64),
    height=np.array([-1.0], np.float32),
  )
  if case == 3:
    a["movement_direction"][:] = 0
    a["allowed_pred_num_tokens"][:] = 0
    a["allowed_pred_num_tokens"][0, -1] = 1
  return a


def session(path=MODEL, gpu=True):
  import onnxruntime as ort

  ort.set_default_logger_severity(3)
  ort.preload_dlls(directory="")
  so = ort.SessionOptions()
  so.intra_op_num_threads = 1
  so.inter_op_num_threads = 1
  so.log_severity_level = 3
  so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  return ort.InferenceSession(
    str(path),
    so,
    providers=[("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"],
  )


def compare(a, b):
  return [
    {
      "shape": list(x.shape),
      "reference_shape": list(y.shape),
      "max_abs": float(np.max(np.abs(x.astype(float) - y.astype(float)))) if x.shape == y.shape else None,
      "rmse": float(np.sqrt(np.mean((x.astype(float) - y.astype(float)) ** 2))) if x.shape == y.shape else None,
      "finite": bool(np.isfinite(x).all()),
    }
    for x, y in zip(a, b)
  ]


def ort_probe(run):
  import onnxruntime as ort
  import torch

  start = time.perf_counter()
  ss = session()
  rec = {
    "providers": ss.get_providers(),
    "ort": ort.__version__,
    "startup_s": time.perf_counter() - start,
    "cases": [],
    "timing": [],
  }
  run.write_json("result.json", rec)
  assert ss.get_providers()[0] == "CUDAExecutionProvider", ss.get_providers()
  print("SESSION", rec, flush=True)
  refs = []
  for c in range(4):
    a = inputs(c)
    t = time.perf_counter()
    out = ss.run(None, a)
    elapsed = time.perf_counter() - t
    refs.append(out)
    np.savez(run.path / f"reference_{c}.npz", **a, ref_qpos=out[0], ref_frames=out[1])
    row = {"case": c, "frames": out[1].tolist(), "first_s": elapsed, "repeat": compare(ss.run(None, a), out)}
    rec["cases"].append(row)
    print("REFERENCE", row, flush=True)
  a = inputs(0)
  for method in ["run", "iobinding"]:
    if method == "run":

      def fn():
        return ss.run(None, a)
    else:
      io = ss.io_binding()
      held = {k: torch.from_numpy(v).cuda() for k, v in a.items()}
      for k, v in held.items():
        io.bind_input(k, "cuda", 0, a[k].dtype, tuple(v.shape), v.data_ptr())
      for o in ss.get_outputs():
        io.bind_output(o.name, "cuda", 0)

      def fn():
        return ss.run_with_iobinding(io)

      fn()
      io.synchronize_outputs()
      rec[method + "_parity"] = compare(io.copy_outputs_to_cpu(), refs[0])
    for _ in range(3):
      fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(12):
      t = time.perf_counter()
      fn()
      torch.cuda.synchronize()
      samples.append(time.perf_counter() - t)
    # Actual 4096 independent full planner calls, retained fixed batch 1.
    t = time.perf_counter()
    for _ in range(4096):
      fn()
    torch.cuda.synchronize()
    total = time.perf_counter() - t
    row = {
      "method": method,
      "batch": 1,
      "population": 4096,
      "population_s": total,
      "plans_s": 4096 / total,
      "latency_p50_s": float(np.median(samples)),
      "latency_p95_s": float(np.quantile(samples, 0.95)),
    }
    rec["timing"].append(row)
    run.write_json("result.json", rec)
    print("TIMING", row, flush=True)
  del ss
  m = onnx.load(MODEL)
  for v in m.graph.input:
    v.type.tensor_type.shape.dim[0].dim_param = "batch"
  for v in m.graph.output:
    v.type.tensor_type.shape.dim[0].dim_param = "batch"
  dyn = run.path / "boundary_dynamic.onnx"
  onnx.save(m, dyn)
  del m
  try:
    ss = session(dyn)
    rec["boundary_dynamic_batch1"] = compare(ss.run(None, inputs()), refs[0])
    rec["boundary_dynamic_batch2"] = compare(
      ss.run(None, {k: np.repeat(v, 2, axis=0) for k, v in inputs().items()}), refs[0]
    )
  except Exception as e:
    rec["boundary_dynamic_error"] = str(e)
    print("DYNAMIC_ERROR", str(e), flush=True)
  run.write_json("result.json", rec)


def convert(run):
  import torch
  from onnx2torch import convert as cv
  from onnx2torch.node_converters.registry import get_converter

  m = onnx.load(MODEL)
  missing = {}
  for op in sorted({n.op_type for n in m.graph.node}):
    try:
      get_converter(op, 17)
    except Exception as e:
      missing[op] = str(e)
  rec = {"missing": missing}
  print("CONVERTER_SUPPORT", rec, flush=True)
  try:
    t = time.perf_counter()
    module = cv(m)
    rec["conversion_s"] = time.perf_counter() - t
    a = inputs()
    y = module(*(torch.from_numpy(a[k.name]) for k in m.graph.input))
    rec["outputs"] = [list(v.shape) for v in y]
  except Exception as e:
    rec["error"] = str(e)
    print("CONVERSION_ERROR", str(e), flush=True)
  run.write_json("result.json", rec)


from crowdmotion.backends.onnx_port import OnnxPort as HandPort  # noqa: E402  (moved 2026-10-18)


def torch_probe(run):
  import torch

  torch.set_num_threads(1)
  precision = os.getenv("SONIC_PRECISION", "fp32")
  torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
  torch.backends.cudnn.allow_tf32 = precision == "tf32"
  rec = {"precision": precision}
  try:
    model = onnx.load(MODEL)
    start = time.perf_counter()
    with torch.inference_mode():
      port = HandPort(model, inputs())
      del model
      rec["startup_s"] = time.perf_counter() - start
      rec["dense_flops_per_plan"] = dict(port.flops)
      refs = []
      ss = session()
      for c in range(4):
        a = inputs(c)
        ref = ss.run(None, a)
        refs.append(ref)
        got = port(*(torch.from_numpy(a[k]).cuda() for k in port.names))
        row = compare([v.cpu().numpy() for v in got], ref)
        rec[f"case_{c}"] = row
        print("HAND_PARITY", c, row, flush=True)
      run.write_json("result.json", rec)
      del ss
      batched = torch.vmap(port)
      batches = (
        [4096]
        if run.record.get("validate_only")
        else ([2, 16, 64, 256, 4096] if precision == "fp32" else [16, 64, 256, 4096])
      )
      for B in batches:
        aa = [inputs(i % 4) for i in range(B)]
        ts = [torch.from_numpy(np.stack([a[k] for a in aa])).cuda() for k in port.names]
        try:
          got = batched(*ts)
          torch.cuda.synchronize()
          rec[f"batch_{B}_shapes"] = [list(v.shape) for v in got]
          for j in range(min(4, B)):
            rec[f"batch{B}_ort_case{j}"] = compare([v[j].cpu().numpy() for v in got], refs[j])
          if B >= 4:
            # Every repeated request and reversed request keeps its own seed/context.
            rec[f"batch{B}_repeat_max_abs"] = float((got[0] - got[0][:4].repeat(B // 4, 1, 1, 1)).abs().max())
          run.write_json("result.json", rec)
          if B == 2:
            for i in range(2):
              ref = port(*(t[i] for t in ts))
              rec[f"batch2_case{i}"] = compare([v[i].cpu().numpy() for v in got], [v.cpu().numpy() for v in ref])
          g = torch.cuda.CUDAGraph()
          # Warm the same allocations before capture.
          for _ in range(0 if run.record.get("validate_only") else 2):
            batched(*ts)
          torch.cuda.synchronize()
          with torch.cuda.graph(g):
            try:
              out = batched(*ts)
            except Exception as e:
              print("CAPTURE_INNER", repr(e), flush=True)
              raise
          g.replay()
          torch.cuda.synchronize()
          for j in range(min(4, B)):
            rec[f"captured_batch{B}_case{j}"] = compare([v[j].cpu().numpy() for v in out], refs[j])
          if run.record.get("validate_only"):
            # In-place buffer changes must affect replay, without changing captured addresses.
            for t in ts:
              t.copy_(t.roll(1, dims=0))
            g.replay()
            torch.cuda.synchronize()
            for j in range(4):
              rec[f"changed_inputs_case{j}"] = compare([v[j].cpu().numpy() for v in out], refs[(j - 1) % 4])
            print(
              "CAPTURE_CHANGED_INPUTS",
              {k: v for k, v in rec.items() if k.startswith(("captured_", "changed_"))},
              flush=True,
            )
            run.write_json("result.json", rec)
            break
          samples = []
          for _ in range(6):
            t = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - t)
          chunks = (4096 + B - 1) // B
          t = time.perf_counter()
          for _ in range(chunks):
            g.replay()
          torch.cuda.synchronize()
          elapsed = time.perf_counter() - t
          row = {
            "batch": B,
            "population": 4096,
            "population_s": elapsed,
            "plans_s": 4096 / elapsed,
            "latency_p50_s": float(np.median(samples)),
            "latency_p95_s": float(np.quantile(samples, 0.95)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
          }
          rec.setdefault("timing", []).append(row)
          print("HAND_TIMING", row, flush=True)
          del g, out, got, ts
          torch.cuda.empty_cache()
        except Exception as e:
          rec[f"batch_{B}_error"] = str(e)
          print("HAND_BATCH_ERROR", B, str(e), flush=True)
          if B == 2:
            break
        run.write_json("result.json", rec)
  except Exception as e:
    rec["error"] = str(e)
    print("HAND_ERROR", str(e), flush=True)
    import traceback

    traceback.print_exc()
  run.write_json("result.json", rec)


def profile(run):
  import onnxruntime as ort

  ort.set_default_logger_severity(3)
  ort.preload_dlls(directory="")
  so = ort.SessionOptions()
  so.intra_op_num_threads = 1
  so.enable_profiling = True
  so.profile_file_prefix = str(run.path / "ort_profile")
  so.log_severity_level = 3
  s = ort.InferenceSession(
    str(MODEL), so, providers=[("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"]
  )
  for _ in range(3):
    s.run(None, inputs())
  path = s.end_profiling()
  events = json.loads(Path(path).read_text())
  tot = Counter()
  counts = Counter()
  ops = Counter()
  for e in events:
    if e.get("cat") == "Node" and "provider" in e.get("args", {}):
      provider = e["args"]["provider"]
      tot[provider] += e.get("dur", 0)
      counts[provider] += 1
      ops[(provider, e["args"].get("op_name", ""))] += e.get("dur", 0)
  rec = {
    "provider_us": dict(tot),
    "node_events": dict(counts),
    "top_ops_us": [(str(k), v) for k, v in ops.most_common(15)],
  }
  run.write_json("result.json", rec)
  print("PROFILE", rec, flush=True)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("task", choices=["inspect", "ort", "convert", "torch", "profile"])
  p.add_argument("--validate-only", action="store_true")
  args = p.parse_args()
  from crowdmotion.core.rundir import RunDir

  with RunDir.create(
    "sonic_" + args.task + "_" + os.getenv("SONIC_PRECISION", "fp32") + ("_validate" if args.validate_only else ""),
    record={"command": sys.argv, "slurm_job": os.getenv("SLURM_JOB_ID"), "validate_only": args.validate_only},
  ) as run:
    (run.path / "probe.py").write_text(Path(__file__).read_text())
    print("RUN", run.path, flush=True)
    import torch

    with torch.autocast("cuda", dtype=torch.float16, enabled=os.getenv("SONIC_PRECISION") == "amp16"):
      globals()[{"ort": "ort_probe", "torch": "torch_probe"}.get(args.task, args.task)](run)


if __name__ == "__main__":
  main()
