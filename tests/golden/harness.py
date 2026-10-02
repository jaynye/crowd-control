"""Golden outputs that pin behavior through the code migration.

  python tests/golden/harness.py capture [--fast] [--slow] [--cases ...] --out <dir>
  python tests/golden/harness.py check   [--fast] [--slow] [--cases ...] --out <dir>

`capture` computes every case with the code under test and writes
`tests/golden/expected/<case>[.fast].json`: dtype, shape, sha256 and summary
statistics of every array, plus the full JSON outputs. `check` recomputes and
compares bit for bit. Both save the full arrays to `--out/<case>.npz` so a
mismatch can be diagnosed against the capture run.

All cases run on CPU with one thread. `--slow` adds real ARDY and Kimodo CPU
cases, which need the checkpoints. The saved motion graph comes from
`--graph`, `$CROWD_GRAPH` or the artifact manifest.

Only the "API adapters" section and the test-double interfaces may change when
code moves. The computations, inputs and expected files must not.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
  os.environ[_var] = "1"

import argparse
import contextlib
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPECTED = HERE / "expected"


# --- API adapters (the only section edited when code moves) -------------------
def api() -> SimpleNamespace:
  from crowdmotion import cli
  from crowdmotion.backends.base import MotionChunk, MotionRequest
  from crowdmotion.backends.registry import ArdyConfig, KimodoConfig, make_backend
  from crowdmotion.behavior import commands
  from crowdmotion.behavior.planner import random_arc_paths
  from crowdmotion.bench import quality, throughput
  from crowdmotion.core.geometry import motion_to_mujoco
  from crowdmotion.core.rng import per_human_noise
  from crowdmotion.graph import GraphConfig, GraphRuntime, MotionGraph
  from crowdmotion.graph.audit import audit_graph
  from crowdmotion.iterate import evaluate
  from crowdmotion.library import generate
  from crowdmotion.sim.capsules import CapsuleSkeleton
  from crowdmotion.viz import replay

  def build_graph(lib, cfg):
    return MotionGraph(lib, GraphConfig(**cfg), device="cpu")

  def graph_tables(graph):
    out = {name: getattr(graph, name) for name in MotionGraph._TABLES}
    if hasattr(graph, "desc"):
      out["desc"] = graph.desc
    return out

  def ardy_backend():
    return make_backend(ArdyConfig(
      "ARDY-Core-RP-20FPS-Horizon40", device="cpu", steps=5, cfg_type="nocfg", precision="fp32", timing=False,
    ))

  def kimodo_backend():
    return make_backend(KimodoConfig(
      "Kimodo-SOMA-RP-v1.1", device="cpu", steps=4, precision="fp32", timing=False, new_frames=20,
    ))

  def plaza_env_cfg(graph_path, **kw):
    from crowdmotion.sim.mjlab.plaza import make_plaza_env_cfg

    return make_plaza_env_cfg(graph_path, **kw)

  def corridor_env_cfg(**kw):
    from crowdmotion.sim.mjlab.corridor.scene import make_crowd_env_cfg

    return make_crowd_env_cfg(**kw)

  return SimpleNamespace(
    MotionRequest=MotionRequest,
    MotionChunk=MotionChunk,
    per_human_noise=per_human_noise,
    build_graph=build_graph,
    graph_tables=graph_tables,
    graph_stats=lambda graph: graph.stats,
    save_graph=lambda graph, path, rest: graph.save(path, rest_joints=rest),
    load_graph=lambda path, overrides=None: MotionGraph.load(path, device="cpu", cfg_overrides=overrides),
    runtime=GraphRuntime,
    steering_paths=random_arc_paths,
    resolved_eval_config=evaluate.resolved_config,
    eval_rollout=lambda graph, cfg, seed: (lambda tr, st: (tr.arrays(), st))(*evaluate.rollout(graph, cfg, seed)),
    behavior_cli=lambda argv: cli.main(["behavior", *argv]),
    audit_graph=audit_graph,
    quality_rollout=quality.run_rollout,
    quality_metrics=quality.metrics,
    branch_check=quality.branch_check,
    generate_streams=generate.generate_streams,
    RandomCommands=commands.RandomCommands,
    CommandRanges=commands.CommandRanges,
    Population=throughput.Population,
    one_call=throughput.one_call,
    merge=throughput.merge,
    run_config=throughput.run_config,
    ardy_backend=ardy_backend,
    kimodo_backend=kimodo_backend,
    plaza_env_cfg=plaza_env_cfg,
    plaza_term_state=lambda term: {
      "boundary": term.state.boundary, "root_xz": term.state.root_xz, "heading": term.state.heading,
      "pose": term.state.pose, "counters": term.state.counters, "seeds": term.state.seeds,
      "frames": term.frames, "vel": term.vel, "command": term.command, "clock": term.clock,
      "edge_t": term.edge_t, "goal": term.behavior.goal, "wait_until": term.behavior.wait_until,
    },
    replay_scene=replay.replay_scene,
    CapsuleSkeleton=CapsuleSkeleton,
    ardy_to_mujoco=motion_to_mujoco,
    corridor_env_cfg=corridor_env_cfg,
  )


# --- Test doubles: their interfaces may follow the backend protocol; their math must not
class FakeOnline:
  """Deterministic stand-in for an online generator.

  Output depends on the seeds, counters, history, root path and fresh-start
  state, so any change in what a harness feeds the backend changes the result.
  """

  J = 8
  D = 12
  C = 4

  def __init__(self, A, flavor: str):
    self.A = A
    self.flavor = flavor
    self.device = torch.device("cpu")
    self.fps = 20.0
    self.frames_per_token = 4
    self.horizon = 40
    self.future_frames = 0
    self.transition_frames = 5
    self.precision = "fp32"
    self.cfg_type = "separated"
    self.steps = 5
    self.cfg_weight = (2.0, 2.0)
    self.conditions_mode = "vectorized"
    self.post_processing = False
    self.to_soma77 = False
    self.model = SimpleNamespace(diffusion=SimpleNamespace(num_base_steps=5))
    self.model_name = "fake"
    self.new_frames = 20
    self.joint_names = [f"j{i}" for i in range(self.J)]
    g = torch.Generator().manual_seed(1234 if flavor == "ardy" else 4321)
    off = torch.randn(self.J, 3, generator=g) * 0.15
    off[0] = 0
    off[:, 1] = torch.tensor([0.0, -0.85, -0.85, -0.87, -0.87, 0.75, 0.4, 0.4])
    self.offsets = off
    self.gain = 0.03 if flavor == "ardy" else 0.05

  def set_cfg_type(self, cfg_type: str) -> None:
    self.cfg_type = cfg_type

  def configure(self, cfg) -> None:
    """Backend protocol: apply a typed config (only frame counts matter to the fake)."""
    for name in ("future_frames", "new_frames", "transition_frames", "precision", "steps"):
      if hasattr(cfg, name):
        setattr(self, name, getattr(cfg, name))

  def skeleton_info(self) -> dict:
    return {
      "parents": [-1, 0, 0, 1, 2, 0, 5, 5],
      "foot_idx": [1, 3, 2, 4],
      "joint_names": list(self.joint_names),
      "head_idx": 5,
      "neutral_head_height": 1.65,
    }

  def empty_text(self, batch: int) -> torch.Tensor:
    return torch.zeros(batch, 1, 8)

  def _generate(self, req, frames: int):
    B = req.batch
    noise = self.A.per_human_noise(req.seeds, req.counters, (frames, self.D), self.device)
    if req.history is None:
      x0 = req.start_xz.float() if req.start_xz is not None else torch.zeros(B, 2)
      h0 = req.start_heading.float() if req.start_heading is not None else torch.zeros(B)
      f0 = torch.zeros(B, self.D)
    else:
      x0, h0 = req.history[:, -1, :2].float(), req.history[:, -1, 2].float()
      f0 = req.history.float().mean(1)
    if req.root_path is not None:
      path = req.root_path[:, :frames].float()
      if path.shape[1] < frames:
        path = torch.cat((path, path[:, -1:].expand(-1, frames - path.shape[1], -1)), 1)
    else:
      t = torch.arange(1, frames + 1, dtype=torch.float32)[None, :, None] / self.fps
      path = x0[:, None] + t * torch.stack((torch.sin(h0), torch.cos(h0)), -1)[:, None]
    xz = path + self.gain * noise[..., :2] + 0.01 * f0[:, None, 3:5]
    prev = torch.cat((x0[:, None], xz[:, :-1]), 1)
    d = xz - prev
    heading = torch.where(d.norm(dim=-1) > 1e-4, torch.atan2(d[..., 0], d[..., 1]), h0[:, None])
    feats = torch.cat((xz, heading[..., None], 0.5 * noise[..., 3:] + 0.1 * f0[:, None, 3:]), -1)
    root = torch.stack((xz[..., 0], 0.9 + 0.01 * noise[..., 2], xz[..., 1]), -1)
    c, s = torch.cos(heading)[..., None], torch.sin(heading)[..., None]
    ox, oy, oz = self.offsets[:, 0], self.offsets[:, 1], self.offsets[:, 3 - 1]
    wob = 0.01 * noise[..., None, :3].expand(-1, -1, self.J, -1) * torch.arange(1, self.J + 1)[:, None] / self.J
    joints = torch.stack(
      (root[..., None, 0] + ox * c + oz * s, root[..., None, 1] + oy, root[..., None, 2] - ox * s + oz * c), -1
    ) + wob
    contacts = noise[..., 4 : 4 + self.C] > 0
    rot = torch.zeros(B, frames, self.J, 3, 3)
    rot[..., 0, 0] = c
    rot[..., 0, 2] = s
    rot[..., 1, 1] = 1
    rot[..., 2, 0] = -s
    rot[..., 2, 2] = c
    return self.A.MotionChunk(
      features=feats, root_pos=root, local_rot=rot, posed_joints=joints, contacts=contacts,
      heading=heading, fps=self.fps,
    )


class FakeArdy(FakeOnline):
  name = "ardy"

  def __init__(self, A):
    super().__init__(A, "ardy")

  @property
  def path_frames(self) -> int:
    return self.horizon + self.future_frames

  @property
  def frames_per_call(self) -> int:
    return self.horizon

  def generate(self, req):
    return self._generate(req, self.horizon + self.future_frames)


class FakeKimodo(FakeOnline):
  name = "kimodo"

  def __init__(self, A):
    super().__init__(A, "kimodo")

  @property
  def path_frames(self) -> int:
    return self.new_frames

  @property
  def frames_per_call(self) -> int:
    return self.new_frames

  def generate(self, req):
    return self._generate(req, self.new_frames)


def synthetic_library(n: int = 24, t: int = 96, seed: int = 0) -> dict:
  """Library dict in the `save_library` layout with a 4-joint foot set and a head."""
  g = torch.Generator().manual_seed(seed)
  fps, J = 20.0, 8
  rate = (torch.rand(n, 1, generator=g) - 0.5) * 2.0
  heading = rate * torch.arange(t)[None] / fps + torch.rand(n, 1, generator=g) * 6
  speed = torch.rand(n, 1, generator=g) * 1.6
  step = torch.stack((torch.sin(heading), torch.cos(heading)), -1) * speed[..., None] / fps
  xz = torch.cumsum(step, 1)
  root = torch.stack((xz[..., 0], torch.full_like(xz[..., 0], 0.9), xz[..., 1]), -1)
  offsets = torch.randn(J, 3, generator=g) * 0.15
  offsets[0] = 0
  offsets[:, 1] = torch.tensor([0.0, -0.85, -0.85, -0.87, -0.87, 0.75, 0.4, 0.4])
  wobble = 0.05 * torch.sin(torch.arange(t)[None, :, None, None] * 0.7 + torch.randn(n, 1, J, 3, generator=g))
  stoop = (torch.rand(n, 1, 1, generator=g) < 0.2).float() * torch.tensor([0, 0, 0, 0, 0, -0.3, 0, 0])[None, None]
  local = offsets[None, None] + wobble
  local[..., 1] = local[..., 1] + stoop
  c, s = torch.cos(heading)[..., None], torch.sin(heading)[..., None]
  wx = local[..., 0] * c + local[..., 2] * s
  wz = -local[..., 0] * s + local[..., 2] * c
  joints = torch.stack((wx + root[..., None, 0], local[..., 1] + 0.9, wz + root[..., None, 2]), -1)
  return {
    "posed_joints": joints,
    "root_pos": root,
    "heading": heading,
    "contacts": torch.rand(n, t, 4, generator=g) > 0.5,
    "meta": {
      "fps": fps, "edge_frames": 8, "history": 4, "joint_names": [f"j{i}" for i in range(J)],
      "parents": [-1, 0, 0, 1, 2, 0, 5, 5], "foot_idx": [1, 3, 2, 4], "head_idx": 5,
      "neutral_head_height": 1.65,
    },
  }


SYNTH_GRAPH = {
  "neighbors": 8, "vel_weight": 0.1, "pose_weight": 1.0, "temperature": 0.02, "inertialize": True,
  "correct_yaw_deg": 5.0, "correct_speed": 0.1, "hunch_ratio": 0.85, "end_vel_weight": 0.25,
  "lookahead": False,
}


# --- Recording ----------------------------------------------------------------
def _as_numpy(x) -> np.ndarray:
  if torch.is_tensor(x):
    x = x.detach().cpu()
    return x.numpy() if x.dtype != torch.bfloat16 else x.float().numpy()
  return np.asarray(x)


def array_digest(a: np.ndarray) -> dict:
  a = np.ascontiguousarray(a)
  head = f"{a.dtype.str}|{a.shape}".encode()
  out = {"dtype": a.dtype.str, "shape": list(a.shape), "sha256": hashlib.sha256(head + a.tobytes()).hexdigest()}
  if a.dtype.kind in "fiub" and a.size:
    f = a.astype(np.float64)
    finite = f[np.isfinite(f)]
    out["stats"] = [float(finite.sum()), float(np.abs(finite).sum()), float(finite.min()) if finite.size else None,
                    float(finite.max()) if finite.size else None, int(f.size - finite.size)]
  return out


def _jsonable(x):
  if torch.is_tensor(x):
    return _as_numpy(x).tolist()
  if isinstance(x, np.ndarray):
    return x.tolist()
  if isinstance(x, (np.floating, np.integer, np.bool_)):
    return x.item()
  if isinstance(x, Path):
    return str(x)
  raise TypeError(type(x))


def canonical(obj) -> str:
  return json.dumps(obj, sort_keys=True, default=_jsonable, allow_nan=True)


class Record:
  def __init__(self):
    self.arrays: dict[str, np.ndarray] = {}
    self.json: dict[str, object] = {}

  def array(self, name: str, x) -> None:
    if name in self.arrays:
      raise KeyError(name)
    self.arrays[name] = _as_numpy(x)

  def tree(self, prefix: str, obj) -> None:
    """Arrays under `prefix/...`; scalars and strings go to JSON `prefix`."""
    scalars = {}
    items = obj.items() if isinstance(obj, dict) else enumerate(obj)
    for key, value in items:
      name = f"{prefix}/{key}"
      if torch.is_tensor(value) or isinstance(value, np.ndarray):
        self.array(name, value)
      elif isinstance(value, dict):
        self.tree(name, value)
      elif isinstance(value, (list, tuple)) and value and all(torch.is_tensor(v) for v in value):
        self.tree(name, list(value))
      else:
        scalars[str(key)] = value
    if scalars:
      self.js(prefix, scalars)

  def js(self, name: str, obj) -> None:
    if name in self.json:
      raise KeyError(name)
    self.json[name] = json.loads(canonical(obj))

  def manifest(self) -> dict:
    return {
      "arrays": {k: array_digest(v) for k, v in sorted(self.arrays.items())},
      "json": {k: {"sha256": hashlib.sha256(canonical(v).encode()).hexdigest(), "value": v}
               for k, v in sorted(self.json.items())},
    }


@contextlib.contextmanager
def cpu_cuda_shims():
  """Harnesses synchronize and read CUDA memory stats; make those no-ops on CPU."""
  names = ("synchronize", "reset_peak_memory_stats", "max_memory_allocated", "empty_cache")
  saved = {n: getattr(torch.cuda, n) for n in names}
  torch.cuda.synchronize = lambda *a, **k: None
  torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
  torch.cuda.max_memory_allocated = lambda *a, **k: 0
  torch.cuda.empty_cache = lambda *a, **k: None
  try:
    yield
  finally:
    for n, f in saved.items():
      setattr(torch.cuda, n, f)


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def model_digest(model) -> str:
  import mujoco

  buf = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
  mujoco.mj_saveModel(model, None, buf)
  return hashlib.sha256(buf.tobytes()).hexdigest()


# --- Cases --------------------------------------------------------------------
DROP_TIMING = ("rollout_wall_s", "isolated_motion_seconds_per_wall_s", "call_wall_mean_s", "elapsed_s")


def _no_timing(d: dict) -> dict:
  return {k: v for k, v in d.items() if k not in DROP_TIMING}


def case_evaluate(A, ctx, rec: Record) -> None:
  """Behavior evaluator rollouts on the saved graph (trace arrays and summaries)."""
  raw = json.loads((ROOT / "configs/behavior/baseline.json").read_text())
  if ctx.fast:
    raw = {**raw, "seeds": [0], "seconds": 5.0}
  cfg = A.resolved_eval_config(raw)
  graph = A.load_graph(ctx.graph, cfg["graph_overrides"])
  for seed in cfg["seeds"]:
    trace, stats = A.eval_rollout(graph, cfg, seed)
    rec.tree(f"seed{seed}", trace)
    rec.js(f"seed{seed}/summary", _no_timing(stats))


def case_audit(A, ctx, rec: Record) -> None:
  """`audit_graph` JSON on the saved graph tables."""
  blob = torch.load(ctx.graph, map_location="cpu", weights_only=False, mmap=True)
  rec.js("audit", A.audit_graph(blob))


RUNTIME_VARIANTS = {
  "saved": {},
  "lookahead": {"lookahead": True},
  "plain": {"correct_yaw_deg": 0.0, "correct_speed": 0.0, "inertialize": False, "end_vel_weight": 0.0},
  "greedy": {"temperature": 0.0},
}


def case_runtime_step(A, ctx, rec: Record) -> None:
  """`GraphRuntime.step` on the saved graph toward per-human random arcs."""
  humans, steps = 64, (20 if ctx.fast else 200)
  for name, overrides in RUNTIME_VARIANTS.items():
    graph = A.load_graph(ctx.graph, overrides)
    runtime = A.runtime(graph)
    gen = torch.Generator().manual_seed(11)
    seeds = torch.arange(humans) * 7 + 100
    xz = torch.rand(humans, 2, generator=gen) * 10 - 5
    heading = torch.rand(humans, generator=gen) * 2 * math.pi - math.pi
    state = runtime.reset(seeds, xz, heading)
    rec.array(f"{name}/start_boundary", state.boundary)
    outs: dict[str, list] = {}
    for _ in range(steps):
      path = A.steering_paths(state.root_xz, state.heading, runtime.path_frames, graph.fps, state.seeds, state.counters)
      out = runtime.step(state, path)
      for k, v in out.items():
        outs.setdefault(k, []).append(v)
    for k, v in outs.items():
      rec.array(f"{name}/{k}", torch.stack(v, 1))
    for k in ("boundary", "root_xz", "heading", "pose", "counters"):
      rec.array(f"{name}/final_{k}", getattr(state, k))


def case_graph_build(A, ctx, rec: Record) -> None:
  """kNN build, tables, stats and save/load on a synthetic library."""
  lib = synthetic_library()
  graph = A.build_graph(lib, SYNTH_GRAPH)
  rec.tree("tables", A.graph_tables(graph))
  rec.js("stats", A.graph_stats(graph))
  with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "g.pt"
    A.save_graph(graph, path, lib["posed_joints"][0, 0])
    loaded = A.load_graph(path)
    tables = A.graph_tables(loaded)
    tables.pop("desc", None)
    rec.tree("loaded", tables)
    rec.array("loaded_rest", loaded.rest_joints)


def case_quality_graph(A, ctx, rec: Record) -> None:
  """Quality-harness graph rollouts and metrics on a synthetic library graph."""
  lib = synthetic_library()
  backend = A.runtime(A.build_graph(lib, SYNTH_GRAPH))
  scenarios = ("cruise", "turn90") if ctx.fast else ("cruise", "stop_start", "turn90", "turn180", "speed_step")
  for scenario in scenarios:
    with cpu_cuda_shims():
      r = A.quality_rollout(backend, {"jitter_changes": True}, scenario, "edge", 6, 9.0 if ctx.fast else 12.0, 5)
      m = A.quality_metrics(r, backend, scenario)
    r = {k: v for k, v in r.items() if k != "call_walls"}
    rec.tree(scenario, r)
    rec.js(f"{scenario}/metrics", _no_timing(m))


ONLINE_CASES = {
  "ardy": (({"history": 4}, "cruise", "refill"), ({"history": 4}, "turn90", 0.5),
           ({"history": 4, "jitter_changes": False}, "stop_start", 1.0), ({"history": 4}, "speed_step", 0.2)),
  "kimodo": (({"new_frames": 20, "history": 5}, "turn180", "refill"), ({"new_frames": 20, "history": 5}, "cruise", 0.5)),
}


def case_quality_online(A, ctx, rec: Record) -> None:
  """Quality-harness online rollouts, metrics and branch checks with fake backends."""
  for flavor, cases in ONLINE_CASES.items():
    backend = FakeArdy(A) if flavor == "ardy" else FakeKimodo(A)
    for i, (cfg, scenario, period) in enumerate(cases[:1] if ctx.fast else cases):
      with cpu_cuda_shims():
        r = A.quality_rollout(backend, cfg, scenario, period, 5, 9.0, 3 + i)
        m = A.quality_metrics(r, backend, scenario)
      rec.tree(f"{flavor}{i}", {k: v for k, v in r.items() if k != "call_walls"})
      rec.js(f"{flavor}{i}/metrics", _no_timing(m))
    with cpu_cuda_shims():
      rec.js(f"{flavor}/branch_check", A.branch_check(backend, cases[0][0], 4, 17))


LIB_COMMANDS = json.loads((ROOT / "configs/library/ardy_a40.json").read_text())["commands"]


def case_library_fake(A, ctx, rec: Record) -> None:
  """Offline library streams and random commands, with a fake ARDY."""
  backend = FakeArdy(A)
  cfg = {"edge_frames": 8, "history": 4, "duration": 4.0 if ctx.fast else 8.0, "seed": 3, "commands": LIB_COMMANDS}
  for offset, humans in ((7, 5), (12, 3)):
    rec.tree(f"streams{offset}", A.generate_streams(backend, cfg, offset, humans))
  seeds = torch.tensor([5, 9, 1_000_017, 2**40 + 3])
  cmds = A.RandomCommands(seeds, torch.tensor([0.0, 1.0, -2.0, 3.0]), A.CommandRanges.from_dict(LIB_COMMANDS))
  speed, heading, modes = [], [], []
  for k in range(100 if ctx.fast else 300):
    s, h = cmds(k * 0.4, "cpu")
    speed.append(s)
    heading.append(h)
    modes.append(torch.as_tensor(cmds.state()["segment"]))
  rec.array("commands/speed", torch.stack(speed, 1))
  rec.array("commands/heading", torch.stack(heading, 1))
  rec.array("commands/segment", torch.stack(modes, 1))


THROUGHPUT_CASES = (
  ("ardy", {"backend": "ardy", "model": "fake", "batch": 6, "microbatch": 4, "history": 4, "steps": 5,
            "cfg_type": "nocfg", "precision": "fp32", "future": 0, "workload": "continue", "seed": 2}),
  ("ardy", {"backend": "ardy", "model": "fake", "batch": 5, "steps": 5, "workload": "fresh", "seed": 1}),
  ("kimodo", {"backend": "kimodo", "model": "fake", "batch": 6, "microbatch": 3, "history": 5, "steps": 4,
              "new_frames": 20, "workload": "continue", "seed": 4}),
)
ROW_KEYS_TIMING = ("wall_p50", "wall_p95", "wall_mean", "wall_min", "R_new", "peak_gb", "stage_sum")


def case_throughput_fake(A, ctx, rec: Record) -> None:
  """Throughput population requests and buffers, and deterministic row fields."""
  defaults = {"reps": 3, "warmup": 1, "time_budget_s": 1e9}
  for i, (flavor, cfg) in enumerate(THROUGHPUT_CASES):
    backend = FakeArdy(A) if flavor == "ardy" else FakeKimodo(A)
    with cpu_cuda_shims():
      row = A.run_config(backend, cfg, defaults)
    rec.js(f"case{i}/row", {k: v for k, v in row.items()
                            if k not in ROW_KEYS_TIMING and not k.startswith(("R_usable", "S_star", "stage_"))})
    backend = FakeArdy(A) if flavor == "ardy" else FakeKimodo(A)
    hist = cfg.get("history", 4) if flavor == "ardy" else backend.transition_frames
    pop = A.Population(backend, cfg["batch"], cfg["seed"], max(hist, 8) + 64)
    for call in range(3 if ctx.fast else 6):
      with cpu_cuda_shims():
        _wall, _stages, chunks = A.one_call(backend, pop, cfg)
      merged = A.merge(chunks)
      rec.array(f"case{i}/call{call}/joints", merged.posed_joints)
      if cfg["workload"] == "continue":
        pop.execute(merged)
    for k in ("features", "root_xz", "heading", "counters"):
      if getattr(pop, k) is not None:
        rec.array(f"case{i}/pop_{k}", getattr(pop, k))


def case_ardy_cpu(A, ctx, rec: Record) -> None:
  """Real ARDY H40 on CPU: library streams, a quality rollout and throughput calls."""
  backend = A.ardy_backend()
  cfg = {"edge_frames": 8, "history": 4, "duration": 2.0, "seed": 0, "commands": LIB_COMMANDS}
  rec.tree("streams", A.generate_streams(backend, cfg, 0, 2))
  with cpu_cuda_shims():
    r = A.quality_rollout(backend, {"history": 4}, "turn90", 1.0, 2, 2.0, 0)
    m = A.quality_metrics(r, backend, "turn90")
  rec.tree("quality", {k: v for k, v in r.items() if k != "call_walls"})
  rec.js("quality/metrics", _no_timing(m))
  tcfg = {"backend": "ardy", "batch": 2, "history": 4, "steps": 5, "cfg_type": "nocfg", "precision": "fp32",
          "workload": "continue", "seed": 0}
  pop = A.Population(backend, 2, 0, 72)
  for call in range(2):
    with cpu_cuda_shims():
      _wall, _stages, chunks = A.one_call(backend, pop, tcfg)
    merged = A.merge(chunks)
    rec.array(f"throughput/call{call}/joints", merged.posed_joints)
    pop.execute(merged)


def case_kimodo_cpu(A, ctx, rec: Record) -> None:
  """Real Kimodo on CPU (4 steps): a quality rollout and throughput calls."""
  backend = A.kimodo_backend()
  cfg = {"new_frames": 20, "history": 5, "steps": 4}
  with cpu_cuda_shims():
    r = A.quality_rollout(backend, cfg, "turn90", "refill", 2, 2.0, 0)
  rec.tree("quality", {k: v for k, v in r.items() if k != "call_walls"})
  tcfg = {"backend": "kimodo", "batch": 2, "history": 5, "steps": 4, "new_frames": 20, "precision": "fp32",
          "workload": "continue", "seed": 0}
  pop = A.Population(backend, 2, 0, 72)
  for call in range(2):
    with cpu_cuda_shims():
      _wall, _stages, chunks = A.one_call(backend, pop, tcfg)
    merged = A.merge(chunks)
    rec.array(f"throughput/call{call}/joints", merged.posed_joints)
    pop.execute(merged)


def case_mjlab_plaza(A, ctx, rec: Record) -> None:
  """mjlab plaza with the GraphCrowd term on CPU: mocap poses, LiDAR and term state."""
  from mjlab.envs import ManagerBasedRlEnv

  cfg = A.plaza_env_cfg(str(ctx.graph), humans=6, num_envs=2, seed=3, width=64, height=48)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    rec.js("model_sha256", model_digest(env.sim.mj_model))
    env.reset()
    term = env.command_manager.get_term("crowd")
    lidar = env.scene["lidar"]
    action = torch.zeros(env.action_space.shape)
    steps = 30 if ctx.fast else 100
    pos, quat, dist = [], [], []
    for step in range(steps):
      if step == steps // 2:
        env.reset(env_ids=torch.tensor([1]))
      env.step(action)
      pos.append(torch.as_tensor(env.sim.data.mocap_pos).clone())
      quat.append(torch.as_tensor(env.sim.data.mocap_quat).clone())
      dist.append(lidar.data.distances.clone())
    rec.array("mocap_pos", torch.stack(pos))
    rec.array("mocap_quat", torch.stack(quat))
    rec.array("lidar", torch.stack(dist))
    rec.tree("term", A.plaza_term_state(term))
  finally:
    env.close()


def case_corridor(A, ctx, rec: Record) -> None:
  """Corridor fixture (primitive humans, CorridorWalk, LiDAR) on CPU."""
  from mjlab.envs import ManagerBasedRlEnv

  cfg = A.corridor_env_cfg(num_envs=2, num_humans=2, seed=7, azimuth_samples=36, elevation_samples=4)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    rec.js("model_sha256", model_digest(env.sim.mj_model))
    env.reset()
    action = torch.zeros(env.action_space.shape)
    qpos, dist = [], []
    for step in range(20 if ctx.fast else 60):
      if step == 10:
        env.reset(env_ids=torch.tensor([0]))
      env.step(action)
      qpos.append(torch.as_tensor(env.sim.data.qpos).clone())
      dist.append(env.scene["lidar"].data.distances.clone())
    rec.array("qpos", torch.stack(qpos))
    rec.array("lidar", torch.stack(dist))
    for name in ("human_0", "human_1"):
      term = env.command_manager.get_term(name)
      rec.array(f"{name}/command", term.command)
      rec.array(f"{name}/time_left", term.time_left)
      rec.array(f"{name}/counter", term.command_counter)
  finally:
    env.close()


def case_replay(A, ctx, rec: Record) -> None:
  """Replay scene model and capsule poses for saved evaluator joints."""
  cfg = A.resolved_eval_config(json.loads((ROOT / "configs/behavior/baseline.json").read_text()))
  graph = A.load_graph(ctx.graph, {})
  skeleton = A.CapsuleSkeleton.from_rest(graph.meta, graph.rest_joints)
  model = A.replay_scene(skeleton, cfg["humans"], cfg["steering"])
  rec.js("model_sha256", model_digest(model))
  trace, _ = A.eval_rollout(graph, {**cfg, "seconds": 2.0}, 4)
  joints = torch.from_numpy(trace["joints"])
  pos, quat = zip(*(skeleton.poses(A.ardy_to_mujoco(joints[:, f])) for f in range(joints.shape[1])))
  rec.array("pos", torch.stack(pos, 1))
  rec.array("quat", torch.stack(quat, 1))


def case_compare(A, ctx, rec: Record) -> None:
  """Evaluate CLI twice (baseline vs override) and the paired comparison outputs."""
  raw = json.loads((ROOT / "configs/behavior/baseline.json").read_text())
  with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for name, overrides in (("a", {}), ("b", {"temperature": 0.03})):
      cfg = {**raw, "seeds": [0, 5], "seconds": 3.0, "graph_overrides": overrides}
      (tmp / f"{name}.json").write_text(json.dumps(cfg))
      A.behavior_cli(["evaluate", "--graph", str(ctx.graph), "--config", str(tmp / f"{name}.json"),
                      "--out", str(tmp / name), "--threads", "1"])
      for seed in cfg["seeds"]:
        with np.load(tmp / name / f"seed_{seed}.npz") as z:
          rec.tree(f"{name}/seed{seed}", {k: z[k] for k in z.files})
      rec.js(f"{name}/skeleton", json.loads((tmp / name / "skeleton.json").read_text()))
    A.behavior_cli(["compare", "--baseline", str(tmp / "a"), "--candidate", str(tmp / "b"), "--out", str(tmp / "cmp")])
    comparison = json.loads((tmp / "cmp" / "comparison.json").read_text())
    rec.js("paired_metrics", comparison["paired_metrics"])
    rec.js("index_html_sha256", hashlib.sha256((tmp / "cmp" / "index.html").read_bytes()).hexdigest())


CASES = {
  "evaluate": (case_evaluate, "graph"),
  "audit": (case_audit, "graph"),
  "runtime_step": (case_runtime_step, "graph"),
  "graph_build": (case_graph_build, None),
  "quality_graph": (case_quality_graph, None),
  "quality_online": (case_quality_online, None),
  "library_fake": (case_library_fake, None),
  "throughput_fake": (case_throughput_fake, None),
  "mjlab_plaza": (case_mjlab_plaza, "graph"),
  "corridor": (case_corridor, None),
  "replay": (case_replay, "graph"),
  "compare": (case_compare, "graph"),
  "ardy_cpu": (case_ardy_cpu, "slow"),
  "kimodo_cpu": (case_kimodo_cpu, "slow"),
}


# --- Driver -------------------------------------------------------------------
def default_graph() -> Path | None:
  """`$CROWD_GRAPH`, else the `graph-default` artifact path when it exists locally."""
  if os.environ.get("CROWD_GRAPH"):
    return Path(os.environ["CROWD_GRAPH"])
  try:
    import tomllib

    manifest = tomllib.loads((ROOT / "artifacts.toml").read_text())
    path = ROOT / manifest["artifact"]["graph-default"]["path"]
  except (OSError, KeyError):
    return None
  return path if path.exists() else None


def expected_path(case: str, fast: bool) -> Path:
  return EXPECTED / f"{case}{'.fast' if fast else ''}.json"


def run_case(A, name: str, ctx) -> Record:
  rec = Record()
  CASES[name][0](A, ctx, rec)
  return rec


def compare_manifests(want: dict, got: dict) -> list[str]:
  problems = []
  for kind in ("arrays", "json"):
    w, g = want[kind], got[kind]
    for key in sorted(set(w) | set(g)):
      if key not in g:
        problems.append(f"{kind} {key}: missing")
      elif key not in w:
        problems.append(f"{kind} {key}: unexpected")
      elif w[key]["sha256"] != g[key]["sha256"]:
        detail = ""
        if kind == "arrays":
          detail = f" (want {w[key]['shape']} {w[key]['dtype']} {w[key].get('stats')}, got {g[key]['shape']} {g[key]['dtype']} {g[key].get('stats')})"
        problems.append(f"{kind} {key}: differs{detail}")
  return problems


def selected_cases(args) -> list[str]:
  names = args.cases or [n for n, (_, need) in CASES.items() if need != "slow" or args.slow]
  for name in names:
    if name not in CASES:
      raise SystemExit(f"unknown case {name}")
  return names


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  parser.add_argument("mode", choices=("capture", "check"))
  parser.add_argument("--out", type=Path, required=True, help="directory for full arrays and the report")
  parser.add_argument("--graph", type=Path, default=default_graph())
  parser.add_argument("--fast", action="store_true", help="smaller variant of every case")
  parser.add_argument("--slow", action="store_true", help="include real ARDY/Kimodo CPU cases")
  parser.add_argument("--cases", nargs="*")
  args = parser.parse_args(argv)
  names = selected_cases(args)
  args.out.mkdir(parents=True, exist_ok=True)
  EXPECTED.mkdir(exist_ok=True)
  ctx = SimpleNamespace(fast=args.fast, graph=args.graph)
  A = api()
  graph_sha = None
  report = {"mode": args.mode, "fast": args.fast, "cases": {}}
  failed = 0
  for name in names:
    if CASES[name][1] == "graph":
      if args.graph is None or not args.graph.exists():
        print(f"SKIP {name}: no saved graph (--graph or CROWD_GRAPH)", flush=True)
        report["cases"][name] = "skipped"
        continue
      graph_sha = graph_sha or file_sha256(args.graph)
    t0 = time.perf_counter()
    torch.manual_seed(0)
    rec = run_case(A, name, ctx)
    got = rec.manifest()
    if CASES[name][1] == "graph":
      got["graph_sha256"] = graph_sha
    np.savez_compressed(args.out / f"{name}{'.fast' if args.fast else ''}.npz",
                        **{k.replace("/", "__"): v for k, v in rec.arrays.items()})
    seconds = time.perf_counter() - t0
    path = expected_path(name, args.fast)
    if args.mode == "capture":
      path.write_text(json.dumps(got, indent=1, sort_keys=True, allow_nan=True) + "\n")
      print(f"CAPTURED {name} ({len(got['arrays'])} arrays, {len(got['json'])} json) in {seconds:.1f} s", flush=True)
      report["cases"][name] = "captured"
      continue
    want = json.loads(path.read_text())
    if want.get("graph_sha256") != got.get("graph_sha256"):
      raise SystemExit(f"{name}: expected file was captured on a different graph")
    problems = compare_manifests(want, got)
    report["cases"][name] = problems or "match"
    if problems:
      failed += 1
      print(f"MISMATCH {name} in {seconds:.1f} s:", *problems[:20], sep="\n  ", flush=True)
    else:
      print(f"MATCH {name} in {seconds:.1f} s", flush=True)
  (args.out / f"report_{args.mode}{'_fast' if args.fast else ''}.json").write_text(json.dumps(report, indent=1))
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main())
