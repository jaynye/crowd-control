"""Turning-library checks (`crowd library turns`), turning coverage (`crowd graph turns`)
and graphs built from several libraries (`graph.build.load_libraries`)."""

from __future__ import annotations

import pytest
import torch

from crowdmotion.backends.base import MotionChunk
from crowdmotion.graph.build import load_libraries
from crowdmotion.graph.config import GraphConfig
from crowdmotion.graph.tables import MotionGraph
from crowdmotion.graph.turns import audit, foot_yaw_deg
from crowdmotion.library import generate
from crowdmotion.library.turns import analyze, write_clips
from synthetic import synthetic_library

FPS = 20.0
# Body-local offsets (x left, y up, z forward) of a small upright skeleton.
BODY = {
  "Hips": (0, 0.95, 0), "Spine": (0, 1.1, 0), "Neck": (0, 1.45, 0.02), "Head": (0, 1.65, 0.05),
  "RightUpLeg": (-0.1, 0.9, 0), "LeftUpLeg": (0.1, 0.9, 0), "RightFoot": (-0.1, 0.08, 0),
  "RightToeBase": (-0.1, 0.02, 0.12), "LeftFoot": (0.1, 0.08, 0), "LeftToeBase": (0.1, 0.02, 0.12),
  "RightArm": (-0.2, 1.4, 0), "LeftArm": (0.2, 1.4, 0),
}
NAMES = list(BODY)
MASK = {"vertical_mask": True, "vertical_pelvis_range_m": 0.15, "vertical_foot_min_m": 0.12,
        "vertical_foot_max_m": 0.6, "vertical_lowest_foot_max_m": 0.12, "posture_cutoff_deg": 15.0,
        "hunch_ratio": 0.85}


class Walker:
  """A fake ARDY that walks the requested path facing its heading targets."""

  device = torch.device("cpu")
  fps = FPS
  frames_per_token = 4
  path_frames = 40
  joint_names = NAMES

  def empty_text(self, batch):
    return torch.zeros(batch, 1, 4)

  def generate(self, req):
    path, heading = req.root_path, torch.nan_to_num(req.root_heading, nan=0.0)
    b, f = path.shape[:2]
    local = torch.tensor(list(BODY.values()), dtype=torch.float32)
    c, s = torch.cos(heading)[..., None], torch.sin(heading)[..., None]
    wx = local[:, 0] * c + local[:, 2] * s + path[..., 0:1]
    wz = -local[:, 0] * s + local[:, 2] * c + path[..., 1:2]
    joints = torch.stack((wx, local[:, 1].expand(b, f, -1), wz), -1)
    root = joints[:, :, 0].clone()
    return MotionChunk(features=torch.zeros(b, f, 4), root_pos=root, local_rot=torch.eye(3).expand(b, f, len(NAMES), 3, 3),
                       posed_joints=joints, contacts=torch.ones(b, f, 4, dtype=torch.bool), heading=heading, fps=FPS)


def turning_library(humans=12, duration=24.0):
  cfg = {"edge_frames": 8, "history": 4, "duration": duration, "seed": 2, "command_source": "turns"}
  lib = generate.generate_streams(Walker(), cfg, 0, humans)
  lib["meta"] = {"fps": FPS, "edge_frames": 8, "joint_names": NAMES, "parents": [-1] + [0] * (len(NAMES) - 1),
                 "foot_idx": [NAMES.index(n) for n in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")],
                 "head_idx": NAMES.index("Head"), "neutral_head_height": 1.65, "command_source": "turns"}
  return lib


def test_turn_report_on_a_perfect_follower(tmp_path):
  lib = turning_library()
  report, clips = analyze(lib, GraphConfig.from_dict(MASK))
  ip, arc = report["in_place"], report["arc"]
  assert ip["segments"] > 3 and arc["segments"] > 3
  # The fake executes the reference exactly: in-place turns reach the command in place.
  assert ip["hip_turn_ratio"]["p50"] == pytest.approx(1.0, abs=0.02)
  assert ip["toe_turn_ratio"]["p50"] == pytest.approx(1.0, abs=0.02)
  assert ip["drift_m_from_rest"]["p90"] < 0.02  # only the tail of an earlier deceleration
  assert arc["radius_ratio"]["p50"] == pytest.approx(1.0, abs=0.05)
  assert arc["forward_fraction"]["p10"] > 0.98
  assert report["heading_abs_error_deg"]["in_place"]["returned_vs_hip"]["p95"] < 0.01
  ground = report["ground"]
  assert ground["in_place"]["admissible"] == 1.0 and ground["in_place"]["admissible_hunched"] == 0.0
  write_clips(tmp_path / "clips.npz", lib, clips, 2)
  if clips:
    import numpy as np

    data = np.load(tmp_path / "clips.npz")
    assert data["joints"].shape[1:] == (80, len(NAMES), 3) and data["planned"].shape[-1] == 2


def test_graph_turn_coverage_counts_in_place_turns():
  lib = turning_library()
  g = MotionGraph(lib, GraphConfig.from_dict({"neighbors": 16, **MASK}), device="cpu")
  out = audit(g)
  edges = out["all"]["edges"]["admissible"]
  assert edges["in_place (speed<.2, turn>=25deg/s)"]["upright"] > 0
  assert sum(v["hunched"] for v in edges.values()) == 0
  total = sum(v["upright"] for v in edges.values())
  assert total == out["all"]["admissible_edges"] == int(g.admissible_edges().sum())
  assert set(out) == {"all"}
  # Feet follow the body here; turning both feet by 60 deg in one edge marks it twisted.
  yaw = foot_yaw_deg(g)
  assert yaw.max() < 1.0
  e = int(torch.nonzero(g.admissible_edges())[3])
  feet = [NAMES.index(n) for n in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")]
  c, s = torch.cos(torch.tensor(1.0472)), torch.sin(torch.tensor(1.0472))
  p = g.edge_pose[e][:, feet]
  g.edge_pose[e][:, feet] = torch.stack((p[..., 0] * c + p[..., 2] * s, p[..., 1], -p[..., 0] * s + p[..., 2] * c), -1)
  yaw = foot_yaw_deg(g)
  assert yaw[e] == pytest.approx(60.0, abs=0.5)
  twisted = audit(g, twist_deg=30.0)["all"]["edges"]["admissible"]
  assert sum(v["twisted"] for v in twisted.values()) == 1
  assert sum(v["twisted_planted"] for v in twisted.values()) == 1  # the fake's contacts are always on


def save(tmp_path, name, lib):
  path = tmp_path / f"{name}.pt"
  torch.save(lib, path)
  return path


def test_load_libraries_merges_and_truncates(tmp_path, capsys):
  a, b = synthetic_library(0), synthetic_library(1)
  # Library b is longer: its streams are cut to a's length (whole edges).
  b = {k: (torch.cat((v, v[:, :10]), 1) if torch.is_tensor(v) else v) for k, v in b.items()}
  pa, pb = save(tmp_path, "a", a), save(tmp_path, "b", b)
  assert load_libraries([pa])["posed_joints"].shape == a["posed_joints"].shape  # one library: unchanged
  merged = load_libraries([pa, pb])
  assert "truncated" in capsys.readouterr().out
  n, t = a["posed_joints"].shape[:2]
  assert merged["posed_joints"].shape[:2] == (2 * n, t)
  assert torch.equal(merged["posed_joints"][n:], b["posed_joints"][:, :t])
  assert [s["streams"] for s in merged["meta"]["sources"]] == [n, n]
  assert set(merged) == {"posed_joints", "root_pos", "heading", "contacts", "meta"}
  cfg = GraphConfig(neighbors=8)
  g, ga = MotionGraph(merged, cfg, device="cpu"), MotionGraph(a, cfg, device="cpu")
  B = ga.valid.shape[0]
  for name in ("edge_root", "edge_yaw", "edge_pose", "valid", "pred_pose"):
    assert torch.equal(getattr(g, name)[:B], getattr(ga, name))
  assert g.streams == 2 * n and g.meta["sources"][1]["library"] == str(pb)
  bad = dict(b, meta={**b["meta"], "edge_frames": 8})
  with pytest.raises(ValueError, match="edge_frames"):
    load_libraries([pa, save(tmp_path, "bad", bad)])


def test_merged_graph_coverage_splits_sources(tmp_path):
  lib = turning_library(humans=6)
  pa, pb = save(tmp_path, "turns", lib), save(tmp_path, "turns2", lib)
  g = MotionGraph(load_libraries([pa, pb]), GraphConfig.from_dict({"neighbors": 16, **MASK}), device="cpu")
  out = audit(g)
  assert set(out) == {"all", "source0", "source1"}
  assert out["source0"]["edges"] == out["source1"]["edges"]  # identical halves
  assert out["all"]["admissible_edges"] == 2 * out["source0"]["admissible_edges"]
