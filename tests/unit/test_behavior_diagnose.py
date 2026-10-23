"""Saved-trace diagnostics, including censoring and known command responses."""

import numpy as np
import pytest

from crowdmotion.iterate.diagnose import command_response, diagnose_trace
from crowdmotion.iterate.evaluate import resolved_config


def _turn_trace():
  vel = np.zeros((1, 20, 2), dtype=np.float32)
  vel[:, :5, 1] = 1
  vel[:, 5:, 0] = 1
  root = np.concatenate((np.zeros((1, 1, 2), dtype=np.float32), np.cumsum(vel / 20, axis=1)), axis=1)
  heading = np.zeros((1, 21), dtype=np.float32)
  heading[:, 6:] = np.pi / 2
  cmd = np.array([[[1, 0], [1, np.pi / 2], [1, np.pi / 2], [1, np.pi / 2]]], dtype=np.float32)
  joints = np.zeros((1, 21, 1, 3), dtype=np.float32)
  joints[..., 0] = root[..., 0, None]
  joints[..., 2] = root[..., 1, None]
  return {"root": root, "heading": heading, "command": cmd, "joints": joints}


def test_response_requires_sustained_motion_and_reports_body_separately():
  out = command_response(_turn_trace(), 20, 5)
  assert out["events"] == out["eligible"] == 1
  assert out["root"]["reached"] == out["body"]["reached"] == 1
  assert out["root"]["delay_s_p50_p95"] == pytest.approx([0.1, 0.1])
  assert out["body"]["delay_s_p50_p95"] == pytest.approx([0.05, 0.05])


def test_response_does_not_count_alignment_after_the_command_is_canceled():
  tr = _turn_trace()
  tr["command"][:, 2:, 0] = 0
  tr["heading"][:, :11] = 0
  out = command_response(tr, 20, 5)
  assert out["body"]["reached"] == 0
  assert out["body"]["not_reached_before_command_changed_or_end"] == 1
  assert out["root"]["reached"] == 0  # Four aligned frames before stop is too short.


def test_seams_progress_and_single_human_context_are_finite():
  out = diagnose_trace(_turn_trace(), resolved_config({}), 20, 5)
  assert out["slow_despite_move_command_percent"] == 0
  assert out["outside_wall_human_frame_percent"] == 0
  assert out["commanded_progress_mps_p50_p95"] == pytest.approx([1, 1])
  assert np.isfinite(out["seams"]["seam_to_interior_step_ratio"])
  assert out["direction_by_neighbor_context"]["close"]["moving_frames"] == 0
