import numpy as np

from crowdmotion.metrics.redirects import ordinary_redirects


def trace(kind):
  frames, L, fps = 80, 8, 20.
  heading = np.zeros((1, frames + 1))
  if kind != "sideways":
    heading[0, :41] = np.linspace(0, np.pi / 2, 41)
    heading[0, 41:] = np.pi / 2
  mid = (heading[:, 1:] + heading[:, :-1]) / 2
  velocity = np.stack((np.sin(mid), np.cos(mid)), -1)
  if kind == "pivot":
    velocity[:, :40] = 0
  if kind == "sideways":
    velocity[:] = [1., 0.]
  root = np.concatenate((np.zeros((1, 1, 2)), np.cumsum(velocity / fps, axis=1)), 1)
  tr = {"root": root, "heading": heading, "goal": np.tile([10., 0.], (1, frames // L, 1))}
  return tr, np.ones((1, frames // L), dtype=bool)


def test_pivot_then_forward_is_distinct_from_sidestep():
  tr, ordinary = trace("pivot")
  out = ordinary_redirects(tr, 20., 8, ordinary)
  assert out["counts"]["body_turn_then_forward"] == 1
  tr, ordinary = trace("sideways")
  out = ordinary_redirects(tr, 20., 8, ordinary)
  assert out["counts"]["lateral_or_backward"] == 1


def test_forward_curve_is_not_sidestep():
  tr, ordinary = trace("curve")
  out = ordinary_redirects(tr, 20., 8, ordinary)
  assert out["counts"]["body_turn_then_forward"] == 1
  assert out["counts"]["lateral_or_backward"] == 0


def test_censoring_and_evasive_exclusion():
  tr, ordinary = trace("pivot")
  ordinary[:, 2:] = False
  out = ordinary_redirects(tr, 20., 8, ordinary)
  assert out["counts"]["unresolved"] == 1
  assert out["records"][0]["end_reason"] == "avoidance_or_wait"
  assert ordinary_redirects(tr, 20., 8, ordinary & False)["attempts"] == 0


def test_new_goal_is_reported_separately_from_initial_or_resumed_alignment():
  tr, ordinary = trace("pivot")
  tr["goal"][:, :2] = [0., 10.]
  out = ordinary_redirects(tr, 20., 8, ordinary)
  assert out["records"][0]["start_s"] == .8
  assert out["records"][0]["trigger"] == "new_goal"
  assert out["by_trigger"]["new_goal"]["body_turn_then_forward"] == 1
  assert not sum(out["by_trigger"]["initial_alignment"].values())
