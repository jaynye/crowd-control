"""Traversal metrics: command following, collisions, time to contact, encounters and falls
of a robot walking to goals through a crowd (`crowd eval traversal`).

The evaluator feeds `TraversalMetrics.step` two snapshots per env step: a `StepState` at
the end of the step's physics, before any auto-reset (a collision or fall is seen as it
happened), and a `CommandState` after the command update (new goals, and the new
episode's start in envs that reset). Everything stays on the env's device; events come
out as column tables.

Definitions (scales in `TraversalEvalCfg`; they label motion for analysis and are not
pass thresholds):

- A **leg** runs from a goal's assignment to its arrival, resample or episode end. Its
  reference is the straight segment from the robot's position at assignment to the
  goal, timed at the leg's commanded speed with the command's ramp-down near the goal.
  Standing envs have no legs.
- **Clearance** is the robot-sphere to human-joint-sphere distance of the collision rule.
  A **collision** is a robot-human pair's clearance falling below `-penetration_m`; a
  pair counts again only after its clearance has returned to 0 (separation).
- **Time to contact (TTC)** is clearance over the closing speed of the robot's root and
  the human's hips along the line between them, while closing faster than
  `closing_min_mps`; 0 in contact and infinite otherwise. (The clearance itself swings
  with the gait, so its finite difference is too noisy to divide by.)
- An **encounter** is a pair within `encounter_clearance_m` or under `encounter_ttc_s`.
  It ends once the pair has been out of that state for `encounter_gap_s`, or with the
  episode. A step is an encounter step when any pair of its env is in that state.
- **Reaction**: within an encounter, from the first moving step with TTC under
  `reaction_ttc_s` to the first step whose smoothed velocity (exponential, time
  constant `reaction_smooth_s`, which removes the gait's speed swing) is
  `reaction_speed_drop` slower than, or turned `reaction_turn_deg` away from, the
  smoothed velocity at that first step. A new goal restarts the measurement, so a turn
  toward it does not count.
- **Fall controls** are random moving or standing times, weighted to the falls'
  distribution over standing status and episode age (`fall_age_bins_s`), so a fall
  early in an episode is compared with control times early in an episode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from crowdmotion.core.config import from_dict

TRAITS = ("oblivious", "stubborn", "cooperative")  # the humans' trait toward the robot
LEG_ENDS = ("arrived", "resampled", "collision", "fall", "time_out", "eval_end", "deadline")
EPISODE_ENDS = ("collision", "fall", "time_out", "eval_end", "success", "deadline")


@dataclass(frozen=True)
class TraversalEvalCfg:
  sigma_m: float = 0.5  # tracking score exp(-(d / sigma)^2)
  encounter_clearance_m: float = 1.0
  encounter_ttc_s: float = 3.0
  encounter_gap_s: float = 0.5
  closing_min_mps: float = 0.05
  ttc_cutoffs_s: tuple[float, ...] = (1.0, 2.0)
  reaction_ttc_s: float = 2.0
  reaction_speed_drop: float = 0.1
  reaction_turn_deg: float = 10.0
  reaction_smooth_s: float = 0.3
  penetration_m: float = 0.05  # the collision rule (`crowd_collision`)
  fall_windows_s: tuple[float, ...] = (1.0, 2.0)
  fall_near_m: float = 2.0
  fall_controls: int = 20_000  # random times for the fall attribution, weighted to match the falls
  fall_age_bins_s: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)
  drift_windows_s: tuple[float, ...] = (1.0, 3.0)
  survival_times_s: tuple[float, ...] = (10.0, 20.0, 30.0, 60.0)
  approach_bins_m: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
  ahead_deg: float = 60.0
  behind_deg: float = 120.0
  deviation_max_m: float = 20.0  # histogram range for quantiles (1 cm bins); the plaza diagonal is 17 m


@dataclass
class StepState:
  """The end of an env step's physics, before any auto-reset (world frame, planar)."""
  pos: torch.Tensor  # [E, 2] robot root
  vel: torch.Tensor  # [E, 2]
  heading: torch.Tensor  # [E]
  vel_b: torch.Tensor  # [E, 3] body vx, vy, wz
  cmd_w: torch.Tensor  # [E, 2] the command acted on during this step
  cmd_b: torch.Tensor  # [E, 3]
  standing: torch.Tensor  # [E] bool
  clearance: torch.Tensor  # [E, M]
  human_pos: torch.Tensor  # [E, M, 2] hips
  human_vel: torch.Tensor  # [E, M, 2]
  trait: torch.Tensor  # [E, M] long, index into TRAITS
  fell: torch.Tensor  # [E] bool, `fell_over` ended the episode
  timed_out: torch.Tensor  # [E] bool
  collided: torch.Tensor  # [E] bool, `crowd_collision` ended the episode
  done: torch.Tensor  # [E] bool, any termination or time-out
  succeeded: torch.Tensor | None = None  # terminal goal success (new persistent-goal task)
  deadline: torch.Tensor | None = None  # a task failure, distinct from rollout truncation


@dataclass
class CommandState:
  """After the command update: the robot's position (for reset envs, the new episode's
  start), its goal, the leg's commanded speed and standing flag, and the episode's goal
  arrivals so far."""
  pos: torch.Tensor  # [E, 2]
  goal: torch.Tensor  # [E, 2]
  speed: torch.Tensor  # [E]
  standing: torch.Tensor  # [E] bool
  arrivals: torch.Tensor  # [E] goals reached this episode


def ideal_time(distance: torch.Tensor, speed: torch.Tensor, slow_m: float, arrive_m: float) -> torch.Tensor:
  """Seconds to reach `arrive_m` of a goal `distance` away along a straight line at the
  goal command's speed profile, `speed * min(d / slow_m, 1)` at distance d."""
  speed = speed.clamp(min=1e-6)
  ramp = (slow_m / speed) * torch.log(torch.clamp(torch.minimum(distance, torch.full_like(distance, slow_m))
                                                  / arrive_m, min=1.0))
  cruise = (distance - slow_m).clamp(min=0.0) / speed
  return cruise + ramp


def segment_distance(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """`(distance from p to segment ab, progress along it in [0, 1])`, batched over rows."""
  ab = b - a
  u = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-9)).clamp(0.0, 1.0)
  return (p - (a + u[..., None] * ab)).norm(dim=-1), u


def kaplan_meier(time: np.ndarray, event: np.ndarray, horizon: float, at: tuple[float, ...] = ()) -> dict:
  """Survival with right censoring: the median (None when survival stays above 0.5 up to
  the last time), the restricted mean to `horizon`, and S(t) at `at`."""
  time, event = np.asarray(time, float), np.asarray(event, bool)
  out = {"median_s": None, "rmst_s": None, "events": int(event.sum()), "n": int(len(time))}
  out.update({f"survival_{t:g}s": None for t in at})
  if not len(time):
    return out
  order = np.argsort(time, kind="stable")
  time, event = time[order], event[order]
  steps, surv, s, i, n = [0.0], [1.0], 1.0, 0, len(time)
  while i < n:
    j = i
    while j < n and time[j] == time[i]:
      j += 1
    d = int(event[i:j].sum())
    if d:
      s *= 1.0 - d / (n - i)
      steps.append(float(time[i]))
      surv.append(s)
    i = j
  steps, surv = np.array(steps), np.array(surv)
  below = np.nonzero(surv <= 0.5)[0]
  if len(below):
    out["median_s"] = float(steps[below[0]])
  edges = np.append(np.clip(steps, 0, horizon), horizon)
  out["rmst_s"] = float((np.diff(edges) * surv).sum())
  covered = time.max()
  for t in at:
    if t <= covered:
      out[f"survival_{t:g}s"] = float(surv[np.searchsorted(steps, t, side="right") - 1])
  return out


def _quantile(hist: torch.Tensor, width: float, q: float) -> float | None:
  total = float(hist.sum())
  if not total:
    return None
  k = int(torch.searchsorted(hist.cumsum(0), torch.tensor(q * total, device=hist.device, dtype=hist.dtype)))
  return (k + 0.5) * width


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float | None:
  keep = (w > 0) & ~np.isnan(x)
  if not keep.any():
    return None
  order = np.argsort(x[keep], kind="stable")
  x, cw = x[keep][order], np.cumsum(w[keep][order])
  return float(x[np.searchsorted(cw, 0.5 * cw[-1])])


def _angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Unsigned angle between planar vectors, 0 where either is near zero."""
  na, nb = a.norm(dim=-1), b.norm(dim=-1)
  cos = (a * b).sum(-1) / (na * nb).clamp(min=1e-9)
  angle = torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))
  return torch.where((na > 0.05) & (nb > 0.05), angle, torch.zeros_like(angle))


class _Table:
  """Columns appended in GPU batches, gathered once at the end."""

  def __init__(self, names: tuple[str, ...]):
    self.names, self.parts = names, {n: [] for n in names}

  def add(self, **cols: torch.Tensor) -> None:
    for name in self.names:
      self.parts[name].append(cols[name].detach().float().flatten())

  def numpy(self) -> dict[str, np.ndarray]:
    for n, p in self.parts.items():
      if len(p) > 1:  # keep the joined column, so repeated snapshots do not re-join every batch
        self.parts[n] = [torch.cat(p)]
    return {n: (p[0].cpu().numpy().copy() if p else np.zeros(0, np.float32)) for n, p in self.parts.items()}


class TraversalMetrics:
  """Accumulates one rollout (module docstring). `steps`: the planned rollout length, which
  sets the rate of matched random times for the fall attribution."""

  def __init__(self, envs: int, humans: int, dt: float, steps: int, slow_m: float, arrive_m: float,
               cfg: TraversalEvalCfg = TraversalEvalCfg(), device="cpu", seed: int = 0):
    self.E, self.M, self.dt, self.cfg, self.dev = envs, humans, dt, cfg, device
    self.slow_m, self.arrive_m = slow_m, arrive_m
    self.t, self.n_steps = 0.0, 0
    self.gen = torch.Generator(device=device)
    self.gen.manual_seed(seed)
    self.control_p = min(1.0, cfg.fall_controls / max(envs * steps, 1))
    E, M, z = envs, humans, lambda *s, dtype=torch.float32: torch.zeros(*s, dtype=dtype, device=device)
    bins = int(round(cfg.deviation_max_m / 0.01)) + 1
    self.dev_hist = z(2, bins)  # free, encounter
    self.dev_sums = z(2, 3)  # free, encounter: steps, sum d, sum d^2 (exact; the histogram is for quantiles)
    self.drift_steps = [max(1, round(w / dt)) for w in cfg.drift_windows_s]
    self.drift_hist = z(len(self.drift_steps), bins)
    self.drift_sums = z(len(self.drift_steps), 2)  # steps, sum
    self.gap_steps = max(1, round(cfg.encounter_gap_s / dt))
    self.fall_steps = [max(1, round(w / dt)) for w in cfg.fall_windows_s]
    # Per-env state.
    self.ep = z(E, dtype=torch.long)
    self.ep_start = z(E)
    self.ep_standing = z(E, dtype=torch.bool)
    self.ep_collisions = z(E)
    self.ep_first_collision = torch.full((E,), math.nan, device=device)
    self.ep_contact_steps = z(E)
    self.arrivals = z(E)
    self.leg_start, self.leg_goal, self.leg_last = z(E, 2), z(E, 2), z(E, 2)
    self.leg_t0, self.leg_speed = z(E), z(E)
    self.leg_standing = z(E, dtype=torch.bool)
    self.leg_sums = z(E, 10)  # path, steps, dev, dev^2, score, free steps, free dev, free score, enc dev, enc score
    self.leg_max = z(E)
    L = max(self.drift_steps) + 1
    self.ring_pos, self.ring_cmd, self.cmd_int = z(E, L, 2), z(E, L, 2), z(E, 2)
    self.drift_valid = z(E, dtype=torch.long)
    self.vel_smooth = z(E, 2)
    self.vel_fresh = torch.ones(E, dtype=torch.bool, device=device)  # seed the smoothing at the next step
    self.colliding = z(E, M, dtype=torch.bool)
    F = max(self.fall_steps)
    self.fall_clear, self.fall_ttc = torch.full((E, F), math.inf, device=device), torch.full((E, F), math.inf,
                                                                                             device=device)
    self.fall_valid = z(E, dtype=torch.long)
    # Encounters, per pair.
    self.enc_active = z(E, M, dtype=torch.bool)
    self.enc_out = z(E, M, dtype=torch.long)  # consecutive steps out of the encounter state
    self.enc = {k: z(E, M) for k in ("t0", "min_clear", "min_ttc", "collided", "contact", "standing", "react_t0",
                                     "react_speed", "react_dx", "react_dy", "latency", "trait")}
    self.cmd_changed = z(E, dtype=torch.bool)  # the previous step gave a new goal or standing flag
    # Totals, on the device until the summary.
    self.tot = {k: z(()) for k in ("robot_steps", "moving_steps", "standing_steps", "contact_steps", "enc_steps",
                                   "moving_enc_steps", "vel_xy", "vel_yaw", "vel_xy_enc", "vel_yaw_enc")}
    self.ttc_below = z(len(cfg.ttc_cutoffs_s))
    A = len(cfg.approach_bins_m) - 1
    self.approach = z(3, A)  # steps, speed sum, toward sum
    self.legs = _Table(("env", "episode", "t0", "x0", "y0", "goal_x", "goal_y", "duration", "straight_m", "path_m",
                        "end", "arrived", "speed",
                        "ideal_s", "lag_s", "time_ratio", "detour", "spl", "dev_mean_m", "dev_rms_m", "dev_max_m",
                        "score", "encounter_share"))
    self.encounters = _Table(("env", "human", "episode", "t0", "duration", "trait", "standing", "min_clear_m",
                              "min_ttc_s", "collided", "contact", "threat", "responded", "latency_s", "censored"))
    self.collisions = _Table(("env", "human", "episode", "t", "episode_t", "trait", "standing", "clearance_m",
                              "robot_speed", "robot_toward", "human_toward", "bearing_deg"))
    attribution = tuple(f"{k}_{w:g}s" for w in cfg.fall_windows_s for k in ("min_clear_m", "min_ttc_s"))
    self.falls = _Table(("env", "episode", "t", "episode_t", "standing", "near", *attribution))
    self.fall_controls = _Table(("env", "episode", "t", "episode_t", "standing", "near", *attribution))
    self.episodes = _Table(("env", "episode", "t0", "duration", "end", "fell", "standing", "collisions",
                            "first_collision_s", "contact_s"))

  # Rollout ---------------------------------------------------------------------------
  def start(self, post: CommandState) -> None:
    """The rollout's first episode in every env (after the initial reset)."""
    every = torch.ones(self.E, dtype=torch.bool, device=self.dev)
    self._new_episode(every, post)

  def step(self, pre: StepState, post: CommandState) -> None:
    self.n_steps += 1
    self.t += self.dt
    cfg = self.cfg
    moving = ~pre.standing
    clear = pre.clearance
    closest = clear.amin(-1)
    rel = self._relative(pre)
    ttc = self._ttc(clear, rel)
    min_ttc = ttc.amin(-1)
    in_enc = (clear < cfg.encounter_clearance_m) | (ttc < cfg.encounter_ttc_s)
    env_enc = in_enc.any(-1)
    self._exposure(pre, moving, closest, min_ttc, env_enc)
    self._command(pre, moving, env_enc)
    onset = self._collisions(pre, clear, rel)
    self._encounters(pre, moving, clear, ttc, in_enc, onset)
    self._approach(pre, moving, clear, rel)
    self._falls(pre, closest, min_ttc)
    # Leg and episode ends, then the new legs and episodes.
    done = pre.done
    arrived = (post.arrivals > self.arrivals) & ~done
    changed = ((post.goal - self.leg_goal).abs().amax(-1) > 1e-6) | (post.standing != self.leg_standing)
    end = torch.full((self.E,), LEG_ENDS.index("resampled"), dtype=torch.long, device=self.dev)
    end = torch.where(arrived, LEG_ENDS.index("arrived"), end)
    cause = self._episode_cause(pre)
    mapping = torch.tensor([2, 3, 4, 5, 0, 6], device=self.dev)
    end = torch.where(done, mapping[cause], end)
    self._close_legs(changed | done | arrived, end, pre.pos)
    self.cmd_changed = (changed | arrived) & ~done
    if done.any():
      self._close_episodes(done, cause, pre.fell)
      self._new_episode(done, post)
    rest = (changed | arrived) & ~done
    if rest.any():
      self._new_legs(rest, post)
    self.arrivals = torch.where(done, post.arrivals, torch.maximum(self.arrivals, post.arrivals))

  def finish(self) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """`(metrics, tables)`; open legs, encounters and episodes end censored."""
    every = torch.ones(self.E, dtype=torch.bool, device=self.dev)
    self._close_legs(every, torch.full((self.E,), LEG_ENDS.index("eval_end"), device=self.dev), self.leg_last)
    self._end_encounters(self.enc_active, censored=True)
    self._close_episodes(every, torch.full((self.E,), EPISODE_ENDS.index("eval_end"), device=self.dev), ~every)
    tables = {name: getattr(self, name).numpy()
              for name in ("legs", "encounters", "collisions", "falls", "fall_controls", "episodes")}
    return self._summary(tables), tables

  def snapshot(self) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """`(metrics, tables)` of what has ended so far, without ending anything: open legs,
    encounters and episodes are left out (a live display; `finish` ends a rollout). Exposure
    totals include the open episodes' steps."""
    tables = {name: getattr(self, name).numpy()
              for name in ("legs", "encounters", "collisions", "falls", "fall_controls", "episodes")}
    return self._summary(tables), tables

  # Per-step pieces -------------------------------------------------------------------
  def _ttc(self, clear: torch.Tensor, rel) -> torch.Tensor:
    robot_toward, human_toward, _ = rel
    closing = robot_toward + human_toward
    ok = (closing > self.cfg.closing_min_mps) & (clear > 0)
    ttc = torch.where(ok, clear / closing.clamp(min=self.cfg.closing_min_mps), torch.full_like(clear, math.inf))
    return torch.where(clear <= 0, torch.zeros_like(clear), ttc)

  def _exposure(self, pre: StepState, moving, closest, min_ttc, env_enc) -> None:
    tot = self.tot
    tot["robot_steps"] += self.E
    tot["moving_steps"] += moving.sum()
    tot["standing_steps"] += (~moving).sum()
    tot["contact_steps"] += (closest < 0).sum()
    tot["enc_steps"] += env_enc.sum()
    tot["moving_enc_steps"] += (env_enc & moving).sum()
    self.ep_contact_steps += (closest < 0).float()
    for i, c in enumerate(self.cfg.ttc_cutoffs_s):
      self.ttc_below[i] += ((min_ttc < c) & moving).sum()
    err_xy = (pre.cmd_b[:, :2] - pre.vel_b[:, :2]).norm(dim=-1)
    err_yaw = (pre.cmd_b[:, 2] - pre.vel_b[:, 2]).abs()
    tot["vel_xy"] += (err_xy * moving).sum()
    tot["vel_yaw"] += (err_yaw * moving).sum()
    tot["vel_xy_enc"] += (err_xy * (moving & env_enc)).sum()
    tot["vel_yaw_enc"] += (err_yaw * (moving & env_enc)).sum()

  def _command(self, pre: StepState, moving, env_enc) -> None:
    cfg, s = self.cfg, self.leg_sums
    on_leg = moving & ~self.leg_standing
    d, _ = segment_distance(pre.pos, self.leg_start, self.leg_goal)
    score = torch.exp(-(d / cfg.sigma_m) ** 2)
    w, free = on_leg.float(), (on_leg & ~env_enc).float()
    enc = (on_leg & env_enc).float()
    step = (pre.pos - self.leg_last).norm(dim=-1)
    s += torch.stack((step * w, w, d * w, d * d * w, score * w, free, d * free, score * free, d * enc, score * enc), -1)
    self.leg_max = torch.where(on_leg, torch.maximum(self.leg_max, d), self.leg_max)
    self.leg_last = pre.pos.clone()
    width, bins = 0.01, self.dev_hist.shape[1]
    idx = (d / width).long().clamp(max=bins - 1)
    for row, mask in ((0, on_leg & ~env_enc), (1, on_leg & env_enc)):
      self.dev_hist[row] += torch.bincount(idx[mask], minlength=bins).float()
      w = mask.float()
      self.dev_sums[row] += torch.stack((w.sum(), (d * w).sum(), (d * d * w).sum()))
    # Drift of the executed displacement from the integrated command over fixed windows.
    self.cmd_int += pre.cmd_w * self.dt
    self.ring_pos = torch.roll(self.ring_pos, 1, dims=1)
    self.ring_cmd = torch.roll(self.ring_cmd, 1, dims=1)
    self.ring_pos[:, 0], self.ring_cmd[:, 0] = pre.pos, self.cmd_int
    self.drift_valid = torch.where(moving, self.drift_valid + 1, torch.zeros_like(self.drift_valid))
    for row, k in enumerate(self.drift_steps):
      ok = self.drift_valid > k
      drift = ((pre.pos - self.ring_pos[:, k]) - (self.cmd_int - self.ring_cmd[:, k])).norm(dim=-1)
      self.drift_hist[row] += torch.bincount((drift[ok] / width).long().clamp(max=bins - 1), minlength=bins).float()
      self.drift_sums[row] += torch.stack((ok.float().sum(), (drift * ok).sum()))

  def _relative(self, pre: StepState):
    """Per pair: unit vector robot to human, robot and human speeds toward each other, and
    the human's bearing in the robot's yaw frame (deg)."""
    rel = pre.human_pos - pre.pos[:, None]
    unit = rel / rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    robot_toward = (pre.vel[:, None] * unit).sum(-1)
    human_toward = -(pre.human_vel * unit).sum(-1)
    c, s = torch.cos(pre.heading)[:, None], torch.sin(pre.heading)[:, None]
    bearing = torch.rad2deg(torch.atan2(-s * rel[..., 0] + c * rel[..., 1], c * rel[..., 0] + s * rel[..., 1]))
    return robot_toward, human_toward, bearing

  def _collisions(self, pre: StepState, clear, rel) -> torch.Tensor:
    now = clear < -self.cfg.penetration_m
    onset = now & ~self.colliding
    self.colliding = (self.colliding & (clear < 0)) | now  # latched until the pair separates
    if onset.any():
      e, h = onset.nonzero(as_tuple=True)
      robot_toward, human_toward, bearing = rel
      first = torch.isnan(self.ep_first_collision[e])
      self.ep_first_collision[e[first]] = self.t - self.ep_start[e[first]]
      self.ep_collisions.index_add_(0, e, torch.ones_like(e, dtype=torch.float32))
      self.collisions.add(env=e, human=h, episode=self.ep[e], t=torch.full((len(e),), self.t, device=self.dev),
                          episode_t=self.t - self.ep_start[e], trait=pre.trait[e, h], standing=pre.standing[e],
                          clearance_m=clear[e, h], robot_speed=pre.vel[e].norm(dim=-1),
                          robot_toward=robot_toward[e, h], human_toward=human_toward[e, h],
                          bearing_deg=bearing[e, h])
    return onset

  def _encounters(self, pre: StepState, moving, clear, ttc, in_enc, onset) -> None:
    cfg, enc = self.cfg, self.enc
    begin = in_enc & ~self.enc_active
    self.enc_out = torch.where(in_enc, 0, self.enc_out + 1)
    if begin.any():
      for k, v in (("t0", self.t), ("min_clear", math.inf), ("min_ttc", math.inf), ("collided", 0.0),
                   ("contact", 0.0), ("react_t0", math.nan), ("react_speed", 0.0), ("react_dx", 0.0),
                   ("react_dy", 0.0), ("latency", math.nan)):
        enc[k] = torch.where(begin, torch.full_like(enc[k], v), enc[k])
      enc["standing"] = torch.where(begin, pre.standing[:, None].float().expand_as(begin), enc["standing"])
      enc["trait"] = torch.where(begin, pre.trait.float(), enc["trait"])
    self.enc_active |= begin
    live = self.enc_active
    enc["min_clear"] = torch.where(live, torch.minimum(enc["min_clear"], clear), enc["min_clear"])
    enc["min_ttc"] = torch.where(live, torch.minimum(enc["min_ttc"], ttc), enc["min_ttc"])
    enc["collided"] = torch.where(live & onset, 1.0, enc["collided"])
    enc["contact"] = torch.where(live & (clear < 0), 1.0, enc["contact"])
    alpha = min(1.0, self.dt / max(cfg.reaction_smooth_s, 1e-6))
    self.vel_smooth = torch.where(self.vel_fresh[:, None], pre.vel, self.vel_smooth + alpha * (pre.vel - self.vel_smooth))
    self.vel_fresh[:] = False
    speed = self.vel_smooth.norm(dim=-1)[:, None].expand_as(clear)
    # A new goal restarts a pending measurement: turning toward it is not a reaction.
    restart = self.cmd_changed[:, None] & torch.isnan(enc["latency"])
    enc["react_t0"] = torch.where(restart, math.nan, enc["react_t0"])
    threat = live & torch.isnan(enc["react_t0"]) & (ttc < cfg.reaction_ttc_s) & moving[:, None]
    enc["react_t0"] = torch.where(threat, self.t, enc["react_t0"])
    enc["react_speed"] = torch.where(threat, speed, enc["react_speed"])
    for i, k in enumerate(("react_dx", "react_dy")):
      enc[k] = torch.where(threat, self.vel_smooth[:, None, i].expand_as(clear), enc[k])
    ref = torch.stack((enc["react_dx"], enc["react_dy"]), -1)
    turned = _angle_deg(self.vel_smooth[:, None].expand_as(ref), ref)
    waiting = live & ~torch.isnan(enc["react_t0"]) & torch.isnan(enc["latency"]) & ~threat
    respond = waiting & ((speed <= enc["react_speed"] - cfg.reaction_speed_drop) | (turned >= cfg.reaction_turn_deg))
    enc["latency"] = torch.where(respond, self.t - enc["react_t0"], enc["latency"])
    self._end_encounters(live & ((self.enc_out >= self.gap_steps) | pre.done[:, None]), censored=False)

  def _end_encounters(self, ending: torch.Tensor, censored: bool) -> None:
    if not ending.any():
      return
    e, h = ending.nonzero(as_tuple=True)
    enc = {k: v[e, h] for k, v in self.enc.items()}
    threat = ~torch.isnan(enc["react_t0"])
    self.encounters.add(env=e, human=h, episode=self.ep[e], t0=enc["t0"], duration=self.t - enc["t0"],
                        trait=enc["trait"], standing=enc["standing"], min_clear_m=enc["min_clear"],
                        min_ttc_s=enc["min_ttc"], collided=enc["collided"], contact=enc["contact"],
                        threat=threat.float(), responded=(~torch.isnan(enc["latency"])).float(),
                        latency_s=enc["latency"], censored=torch.full_like(enc["t0"], float(censored)))
    self.enc_active[e, h] = False

  def _approach(self, pre: StepState, moving, clear, rel) -> None:
    near = clear.argmin(-1)
    rows = torch.arange(self.E, device=self.dev)
    robot_toward, _, bearing = rel
    ahead = moving & (bearing[rows, near].abs() < self.cfg.ahead_deg)
    c = clear[rows, near]
    edges = torch.tensor(self.cfg.approach_bins_m, device=self.dev)
    b = torch.bucketize(c, edges, right=True) - 1
    ok = ahead & (b >= 0) & (b < len(edges) - 1)
    A = len(edges) - 1
    for row, value in ((0, torch.ones_like(c)), (1, pre.vel.norm(dim=-1)), (2, robot_toward[rows, near])):
      self.approach[row] += torch.bincount(b[ok], weights=value[ok], minlength=A)[:A]

  def _falls(self, pre: StepState, closest, min_ttc) -> None:
    self.fall_clear = torch.roll(self.fall_clear, 1, dims=1)
    self.fall_ttc = torch.roll(self.fall_ttc, 1, dims=1)
    self.fall_clear[:, 0], self.fall_ttc[:, 0] = closest, min_ttc
    self.fall_valid += 1
    pick = (torch.rand(self.E, generator=self.gen, device=self.dev) < self.control_p) & ~pre.done
    for table, mask in ((self.falls, pre.fell), (self.fall_controls, pick)):
      if not mask.any():
        continue
      e = mask.nonzero().flatten()
      cols, near = {}, None
      for w, k in zip(self.cfg.fall_windows_s, self.fall_steps):
        span = torch.arange(k, device=self.dev)[None] < self.fall_valid[e, None]
        mc = torch.where(span, self.fall_clear[e, :k], math.inf).amin(1)
        mt = torch.where(span, self.fall_ttc[e, :k], math.inf).amin(1)
        cols[f"min_clear_m_{w:g}s"], cols[f"min_ttc_s_{w:g}s"] = mc, mt
        near = mc < self.cfg.fall_near_m  # the widest window, listed last
      table.add(env=e, episode=self.ep[e], t=torch.full((len(e),), self.t, device=self.dev),
                episode_t=self.t - self.ep_start[e], standing=pre.standing[e], near=near, **cols)

  # Legs and episodes -------------------------------------------------------------------
  def _episode_cause(self, pre: StepState) -> torch.Tensor:
    cause = torch.full((self.E,), EPISODE_ENDS.index("time_out"), dtype=torch.long, device=self.dev)
    if pre.succeeded is not None:
      cause = torch.where(pre.succeeded, EPISODE_ENDS.index("success"), cause)
    if pre.deadline is not None:
      cause = torch.where(pre.deadline, EPISODE_ENDS.index("deadline"), cause)
    cause = torch.where(pre.fell, EPISODE_ENDS.index("fall"), cause)
    return torch.where(pre.collided, EPISODE_ENDS.index("collision"), cause)

  def _new_episode(self, envs: torch.Tensor, post: CommandState) -> None:
    self.ep = torch.where(envs, self.ep + 1, self.ep)
    self.ep_start = torch.where(envs, self.t, self.ep_start)
    self.ep_standing = torch.where(envs, post.standing, self.ep_standing)
    self.ep_collisions = torch.where(envs, 0.0, self.ep_collisions)
    self.ep_first_collision = torch.where(envs, math.nan, self.ep_first_collision)
    self.ep_contact_steps = torch.where(envs, 0.0, self.ep_contact_steps)
    for name in ("drift_valid", "fall_valid"):
      setattr(self, name, torch.where(envs, 0, getattr(self, name)))
    self.vel_fresh |= envs
    self.colliding &= ~envs[:, None]
    self.cmd_int = torch.where(envs[:, None], 0.0, self.cmd_int)
    self.fall_clear[envs], self.fall_ttc[envs] = math.inf, math.inf
    self._new_legs(envs, post)

  def _new_legs(self, envs: torch.Tensor, post: CommandState) -> None:
    m = envs[:, None]
    self.leg_start = torch.where(m, post.pos, self.leg_start)
    self.leg_last = torch.where(m, post.pos, self.leg_last)
    self.leg_goal = torch.where(m, post.goal, self.leg_goal)
    self.leg_t0 = torch.where(envs, self.t, self.leg_t0)
    self.leg_speed = torch.where(envs, post.speed, self.leg_speed)
    self.leg_standing = torch.where(envs, post.standing, self.leg_standing)
    self.leg_sums = torch.where(m, 0.0, self.leg_sums)
    self.leg_max = torch.where(envs, 0.0, self.leg_max)

  def _close_legs(self, ending: torch.Tensor, end: torch.Tensor, last_pos: torch.Tensor) -> None:
    ending = ending & ~self.leg_standing & (self.leg_sums[:, 1] > 0)  # legs with moving steps
    if not ending.any():
      return
    e = ending.nonzero().flatten()
    s = self.leg_sums[e]
    steps = s[:, 1].clamp(min=1)
    straight = ((self.leg_goal[e] - self.leg_start[e]).norm(dim=-1) - self.arrive_m).clamp(min=0.0)
    path = s[:, 0] + (last_pos[e] - self.leg_last[e]).norm(dim=-1)
    arrived = end[e] == LEG_ENDS.index("arrived")
    duration = self.t - self.leg_t0[e]
    ideal = ideal_time(straight + self.arrive_m, self.leg_speed[e], self.slow_m, self.arrive_m)
    nan = torch.full_like(duration, math.nan)
    start, goal = self.leg_start[e], self.leg_goal[e]
    self.legs.add(env=e, episode=self.ep[e], t0=self.leg_t0[e], x0=start[:, 0], y0=start[:, 1], goal_x=goal[:, 0],
                  goal_y=goal[:, 1], duration=duration, straight_m=straight, path_m=path,
                  end=end[e], arrived=arrived, speed=self.leg_speed[e], ideal_s=ideal,
                  lag_s=torch.where(arrived, duration - ideal, nan),
                  time_ratio=torch.where(arrived, duration / ideal.clamp(min=1e-6), nan),
                  detour=torch.where(arrived, path / straight.clamp(min=1e-6), nan),
                  spl=arrived.float() * straight / torch.maximum(path, straight).clamp(min=1e-6),
                  dev_mean_m=s[:, 2] / steps, dev_rms_m=(s[:, 3] / steps).sqrt(), dev_max_m=self.leg_max[e],
                  score=s[:, 4] / steps, encounter_share=1 - s[:, 5] / steps)

  def _close_episodes(self, ending: torch.Tensor, cause: torch.Tensor, fell: torch.Tensor) -> None:
    """`cause` names one end (a collision wins a tie); `fell` flags a fall independently."""
    e = (ending & (self.ep_start < self.t - 1e-9)).nonzero().flatten()  # episodes with steps
    if not len(e):
      return
    self._end_encounters(self.enc_active & ending[:, None], censored=False)
    self.episodes.add(env=e, episode=self.ep[e], t0=self.ep_start[e], duration=self.t - self.ep_start[e],
                      end=cause[e], fell=fell[e], standing=self.ep_standing[e], collisions=self.ep_collisions[e],
                      first_collision_s=self.ep_first_collision[e], contact_s=self.ep_contact_steps[e] * self.dt)

  # Summary -----------------------------------------------------------------------------
  def _summary(self, tables: dict[str, dict[str, np.ndarray]]) -> dict:
    cfg, dt = self.cfg, self.dt
    tot = {k: float(v) for k, v in self.tot.items()}
    robot_min = tot["robot_steps"] * dt / 60
    moving_min, standing_min = tot["moving_steps"] * dt / 60, tot["standing_steps"] * dt / 60
    out: dict[str, float | int | None] = {
      "exposure/robot_min": robot_min, "exposure/moving_min": moving_min, "exposure/standing_min": standing_min,
      "exposure/encounter_share": tot["enc_steps"] / max(tot["robot_steps"], 1),
    }
    per = lambda n, minutes: n / minutes if minutes else None  # noqa: E731
    med = lambda x: float(np.nanmedian(x)) if np.isfinite(x).any() else None  # noqa: E731
    mean = lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else None  # noqa: E731
    share = lambda x: float(np.mean(x)) if len(x) else None  # noqa: E731

    legs = tables["legs"]
    arrived = legs["arrived"] > 0
    out.update({
      "command/legs": int(len(arrived)), "command/arrivals": int(arrived.sum()),
      "command/success": share(arrived), "command/spl": share(legs["spl"]),
      "command/goals_per_moving_min": per(int(arrived.sum()), moving_min),
      "command/lag_median_s": med(legs["lag_s"]), "command/time_ratio_median": med(legs["time_ratio"]),
      "command/detour_ratio_median": med(legs["detour"]),
      "command/vel_error_xy": tot["vel_xy"] / tot["moving_steps"] if tot["moving_steps"] else None,
      "command/vel_error_yaw": tot["vel_yaw"] / tot["moving_steps"] if tot["moving_steps"] else None,
      "command/encounter/vel_error_xy": tot["vel_xy_enc"] / tot["moving_enc_steps"] if tot["moving_enc_steps"]
      else None,
    })
    width = 0.01
    centers = (torch.arange(self.dev_hist.shape[1], device=self.dev) + 0.5) * width
    score = torch.exp(-(centers / cfg.sigma_m) ** 2)  # 1 cm bins: within 1e-3 of the exact mean score
    sums = self.dev_sums.cpu().tolist()
    for name, hist, (n, d, d2) in (("all", self.dev_hist.sum(0), [a + b for a, b in zip(*sums)]),
                                   ("free", self.dev_hist[0], sums[0]), ("encounter", self.dev_hist[1], sums[1])):
      prefix = "command/" if name == "all" else f"command/{name}/"
      out[prefix + "steps"] = int(n)
      out[prefix + "cross_track_mean_m"] = d / n if n else None
      out[prefix + "cross_track_rms_m"] = math.sqrt(d2 / n) if n else None
      out[prefix + "cross_track_p95_m"] = _quantile(hist, width, 0.95)
      out[prefix + "tracking_score"] = float((hist * score).sum()) / n if n else None
    for w, hist, (n, d) in zip(cfg.drift_windows_s, self.drift_hist, self.drift_sums.cpu().tolist()):
      out[f"command/drift_{w:g}s_mean_m"] = d / n if n else None
      out[f"command/drift_{w:g}s_p95_m"] = _quantile(hist, width, 0.95)

    col = tables["collisions"]
    n_col = len(col["env"])
    standing = col["standing"] > 0
    out.update({
      "collision/count": n_col, "collision/per_robot_min": per(n_col, robot_min),
      "collision/moving_per_min": per(int((~standing).sum()), moving_min),
      "collision/standing_per_min": per(int(standing.sum()), standing_min),
      "collision/contact_share": tot["contact_steps"] / max(tot["robot_steps"], 1),
      "collision/human_led_share": share(col["human_toward"] > col["robot_toward"]),
      "collision/behind_share": share(np.abs(col["bearing_deg"]) > cfg.behind_deg),
      "collision/robot_still_share": share(col["robot_speed"] < 0.15),
    })
    for i, trait in enumerate(TRAITS):
      out[f"collision/{trait}_per_robot_min"] = per(int((col["trait"] == i).sum()), robot_min)

    enc = tables["encounters"]
    n_enc = len(enc["env"])
    out.update({
      "encounter/count": n_enc, "encounter/per_robot_min": per(n_enc, robot_min),
      "encounter/per_moving_min": per(int((enc["standing"] == 0).sum()), moving_min),
      "collision/per_encounter": n_col / n_enc if n_enc else None,
      "encounter/min_clear_median_m": med(enc["min_clear_m"]),
      # Infinite (never closing) minima count; the median is null when half never close.
      "encounter/min_ttc_median_s": float(np.median(enc["min_ttc_s"])) if n_enc else None,
    })
    for c, below in zip(cfg.ttc_cutoffs_s, self.ttc_below):
      out[f"encounter/min_ttc_lt_{c:g}s_share"] = share(enc["min_ttc_s"] < c)
      out[f"ttc/moving_steps_lt_{c:g}s_share"] = float(below) / tot["moving_steps"] if tot["moving_steps"] else None
    threat = enc["threat"] > 0
    out.update({
      "reaction/threats": int(threat.sum()),
      "reaction/response_share": share(enc["responded"][threat] > 0),
      "reaction/latency_median_s": med(enc["latency_s"][threat]),
    })
    edges = cfg.approach_bins_m
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
      n = float(self.approach[0, i])
      out[f"approach/{lo:g}-{hi:g}m/steps"] = int(n)
      out[f"approach/{lo:g}-{hi:g}m/speed"] = float(self.approach[1, i]) / n if n else None
      out[f"approach/{lo:g}-{hi:g}m/toward"] = float(self.approach[2, i]) / n if n else None

    falls, controls = tables["falls"], tables["fall_controls"]
    w = cfg.fall_windows_s[-1]
    out.update({"fall/count": len(falls["env"]), "fall/per_robot_min": per(len(falls["env"]), robot_min)})
    # Controls weighted to the falls' strata (standing, episode age): within a stratum,
    # each control counts falls / controls.
    stratum = lambda t: np.digitize(t["episode_t"], cfg.fall_age_bins_s) * 2 + (t["standing"] > 0)  # noqa: E731
    fs, cs = stratum(falls), stratum(controls)
    weight = np.zeros(len(cs))
    for k in np.unique(fs):
      inside = cs == k
      if inside.any():
        weight[inside] = (fs == k).sum() / inside.sum()
    out["fall/control_samples"] = len(controls["env"])
    out["fall/control_coverage"] = share(np.isin(fs, cs)) if len(fs) else None  # falls with matching controls
    for name, t, wt in (("fall", falls, np.ones(len(fs))), ("fall/control", controls, weight)):
      ok = wt.sum() > 0
      wshare = lambda x: float((wt * x).sum() / wt.sum()) if ok else None  # noqa: E731,B023
      out[f"{name}/near_share"] = wshare(t["near"] > 0)
      out[f"{name}/min_clear_median_m_{w:g}s"] = _weighted_median(t[f"min_clear_m_{w:g}s"], wt) if ok else None
      for c in cfg.ttc_cutoffs_s:
        out[f"{name}/min_ttc_lt_{c:g}s_share_{w:g}s"] = wshare(t[f"min_ttc_s_{w:g}s"] < c)

    ep = tables["episodes"]
    horizon = float(ep["duration"].max()) if len(ep["duration"]) else 0.0
    collided = ~np.isnan(ep["first_collision_s"])
    out.update({"episode/count": len(ep["env"]), "episode/mean_s": mean(ep["duration"]),
                "episode/collisions_mean": mean(ep["collisions"])})
    for i, end in enumerate(EPISODE_ENDS):
      out[f"episode/end_{end}_share"] = share(ep["end"] == i)
    success = ep["end"] == EPISODE_ENDS.index("success")
    completed = ep["end"] != EPISODE_ENDS.index("eval_end")
    out.update({"goal/completed_episodes": int(completed.sum()),
                "goal/censored_episodes": int((~completed).sum()),
                "goal/success_fraction": float(success.sum() / completed.sum()) if completed.any() else None,
                "goal/success_per_robot_min": per(int(success.sum()), robot_min),
                "goal/success_time_mean_s": mean(ep["duration"][success])})
    km = kaplan_meier(np.where(collided, ep["first_collision_s"], ep["duration"]), collided, horizon,
                      cfg.survival_times_s)
    out.update({f"first_collision/{k}": v for k, v in km.items()})
    km = kaplan_meier(ep["duration"], ep["fell"] > 0, horizon, cfg.survival_times_s)  # a tie with a collision counts
    out.update({f"first_fall/{k}": v for k, v in km.items()})
    out["first_collision/horizon_s"] = horizon
    return out


def cfg_from_dict(d: dict | None) -> TraversalEvalCfg:
  """A strict `TraversalEvalCfg` from JSON: unknown keys are rejected, lists become tuples."""
  return from_dict(TraversalEvalCfg, d, what="traversal evaluation")
