"""Encounter behavior: conditioned travel, a collision bubble and committed decisions.

The alternative to the plaza design (`behavior.plaza`: random goals with continuous
time-to-closest-approach avoidance). Each human follows a condition and reacts to
neighbors through discrete decisions that it keeps once made:

1. **Conditioning** gives the preferred travel: toward the current waypoint (random
   waypoints, or fixed routes) or along a commanded velocity. A large change of travel
   direction becomes a turn of a sampled radius; a zero radius turns in place.
2. **Bubble.** A neighbor is a threat when it is inside the collision bubble
   (`bubble_radius_m`) and is approaching, and when the two would come within
   `pass_margin_m` of each other's bodies within `horizon_s` if both kept their
   intended velocities.
3. **Switching.** At each step a cruising human with a threat switches to an
   intervention with probability `1 - exp(-rate * dt)`, where
   `rate = switch_rate_hz * (1 - assertiveness) * urgency ** urgency_exponent`.
   `urgency` is 0 when the closest threat is at the bubble's edge and rises to 1 at body
   contact. Assertiveness is drawn per human at reset. While a human is blocked, its
   effective assertiveness decays over `patience_s`, so standoffs resolve.
4. **Strategy.** The intervention is drawn from weights interpolated by urgency between
   `strategy_far` and `strategy_near` (order `STRATEGIES`):
   - sidestep and continue;
   - a gradual turn away and back (`veer`), with its radius interpolated by urgency;
   - an in-place turn away and back (`pivot`);
   - `yield`: slow or stop until clear.

   The side avoids the threat's closest-approach point and is checked against walls
   and obstacles. The human keeps the intervention until it finishes; it does not
   re-decide partway.
5. **Stop rule.** It overrides every command. Motion toward a neighbor that would bring
   the bodies within `stop_clearance_m` within `stop_horizon_s` is removed, whether
   that motion is intended or still executing; sliding past is allowed. A human stops
   when a conflict remains, and it may still turn in place. Both humans in a conflict
   may decide at once. Nothing breaks the tie except this rule. Bodies that end up
   closer than the clearance, because executed motion lags commands, step apart at
   `separation_speed_mps`. Commands are requests, so the graph's clearance cost
   (`clearance_weight`) applies the same rule when clips are selected.

Bodies are vertical cylinders of radius `body_radius_m` for these rules. Decisions are
made once per command step (one graph edge). Randomness uses per-human counter streams
(`human_seeds` scheme `"encounter"`). Each command step draws one fixed block of
uniforms per human, so results never depend on batch order or composition. These are
controllable heuristics, not calibrated pedestrian behavior. Tensors are `[E, M, ...]`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from crowdmotion.behavior.planner import plan_arc
from crowdmotion.core.config import from_dict
from crowdmotion.core.geometry import wrap
from crowdmotion.core.rng import hash_uniform, human_seeds

STRATEGIES = ("sidestep", "veer", "pivot", "yield")
MODES = ("cruise", *STRATEGIES)
CRUISE, SIDESTEP, VEER, PIVOT, YIELD = range(len(MODES))
CONDITIONING = ("random_waypoints", "waypoints", "velocity")

# Uniform slots of each human's per-step draw.
(S_HAZARD, S_STRATEGY, S_RIGHT, S_NOISE, S_ANGLE, S_RADIUS_JITTER, S_HOLD, S_SIDESTEP, S_RATE, S_YIELD,
 S_WAIT, S_WAIT_S, S_SPEED, S_INPLACE, S_TURN_RADIUS, S_WAYPOINT) = range(16)
WAYPOINT_CANDIDATES = 8
SLOTS = S_WAYPOINT + 2 * WAYPOINT_CANDIDATES
SPAWN_CANDIDATES = 32
FINISH_TIMEOUT_S = 1.5  # extra time allowed for a turn phase to reach its target
LEAD_S = 0.5  # predict a turning human's motion along the direction reached this far into its arc
NEIGHBORS = 4  # nearest neighbors passed to the graph's clearance cost
PUSH_MPS = 0.6  # largest push back out of a wall or obstacle margin


def _pairs(value) -> tuple:
  return tuple(tuple(float(x) for x in item) for item in value)


@dataclass
class EncounterCfg:
  # Scene (same names as `PlazaSteeringCfg`).
  half_x: float = 7.0
  half_z: float = 5.0
  wall_margin_m: float = 0.8
  obstacles: tuple[tuple[float, float, float], ...] = ()  # (x, z, radius)
  # Conditioning.
  conditioning: str = "random_waypoints"  # one of CONDITIONING
  speed_range: tuple[float, float] = (0.8, 1.6)  # preferred speed per leg (waypoints)
  arrive_m: float = 0.6
  arrive_blocked_m: float = 1.5  # a human blocked by the stop rule this close to its waypoint has arrived
  wait_probability: float = 0.3  # chance of waiting on reaching a waypoint
  wait_s: tuple[float, float] = (1.5, 4.0)
  waypoint_min_leg_m: float = 3.0  # random waypoints: preferred minimum leg length
  waypoints: tuple = ()  # "waypoints": per human, ((x, z), ...)
  loop_waypoints: bool = True  # "waypoints": restart the route, else stop at its end
  velocities: tuple = ()  # "velocity": per human, (speed m/s, heading deg)
  starts: tuple = ()  # optional per human (x, z, heading deg); default: random spawn
  spawn_gap_m: float = 1.2
  # Ordinary turns toward the condition.
  turn_start_deg: float = 30.0
  turn_finish_deg: float = 10.0
  turn_radius_m: tuple[float, float] = (0.5, 2.0)  # gradual turns
  inplace_min_deg: float = 75.0  # turns at least this large may be made in place
  inplace_probability: float = 0.5
  inplace_rate_deg_s: tuple[float, float] = (90.0, 150.0)
  max_turn_rate_deg_s: float = 180.0  # gradual turns slow down to respect it
  # Bubble and decisions.
  body_radius_m: float = 0.35  # reach of swinging arms: per-human max joint reach is about 0.4 m from the root
  bubble_radius_m: float = 2.5
  pass_margin_m: float = 0.3
  horizon_s: float = 3.0
  switch_rate_hz: float = 2.0
  urgency_exponent: float = 1.0
  assertiveness_range: tuple[float, float] = (0.0, 0.9)  # drawn per human at reset
  patience_s: float = 3.0
  strategy_far: tuple[float, float, float, float] = (0.2, 0.5, 0.05, 0.25)  # STRATEGIES order
  strategy_near: tuple[float, float, float, float] = (0.35, 0.1, 0.25, 0.3)
  prefer_right: float = 0.6  # head-on: chance of passing on one's right
  side_noise: float = 0.1  # chance of taking the side toward the threat
  # Interventions.
  sidestep_m: tuple[float, float] = (0.4, 0.9)
  sidestep_speed_mps: float = 0.6
  sidestep_forward_scale: float = 0.5
  veer_angle_deg: tuple[float, float] = (20.0, 45.0)
  veer_radius_m: tuple[float, float] = (3.0, 0.8)  # (far, near): radius by urgency
  pivot_angle_deg: tuple[float, float] = (45.0, 90.0)
  hold_s: tuple[float, float] = (0.8, 2.0)  # straight walking after a veer or pivot
  yield_speed_scale: tuple[float, float] = (0.0, 0.5)
  yield_max_s: float = 3.0
  # Stop rule.
  stop_clearance_m: float = 0.01
  stop_horizon_s: float = 1.0
  separation_speed_mps: float = 0.3  # bodies closer than the clearance step apart at this speed
  approach_time_s: float = 0.8  # approach anyone no faster than (gap - clearance) / this; 0 disables

  def __post_init__(self):
    self.waypoints = tuple(_pairs(route) for route in self.waypoints)
    self.velocities, self.starts = _pairs(self.velocities), _pairs(self.starts)
    self.obstacles = _pairs(self.obstacles)
    if self.conditioning not in CONDITIONING:
      raise ValueError(f"conditioning must be one of {CONDITIONING}")
    if self.conditioning == "waypoints" and (not self.waypoints or any(not r for r in self.waypoints)):
      raise ValueError("waypoint conditioning needs a nonempty route per human")
    if any(len(p) != 2 for r in self.waypoints for p in r):
      raise ValueError("waypoints are (x, z) pairs")
    if self.conditioning == "velocity" and not self.velocities:
      raise ValueError("velocity conditioning needs (speed, heading_deg) per human")
    if any(len(v) != 2 or v[0] < 0 for v in self.velocities) or any(len(s) != 3 for s in self.starts):
      raise ValueError("velocities are (speed >= 0, heading_deg); starts are (x, z, heading_deg)")
    for name in ("speed_range", "wait_s", "turn_radius_m", "inplace_rate_deg_s", "assertiveness_range",
                 "sidestep_m", "veer_angle_deg", "pivot_angle_deg", "hold_s", "yield_speed_scale"):
      lo, hi = getattr(self, name)
      if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi):
        raise ValueError(f"{name} must be an ordered nonnegative finite range")
    if any(not 0 < r for r in self.veer_radius_m) or len(self.veer_radius_m) != 2:
      raise ValueError("veer_radius_m is a positive (far, near) pair")
    if self.assertiveness_range[1] > 1 or self.yield_speed_scale[1] > 1:
      raise ValueError("assertiveness and yield speed scale lie in [0, 1]")
    for name in ("strategy_far", "strategy_near"):
      w = getattr(self, name)
      if len(w) != len(STRATEGIES) or any(not math.isfinite(x) or x < 0 for x in w) or sum(w) <= 0:
        raise ValueError(f"{name} needs {len(STRATEGIES)} nonnegative weights ({STRATEGIES}) with a positive sum")
    for name in ("wait_probability", "inplace_probability", "prefer_right", "side_noise"):
      if not 0 <= getattr(self, name) <= 1:
        raise ValueError(f"{name} is a probability")
    if not 0 < self.turn_finish_deg < self.turn_start_deg <= 180:
      raise ValueError("turns need 0 < finish < start <= 180 degrees")
    if not 0 < self.body_radius_m < self.bubble_radius_m / 2:
      raise ValueError("the bubble must be larger than body contact (2 * body_radius_m)")
    positive = ("horizon_s", "patience_s", "stop_horizon_s", "sidestep_speed_mps", "max_turn_rate_deg_s",
                "arrive_m", "half_x", "half_z", "yield_max_s")
    if any(not getattr(self, n) > 0 for n in positive):
      raise ValueError(f"{positive} must be positive")
    nonnegative = ("switch_rate_hz", "urgency_exponent", "pass_margin_m", "stop_clearance_m", "wall_margin_m",
                   "waypoint_min_leg_m", "spawn_gap_m", "sidestep_forward_scale", "arrive_blocked_m",
                   "separation_speed_mps", "approach_time_s")
    if any(not (math.isfinite(getattr(self, n)) and getattr(self, n) >= 0) for n in nonnegative):
      raise ValueError(f"{nonnegative} must be nonnegative and finite")

  @classmethod
  def from_dict(cls, d: dict | None) -> "EncounterCfg":
    return from_dict(cls, d, what="encounter")

  def as_json(self) -> dict:
    return asdict(self)


def _lerp(bounds, u):
  return bounds[0] + (bounds[1] - bounds[0]) * u


def _direction(heading):
  return torch.stack((torch.sin(heading), torch.cos(heading)), -1)


def _right(heading):
  return torch.stack((torch.cos(heading), -torch.sin(heading)), -1)


def _closest(r, rv, horizon):
  """Closest approach distance of relative positions `r` moving at `rv` within `horizon`."""
  tau = (-(r * rv).sum(-1) / (rv * rv).sum(-1).clamp(min=1e-9)).clamp(0.0, horizon)
  point = r + rv * tau[..., None]
  return point.norm(dim=-1), point


class EncounterBehavior:
  """Command source for `E` environments of `M` humans (see the module docstring)."""

  def __init__(self, envs: int, humans: int, cfg: EncounterCfg, seed: int, device):
    self.cfg, self.E, self.M, self.device = cfg, envs, humans, torch.device(device)
    self.base_seed = seed
    for name in ("waypoints", "velocities", "starts"):
      items = getattr(cfg, name)
      if items and len(items) != humans:
        raise ValueError(f"encounter {name} lists {len(items)} humans; the crowd has {humans}")
    dev, shape = self.device, (envs, humans)

    def zeros(dtype=torch.float32):
      return torch.zeros(shape, dtype=dtype, device=dev)

    obs = cfg.obstacles or ((1e6, 1e6, 0.0),)
    self.obstacles = torch.tensor(obs, dtype=torch.float32, device=dev)  # [O, 3]
    self.goal = torch.zeros(envs, humans, 2, device=dev)
    self.speed = zeros()  # preferred speed of the current leg
    self.wait_until = torch.full(shape, -1.0, device=dev)
    k = max([len(r) for r in cfg.waypoints] or [1])
    self.route = torch.zeros(envs, humans, k, 2, device=dev)
    self.route_len = torch.ones(shape, dtype=torch.long, device=dev)
    self.route_index = zeros(torch.long)
    if cfg.waypoints:
      for i, r in enumerate(cfg.waypoints):
        self.route[:, i, :len(r)] = torch.tensor(r, device=dev)
        self.route_len[:, i] = len(r)
    self.velocity = torch.zeros(envs, humans, 2, device=dev)  # conditioned (speed, heading)
    if cfg.velocities:
      v = torch.tensor(cfg.velocities, device=dev)
      self.velocity[:] = torch.stack((v[:, 0], torch.deg2rad(v[:, 1])), -1)
    self.finished = zeros(torch.bool)  # reached the end of a non-looping route
    # Ordinary turns.
    self.turning, self.turn_radius, self.turn_rate, self.turn_until = zeros(torch.bool), zeros(), zeros(), zeros()
    # Interventions.
    self.mode, self.phase = zeros(torch.long), zeros(torch.long)
    self.side, self.target, self.radius, self.rate = zeros(), zeros(), zeros(), zeros()
    self.started, self.until, self.hold, self.scale = zeros(), zeros(), zeros(), zeros()
    self.assertiveness = zeros()
    self.blocked_since = torch.full(shape, -1.0, device=dev)
    self.seeds, self.counters = zeros(torch.long), zeros(torch.long)
    self.episodes = torch.full((envs,), -1, dtype=torch.long, device=dev)
    self.last_t = zeros()
    # Per-step records (for traces, diagnostics and the debug view).
    self.threat, self.eligible, self.switched, self.stopped, self.separating, self.evasive = (
      zeros(torch.bool) for _ in range(6))
    self.urgency = zeros()
    self.turn_active, self.turn_target, self.orientation = zeros(torch.bool), zeros(), torch.ones(shape, device=dev)
    self._plan = self._neighbors = None

  # Random streams --------------------------------------------------------------------
  def _draw(self, env_ids, counter: int | None = None, slots: int = SLOTS) -> torch.Tensor:
    seeds = self.seeds[env_ids].reshape(-1)
    counters = self.counters[env_ids].reshape(-1) if counter is None else torch.full_like(seeds, counter)
    return hash_uniform(seeds, counters, slots).view(len(env_ids), self.M, slots).to(self.device)

  # Sampling ------------------------------------------------------------------------------
  def _box(self, margin: float):
    c = self.cfg
    lo = torch.tensor([-c.half_x + margin, -c.half_z + margin], device=self.device)
    return lo, -lo

  def _clear(self, points, margin: float):
    """Points `[..., 2]` at least `margin` from every obstacle's edge."""
    d = (points[..., None, :] - self.obstacles[:, :2]).norm(dim=-1)
    return (d >= self.obstacles[:, 2] + margin).all(-1)

  def _spawn(self, env_ids) -> torch.Tensor:
    """Greedy non-overlapping starts from each human's own candidates."""
    e, M = len(env_ids), self.M
    u = self._draw(env_ids, counter=-2, slots=2 * SPAWN_CANDIDATES).view(e, M, SPAWN_CANDIDATES, 2)
    lo, hi = self._box(0.8)
    cand = lo + (hi - lo) * u
    clear = self._clear(cand, 0.8)
    placed = torch.zeros(e, M, 2, device=self.device)
    rows = torch.arange(e, device=self.device)
    for i in range(M):
      ci = cand[:, i]
      if i:
        gap = (ci[:, :, None] - placed[:, None, :i]).norm(dim=-1).amin(-1)
      else:
        gap = torch.full(ci.shape[:2], float("inf"), device=self.device)
      good = clear[:, i] & (gap > self.cfg.spawn_gap_m)
      fallback = torch.where(clear[:, i], gap, -torch.inf).argmax(1)
      pick = torch.where(good.any(1), good.float().argmax(1), fallback)
      placed[:, i] = ci[rows, pick]
    return placed

  def _waypoints(self, xz, u) -> torch.Tensor:
    """Random waypoints `[n, 2]`: the first candidate clear of obstacles at least the
    minimum leg away, else the farthest clear candidate."""
    n = len(xz)
    lo, hi = self._box(1.0)
    cand = lo + (hi - lo) * u.reshape(n, WAYPOINT_CANDIDATES, 2)
    clear = self._clear(cand, 0.8)
    leg = (cand - xz[:, None]).norm(dim=-1)
    good = clear & (leg >= self.cfg.waypoint_min_leg_m)
    pick = torch.where(good.any(1), good.float().argmax(1), torch.where(clear, leg, -1.0).argmax(1))
    return cand[torch.arange(n, device=self.device), pick]

  # External conditioning ---------------------------------------------------------------
  def set_waypoints(self, env_ids, routes: torch.Tensor, lengths: torch.Tensor | None = None) -> None:
    """Replace the routes `[e, M, K, 2]` of `env_ids` (conditioning "waypoints")."""
    k = routes.shape[2]
    if k > self.route.shape[2]:
      pad = torch.zeros(self.E, self.M, k - self.route.shape[2], 2, device=self.device)
      self.route = torch.cat((self.route, pad), 2)
    self.route[env_ids, :, :k] = routes.to(self.device)
    self.route_len[env_ids] = k if lengths is None else lengths.to(self.device)
    self.route_index[env_ids] = 0
    self.finished[env_ids] = False
    self.goal[env_ids] = self.route[env_ids, :, 0]

  def set_velocities(self, env_ids, speed: torch.Tensor, heading: torch.Tensor) -> None:
    """Replace the conditioned velocity `(speed [e, M], heading [e, M])` (conditioning "velocity")."""
    self.velocity[env_ids] = torch.stack((speed, heading), -1).to(self.device)

  # Reset ---------------------------------------------------------------------------------
  def reset(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """New starts `xz [e, M, 2]`, `heading [e, M]` and conditions for `env_ids`."""
    c, M, dev = self.cfg, self.M, self.device
    env_ids = env_ids.to(dev)
    e = len(env_ids)
    self.episodes[env_ids] += 1
    ids = env_ids[:, None] * M + torch.arange(M, device=dev)
    self.seeds[env_ids] = human_seeds(self.base_seed, ids, "encounter", episode=self.episodes[env_ids][:, None])
    self.counters[env_ids] = 0
    u = self._draw(env_ids, counter=-1)
    if c.starts:
      starts = torch.tensor(c.starts, device=dev).expand(e, M, 3)
      xz, heading = starts[..., :2].clone(), torch.deg2rad(starts[..., 2])
    else:
      xz, heading = self._spawn(env_ids), None
    self.assertiveness[env_ids] = _lerp(c.assertiveness_range, u[..., 0])
    self.speed[env_ids] = _lerp(c.speed_range, u[..., 1])
    self.wait_until[env_ids] = -1.0
    self.finished[env_ids] = False
    if c.conditioning == "random_waypoints":
      block = self._draw(env_ids, counter=-3, slots=2 * WAYPOINT_CANDIDATES)
      self.goal[env_ids] = self._waypoints(xz.reshape(-1, 2), block.reshape(e * M, -1)).view(e, M, 2)
    elif c.conditioning == "waypoints":
      self.route_index[env_ids] = 0
      self.goal[env_ids] = self.route[env_ids, :, 0]
    else:
      self.speed[env_ids] = self.velocity[env_ids, :, 0]
    if heading is None:
      to_goal = self.goal[env_ids] - xz
      heading = (self.velocity[env_ids, :, 1] if c.conditioning == "velocity"
                 else torch.atan2(to_goal[..., 0], to_goal[..., 1]))
    if c.conditioning == "velocity":
      self.goal[env_ids] = xz + 3.0 * _direction(self.velocity[env_ids, :, 1])
    for name in ("turning", "threat", "eligible", "switched", "stopped", "separating", "evasive", "turn_active"):
      getattr(self, name)[env_ids] = False
    for name in ("mode", "phase"):
      getattr(self, name)[env_ids] = 0
    for name in ("turn_radius", "turn_rate", "turn_until", "side", "radius", "rate", "started", "until", "hold",
                 "scale", "urgency", "last_t"):
      getattr(self, name)[env_ids] = 0.0
    self.target[env_ids] = heading
    self.turn_target[env_ids] = heading
    self.orientation[env_ids] = 1.0
    self.blocked_since[env_ids] = -1.0
    return xz, heading

  # Conditioning --------------------------------------------------------------------------
  def _condition(self, env_ids, now, xz, u):
    """Arrivals, waits and next waypoints; returns `(preferred speed, heading, waiting)`."""
    c = self.cfg
    if c.conditioning == "velocity":
      vel = self.velocity[env_ids]
      self.goal[env_ids] = xz + 3.0 * _direction(vel[..., 1])
      return vel[..., 0], vel[..., 1], torch.zeros_like(now, dtype=torch.bool)
    goal, wait = self.goal[env_ids], self.wait_until[env_ids]
    dist = (goal - xz).norm(dim=-1)
    blocked = self.blocked_since[env_ids] >= 0  # stopped by the rule at the last command
    arrived = ((dist < c.arrive_m) | (blocked & (dist < c.arrive_blocked_m))) & ~self.finished[env_ids]
    start_wait = arrived & (wait < 0) & (u[..., S_WAIT] < c.wait_probability)
    wait = torch.where(start_wait, now + _lerp(c.wait_s, u[..., S_WAIT_S]), wait)
    done = arrived & ~start_wait & (wait < now)
    if c.conditioning == "random_waypoints":
      fresh = self._waypoints(xz.reshape(-1, 2), u[..., S_WAYPOINT:].reshape(-1, 2 * WAYPOINT_CANDIDATES))
      goal = torch.where(done[..., None], fresh.view_as(goal), goal)
      self.speed[env_ids] = torch.where(done, _lerp(c.speed_range, u[..., S_SPEED]), self.speed[env_ids])
    else:
      index = self.route_index[env_ids] + done.long()
      length = self.route_len[env_ids]
      end = index >= length
      if c.loop_waypoints:
        index = torch.where(end, 0, index)
      else:
        self.finished[env_ids] |= end
        index = torch.minimum(index, length - 1)
      self.route_index[env_ids] = index
      route = self.route[env_ids]
      goal = route.gather(2, index[..., None, None].expand(*index.shape, 1, 2))[:, :, 0]
    wait = torch.where(done, -1.0, wait)
    wait = torch.where(self.finished[env_ids], float("inf"), wait)  # a finished route waits for good
    self.goal[env_ids], self.wait_until[env_ids] = goal, wait
    to_goal = goal - xz
    dist = to_goal.norm(dim=-1)
    waiting = wait >= now
    speed = self.speed[env_ids] * (dist / 1.0).clamp(max=1.0)
    speed = torch.where(waiting, torch.zeros_like(speed), speed)
    return speed, torch.atan2(to_goal[..., 0], to_goal[..., 1]), waiting

  # Interventions -------------------------------------------------------------------------
  def _advance_maneuvers(self, env_ids, now, xz, heading, threat_before):
    """Phase changes and completion of ongoing interventions."""
    c = self.cfg
    mode, phase = self.mode[env_ids], self.phase[env_ids]
    until, target = self.until[env_ids], self.target[env_ids]
    aligned = wrap(target - heading).abs() <= math.radians(c.turn_finish_deg)
    turned = ((mode == VEER) | (mode == PIVOT)) & (phase == 0) & (aligned | (now >= until))
    phase = torch.where(turned, 1, phase)
    until = torch.where(turned, now + self.hold[env_ids], until)
    # Walking on toward a wall or obstacle ends a veer, pivot or sidestep early.
    ahead = torch.where((mode == SIDESTEP)[..., None], _right(target) * self.side[env_ids, ..., None] * 0.6,
                        _direction(torch.where(phase == 1, target, heading)))
    walled = ((mode == SIDESTEP) | (mode == VEER) | ((mode == PIVOT) & (phase == 1))) & ~self._inside(xz + ahead)
    over = (((mode == SIDESTEP) & (now >= until))
            | (((mode == VEER) | (mode == PIVOT)) & (phase == 1) & (now >= until) & ~turned)
            | ((mode == YIELD) & ((now >= until) | (~threat_before & (now - self.started[env_ids] >= 0.4))))
            | walled)
    self.mode[env_ids] = torch.where(over, CRUISE, mode)
    self.phase[env_ids] = torch.where(over, 0, phase)
    self.until[env_ids] = until

  def _intended(self, env_ids, now, xz, heading, pref_speed, pref_heading, waiting, u):
    """Intended `(speed, travel heading, yaw rate, max turn, turn active, turn target,
    orientation weight, wanted heading)` before the stop rule, for every human's current
    mode. The wanted heading is where the current phase finally faces."""
    c = self.cfg
    mode, phase = self.mode[env_ids], self.phase[env_ids]
    cruise = mode == CRUISE
    # Ordinary turns toward the condition (cruising humans only).
    error = wrap(pref_heading - heading)
    moving = ~waiting & (pref_speed >= 0.15)
    turning = self.turning[env_ids] & cruise & moving & (error.abs() > math.radians(c.turn_finish_deg))
    begin = cruise & moving & ~turning & (error.abs() >= math.radians(c.turn_start_deg))
    radius = _lerp(c.turn_radius_m, u[..., S_TURN_RADIUS])
    if c.conditioning != "velocity":
      radius = torch.minimum(radius, (self.goal[env_ids] - xz).norm(dim=-1) / 2)
    in_place = ((error.abs() >= math.radians(c.inplace_min_deg)) & (u[..., S_INPLACE] < c.inplace_probability)
                | (radius < 0.25) | ~self._inside(xz + _direction(heading) * radius[..., None].clamp(max=1.5)))
    rate = torch.deg2rad(_lerp(c.inplace_rate_deg_s, u[..., S_RATE]))
    self.turn_radius[env_ids] = torch.where(begin, torch.where(in_place, 0.0, radius), self.turn_radius[env_ids])
    self.turn_rate[env_ids] = torch.where(begin, rate, self.turn_rate[env_ids])
    self.turn_until[env_ids] = torch.where(begin, now + error.abs() / rate.clamp(min=1e-3) + FINISH_TIMEOUT_S,
                                           self.turn_until[env_ids])
    # An in-place turn the graph cannot realize in time continues as a tight arc.
    stalled = turning & (self.turn_radius[env_ids] == 0) & (now >= self.turn_until[env_ids])
    self.turn_radius[env_ids] = torch.where(stalled, max(c.turn_radius_m[0], 0.5), self.turn_radius[env_ids])
    turning = turning | begin
    self.turning[env_ids] = turning
    turn_radius, turn_rate = self.turn_radius[env_ids], self.turn_rate[env_ids]
    max_rate = math.radians(c.max_turn_rate_deg_s)

    speed = torch.where(cruise, pref_speed, self.speed[env_ids])
    travel = torch.where(cruise, pref_heading, heading)
    yaw_rate = torch.zeros_like(speed)
    max_turn = torch.zeros_like(speed)
    turn_active = torch.zeros_like(cruise)
    turn_target = heading.clone()
    orientation = torch.ones_like(speed)

    # Cruise turns: in place, or an arc of the drawn radius from the current facing.
    pivot = turning & (turn_radius == 0)
    arc = turning & (turn_radius > 0)
    arc_speed = torch.minimum(pref_speed, max_rate * turn_radius)
    speed = torch.where(pivot, 0.0, torch.where(arc, arc_speed, speed))
    travel = torch.where(turning, heading, travel)
    rate = torch.where(pivot, turn_rate, arc_speed / turn_radius.clamp(min=1e-3))
    yaw_rate = torch.where(turning, torch.sign(error) * rate, yaw_rate)
    max_turn = torch.where(turning, error.abs(), max_turn)
    turn_active = turn_active | pivot
    turn_target = torch.where(pivot, pref_heading, turn_target)

    # Sidestep: hold the body facing, translate sideways while walking on slowly.
    target, side = self.target[env_ids], self.side[env_ids]
    step = mode == SIDESTEP
    velocity = (_direction(target) * (self.speed[env_ids] * c.sidestep_forward_scale)[..., None]
                + _right(target) * (side * c.sidestep_speed_mps)[..., None])
    speed = torch.where(step, velocity.norm(dim=-1), speed)
    travel = torch.where(step, torch.atan2(velocity[..., 0], velocity[..., 1]), travel)
    turn_active = turn_active | step
    turn_target = torch.where(step, target, turn_target)
    orientation = torch.where(step, 0.0, orientation)

    # Veer and pivot: turn toward the side target, then walk straight along it.
    away = wrap(target - heading)
    veer_out = (mode == VEER) & (phase == 0)
    pivot_out = (mode == PIVOT) & (phase == 0)
    straight = ((mode == VEER) | (mode == PIVOT)) & (phase == 1)
    veer_speed = torch.minimum(self.speed[env_ids], max_rate * self.radius[env_ids])
    speed = torch.where(veer_out, veer_speed, torch.where(pivot_out, 0.0, speed))
    travel = torch.where(veer_out | pivot_out, heading, torch.where(straight, target, travel))
    out_rate = torch.where(pivot_out, self.rate[env_ids], veer_speed / self.radius[env_ids].clamp(min=1e-3))
    yaw_rate = torch.where(veer_out | pivot_out, torch.sign(away) * out_rate, yaw_rate)
    max_turn = torch.where(veer_out | pivot_out, away.abs(), max_turn)
    turn_active = turn_active | pivot_out
    turn_target = torch.where(pivot_out, target, turn_target)

    # Yield: keep the travel direction at a reduced speed (possibly zero).
    give = mode == YIELD
    speed = torch.where(give, pref_speed * self.scale[env_ids], speed)
    travel = torch.where(give, pref_heading, travel)
    want = torch.where(cruise | give, pref_heading, target)
    return speed, travel, yaw_rate, max_turn, turn_active, turn_target, orientation, want

  def _confine(self, xz, velocity):
    """Remove travel into walls and obstacles and push back out of their margins."""
    c = self.cfg
    lim = torch.tensor([c.half_x - c.wall_margin_m, c.half_z - c.wall_margin_m], device=self.device)
    sign = torch.sign(xz)
    excess = (xz.abs() - lim).clamp(min=0.0)
    outward = (velocity * sign).clamp(min=0.0) * (excess > 0)
    velocity = velocity - sign * (outward + (3.0 * excess).clamp(max=PUSH_MPS))
    away = xz[..., None, :] - self.obstacles[:, :2]
    dd = away.norm(dim=-1).clamp(min=1e-6)
    normal = away / dd[..., None]
    reach = self.obstacles[:, 2] + 0.6
    inside = dd < reach
    inward = (-(velocity[..., None, :] * normal).sum(-1)).clamp(min=0.0) * inside
    push = (3.0 * (reach - dd)).clamp(0.0, PUSH_MPS)
    velocity = velocity + (normal * (inward + push)[..., None]).sum(-2)
    return velocity, (excess > 0).any(-1) | inside.any(-1)

  def _inside(self, points) -> torch.Tensor:
    """Points `[..., 2]` inside the wall margin and clear of obstacles."""
    c = self.cfg
    lim = torch.tensor([c.half_x - c.wall_margin_m, c.half_z - c.wall_margin_m], device=self.device)
    return (points.abs() <= lim).all(-1) & self._clear(points, 0.3)

  def _side_feasible(self, xz, heading, side):
    """Whether a sideways move on `side` stays inside the walls and clear of obstacles."""
    return self._inside(xz + _right(heading) * side[..., None] + _direction(heading))

  # Command -------------------------------------------------------------------------------
  @torch.no_grad()
  def encounter_command(self, t, xz, vel, heading, env_ids):
    """`(speed [e, M], heading [e, M])` for environments `env_ids` at their times `t [e]`."""
    c, M = self.cfg, self.M
    e = len(env_ids)
    now = t.to(self.device)[:, None].expand(e, M)
    dt = (now - self.last_t[env_ids]).clamp(min=0.0)
    self.last_t[env_ids] = now
    u = self._draw(env_ids)
    self.counters[env_ids] += 1
    pref_speed, pref_heading, waiting = self._condition(env_ids, now, xz, u)
    self._advance_maneuvers(env_ids, now, xz, heading, self.threat[env_ids])

    def intended():
      """Commands for every human's mode, and the velocity used to predict its motion:
      the travel direction reached `LEAD_S` into a turning arc, kept clear of walls."""
      speed, travel, yaw_rate, max_turn, *rest = self._intended(
        env_ids, now, xz, heading, pref_speed, pref_heading, waiting, u)
      lead = torch.maximum(torch.minimum(yaw_rate * LEAD_S, max_turn), -max_turn)
      velocity, confined = self._confine(xz, _direction(travel + lead) * speed[..., None])
      return velocity, confined, speed, travel, yaw_rate, max_turn, rest

    velocity = intended()[0]

    # Threats in the bubble, judged on intended velocities.
    eye = torch.eye(M, dtype=torch.bool, device=self.device)
    r = xz[:, None] - xz[:, :, None]  # [e, i, j, 2]: j relative to i
    rv = velocity[:, None] - velocity[:, :, None]
    dist = r.norm(dim=-1).masked_fill(eye, float("inf"))
    contact = 2 * c.body_radius_m
    miss, point = _closest(r, rv, c.horizon_s)
    approaching = (r * rv).sum(-1) < 0
    threats = (dist < c.bubble_radius_m) & approaching & (miss < contact + c.pass_margin_m)
    near = torch.where(threats, dist, float("inf")).min(-1)
    threat = threats.any(-1)
    urgency = ((c.bubble_radius_m - near.values) / (c.bubble_radius_m - contact)).clamp(0.0, 1.0)
    urgency = torch.where(threat, urgency, 0.0)

    # Switching hazard for cruising humans.
    blocked_s = torch.where(self.blocked_since[env_ids] >= 0, now - self.blocked_since[env_ids], 0.0)
    assertive = self.assertiveness[env_ids] * torch.exp(-blocked_s / c.patience_s)
    rate = c.switch_rate_hz * (1 - assertive) * urgency.pow(c.urgency_exponent)
    eligible = (self.mode[env_ids] == CRUISE) & ~waiting & threat
    switch = eligible & (u[..., S_HAZARD] < 1 - torch.exp(-rate * dt))
    if switch.any():
      self._begin(env_ids, switch, now, xz, heading, urgency, near.indices, r, point, u)
    velocity, confined, speed, travel, yaw_rate, max_turn, rest = intended()
    turn_active, turn_target, orientation, want = rest

    # Stop rule. Others are predicted at their executed velocities. Motion conflicts
    # when the intended or the current (still executing) velocity would bring the bodies
    # within the clearance and standing would keep them further apart. The intended
    # velocity loses its component toward each conflicting neighbor (sliding past is
    # allowed); a human stops if a conflict remains or little motion is left.
    threshold = contact + c.stop_clearance_m
    others = vel[:, None].expand_as(r)  # [e, i, j, 2]: j's executed velocity
    still_gap, _ = _closest(r, others, c.stop_horizon_s)

    def conflicts(own):
      gap, _ = _closest(r, others - own[:, :, None], c.stop_horizon_s)
      return ((gap < threshold) & (gap < still_gap - 1e-3)).masked_fill(eye, False)

    unit = r / r.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    if c.approach_time_s > 0:
      # Approach each neighbor no faster than the remaining gap allows; passing is free.
      allowed = ((r.norm(dim=-1) - threshold).clamp(min=0.0) / c.approach_time_s).masked_fill(eye, float("inf"))
      excess = ((velocity[:, :, None] * unit).sum(-1) - allowed).clamp(min=0.0)
      slowed = (excess > 0).any(-1)
      velocity = velocity - (excess[..., None] * unit).sum(2)
      speed = torch.where(slowed, velocity.norm(dim=-1), speed)
      travel = torch.where(slowed, torch.atan2(velocity[..., 0], velocity[..., 1]), travel)
      yaw_rate = torch.where(slowed, 0.0, yaw_rate)
    conflict = conflicts(velocity) | conflicts(vel)
    closing = ((velocity[:, :, None] * unit).sum(-1).clamp(min=0.0) * conflict)[..., None]
    slid = velocity - (closing * unit).sum(2)
    held = conflict.any(-1)
    stop = held & (conflicts(slid).any(-1) | (slid.norm(dim=-1) < 0.15))
    velocity = torch.where(stop[..., None], 0.0, torch.where(held[..., None], slid, velocity))
    speed = torch.where(stop, 0.0, torch.where(held, velocity.norm(dim=-1), speed))
    redirected = confined | held
    travel = torch.where(redirected, torch.atan2(velocity[..., 0], velocity[..., 1]), travel)
    speed = torch.where(confined & ~stop, velocity.norm(dim=-1), speed)
    yaw_rate = torch.where(redirected, 0.0, yaw_rate)
    # A stopped human keeps turning toward its wanted heading, in place.
    pivot = stop & (wrap(want - heading).abs() > math.radians(c.turn_finish_deg))
    turn_active = turn_active | pivot
    turn_target = torch.where(pivot, want, turn_target)
    # Bodies already within the clearance step apart, facing unchanged.
    overlap = (threshold - r.norm(dim=-1)).clamp(min=0.0).masked_fill(eye, 0.0)  # [e, i, j]
    away = -(r / r.norm(dim=-1, keepdim=True).clamp(min=1e-6) * overlap[..., None]).sum(2)
    separate = (overlap > 0).any(-1) & (c.separation_speed_mps > 0)
    push = away / away.norm(dim=-1, keepdim=True).clamp(min=1e-6) * c.separation_speed_mps
    velocity = torch.where(separate[..., None], velocity + push, velocity)
    speed = torch.where(separate, velocity.norm(dim=-1), speed)
    travel = torch.where(separate, torch.atan2(velocity[..., 0], velocity[..., 1]), travel)
    yaw_rate = torch.where(separate, 0.0, yaw_rate)
    turn_target = torch.where(separate & ~turn_active, heading, turn_target)
    turn_active = turn_active | separate
    orientation = torch.where(separate, 0.0, orientation)
    speed = speed.clamp(max=max([c.speed_range[1], *(v[0] for v in c.velocities)]))
    still = speed < 0.05
    travel = torch.where(still, heading, travel)
    speed = speed.masked_fill(still, 0.0)

    blocked = stop & (pref_speed >= 0.15)
    self.blocked_since[env_ids] = torch.where(
      blocked, torch.where(self.blocked_since[env_ids] >= 0, self.blocked_since[env_ids], now), -1.0)
    self.threat[env_ids], self.urgency[env_ids], self.switched[env_ids] = threat, urgency, switch
    self.eligible[env_ids] = eligible
    self.stopped[env_ids], self.separating[env_ids] = stop, separate
    self.evasive[env_ids] = (self.mode[env_ids] != CRUISE) | stop | separate
    self.turn_active[env_ids], self.turn_target[env_ids] = turn_active, turn_target
    self.orientation[env_ids] = orientation
    self._plan = (yaw_rate.reshape(-1), max_turn.reshape(-1))
    self._neighbors = self._nearest(xz, vel, dist, threshold)
    return speed, travel

  def _nearest(self, xz, vel, dist, threshold):
    """`(positions, executed velocities, clearance distance)` of each human's nearest
    neighbors and wall mirrors, flattened to rows, for the graph's clearance cost."""
    e, M = xz.shape[:2]
    n = min(NEIGHBORS, M - 1)
    idx = dist.topk(n, dim=-1, largest=False).indices  # [e, M, n]; self is at infinity
    pick = lambda x: x[:, None].expand(e, M, M, 2).gather(2, idx[..., None].expand(e, M, n, 2))  # noqa: E731
    # Stationary mirror images behind the nearest x and z walls: a mirror is as close
    # as twice the gap to its wall, so the same clearance keeps bodies off the walls.
    half = torch.tensor([self.cfg.half_x, self.cfg.half_z], device=self.device)
    side = torch.where(xz >= 0, 1.0, -1.0)
    mirror_x = torch.stack((2 * side[..., 0] * half[0] - xz[..., 0], xz[..., 1]), -1)
    mirror_z = torch.stack((xz[..., 0], 2 * side[..., 1] * half[1] - xz[..., 1]), -1)
    points = torch.cat((pick(xz), mirror_x[:, :, None], mirror_z[:, :, None]), 2)
    speeds = torch.cat((pick(vel), torch.zeros(e, M, 2, 2, device=self.device)), 2)
    limit = torch.full((e * M,), threshold, device=self.device)
    return points.reshape(e * M, n + 2, 2), speeds.reshape(e * M, n + 2, 2), limit

  def _begin(self, env_ids, switch, now, xz, heading, urgency, nearest, r, point, u):
    """Draw and start an intervention for the humans in `switch`. A human already
    blocked by the stop rule does not draw yield, which would change nothing."""
    c = self.cfg
    far = torch.tensor(c.strategy_far, device=self.device)
    close = torch.tensor(c.strategy_near, device=self.device)
    weights = far + (close - far) * urgency[..., None]
    blocked = (self.blocked_since[env_ids] >= 0)[..., None]
    no_yield = weights.clone()
    no_yield[..., YIELD - 1] = 0.0
    weights = torch.where(blocked & (no_yield.sum(-1, keepdim=True) > 0), no_yield, weights)
    cdf = weights.cumsum(-1) / weights.sum(-1, keepdim=True)
    strategy = (u[..., S_STRATEGY, None] >= cdf).sum(-1).clamp(max=len(STRATEGIES) - 1) + 1
    # Side away from the closest threat's closest-approach point (body frame).
    pick = lambda x: x.gather(2, nearest[..., None, None].expand(*nearest.shape, 1, 2))[:, :, 0]  # noqa: E731
    lateral = (pick(point) * _right(heading)).sum(-1)
    now_lateral = (pick(r) * _right(heading)).sum(-1)
    lateral = torch.where(lateral.abs() < 0.1, now_lateral, lateral)
    head_on = lateral.abs() < 0.1
    side = torch.where(head_on, torch.where(u[..., S_RIGHT] < c.prefer_right, 1.0, -1.0), -torch.sign(lateral))
    side = torch.where(u[..., S_NOISE] < c.side_noise, -side, side)
    ok = self._side_feasible(xz, heading, side)
    other = self._side_feasible(xz, heading, -side)
    side = torch.where(ok | ~other, side, -side)
    strategy = torch.where(ok | other, strategy, YIELD)
    angle = torch.where(strategy == PIVOT, _lerp(c.pivot_angle_deg, u[..., S_ANGLE]),
                        _lerp(c.veer_angle_deg, u[..., S_ANGLE]))
    radius = _lerp(c.veer_radius_m, urgency) * (0.8 + 0.4 * u[..., S_RADIUS_JITTER])
    sidestep_s = _lerp(c.sidestep_m, u[..., S_SIDESTEP]) / c.sidestep_speed_mps
    pivot_rate = torch.deg2rad(_lerp(c.inplace_rate_deg_s, u[..., S_RATE]))
    turn_s = torch.where(strategy == PIVOT, torch.deg2rad(angle) / pivot_rate,
                         torch.deg2rad(angle) * radius / self.speed[env_ids].clamp(min=0.3))
    until = torch.where(strategy == SIDESTEP, now + sidestep_s,
                        torch.where(strategy == YIELD, now + c.yield_max_s, now + turn_s + FINISH_TIMEOUT_S))
    target = torch.where(strategy == SIDESTEP, heading, heading + side * torch.deg2rad(angle))
    sel = lambda new, old: torch.where(switch, new, old)  # noqa: E731
    self.mode[env_ids] = sel(strategy, self.mode[env_ids])
    self.phase[env_ids] = sel(torch.zeros_like(strategy), self.phase[env_ids])
    self.side[env_ids] = sel(side, self.side[env_ids])
    self.target[env_ids] = sel(target, self.target[env_ids])
    self.radius[env_ids] = sel(radius, self.radius[env_ids])
    self.rate[env_ids] = sel(pivot_rate, self.rate[env_ids])
    self.started[env_ids] = sel(now, self.started[env_ids])
    self.until[env_ids] = sel(until, self.until[env_ids])
    self.hold[env_ids] = sel(_lerp(c.hold_s, u[..., S_HOLD]), self.hold[env_ids])
    self.scale[env_ids] = sel(_lerp(c.yield_speed_scale, u[..., S_YIELD]), self.scale[env_ids])
    self.turning[env_ids] &= ~switch

  # Controller interface ------------------------------------------------------------------
  def command(self, t: torch.Tensor, kin, groups: torch.Tensor):
    """Controller interface: flat `[e*M]` kinematics for env ids `groups`."""
    e, M = len(groups), self.M
    speed, heading = self.encounter_command(
      t, kin.root_xz.view(e, M, 2), kin.vel.view(e, M, 2), kin.heading.view(e, M), groups)
    self._rows = groups
    return speed.reshape(-1), heading.reshape(-1)

  def plan(self, xz, vel, speed, heading, frames: int, fps: float):
    """Planner for the rows just commanded: arcs for gradual turns, in place at zero speed."""
    yaw_rate, max_turn = self._plan
    path, _ = plan_arc(xz, vel, speed, heading, yaw_rate, frames, fps, max_turn=max_turn)
    return path

  def motion_options(self) -> dict:
    """Graph preferences for the rows just commanded: a retained body target during
    in-place turns and sidesteps, no forward-travel preference while sidestepping, and
    the nearest neighbors for the clearance cost (`graph_overrides.clearance_weight`)."""
    g = self._rows
    return {"orientation_weight": self.orientation[g].reshape(-1),
            "turn_target": self.turn_target[g].reshape(-1),
            "turn_active": self.turn_active[g].reshape(-1),
            "neighbors": self._neighbors}

  def trace_records(self, env: int = 0) -> dict[str, torch.Tensor]:
    """Per-human state of environment `env` after the last command, for traces."""
    return {
      "encounter_mode": self.mode[env].clone(),
      "encounter_phase": self.phase[env].clone(),
      "encounter_threat": self.threat[env].clone(),
      "encounter_urgency": self.urgency[env].clone(),
      "encounter_eligible": self.eligible[env].clone(),
      "encounter_switch": self.switched[env].clone(),
      "encounter_stopped": self.stopped[env].clone(),
      "encounter_separating": self.separating[env].clone(),
      "encounter_assertiveness": self.assertiveness[env].clone(),
      "encounter_side": self.side[env].clone(),
      "encounter_ordinary_turn": self.turning[env].clone(),
      "encounter_ordinary_turn_radius": self.turn_radius[env].clone(),
      "steering_evasive": self.evasive[env].clone(),
      "turn_active": self.turn_active[env].clone(),
      "turn_target": self.turn_target[env].clone(),
    }
