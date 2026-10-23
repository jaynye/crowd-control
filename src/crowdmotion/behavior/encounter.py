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

Optional decisive behavior adds executed-stall recovery, retained veer targets,
all-neighbor geometric avoidance, and bounded completion when the direct goal path
clears. A separate reset draw can designate a small stubborn population; finite
stall patience retires that trait before forcing a resolution attempt. Every new
mechanism defaults off, preserving the original command and random-draw sequence.

**External agents** (the robot) are bodies the humans perceive but do not control:
`set_external` supplies their positions and velocities, which stand in for their
intended motion as well. Each external agent is one more neighbor column, a cylinder of
`robot_radius_m`, in the threat test, stop rule, separation, geometric avoidance, bypass
completion and the graph's clearance cost. A human's attitude toward external agents is
a reset trait, drawn apart from its stubbornness toward humans: oblivious (does not
perceive them at all), stubborn (obeys the stop rule and separation but never intervenes
for them), or cooperative with its own `robot_assertiveness_range`, which sets the
switch rate when an external agent is the closest threat. Without `set_external`
nothing changes.
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
HIDDEN_M = 1e6  # an external agent a human does not perceive sits this far away


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
  stubborn_fraction: float = 0.0  # opt-in population; traits and ranges apply at reset
  nonstubborn_assertiveness_range: tuple[float, float] = (0.0, 0.35)
  stubborn_patience_s: tuple[float, float] = (8.0, 14.0)  # then retires stubbornness for this episode
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
  blocked_turn_first: bool = False  # constrained veer: retain its target and pivot before translation
  retain_avoidance_heading: bool = False  # graph retains the veer target even when its arc is slowed
  minimal_avoidance: bool = False  # choose the least predicted-clear deviation from the goal heading
  avoidance_margin_m: tuple[float, float] = (0.02, 0.08)  # stochastic margin beyond pass_margin_m
  stall_recovery_s: float = 0.0  # opt-in: failed mutual near-stop attempts finish and pivot after this duration
  project_safe_velocity: bool = False  # closest velocity satisfying all radial approach/stop bounds
  pass_commitment_s: float = 0.0  # opt-in: leave bypass when goal path clears; bound its duration
  pass_commitment_jitter: float = 0.0  # earlier independent deadline for co-motion without goal progress
  anticipate_braking: bool = False  # graph checks both executed and braking neighbor predictions
  # External agents (the robot; only when `set_external` supplies them). Traits apply at reset.
  robot_radius_m: float = 0.3
  robot_oblivious_fraction: float = 0.0  # do not perceive external agents at all
  robot_stubborn_fraction: float = 0.0  # perceive and stop for them, but never intervene for them
  robot_assertiveness_range: tuple[float, float] = (0.0, 0.35)  # the cooperative rest
  reactive: bool = True  # opt-out: independent walkers, no human/robot response or graph clearance
  flow_fraction: float = 0.0  # persistent shared-heading assignment, reflected at boundaries
  flow_heading_deg: float | None = None  # None: independent random common heading per environment/reset
  flow_spread_deg: float = 10.0

  def __post_init__(self):
    if not 0 <= self.flow_fraction <= 1 or not 0 <= self.flow_spread_deg <= 180:
      raise ValueError("flow_fraction must lie in [0, 1], flow_spread_deg in [0, 180]")
    if self.flow_heading_deg is not None and not math.isfinite(self.flow_heading_deg):
      raise ValueError("flow_heading_deg must be finite")
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
                 "sidestep_m", "veer_angle_deg", "pivot_angle_deg", "hold_s", "yield_speed_scale",
                 "avoidance_margin_m", "nonstubborn_assertiveness_range", "stubborn_patience_s",
                 "robot_assertiveness_range"):
      lo, hi = getattr(self, name)
      if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi):
        raise ValueError(f"{name} must be an ordered nonnegative finite range")
    if any(not 0 < r for r in self.veer_radius_m) or len(self.veer_radius_m) != 2:
      raise ValueError("veer_radius_m is a positive (far, near) pair")
    if self.assertiveness_range[1] > 1 or self.yield_speed_scale[1] > 1:
      raise ValueError("assertiveness and yield speed scale lie in [0, 1]")
    if self.nonstubborn_assertiveness_range[1] > 1 or self.stubborn_patience_s[0] <= 0:
      raise ValueError("nonstubborn assertiveness lies in [0, 1]; stubborn patience must be positive")
    if self.robot_assertiveness_range[1] > 1 or not self.robot_radius_m > 0:
      raise ValueError("robot assertiveness lies in [0, 1]; robot_radius_m must be positive")
    if self.robot_oblivious_fraction + self.robot_stubborn_fraction > 1:
      raise ValueError("robot_oblivious_fraction + robot_stubborn_fraction must not exceed 1")
    for name in ("strategy_far", "strategy_near"):
      w = getattr(self, name)
      if len(w) != len(STRATEGIES) or any(not math.isfinite(x) or x < 0 for x in w) or sum(w) <= 0:
        raise ValueError(f"{name} needs {len(STRATEGIES)} nonnegative weights ({STRATEGIES}) with a positive sum")
    for name in ("wait_probability", "inplace_probability", "prefer_right", "side_noise", "stubborn_fraction",
                 "pass_commitment_jitter", "robot_oblivious_fraction", "robot_stubborn_fraction"):
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
                   "separation_speed_mps", "approach_time_s", "stall_recovery_s", "pass_commitment_s")
    if any(not (math.isfinite(getattr(self, n)) and getattr(self, n) >= 0) for n in nonnegative):
      raise ValueError(f"{nonnegative} must be nonnegative and finite")

  @classmethod
  def from_dict(cls, d: dict | None) -> "EncounterCfg":
    return from_dict(cls, d, what="encounter")

  def as_json(self) -> dict:
    return asdict(self)


@dataclass
class _External:
  """External agents of a batch of environments, as neighbor columns `M..N-1`."""
  xz: torch.Tensor  # [e, K, 2]
  vel: torch.Tensor  # [e, K, 2], also their predicted (intended) velocity
  shift: torch.Tensor  # [N]: subtracted from distances so each body counts as human-sized
  hidden: torch.Tensor  # [e, M, K]: not perceived
  decides: torch.Tensor  # [e, M, N]: columns a human may intervene for

  def velocities(self, humans: torch.Tensor) -> torch.Tensor:
    """Per-column velocities `[e, N, 2]`: the humans' `[e, M, 2]`, then the agents'."""
    return torch.cat((humans, self.vel), 1)


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


def project_velocity(velocity, normals, bounds):
  """Euclidean projection onto 2-D half-planes with unit normals, ``normal @ v <= bound``.

  Zero must be feasible (all bounds nonnegative). The closest point is the original
  velocity, a projection onto one line, or the intersection of two lines. Enumerating
  active sets avoids summing corrections from several neighbors, which can reverse
  a command. It grants no human precedence and is independent of constraint order
  apart from floating-point roundoff. Work only on rows violating a constraint.
  """
  shape = velocity.shape
  original = velocity.reshape(-1, 2)
  normal = normals.reshape(len(original), -1, 2)
  limit = bounds.reshape(len(original), -1)
  rows = ((original[:, None] * normal).sum(-1) > limit).any(-1).nonzero().flatten()
  if not len(rows):
    return velocity
  v, n, b = original[rows], normal[rows], limit[rows]
  finite = torch.isfinite(b)
  b = torch.where(finite, b, 1e6)
  excess = ((v[:, None] * n).sum(-1) - b).clamp(min=0.)
  single = v[:, None] - excess[..., None] * n
  i, j = torch.triu_indices(n.shape[1], n.shape[1], 1, device=v.device)
  ni, nj, bi, bj = n[:, i], n[:, j], b[:, i], b[:, j]
  det = ni[..., 0] * nj[..., 1] - ni[..., 1] * nj[..., 0]
  independent = (det.abs() > 1e-6) & finite[:, i] & finite[:, j]
  safe_det = torch.where(independent, det, 1.)
  double = torch.stack(((bi * nj[..., 1] - bj * ni[..., 1]) / safe_det,
                        (ni[..., 0] * bj - nj[..., 0] * bi) / safe_det), -1)
  candidates = torch.cat((torch.zeros_like(v[:, None]), single, double), 1)
  feasible = ((candidates[:, :, None] * n[:, None]).sum(-1) <= b[:, None] + 1e-6).all(-1)
  feasible[:, 1 + n.shape[1]:] &= independent
  distance = ((candidates - v[:, None]) ** 2).sum(-1).masked_fill(~feasible, torch.inf)
  pick = distance.argmin(-1)
  result = original.clone()
  result[rows] = candidates[torch.arange(len(rows), device=v.device), pick]
  return result.reshape(shape)


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
    self.flow_member, self.flow_heading = zeros(torch.bool), zeros()
    # Ordinary turns.
    self.turning, self.turn_radius, self.turn_rate, self.turn_until = zeros(torch.bool), zeros(), zeros(), zeros()
    # Interventions.
    self.mode, self.phase = zeros(torch.long), zeros(torch.long)
    self.side, self.target, self.radius, self.rate = zeros(), zeros(), zeros(), zeros()
    self.started, self.until, self.hold, self.scale = zeros(), zeros(), zeros(), zeros()
    self.assertiveness = zeros()
    self.stubborn, self.stubborn_released = zeros(torch.bool), zeros(torch.bool)
    self.stubborn_patience = zeros()
    # Attitude toward external agents (reset traits) and their state (`set_external`).
    self.robot_perceives, self.robot_decides, self.robot_assertiveness = zeros(torch.bool), zeros(torch.bool), zeros()
    self.external_xz = self.external_vel = None
    self.stall_since = torch.full(shape, -1.0, device=dev)
    self.resolution_turn = zeros(torch.bool)
    self.pass_deadline = zeros()
    self.pass_duration = zeros()
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

  # External agents -----------------------------------------------------------------------
  def set_external(self, xz: torch.Tensor | None, vel: torch.Tensor | None = None) -> None:
    """External agents of every environment, `xz [E, K, 2]` and `vel [E, K, 2]` in the
    library frame; `None` removes them. Positions at `HIDDEN_M` or beyond are absent."""
    if xz is None:
      self.external_xz = self.external_vel = None
      return
    if xz.shape[:1] != (self.E,) or xz.shape[-1] != 2 or xz.ndim != 3:
      raise ValueError(f"external positions are [E={self.E}, K, 2], got {tuple(xz.shape)}")
    self.external_xz = xz.to(self.device, torch.float32)
    self.external_vel = (torch.zeros_like(self.external_xz) if vel is None
                         else vel.to(self.device, torch.float32).expand_as(self.external_xz))

  def _external(self, env_ids):
    """External agents of `env_ids` with each human's view of them, or None."""
    if self.external_xz is None:
      return None
    xz, vel = self.external_xz[env_ids], self.external_vel[env_ids]
    K, M = xz.shape[1], self.M
    shift = torch.cat((torch.zeros(M, device=self.device),
                       torch.full((K,), self.cfg.robot_radius_m - self.cfg.body_radius_m, device=self.device)))
    ones = torch.ones(len(env_ids), M, M, dtype=torch.bool, device=self.device)
    return _External(xz=xz, vel=vel, shift=shift,
                     hidden=~self.robot_perceives[env_ids][..., None].expand(-1, -1, K),
                     decides=torch.cat((ones, self.robot_decides[env_ids][..., None].expand(-1, -1, K)), -1))

  def _relative(self, xz, ext):
    """Positions of each human's neighbor columns relative to it, `[e, M, N, 2]`: the
    humans, then the external agents (moved out of reach where not perceived)."""
    if ext is None:
      return xz[:, None] - xz[:, :, None]
    r = torch.cat((xz, ext.xz), 1)[:, None] - xz[:, :, None]
    away = torch.tensor([HIDDEN_M, 0.0], device=self.device)
    r[:, :, self.M:] = torch.where(ext.hidden[..., None], away, r[:, :, self.M:])
    return r

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
    if c.flow_fraction:
      flow = self._draw(env_ids, counter=-6, slots=2)
      shared_seeds = human_seeds(self.base_seed, env_ids, "crowd_flow", episode=self.episodes[env_ids])
      common = hash_uniform(shared_seeds, torch.zeros_like(env_ids), 1) * (2 * math.pi)
      if c.flow_heading_deg is not None:
        common.fill_(math.radians(c.flow_heading_deg))
      self.flow_member[env_ids] = flow[..., 0] < c.flow_fraction
      self.flow_heading[env_ids] = common + (2 * flow[..., 1] - 1) * math.radians(c.flow_spread_deg)
    u = self._draw(env_ids, counter=-1)
    if c.starts:
      starts = torch.tensor(c.starts, device=dev).expand(e, M, 3)
      xz, heading = starts[..., :2].clone(), torch.deg2rad(starts[..., 2])
    else:
      xz, heading = self._spawn(env_ids), None
    self.assertiveness[env_ids] = _lerp(c.assertiveness_range, u[..., 0])
    self.stubborn[env_ids] = False
    self.stubborn_released[env_ids] = False
    self.stall_since[env_ids] = -1.0
    if c.stubborn_fraction > 0:
      # A separate reset counter in the same encounter stream leaves every legacy
      # slot and subsequent draw untouched, including when this option is off.
      traits = self._draw(env_ids, counter=-4, slots=2)
      self.stubborn[env_ids] = traits[..., 0] < c.stubborn_fraction
      self.stubborn_patience[env_ids] = _lerp(c.stubborn_patience_s, traits[..., 1])
      self.assertiveness[env_ids] = _lerp(c.nonstubborn_assertiveness_range, u[..., 0])
    # Attitude toward external agents: its own reset counter, so no other draw moves.
    robot = self._draw(env_ids, counter=-5, slots=2)
    oblivious = robot[..., 0] < c.robot_oblivious_fraction
    self.robot_perceives[env_ids] = ~oblivious
    self.robot_decides[env_ids] = robot[..., 0] >= c.robot_oblivious_fraction + c.robot_stubborn_fraction
    self.robot_assertiveness[env_ids] = _lerp(c.robot_assertiveness_range, robot[..., 1])
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
    for name in ("turning", "threat", "eligible", "switched", "stopped", "separating", "evasive", "turn_active",
                 "resolution_turn"):
      getattr(self, name)[env_ids] = False
    for name in ("mode", "phase"):
      getattr(self, name)[env_ids] = 0
    for name in ("turn_radius", "turn_rate", "turn_until", "side", "radius", "rate", "started", "until", "hold",
                 "scale", "urgency", "last_t", "pass_deadline", "pass_duration"):
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
  def _advance_maneuvers(self, env_ids, now, xz, heading, threat_before,
                         *, vel=None, pref_speed=None, pref_heading=None, ext=None):
    """Phase changes and completion of ongoing interventions."""
    c = self.cfg
    mode, phase = self.mode[env_ids], self.phase[env_ids]
    until, target = self.until[env_ids], self.target[env_ids]
    aligned = wrap(target - heading).abs() <= math.radians(c.turn_finish_deg)
    turned = ((mode == VEER) | (mode == PIVOT)) & (phase == 0) & (aligned | (now >= until))
    phase = torch.where(turned, 1, phase)
    until = torch.where(turned, now + self.hold[env_ids], until)
    straight_done = (now >= until) & ~turned
    if c.pass_commitment_s > 0:
      self.pass_deadline[env_ids] = torch.where(turned, now + c.pass_commitment_s, self.pass_deadline[env_ids])
      relative = self._relative(xz, ext)
      distance = relative.norm(dim=-1)
      own = _direction(pref_heading) * pref_speed[..., None]
      seen = vel if ext is None else ext.velocities(vel)
      miss, _ = _closest(relative, seen[:, None] - own[:, :, None], c.horizon_s)
      if ext is None:
        required = distance.clamp(max=2 * c.body_radius_m + c.pass_margin_m) - 1e-4
      else:
        miss = miss - ext.shift
        required = (distance - ext.shift).clamp(max=2 * c.body_radius_m + c.pass_margin_m) - 1e-4
      near = (distance > 1e-4) & (distance < c.bubble_radius_m)
      clear = ((miss >= required) | ~near).all(-1)
      if c.pass_commitment_jitter > 0:
        executed_speed = vel.norm(dim=-1)
        parallel = (vel[:, :, None] * vel[:, None]).sum(-1) > (
          math.cos(math.radians(20.)) * executed_speed[:, :, None] * executed_speed[:, None])
        humans = distance[..., :self.M]  # co-motion is between humans
        close = (humans > 1e-4) & (humans < 2 * c.body_radius_m + c.pass_margin_m + .1)
        moving = (executed_speed[:, :, None] > .15) & (executed_speed[:, None] > .15)
        low_progress = (vel * _direction(pref_heading)).sum(-1) < .25 * executed_speed
        co_motion = (parallel & close & moving).any(-1) & low_progress & (phase == 1) & ~clear
        earlier = until - self.hold[env_ids] + self.pass_duration[env_ids]
        self.pass_deadline[env_ids] = torch.where(
          co_motion, torch.minimum(self.pass_deadline[env_ids], earlier), self.pass_deadline[env_ids])
      straight_done = (clear | (now >= self.pass_deadline[env_ids])) & ~turned
    # Walking on toward a wall or obstacle ends a veer, pivot or sidestep early.
    ahead = torch.where((mode == SIDESTEP)[..., None], _right(target) * self.side[env_ids, ..., None] * 0.6,
                        _direction(torch.where(phase == 1, target, heading)))
    walled = ((mode == SIDESTEP) | (mode == VEER) | ((mode == PIVOT) & (phase == 1))) & ~self._inside(xz + ahead)
    over = (((mode == SIDESTEP) & (now >= until))
            | (((mode == VEER) | (mode == PIVOT)) & (phase == 1) & straight_done)
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
    if c.retain_avoidance_heading:
      turn_active = turn_active | veer_out
      turn_target = torch.where(veer_out, target, turn_target)

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
    ext = self._external(env_ids)
    pref_speed, pref_heading, waiting = self._condition(env_ids, now, xz, u)
    if c.flow_fraction:
      direction = _direction(self.flow_heading[env_ids])
      limit = torch.tensor([c.half_x, c.half_z], device=self.device) - c.wall_margin_m - 1.5
      reflect = (xz.abs() > limit) & (direction * xz > 0)
      direction = torch.where(reflect, -direction, direction)
      self.flow_heading[env_ids] = torch.atan2(direction[..., 0], direction[..., 1])
      member = self.flow_member[env_ids]
      pref_heading = torch.where(member, self.flow_heading[env_ids], pref_heading)
      pref_speed = torch.where(member, self.speed[env_ids], pref_speed)
      waiting = waiting & ~member
      self.goal[env_ids] = torch.where(member[..., None], xz + 3 * direction, self.goal[env_ids])
    if not c.reactive:
      # Deliberately bypass every neighbor mechanism, including stop/separation and
      # the graph clearance cost. Ordinary turns and world confinement still apply.
      self.mode[env_ids] = CRUISE
      for field in ("threat", "eligible", "switched", "stopped", "separating", "evasive"):
        getattr(self, field)[env_ids] = False
      self.blocked_since[env_ids] = -1.
      speed, travel, yaw, turn, active, target, orientation, _ = self._intended(
        env_ids, now, xz, heading, pref_speed, pref_heading, waiting, u)
      velocity, confined = self._confine(xz, _direction(travel) * speed[..., None])
      speed = torch.where(confined, velocity.norm(dim=-1), speed)
      travel = torch.where(confined, torch.atan2(velocity[..., 0], velocity[..., 1]), travel)
      self.turn_active[env_ids], self.turn_target[env_ids] = active & ~confined, target
      self.orientation[env_ids] = orientation
      self.robot_perceives[env_ids] = False
      self.robot_decides[env_ids] = False
      self._plan = (torch.where(confined, 0., yaw).reshape(-1), turn.reshape(-1))
      self._neighbors = None
      return speed, travel
    self._advance_maneuvers(env_ids, now, xz, heading, self.threat[env_ids],
                            vel=vel, pref_speed=pref_speed, pref_heading=pref_heading, ext=ext)

    def body(d):
      """Distances `[..., N]` to neighbor columns, as if every body were human-sized."""
      return d if ext is None else d - ext.shift

    columns = (lambda v: v) if ext is None else ext.velocities  # noqa: E731

    def intended():
      """Commands for every human's mode, and the velocity used to predict its motion:
      the travel direction reached `LEAD_S` into a turning arc, kept clear of walls."""
      speed, travel, yaw_rate, max_turn, *rest = self._intended(
        env_ids, now, xz, heading, pref_speed, pref_heading, waiting, u)
      lead = torch.maximum(torch.minimum(yaw_rate * LEAD_S, max_turn), -max_turn)
      velocity, confined = self._confine(xz, _direction(travel + lead) * speed[..., None])
      return velocity, confined, speed, travel, yaw_rate, max_turn, rest

    velocity = intended()[0]

    # Threats in the bubble, judged on intended velocities. Columns j are the other
    # humans, then any external agents (whose intended velocity is their current one).
    eye = (torch.eye(M, dtype=torch.bool, device=self.device) if ext is None
           else torch.eye(M, len(ext.shift), dtype=torch.bool, device=self.device))
    r = self._relative(xz, ext)  # [e, i, j, 2]: j relative to i
    rv = columns(velocity)[:, None] - velocity[:, :, None]
    dist = r.norm(dim=-1).masked_fill(eye, float("inf"))
    contact = 2 * c.body_radius_m
    miss, point = _closest(r, rv, c.horizon_s)
    approaching = (r * rv).sum(-1) < 0
    threats = (dist < c.bubble_radius_m) & approaching & (body(miss) < contact + c.pass_margin_m)
    # Bodies a human may intervene for, and distances to them (robot-stubborn humans
    # never intervene for an external agent; they still stop and separate).
    acting = dist if ext is None else body(dist).masked_fill(~ext.decides, float("inf"))
    if ext is not None:
      threats = threats & ext.decides
    near = torch.where(threats, body(dist), float("inf")).min(-1)
    threat = threats.any(-1)
    urgency = ((c.bubble_radius_m - near.values) / (c.bubble_radius_m - contact)).clamp(0.0, 1.0)
    urgency = torch.where(threat, urgency, 0.0)
    # Actual near-stationarity is distinct from the stop-rule flag: graph selection
    # may hold a nonzero command. Only nearby, pending-goal stalls spend stubborn
    # patience. After that independently drawn finite time the trait is retired for
    # this episode and a pivot is initiated, even at zero stochastic switch rate.
    stalled = (vel.norm(dim=-1) < .15) & (pref_speed >= .15) & ~waiting & (acting.min(-1).values < contact + c.pass_margin_m + .1)
    previous = self.stall_since[env_ids]
    self.stall_since[env_ids] = torch.where(stalled, torch.where(previous >= 0, previous, now), -1.0)
    stall_s = torch.where(stalled, now - self.stall_since[env_ids], 0.0)
    stubborn = self.stubborn[env_ids] & ~self.stubborn_released[env_ids]
    release = stubborn & stalled & (stall_s >= self.stubborn_patience[env_ids])
    self.stubborn_released[env_ids] |= release
    stubborn = stubborn & ~release

    # Switching hazard for cruising humans.
    blocked_s = torch.where(self.blocked_since[env_ids] >= 0, now - self.blocked_since[env_ids], 0.0)
    if ext is None:
      assertive = self.assertiveness[env_ids] * torch.exp(-blocked_s / c.patience_s)
      assertive = torch.where(stubborn, 1.0, assertive)
    else:
      # The closest threat's kind picks the trait: robot assertiveness for an external
      # agent; stubbornness toward humans does not apply to it.
      external = threat & (near.indices >= M)
      base = torch.where(external, self.robot_assertiveness[env_ids], self.assertiveness[env_ids])
      assertive = base * torch.exp(-blocked_s / c.patience_s)
      assertive = torch.where(stubborn & ~external, 1.0, assertive)
    rate = c.switch_rate_hz * (1 - assertive) * urgency.pow(c.urgency_exponent)
    eligible = (self.mode[env_ids] == CRUISE) & ~waiting & threat
    switch = eligible & (u[..., S_HAZARD] < 1 - torch.exp(-rate * dt))
    recovery = torch.zeros_like(switch)
    if c.stall_recovery_s > 0:
      other_still = columns(vel).norm(dim=-1)[:, None] < .15
      mutual = ((acting < contact + c.pass_margin_m + .1) & other_still).any(-1)
      mode, phase = self.mode[env_ids], self.phase[env_ids]
      turning_phase = ((mode == VEER) | (mode == PIVOT)) & (phase == 0)
      straight_ready = (phase != 1) | (now - (self.until[env_ids] - self.hold[env_ids]) >= c.stall_recovery_s)
      recovery = (stalled & mutual & ~stubborn & ~turning_phase & straight_ready
                  & (stall_s >= c.stall_recovery_s) & (now - self.started[env_ids] >= c.stall_recovery_s))
    forced = release | recovery
    switch = switch | forced
    if switch.any():
      selected_neighbor = torch.where(forced, acting.argmin(-1), near.indices)
      self._begin(env_ids, switch, now, xz, heading, urgency, selected_neighbor, r, point, u,
                  pref_heading=pref_heading, vel=vel, force_pivot=forced,
                  others=None if ext is None else ext.velocities(vel), shift=None if ext is None else ext.shift)
    velocity, confined, speed, travel, yaw_rate, max_turn, rest = intended()
    turn_active, turn_target, orientation, want = rest
    # Observations only: do not consume randomness or feed these back into control.
    self._decision_trace = {
      "encounter_pair_threat": threats.clone(),
      "encounter_effective_assertiveness": assertive.clone(),
      "encounter_blocked_s": blocked_s.clone(),
      "encounter_switch_probability": (1 - torch.exp(-rate * dt)).clone(),
      "encounter_preferred_velocity": (_direction(pref_heading) * pref_speed[..., None]).clone(),
      "encounter_intended_velocity": velocity.clone(),
      "encounter_executed_velocity": vel.clone(),
      "encounter_target": self.target[env_ids].clone(),
      "encounter_until": self.until[env_ids].clone(),
      "encounter_stall_s": stall_s.clone(),
      "encounter_stubborn_active": stubborn.clone(),
      "encounter_stubborn_release": release.clone(),
      "encounter_stall_recovery": recovery.clone(),
    }
    self._decision_trace_envs = env_ids.clone()

    # Stop rule. Others are predicted at their executed velocities. Motion conflicts
    # when the intended or the current (still executing) velocity would bring the bodies
    # within the clearance and standing would keep them further apart. The intended
    # velocity loses its component toward each conflicting neighbor (sliding past is
    # allowed); a human stops if a conflict remains or little motion is left.
    threshold = contact + c.stop_clearance_m
    others = columns(vel)[:, None].expand_as(r)  # [e, i, j, 2]: j's executed velocity
    still_gap, _ = _closest(r, others, c.stop_horizon_s)

    def conflicts(own):
      gap, _ = _closest(r, others - own[:, :, None], c.stop_horizon_s)
      return ((body(gap) < threshold) & (gap < still_gap - 1e-3)).masked_fill(eye, False)

    unit = r / r.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    if c.approach_time_s > 0:
      # Approach each neighbor no faster than the remaining gap allows; passing is free.
      allowed = ((body(r.norm(dim=-1)) - threshold).clamp(min=0.0) / c.approach_time_s).masked_fill(eye, float("inf"))
      excess = ((velocity[:, :, None] * unit).sum(-1) - allowed).clamp(min=0.0)
      slowed = (excess > 0).any(-1)
      velocity = (project_velocity(velocity, unit, allowed) if c.project_safe_velocity
                  else velocity - (excess[..., None] * unit).sum(2))
      speed = torch.where(slowed, velocity.norm(dim=-1), speed)
      travel = torch.where(slowed, torch.atan2(velocity[..., 0], velocity[..., 1]), travel)
      yaw_rate = torch.where(slowed, 0.0, yaw_rate)
    conflict = conflicts(velocity) | conflicts(vel)
    closing = ((velocity[:, :, None] * unit).sum(-1).clamp(min=0.0) * conflict)[..., None]
    slid = (project_velocity(velocity, unit, torch.where(conflict, 0., torch.inf)) if c.project_safe_velocity
            else velocity - (closing * unit).sum(2))
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
    if c.blocked_turn_first:
      veer_out = (self.mode[env_ids] == VEER) & (self.phase[env_ids] == 0)
      constrained = held | (slowed if c.approach_time_s > 0 else torch.zeros_like(held))
      resolve = veer_out & (self.resolution_turn[env_ids] | (constrained & (vel.norm(dim=-1) < 0.15)))
      self.resolution_turn[env_ids] = resolve
      # The approach limiter zeros the arc yaw rate. Keep the body target explicit
      # while asking for zero translation, until the committed veer reaches phase 1.
      velocity = torch.where(resolve[..., None], 0.0, velocity)
      speed = torch.where(resolve, 0.0, speed)
      yaw_rate = torch.where(resolve, 0.0, yaw_rate)
      turn_active = turn_active | resolve
      turn_target = torch.where(resolve, self.target[env_ids], turn_target)
    # Bodies already within the clearance step apart, facing unchanged.
    overlap = (threshold - body(r.norm(dim=-1))).clamp(min=0.0).masked_fill(eye, 0.0)  # [e, i, j]
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
    braked = None
    if c.anticipate_braking:
      executed_speed = vel.norm(dim=-1)
      ratio = (speed / executed_speed.clamp(min=1e-6)).clamp(max=1.)
      reversing = ((_direction(travel) * vel).sum(-1) < 0) & (speed > .05)
      ratio = torch.where(stop | reversing, 0., ratio)
      braked = vel * ratio[..., None]
    self._neighbors = self._nearest(xz, vel, dist, threshold, braked=braked, ext=ext, r=r)
    self._decision_trace.update(
      encounter_conflict=conflict.clone(), encounter_held=held.clone(),
      encounter_command_velocity=(_direction(travel) * speed[..., None]).clone(),
      encounter_approach_slowed=slowed.clone() if c.approach_time_s > 0 else torch.zeros_like(stop),
    )
    if ext is not None:
      self._decision_trace.update(encounter_external_threat=threats[..., M:].any(-1).clone(),
                                  encounter_external_conflict=conflict[..., M:].any(-1).clone())
    return speed, travel

  def _nearest(self, xz, vel, dist, threshold, *, braked=None, ext=None, r=None):
    """`(positions, executed velocities, clearance distance)` of each human's nearest
    neighbors, wall mirrors and external agents, flattened to rows, for the graph's
    clearance cost. With external agents the clearance is per neighbor."""
    e, M = xz.shape[:2]
    n = min(NEIGHBORS, M - 1)
    idx = dist[..., :M].topk(n, dim=-1, largest=False).indices  # [e, M, n]; self is at infinity
    pick = lambda x: x[:, None].expand(e, M, M, 2).gather(2, idx[..., None].expand(e, M, n, 2))  # noqa: E731
    # Stationary mirror images behind the nearest x and z walls: a mirror is as close
    # as twice the gap to its wall, so the same clearance keeps bodies off the walls.
    half = torch.tensor([self.cfg.half_x, self.cfg.half_z], device=self.device)
    side = torch.where(xz >= 0, 1.0, -1.0)
    mirror_x = torch.stack((2 * side[..., 0] * half[0] - xz[..., 0], xz[..., 1]), -1)
    mirror_z = torch.stack((xz[..., 0], 2 * side[..., 1] * half[1] - xz[..., 1]), -1)
    points = torch.cat((pick(xz), mirror_x[:, :, None], mirror_z[:, :, None]), 2)
    speeds = torch.cat((pick(vel), torch.zeros(e, M, 2, 2, device=self.device)), 2)
    if braked is not None:
      braking = (pick(braked) - pick(vel)).norm(dim=-1) > 1e-3
      # Keep executed motion too: an instant-stop-only forecast would miss someone
      # still approaching. Unchanged forecasts are padded away, not counted twice.
      extra = torch.where(braking[..., None], pick(xz), 1e6)
      points = torch.cat((points, extra), 2)
      speeds = torch.cat((speeds, pick(braked)), 2)
    limit = torch.full((e * M,), threshold, device=self.device)
    if ext is not None:
      K = ext.xz.shape[1]
      points = torch.cat((points, xz[:, :, None] + r[:, :, M:]), 2)  # unperceived agents sit far away
      speeds = torch.cat((speeds, ext.vel[:, None].expand(e, M, K, 2)), 2)
      limit = torch.cat((limit[:, None].expand(e * M, points.shape[2] - K),
                         (threshold + ext.shift[M:]).expand(e * M, K)), 1)
    return points.reshape(e * M, -1, 2), speeds.reshape(e * M, -1, 2), limit

  def _minimal_heading(self, xz, pref_heading, speed, relative, other_velocity, side, head_on, margin,
                       shift=None):
    """One-degree velocity-obstacle search against all supplied nearby bodies.

    Inside the passing margin, admissibility means no further closing. If no
    direction clears, choose maximum predicted clearance and keep the stop rule.
    Head-on keeps each human's independently sampled passing side; other cases
    prefer the shorter detour from its goal. No pair precedence is introduced.
    """
    c = self.cfg
    if relative.ndim == xz.ndim:
      relative, other_velocity = relative[..., None, :], other_velocity[..., None, :]
    angles = torch.deg2rad(torch.arange(0, 181, device=self.device, dtype=xz.dtype))
    offsets = torch.stack((-angles, angles), 0)  # [side, angle]
    targets = pref_heading[..., None, None] + offsets
    own = _direction(targets) * speed[..., None, None, None]
    rel = relative[..., None, None, :, :]
    rv = other_velocity[..., None, None, :, :] - own[..., None, :]
    miss, _ = _closest(rel, rv, c.horizon_s)
    distance = relative.norm(dim=-1)
    clearance = 2 * c.body_radius_m + c.pass_margin_m + margin
    if shift is None:
      required = torch.minimum(clearance[..., None], distance) - 1e-4
    else:  # bodies of other sizes (external agents) count as human-sized
      miss = miss - shift
      required = torch.minimum(clearance[..., None], distance - shift) - 1e-4
    nearby = (distance > 1e-4) & (distance < c.bubble_radius_m)
    safe_margin = (miss - required[..., None, None, :]).masked_fill(~nearby[..., None, None, :], torch.inf).amin(-1)
    nearest = distance.masked_fill(~nearby, torch.inf).amin(-1).clamp(.6, 1.5)
    endpoint = xz[..., None, None, :] + _direction(targets) * nearest[..., None, None, None]
    feasible = self._inside(endpoint)
    clear = (safe_margin >= 0) & feasible
    allowed_side = (~head_on[..., None]) | (torch.tensor([-1., 1.], device=self.device) == side[..., None])
    clear &= allowed_side[..., None]
    # Primary key: deviation from goal. Stochastic side resolves equal detours.
    cost = angles[None] + (torch.tensor([-1., 1.], device=self.device) != side[..., None])[..., :, None] * 1e-5
    cost = cost.expand_as(safe_margin).masked_fill(~clear, float("inf"))
    flat = cost.flatten(-2)
    best = flat.argmin(-1)
    fallback = safe_margin.masked_fill(~feasible | ~allowed_side[..., None], -torch.inf).flatten(-2).argmax(-1)
    best = torch.where(clear.flatten(-2).any(-1), best, fallback)
    target = targets.flatten(-2).gather(-1, best[..., None])[..., 0]
    chosen_side = torch.where(wrap(target - pref_heading) >= 0, 1., -1.)
    return target, chosen_side

  def _begin(self, env_ids, switch, now, xz, heading, urgency, nearest, r, point, u,
             *, pref_heading=None, vel=None, force_pivot=None, others=None, shift=None):
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
    if force_pivot is not None:
      strategy = torch.where(force_pivot, PIVOT, strategy)
    angle = torch.where(strategy == PIVOT, _lerp(c.pivot_angle_deg, u[..., S_ANGLE]),
                        _lerp(c.veer_angle_deg, u[..., S_ANGLE]))
    geometric_target = None
    step_distance = _lerp(c.sidestep_m, u[..., S_SIDESTEP])
    if c.minimal_avoidance:
      other_velocity = (vel if others is None else others)[:, None].expand_as(r)
      geometric_target = heading.clone()
      chosen_target, chosen_side = self._minimal_heading(
        xz[switch], pref_heading[switch], self.speed[env_ids][switch], r[switch], other_velocity[switch],
        side[switch], head_on[switch], _lerp(c.avoidance_margin_m, u[..., S_ANGLE])[switch], shift=shift)
      geometric_target[switch] = chosen_target
      side[switch] = chosen_side
      angle = torch.rad2deg(wrap(geometric_target - heading).abs())
      # Closest-point lateral deficit, plus the independently sampled small margin.
      deficit = 2 * c.body_radius_m + c.pass_margin_m - lateral.abs()
      if shift is not None:
        deficit = deficit + shift[nearest]
      step_distance = (deficit + _lerp(c.avoidance_margin_m, u[..., S_SIDESTEP])).clamp(.1, c.sidestep_m[1])
    radius = _lerp(c.veer_radius_m, urgency) * (0.8 + 0.4 * u[..., S_RADIUS_JITTER])
    sidestep_s = step_distance / c.sidestep_speed_mps
    pivot_rate = torch.deg2rad(_lerp(c.inplace_rate_deg_s, u[..., S_RATE]))
    turn_s = torch.where(strategy == PIVOT, torch.deg2rad(angle) / pivot_rate,
                         torch.deg2rad(angle) * radius / self.speed[env_ids].clamp(min=0.3))
    until = torch.where(strategy == SIDESTEP, now + sidestep_s,
                        torch.where(strategy == YIELD, now + c.yield_max_s, now + turn_s + FINISH_TIMEOUT_S))
    target = torch.where(strategy == SIDESTEP, heading, heading + side * torch.deg2rad(angle))
    if geometric_target is not None:
      target = torch.where((strategy == VEER) | (strategy == PIVOT), geometric_target, target)
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
    duration = c.pass_commitment_s * (1 - c.pass_commitment_jitter * (1 - u[..., S_HOLD]))
    self.pass_duration[env_ids] = sel(duration, self.pass_duration[env_ids])
    self.scale[env_ids] = sel(_lerp(c.yield_speed_scale, u[..., S_YIELD]), self.scale[env_ids])
    self.turning[env_ids] &= ~switch
    self.resolution_turn[env_ids] &= ~switch

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
    rows = (getattr(self, "_decision_trace_envs", torch.empty(0, device=self.device)) == env).nonzero()
    decision = ({key: value[int(rows[0, 0])].clone() for key, value in self._decision_trace.items()}
                if len(rows) else {})
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
      "encounter_stubborn": self.stubborn[env].clone(),
      "encounter_side": self.side[env].clone(),
      "encounter_resolution_turn": self.resolution_turn[env].clone(),
      "encounter_ordinary_turn": self.turning[env].clone(),
      "encounter_ordinary_turn_radius": self.turn_radius[env].clone(),
      "steering_evasive": self.evasive[env].clone(),
      "turn_active": self.turn_active[env].clone(),
      "turn_target": self.turn_target[env].clone(),
      **decision,
    }
