"""What `crowd play traversal`'s live Viser viewer offers, and what a switch needs.

The viewer (`live_viewer`) plays one traversal policy in one crowd and switches either
while the Viser server stays up. This module holds its parts that need no simulator:

- **Policies** (`PolicyRun`, `discover_runs`): `crowd train traversal` run directories and
  their own checkpoints. A policy plays in the env its `run.json` records
  (`play.run_policy`), as `crowd eval traversal` plays it: its task (velocity goals or
  persistent navigation goals), observation groups, planner rate and graph.
- **Crowds** (`crowd_choices`, `resolve_crowd`): `trained`, the policy's own crowd, and the
  fixed evaluation crowds of the navigation ablations, built by the same code as the
  campaign's (`ablate_navigation.navigation_conditions` and `condition_behavior`) or read
  from a campaign directory's frozen `conditions`. A crowd sets the number of humans,
  their behavior config and their perception of the robot (`aware` or `blind`), as the
  evaluator's `--humans`, `--behavior` and crowd do.
- **Refusals** (`incompatibility`): a privileged teacher reads its `min(humans, 6)` nearest
  humans, so it cannot play in a crowd that changes that number. A checkpoint whose input
  width differs from the env's observation (`checkpoint_inputs`) is refused when loaded.
- **Switches** (`plan_switch`): a new checkpoint, or a policy whose env is the same, loads
  in place. A crowd that differs only in encounter settings a running crowd reads afresh
  (`LIVE_FIELDS`: reactivity, flow, the robot traits and the other live controls) is a
  queued edit (`LiveCrowdTuning.submit_encounter`). Anything else (number of humans, robot
  perception, graph settings, task, planner rate, observation group) rebuilds the env in
  the same process. Every switch restarts the episodes (rewound, as the evaluator does for
  each checkpoint) and the counters.
- **Panel** (`panel_rows`): the evaluator's `TraversalMetrics` over the episodes played
  since the last switch.
"""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from crowdmotion.metrics.traversal import EPISODE_ENDS
from crowdmotion.sim.mjlab.ablate_navigation import BASE_BEHAVIOR, condition_behavior, navigation_conditions
from crowdmotion.sim.mjlab.evaluate import run_checkpoints
from crowdmotion.sim.mjlab.play import label, run_policy
from crowdmotion.sim.mjlab.traversal import load_behavior, make_traversal_env_cfg
from crowdmotion.sim.mjlab.tuning import ENCOUNTER_CONTROLS

RUN_GLOB = "*_train_traversal_*"
TRAINED = "trained"
# The teacher's `crowd_nearest` reads min(humans, this) humans (make_traversal_env_cfg).
TEACHER_NEIGHBORS = inspect.signature(make_traversal_env_cfg).parameters["teacher_neighbors"].default
NEAREST_VALUES = 5  # per neighbor: position, velocity and clearance (`traversal.crowd_nearest`)
# NavigationCfg fields that only shape rewards or PPO's credit assignment; navigation
# policies that differ in nothing else play in the same env. Other fields (deadline, goal
# distances, speed, and any added later) change the task, so they rebuild it.
NAV_REWARD_FIELDS = frozenset({"reward_mode", "gamma_per_s", "gae_lambda_per_s", "rollout_s", "success_reward",
                               "time_bonus", "failure_reward", "potential_weight", "action_l2", "action_rate"})
# Encounter fields a running crowd reads afresh (at its next decision, or at its next
# reset, which follows every switch): the live controls, reactivity and flow.
LIVE_FIELDS = frozenset({c.field for c in ENCOUNTER_CONTROLS}
                        | {"reactive", "flow_fraction", "flow_heading_deg", "flow_spread_deg"})


# Policies -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class PolicyRun:
  """A `crowd train traversal` run as a playable policy."""
  path: Path
  trained: str  # the recorded policy: lidar, teacher or student
  plays_as: str  # lidar or teacher (a distilled student plays as the LiDAR policy)
  env: dict  # env options from run.json (`train.env_options`)
  checkpoints: tuple[Path, ...]  # its own model_<i>.pt by iteration

  @property
  def name(self) -> str:
    return label(self.path)

  @property
  def navigation(self) -> bool:
    return self.env.get("navigation") is not None

  def describe(self) -> str:
    rate = self.env.get("planner_rate") or 0.0
    who = self.plays_as if self.trained == self.plays_as else f"{self.trained} as {self.plays_as}"
    crowd = "aware" if self.env.get("perceive_robot") else "blind"
    return " · ".join((who, f"planner {rate:g} Hz" if rate else "50 Hz joints",
                       "navigation goals" if self.navigation else "velocity goals",
                       f"trained with {self.env['humans']} humans, {crowd}", f"{len(self.checkpoints)} checkpoints"))


def load_run(path: str | Path) -> PolicyRun:
  """A run as a policy. Its graph and rotations paths are resolved, so that one file named
  through two paths (a symlinked home) is one env and one cached graph."""
  path = Path(path).resolve()
  trained, plays_as, env = run_policy(path)
  env = {**env, **{k: str(Path(env[k]).resolve()) for k in ("graph", "rotations") if env.get(k)}}
  return PolicyRun(path, trained, plays_as, env, tuple(run_checkpoints(path)))


def discover_runs(root: str | Path, pattern: str = RUN_GLOB) -> list[PolicyRun]:
  """Playable runs under `root`, newest first: those with a `run.json` naming a known
  policy and at least one checkpoint of their own."""
  found = []
  for path in sorted(Path(root).glob(pattern), reverse=True):
    if not (path / "run.json").is_file():
      continue
    try:
      found.append(load_run(path))
    except (FileNotFoundError, ValueError, KeyError, TypeError, json.JSONDecodeError):
      continue
  return found


def run_labels(runs: list[PolicyRun]) -> dict[str, PolicyRun]:
  """Unique dropdown labels: the run's name, with its directory's time stamp when two share it."""
  counts: dict[str, int] = {}
  for run in runs:
    counts[run.name] = counts.get(run.name, 0) + 1
  return {(run.name if counts[run.name] == 1 else f"{run.name} ({run.path.name.split('_', 1)[0]})"): run
          for run in runs}


def resolve_checkpoint(policy: PolicyRun, name: str | Path | None) -> Path:
  """`name` among the run's checkpoints (a file name such as `model_95.pt`), a path, or the
  newest for None or an empty name."""
  if not name:
    return policy.checkpoints[-1]
  for ckpt in policy.checkpoints:
    if ckpt.name == str(name):
      return ckpt
  if Path(name).is_file():
    return Path(name)
  raise FileNotFoundError(f"{policy.name} has no checkpoint {name}")


def checkpoint_inputs(path: str | Path) -> int | None:
  """The actor's input width in a checkpoint (its first layer), or None if it has none."""
  import torch

  state = torch.load(path, map_location="cpu", weights_only=False)
  actor = state.get("student_state_dict") or state.get("actor_state_dict") or {}
  weight = actor.get("mlp.0.weight")
  return None if weight is None else int(weight.shape[1])


# Crowds -----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CrowdChoice:
  """A crowd on offer. Unset fields take the policy's own (the `trained` crowd)."""
  name: str
  description: str
  humans: int | None = None
  perception: str | None = None  # aware or blind
  behavior: dict | str | None = field(default=None, compare=False)  # a config, a path, or the policy's


def _describe(condition: dict, behavior: dict) -> str:
  enc = behavior.get("encounter", {})
  kind = "reactive" if condition["reactive"] else "independent walkers (no responses; humans may overlap)"
  flow = f", {condition['flow']:.0%} in one coherent flow" if condition["flow"] else ""
  if condition["crowd"] == "aware":
    oblivious, stubborn = enc.get("robot_oblivious_fraction", 0.0), enc.get("robot_stubborn_fraction", 0.0)
    robot = (f"robot-aware: {oblivious:.0%} oblivious, {stubborn:.0%} stubborn, "
             f"{1 - oblivious - stubborn:.0%} cooperative toward it")
  else:
    robot = "robot-blind"
  return f"{condition['humans']} humans, {kind}{flow}, {robot}"


def crowd_choices(campaign: str | Path | None = None) -> list[CrowdChoice]:
  """`trained`, then the navigation campaign's evaluation crowds: built here by the
  campaign's code, or the frozen `conditions` of a campaign directory (its `run.json`)."""
  out = [CrowdChoice(TRAINED, "the policy's own training crowd (its run.json)")]
  if campaign is not None:
    record = json.loads((Path(campaign) / "run.json").read_text())
    for c in record["conditions"]:
      behavior = json.loads(Path(c["behavior"]).read_text())
      out.append(CrowdChoice(c["name"], _describe(c, behavior), c["humans"], c["crowd"], c["behavior"]))
    return out
  base = json.loads(BASE_BEHAVIOR.read_text())
  for c in navigation_conditions():
    behavior = condition_behavior(base, c)
    out.append(CrowdChoice(c["name"], _describe(c, behavior), c["humans"], c["crowd"], behavior))
  return out


@dataclass(frozen=True)
class ResolvedCrowd:
  """A crowd for one policy: what the env is built with."""
  name: str
  humans: int
  perception: str  # aware or blind
  behavior: str  # behavior config path
  overrides: dict = field(compare=False)  # its graph overrides
  encounter: object = field(compare=False)  # its EncounterCfg as the env uses it (obstacles dropped)


def resolve_crowd(choice: CrowdChoice, policy: PolicyRun, scratch: Path) -> ResolvedCrowd:
  """`choice` for `policy`; a behavior given as a config is written to `scratch`."""
  if choice.behavior is None:
    path = Path(policy.env["behavior"])
  elif isinstance(choice.behavior, dict):
    path = Path(scratch) / f"{choice.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(choice.behavior, indent=2))
  else:
    path = Path(choice.behavior)
  overrides, encounter = load_behavior(path)
  perception = choice.perception or ("aware" if policy.env.get("perceive_robot") else "blind")
  if perception not in ("aware", "blind"):
    raise ValueError(f"crowd perception must be aware or blind, got {perception!r}")
  return ResolvedCrowd(choice.name, int(choice.humans or policy.env["humans"]), perception, str(path),
                       overrides, replace(encounter, obstacles=()))


def incompatibility(policy: PolicyRun, humans: int) -> str | None:
  """Why `policy` cannot play among `humans` humans, or None."""
  if policy.plays_as != "teacher":
    return None
  trained, now = min(policy.env["humans"], TEACHER_NEIGHBORS), min(humans, TEACHER_NEIGHBORS)
  if trained == now:
    return None
  width = NEAREST_VALUES * abs(trained - now)
  fits = f"at least {TEACHER_NEIGHBORS}" if trained == TEACHER_NEIGHBORS else f"exactly {trained}"
  return (f"{policy.name} is a privileged teacher: it observes its {trained} nearest humans, but a crowd of "
          f"{humans} gives it {now} ({width} observation values {'fewer' if now < trained else 'more'}). "
          f"It plays among {fits} humans; a LiDAR policy (a distilled student) plays in any crowd.")


# Switches -----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Target:
  """What the viewer plays: a policy's checkpoint in a crowd."""
  policy: PolicyRun
  checkpoint: Path
  crowd: ResolvedCrowd

  def describe(self) -> str:
    return f"{self.policy.name} {self.checkpoint.stem} in {self.crowd.name}"


def env_signature(policy: PolicyRun, crowd: ResolvedCrowd) -> dict:
  """Everything the built env and runner depend on, by a readable name."""
  e = policy.env
  nav = e.get("navigation")
  return {
    "graph": str(e.get("graph")), "rotations": str(e.get("rotations")),
    "policy observations": policy.plays_as,
    "forward-first command": bool(e.get("forward_first")), "teacher LiDAR": bool(e.get("teacher_lidar")),
    "no-avoidance baseline": bool(e.get("no_avoidance")), "planner rate": float(e.get("planner_rate") or 0.0),
    "task": "velocity goals" if nav is None else {k: v for k, v in nav.items() if k not in NAV_REWARD_FIELDS},
    "humans": crowd.humans, "robot perception": crowd.perception, "graph settings": dict(crowd.overrides),
    "crowd settings": {k: v for k, v in asdict(crowd.encounter).items() if k not in LIVE_FIELDS},
  }


def _change(name: str, before, after) -> str:
  if isinstance(before, dict) and isinstance(after, dict):
    keys = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    return f"{name} ({', '.join(keys)})"
  short = lambda v: "navigation goals" if isinstance(v, dict) else v  # noqa: E731
  return f"{name} {short(before)} → {short(after)}"


@dataclass(frozen=True)
class Switch:
  rebuild: bool  # the env must be built again
  load: bool  # another checkpoint loads into the runner
  fields: dict  # encounter fields to edit on the running crowd
  changes: tuple[str, ...]  # what changes, for the user


def plan_switch(current: Target, target: Target) -> Switch:
  before, after = env_signature(current.policy, current.crowd), env_signature(target.policy, target.crowd)
  rebuild = tuple(_change(k, before[k], after[k]) for k in before if before[k] != after[k])
  policy = (Path(target.policy.path).resolve(), target.checkpoint.resolve()) != (
    Path(current.policy.path).resolve(), current.checkpoint.resolve())
  was, now = asdict(current.crowd.encounter), asdict(target.crowd.encounter)
  fields = {k: getattr(target.crowd.encounter, k) for k in sorted(LIVE_FIELDS) if was[k] != now[k]}
  changes = []
  if policy:
    changes.append(f"policy {current.policy.name} {current.checkpoint.stem} → "
                   f"{target.policy.name} {target.checkpoint.stem}")
  if current.crowd.name != target.crowd.name:
    changes.append(f"crowd {current.crowd.name} → {target.crowd.name}")
  if rebuild:
    return Switch(True, policy, {}, tuple(changes) + rebuild)
  changes += [f"{k} {was[k]} → {now[k]}" for k in fields]
  return Switch(False, policy, fields, tuple(changes))


# Panel ---------------------------------------------------------------------------------------------
def _num(value, fmt: str = ".2f") -> str:
  return "–" if value is None or (isinstance(value, float) and not math.isfinite(value)) else format(value, fmt)


def outcome_counts(tables: dict) -> dict[str, int]:
  """Ended episodes by cause (`EPISODE_ENDS`, without the evaluator's censoring end)."""
  ends = tables["episodes"]["end"]
  return {end: int((ends == i).sum()) for i, end in enumerate(EPISODE_ENDS) if end != "eval_end"}


def panel_rows(metrics: dict, tables: dict, navigation: bool, seconds: float, envs: int,
               realtime: float | None) -> list[tuple[str, str]]:
  """The live panel: `(label, value)` rows from a `TraversalMetrics.snapshot()`."""
  robot_min = metrics.get("exposure/robot_min") or 0.0
  ends = outcome_counts(tables)
  rows = [("Simulated", f"{seconds:.1f} s × {envs} env{'s' if envs > 1 else ''} = {robot_min:.2f} robot-min"),
          ("Real-time factor", "–" if not realtime else f"{realtime:.2f}×"),
          ("Collisions", f"{metrics.get('collision/count', 0)} ({_num(metrics.get('collision/per_robot_min'))} "
                         "/ robot-min)"),
          ("Falls", f"{metrics.get('fall/count', 0)} ({_num(metrics.get('fall/per_robot_min'))} / robot-min)")]
  if navigation:
    done = metrics.get("goal/completed_episodes", 0)
    rows += [("Goals reached", f"{ends['success']} of {done} episodes "
                               f"({_num(metrics.get('goal/success_fraction'), '.0%')})"),
             ("Goals / robot-min", _num(metrics.get("goal/success_per_robot_min"))),
             ("Outcomes", " · ".join(f"{k.replace('_', '-')} {ends[k]}"
                                     for k in ("success", "collision", "fall", "deadline")))]
  else:
    rows += [("Goals reached", f"{metrics.get('command/arrivals', 0)} of {metrics.get('command/legs', 0)} legs "
                               f"({_num(metrics.get('command/goals_per_moving_min'))} / moving-min)"),
             ("Episode ends", " · ".join(f"{k.replace('_', '-')} {ends[k]}"
                                         for k in ("collision", "fall", "time_out")))]
  return rows


def panel_html(title: str, rows: list[tuple[str, str]]) -> str:
  cells = "".join(f"<tr><td style='padding-right:0.8em;white-space:nowrap'><b>{k}</b></td><td>{v}</td></tr>"
                  for k, v in rows)
  return (f"<div style='font-size:0.85em;line-height:1.3;padding:0 1em 0.5em 1em'>{title}"
          f"<table style='margin-top:0.3em'>{cells}</table></div>")
