"""Behavior designs: build either crowd command source from a resolved behavior config.

- `plaza` (the original, and the default when a config names no design): random
  goals and waits (`GoalIntent`, optionally committed turns) with continuous
  time-to-closest-approach avoidance (`LocalSteering`). Section `steering`
  (`PlazaSteeringCfg`).
- `encounter`: waypoint or velocity conditioning, a collision bubble and committed
  stochastic interventions with a stop rule (`EncounterBehavior`). Section
  `encounter` (`EncounterCfg`).

Both designs drive the same `CrowdController` and graph, so every consumer
(evaluator, mjlab `GraphCrowd`, demos) can run either one.
"""

from __future__ import annotations

from crowdmotion.behavior.encounter import EncounterBehavior, EncounterCfg
from crowdmotion.behavior.planner import plan_path
from crowdmotion.behavior.plaza import PlazaBehavior, PlazaSteeringCfg

DESIGNS = ("plaza", "encounter")
SECTIONS = {"plaza": "steering", "encounter": "encounter"}


def design_of(cfg: dict) -> str:
  """The design a resolved or raw behavior config selects (default `plaza`)."""
  design = cfg.get("design", "plaza")
  if design not in DESIGNS:
    raise ValueError(f"design must be one of {DESIGNS}, got {design!r}")
  return design


def scene_section(cfg: dict) -> dict:
  """The design's own section; both carry the scene (half_x, half_z, obstacles, arrive_m)."""
  return cfg[SECTIONS[design_of(cfg)]]


def context_radii(cfg: dict) -> tuple[float, float]:
  """`(close, clear)` root-gap radii for diagnostics: contact/personal radii in the plaza
  design; body contact plus passing margin, and the bubble, in the encounter design."""
  section = scene_section(cfg)
  if design_of(cfg) == "encounter":
    return 2 * section["body_radius_m"] + section["pass_margin_m"], section["bubble_radius_m"]
  return section["contact_m"], section["personal_m"]


def make_behavior(design: str, section, envs: int, humans: int, seed: int, device):
  """`(command source, planner)` for `envs` environments of `humans` humans.

  `section`: the design's config object (`PlazaSteeringCfg` or `EncounterCfg`); it is
  used by reference, so live edits to it take effect at the next command.
  """
  if design == "encounter":
    if not isinstance(section, EncounterCfg):
      raise TypeError("the encounter design needs an EncounterCfg")
    behavior = EncounterBehavior(envs, humans, section, seed, device)
    return behavior, behavior.plan
  if design == "plaza":
    if not isinstance(section, PlazaSteeringCfg):
      raise TypeError("the plaza design needs a PlazaSteeringCfg")
    return PlazaBehavior(envs, humans, section, seed, device), plan_path
  raise ValueError(f"design must be one of {DESIGNS}, got {design!r}")


def section_cfg(cfg: dict):
  """The typed section of a resolved config."""
  if design_of(cfg) == "encounter":
    return EncounterCfg.from_dict(cfg["encounter"])
  return PlazaSteeringCfg(**cfg["steering"])
