"""Viser crowd controls, queued for application on the simulation thread.

Only mutable runtime settings are exposed. Saved graph tables and their baked
configuration stay unchanged; buffered motion finishes before the next selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from queue import Empty, SimpleQueue


@dataclass(frozen=True)
class Control:
  owner: str
  field: str
  label: str
  minimum: float = 0.
  maximum: float = 1.
  step: float = .01
  boolean: bool = False
  index: int | None = None  # one element of a tuple field

  @property
  def key(self) -> tuple:
    return _key(self.owner, self.field, self.index)


def _key(owner: str, field: str, index: int | None) -> tuple:
  return (owner, field) if index is None else (owner, field, index)


# Graph fields that change which edges are admissible or which starts a style can draw.
MASK_FIELDS = ("vertical_mask", "posture_enabled", "twist_mask", "twist_max_foot_yaw_deg")


CONTROLS = (
  Control("graph", "temperature", "Sampling temperature", 0, .2, .001),
  Control("graph", "pose_weight", "Pose continuity weight", 0, 10, .05),
  Control("graph", "end_vel_weight", "End velocity weight", 0, 5, .05),
  Control("graph", "start_vel_weight", "Executed entry velocity weight", 0, 5, .05),
  Control("graph", "facing_weight", "Forward preference / sidestep rarity", 0, 5, .05),
  Control("graph", "heading_weight", "Body heading weight", 0, 5, .05),
  Control("graph", "heading_deadband_deg", "Heading deadband (deg)", 0, 90, 1),
  Control("steering", "forward_steering", "Forward steering", boolean=True),
  Control("steering", "forward_lead_deg", "Travel lead over body (deg)", 1, 90, 1),
  Control("steering", "forward_speed_floor", "Forward speed floor", .01, 1, .01),
  Control("steering", "relax_orientation_for_avoidance", "Release orientation costs for avoidance", boolean=True),
  Control("steering", "turn_before_walk", "Retain turn before walking", boolean=True),
  Control("steering", "redirect_start_deg", "Start body turn (deg)", 2, 180, 1),
  Control("steering", "redirect_finish_deg", "Finish body turn (deg)", 1, 89, 1),
  Control("steering", "redirect_speed_mps", "Requested speed during turn (m/s)", 0, 2, .05),
  Control("graph", "turn_weight", "Retained turn weight", 0, 10, .1),
  Control("graph", "turn_preserve_yaw", "Preserve native yaw during turn", boolean=True),
  Control("graph", "posture_enabled", "Sticky posture style", boolean=True),
  Control("graph", "posture_weight", "Posture preference weight", 0, 2, .01),
  Control("graph", "posture_upright_to_hunched_rate", "Upright to hunched rate (/s)", 0, .2, .001),
  Control("graph", "posture_hunched_to_upright_rate", "Hunched to upright rate (/s)", 0, .2, .001),
  Control("graph", "posture_initial_hunched", "Initial hunched share (next reset)", 0, 1, .01),
  Control("graph", "vertical_mask", "Flat-ground vertical mask", boolean=True),
  Control("graph", "twist_mask", "Planted-twist mask", boolean=True),
  Control("graph", "twist_max_foot_yaw_deg", "Planted-twist foot yaw limit (deg)", 0, 180, .5),
  Control("graph", "clearance_weight", "Clearance weight (encounter neighbors)", 0, 100, .5),
)

# Encounter design (`behavior.encounter`): shown instead of the plaza steering controls.
ENCOUNTER_CONTROLS = (
  Control("encounter", "bubble_radius_m", "Collision bubble radius (m)", .6, 8, .05),
  Control("encounter", "pass_margin_m", "Passing margin between bodies (m)", 0, 2, .05),
  Control("encounter", "horizon_s", "Collision look-ahead (s)", .2, 8, .1),
  Control("encounter", "switch_rate_hz", "Switch rate at contact (/s)", 0, 20, .1),
  Control("encounter", "urgency_exponent", "Urgency exponent", 0, 5, .05),
  Control("encounter", "assertiveness_range", "Assertiveness low (next reset)", 0, 1, .01, index=0),
  Control("encounter", "assertiveness_range", "Assertiveness high (next reset)", 0, 1, .01, index=1),
  Control("encounter", "patience_s", "Patience when blocked (s)", .1, 60, .1),
  *(Control("encounter", f"strategy_{where}", f"{name.capitalize()} weight ({where})", 0, 10, .05, index=k)
    for where in ("far", "near") for k, name in enumerate(("sidestep", "veer", "pivot", "yield"))),
  Control("encounter", "veer_radius_m", "Veer radius far (m)", .1, 10, .05, index=0),
  Control("encounter", "veer_radius_m", "Veer radius near (m)", .1, 10, .05, index=1),
  Control("encounter", "prefer_right", "Head-on: pass on the right", 0, 1, .01),
  Control("encounter", "inplace_probability", "Large ordinary turns in place", 0, 1, .01),
  Control("encounter", "turn_radius_m", "Ordinary turn radius low (m)", 0, 10, .05, index=0),
  Control("encounter", "turn_radius_m", "Ordinary turn radius high (m)", 0, 10, .05, index=1),
  Control("encounter", "body_radius_m", "Body radius (m)", .05, .5, .01),
  Control("encounter", "stop_clearance_m", "Stop clearance (m)", 0, .5, .005),
  Control("encounter", "stop_horizon_s", "Stop look-ahead (s)", .1, 4, .05),
  Control("encounter", "wait_probability", "Wait at waypoints", 0, 1, .01),
)


class LiveCrowdTuning:
  """GUI callbacks enqueue scalars; `apply_pending` runs beside crowd stepping."""

  def __init__(self, graph, steering, encounter=None):
    """`encounter`: the encounter design's live `EncounterCfg`; its controls replace the
    plaza steering controls."""
    self.graph, self.steering, self.encounter = graph, steering, encounter
    self.pending = SimpleQueue()
    shown = CONTROLS if encounter is None else (
      *(c for c in CONTROLS if c.owner != "steering"), *ENCOUNTER_CONTROLS)
    self.controls = {c.key: c for c in shown}
    self.handles = {}
    self.status = None
    self.last_result = "No live changes yet."

  def _owner(self, owner: str):
    return {"graph": self.graph.cfg, "steering": self.steering, "encounter": self.encounter}[owner]

  def submit(self, owner: str, field: str, value, index: int | None = None) -> None:
    control = self.controls[_key(owner, field, index)]
    if control.boolean:
      if not isinstance(value, bool):
        raise ValueError(f"{field} requires a boolean")
    else:
      value = float(value)
      if not math.isfinite(value) or not control.minimum <= value <= control.maximum:
        raise ValueError(f"{field} must be in [{control.minimum}, {control.maximum}]")
    self.pending.put((owner, field, index, value))

  def apply_pending(self) -> int:
    """Validate each queued edit before mutation; return the number applied."""
    applied = 0
    while True:
      try:
        owner, field, index, value = self.pending.get_nowait()
      except Empty:
        break
      cfg = self._owner(owner)
      if index is not None:
        items = list(getattr(cfg, field))
        items[index] = value
        value = tuple(items)
      try:
        trial = replace(cfg, **{field: value})  # dataclass validation runs on the trial
        if owner == "steering":
          # Check the angle pair even when its mechanism is disabled, so a later
          # toggle cannot enable an invalid intermediate configuration.
          if not 0 < trial.redirect_finish_deg < trial.redirect_start_deg <= 180:
            raise ValueError("finish angle must be smaller than start angle")
        elif field in MASK_FIELDS and (trial.vertical_mask or trial.posture_enabled or trial.twist_mask):
          # Check every mask edit while any mask is on: a lower twist limit can also
          # empty the start pool.
          if trial.vertical_mask or trial.posture_enabled:
            self.graph.ensure_annotations()
          if trial.twist_mask:
            self.graph.ensure_twist_annotations()
          allowed = self.graph.admissible_edges(cfg=trial)
          if not allowed.any():
            raise ValueError("mask leaves no admissible start edges")
          if trial.posture_enabled:
            ann = self.graph.annotations
            edge_style = ann.trunk_pitch_deg >= trial.posture_cutoff_deg
            start_style = ann.start_pitch_deg >= trial.posture_cutoff_deg
            for style in (False, True):
              if not (allowed & (edge_style == style) & (start_style == style)).any():
                raise ValueError(f"no admissible starts for hunched={style}")
        setattr(cfg, field, value)
        applied += 1
        suffix = " (next reset)" if field in ("posture_initial_hunched", "assertiveness_range") else ""
        self.last_result = f"Applied {field} = {value}{suffix}."
      except ValueError as exc:
        self.last_result = f"Rejected {field}: {exc}."
        handle = self.handles.get(_key(owner, field, index))
        if handle is not None:
          current = getattr(cfg, field)
          handle.value = current if index is None else current[index]
      if self.status is not None:
        self.status.content = self.last_result
    return applied

  def create_gui(self, server, name="Crowd") -> None:
    with server.gui.add_folder(f"{name.capitalize()} behavior"):
      server.gui.add_markdown(
        "Changes affect every environment at its next motion edge (up to 0.4 s). "
        "Initial posture share applies at reset. Higher forward preference makes "
        "ordinary sidesteps rarer; close avoidance can still use them."
      )
      for control in self.controls.values():
        value = getattr(self._owner(control.owner), control.field)
        if control.index is not None:
          value = value[control.index]
        if control.boolean:
          handle = server.gui.add_checkbox(control.label, initial_value=value)
        else:
          handle = server.gui.add_number(control.label, initial_value=value, step=control.step,
                                         min=control.minimum, max=control.maximum)
        self.handles[control.key] = handle

        @handle.on_update
        def changed(_event, c=control, h=handle):
          self.submit(c.owner, c.field, h.value, c.index)

      self.status = server.gui.add_markdown(self.last_result)
