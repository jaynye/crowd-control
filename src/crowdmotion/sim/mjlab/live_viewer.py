"""`crowd play traversal`'s Viser viewer: one policy in one crowd, switched from the GUI while
the server stays up, with the evaluator's metrics on a live panel (`live` describes what is
on offer and what each switch needs).

- **Env**: `evaluate.eval_env_cfg`, the config `crowd eval traversal` builds for the policy's
  run, with the chosen crowd's humans, behavior and robot perception; the viewer changes only
  display settings (goal arrows on, follow camera, LiDAR drawing) and, for velocity-goal
  tasks, the episode limit (none, as before). Episodes end on a collision unless
  `--no-end-on-collision`. Goals are paired (`CrowdGoalCommandCfg.paired_seed`), and every
  switch rewinds the episodes, so two policies in one crowd meet the same goal chain.
- **Planning arms** step one env step per drawn step (`PlanningVecEnv.decide`): the policy
  acts once per decision and the tracker walks in between.
- **Switches** (`Live`): a checkpoint or a policy with the same env loads in place; a crowd
  that differs only in live encounter settings is a queued edit on the running crowd
  (`LiveCrowdTuning.submit_encounter`); anything else ends this viewer, and the loop builds
  the new env and a new viewer on the same server (the browser shows a building notice,
  then the new scene). A failed build returns to the previous selection.
- **Panel**: `TraversalMetrics` (the evaluator's definitions) since the last switch, at 1 Hz.

The viewer writes nothing and stays off W&B.
"""

from __future__ import annotations

import gc
import socket
import tempfile
import time
import traceback
from argparse import Namespace
from dataclasses import asdict, replace
from pathlib import Path

import torch
import viser
from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.base import ViewerAction

from crowdmotion.metrics.traversal import TraversalEvalCfg, TraversalMetrics
from crowdmotion.sim.mjlab.evaluate import build_env, command_state, eval_env_cfg, rewind
from crowdmotion.sim.mjlab.live import (
  TRAINED,
  Target,
  checkpoint_inputs,
  crowd_choices,
  discover_runs,
  incompatibility,
  load_run,
  outcome_counts,
  panel_html,
  panel_rows,
  plan_switch,
  resolve_checkpoint,
  resolve_crowd,
  run_labels,
)
from crowdmotion.sim.mjlab.play import FOLLOW
from crowdmotion.sim.mjlab.train import actor_checkpoint

PLAY_EPISODE_S = 1e9  # velocity-goal episodes end only on a collision or fall, as before
EVAL_CFG = replace(TraversalEvalCfg(), fall_controls=0)  # the panel shows no fall attribution
PANEL_S = 1.0  # panel refresh period (a snapshot syncs the GPU)


class Refused(Exception):
  """A switch the viewer will not make; the message says why."""


class LiveMetrics:
  """The evaluator's `TraversalMetrics` over an open-ended play, restarted on each switch.
  `record` takes the step the env's `eval_capture` event saw (`evaluate.capture`)."""

  def __init__(self, env, seed: int):
    self.env, self.seed = env, seed
    self.restart()

  def restart(self) -> None:
    env = self.env
    twist = env.command_manager.get_term("twist")
    self.m = TraversalMetrics(env.num_envs, env.command_manager.get_term("crowd").M, env.step_dt, 1,
                              twist.cfg.slow_m, twist.cfg.arrive_m, EVAL_CFG, env.device, seed=self.seed)
    self.m.start(command_state(env))

  def record(self) -> None:
    self.m.step(self.env._eval_pre, command_state(self.env))

  @property
  def seconds(self) -> float:
    return self.m.t

  def snapshot(self):
    return self.m.snapshot()


class Session:
  """One built env for a `Target`, its runner and loaded checkpoint, stepped one env step at
  a time. It is the viewer's env (`mjlab.viewer.base.EnvProtocol`)."""

  def __init__(self, target: Target, args, planner=None):
    from mjlab.rl import MjlabOnPolicyRunner

    from crowdmotion.sim.mjlab.plaza import set_hit_sphere_radius
    from crowdmotion.sim.mjlab.traversal import traversal_rl_cfg

    self.target, self.args = target, args
    policy, crowd = target.policy, target.crowd
    opts = policy.env
    nav = opts.get("navigation")
    spec = Namespace(behavior=crowd.behavior, humans=crowd.humans, seed=args.seed, envs=args.num_envs,
                     seconds=None, episode_s=None if nav is not None else PLAY_EPISODE_S, paired=True, cfg=EVAL_CFG)
    cfg = eval_env_cfg(opts, crowd.perception, "end" if args.end_on_collision else "continue", spec)
    cfg.commands["twist"].debug_vis = True  # the goal command (the evaluator hides it)
    cfg.viewer.max_extra_envs = 0  # every world sits at the origin
    cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = (FOLLOW[k] for k in ("distance", "elevation",
                                                                                       "azimuth"))
    agent = traversal_rl_cfg(teacher=policy.plays_as == "teacher", planner_rate=opts.get("planner_rate") or 0.0,
                             navigation=nav)
    self.raw, self.wrapped = build_env(cfg, opts, agent.clip_actions, args.device, planner=planner)
    try:
      set_hit_sphere_radius(self.raw)
      self.raw.scene["lidar"].cfg.viz.stride = args.lidar_stride
      self.runner = MjlabOnPolicyRunner(self.wrapped, asdict(agent), device=args.device)
      self.group = "teacher" if policy.plays_as == "teacher" else "actor"
      self.planning = hasattr(self.wrapped, "decide")
      self.metrics = LiveMetrics(self.raw, args.seed)
      if self.planning:  # as the evaluator: every env step of a decision is measured
        self.wrapped.on_substep = self.metrics.record
      self._decision = None
      self.reset()
      self.load(target.checkpoint)
    except BaseException:
      self.close()
      raise

  # The viewer's env.
  @property
  def num_envs(self) -> int:
    return self.raw.num_envs

  @property
  def device(self):
    return self.raw.device

  @property
  def cfg(self):
    return self.raw.cfg

  @property
  def unwrapped(self):
    return self.raw

  def get_observations(self):
    return self.obs

  def step(self, actions=None) -> None:
    """One env step; the policy acts at its own rate (every step, or once per decision)."""
    del actions
    if self.planning:
      if self._decision is None:
        self._decision = self.wrapped.decide(self.policy(self.obs))
      try:
        next(self._decision)
      except StopIteration as stop:
        self.obs, self._decision = stop.value[0], None
    else:
      self.obs = self.wrapped.step(self.policy(self.obs))[0]
      self.metrics.record()

  def reset(self) -> None:
    """Rewind the episodes (`evaluate.rewind`: the same paired goals and crowd draws as a
    fresh env), reset every env and restart the counters."""
    if self._decision is not None:
      self._decision.close()
      self._decision = None
    rewind(self.raw, self.args.seed)
    self.obs, _ = self.wrapped.reset()
    self.metrics.restart()

  def close(self) -> None:
    raw = getattr(self, "raw", None)
    if raw is not None:
      raw.close()
    for name in ("raw", "wrapped", "runner", "policy", "metrics", "obs", "_decision"):
      self.__dict__.pop(name, None)
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

  # Policies and crowds.
  def check(self, checkpoint: Path) -> None:
    """Refuse a checkpoint whose actor reads another observation width than this env's."""
    want, have = checkpoint_inputs(checkpoint), int(self.obs[self.group].shape[-1])
    if want is not None and want != have:
      raise Refused(f"{checkpoint.name} reads {want} observation values; this env gives its "
                    f"{self.group} group {have}")

  def load(self, checkpoint: Path) -> None:
    self.check(checkpoint)
    self.runner.load(str(actor_checkpoint(checkpoint)), load_cfg={"actor": True}, strict=True,
                     map_location=self.args.device)
    self.policy = self.runner.get_inference_policy(device=self.args.device)

  def edit_crowd(self, fields: dict) -> None:
    """Queue encounter fields on the running crowd; they apply at the next reset."""
    self.raw.command_manager.get_term("crowd").live_tuning.submit_encounter(fields)

  def crowd_result(self) -> str:
    return self.raw.command_manager.get_term("crowd").live_tuning.last_result


class LiveViewer(ViserPlayViewer):
  """mjlab's Viser play viewer with a policy and crowd switcher and a live panel above its tabs."""

  def __init__(self, live: "Live", session: Session, message: tuple[str, bool] = ("", False)):
    super().__init__(session, session.step, frame_rate=getattr(live.args, "frame_rate", 60.0),
                     viser_server=live.server)
    self.live, self.session = live, session
    self.rebuild: tuple[Target, object] | None = None
    self.message = message  # (text, is an error), shown when the viewer starts
    self.panel: dict = {}  # the last panel values, plain Python (read by headless checks)
    self.ready = False
    self._panel_time = 0.0
    self._syncing = False  # selection widgets being set from the current target

  # Setup -------------------------------------------------------------------------------
  def setup(self) -> None:
    gui = self._server.gui
    live, target = self.live, self.session.target
    with gui.add_folder("Live play"):
      self._now = gui.add_html("")
      self._perf = gui.add_html("")
      self._note = gui.add_html("")
    with gui.add_folder("Policy and crowd"):
      self._labels = live.run_options(target.policy)
      current = next(k for k, v in self._labels.items() if v.path == target.policy.path)
      self._run = gui.add_dropdown("Run", options=tuple(self._labels), initial_value=current)
      self._path = gui.add_text("Or run path", initial_value="", hint="any crowd train traversal run directory")
      self._run_info = gui.add_markdown(target.policy.describe())
      names = [c.name for c in target.policy.checkpoints]
      self._ckpt = gui.add_dropdown("Checkpoint", options=tuple(names), initial_value=target.checkpoint.name
                                    if target.checkpoint.name in names else names[-1])
      self._crowd = gui.add_dropdown("Crowd", options=tuple(live.choices), initial_value=target.crowd.name)
      self._crowd_info = gui.add_markdown(live.choices[target.crowd.name].description)
      self._apply = gui.add_button("Apply", hint="switch to this policy and crowd; counters restart")
      self._rescan = gui.add_button("Rescan runs")

    @self._run.on_update
    def _(_) -> None:
      if self._syncing:
        return
      self._syncing = True
      try:  # a run picked from the menu replaces a typed path
        self._path.value = ""
      finally:
        self._syncing = False
      self._show_run(self._labels.get(self._run.value))

    @self._path.on_update
    def _(_) -> None:
      if self._syncing:
        return
      text = self._path.value.strip()
      if not text:
        return self._show_run(self._labels.get(self._run.value))
      try:
        self._show_run(load_run(text))
      except Exception as exc:  # noqa: BLE001 (shown to the user)
        self._run_info.content = f"**Not a playable run:** {exc}"

    @self._crowd.on_update
    def _(_) -> None:
      choice = self.live.choices.get(self._crowd.value)
      self._crowd_info.content = choice.description if choice else ""

    @self._apply.on_click
    def _(_) -> None:
      text = self._path.value.strip()
      run = text or str(self._labels[self._run.value].path)
      self._actions.append((ViewerAction.CUSTOM, {"type": "live_switch", "run": run,
                                                  "checkpoint": self._ckpt.value, "crowd": self._crowd.value}))

    @self._rescan.on_click
    def _(_) -> None:  # new runs, and new checkpoints of the selected one (an arm still training)
      self.live.rescan()
      self._labels = self.live.run_options(self.session.target.policy)
      self._syncing = True
      try:
        self._run.options = tuple(self._labels)
      finally:
        self._syncing = False
      text = self._path.value.strip()
      try:
        self._show_run(load_run(text) if text else self._labels.get(self._run.value))
      except Exception as exc:  # noqa: BLE001 (shown to the user)
        self._run_info.content = f"**Not a playable run:** {exc}"

    super().setup()
    self._say(*self.message)
    self._show_target()
    self.ready = True

  def _show_run(self, policy) -> None:
    """Show a run's description and checkpoints (the GUI thread; nothing in the sim changes)."""
    if policy is None:
      return
    self._run_info.content = policy.describe()
    names = tuple(c.name for c in policy.checkpoints)
    self._ckpt.options = names
    self._ckpt.value = names[-1]

  def _sync_selection(self) -> None:
    """Point the selection widgets at what plays now."""
    target = self.session.target
    self._syncing = True
    try:
      self._run.options = tuple(self._labels)
      label = next((k for k, v in self._labels.items() if v.path == target.policy.path), None)
      if label is not None:
        self._run.value = label
        self._path.value = ""
      else:
        self._path.value = str(target.policy.path)
      self._run_info.content = target.policy.describe()
      self._ckpt.options = tuple(c.name for c in target.policy.checkpoints)
      self._ckpt.value = target.checkpoint.name
      self._crowd.value = target.crowd.name
      self._crowd_info.content = self.live.choices[target.crowd.name].description
    finally:
      self._syncing = False

  def _show_target(self) -> None:
    target = self.session.target
    self._now.content = (
      "<div style='font-size:0.85em;line-height:1.3;padding:0 1em 0.3em 1em'>"
      f"<b>Policy</b> {target.policy.name} · {target.checkpoint.stem}<br/>{target.policy.describe()}<br/>"
      f"<b>Crowd</b> {target.crowd.name}: {self.live.choices[target.crowd.name].description}</div>")
    self._panel_time = 0.0

  def _say(self, text: str, error: bool = False) -> None:
    if text:
      print(("REFUSED " if error else "") + text, flush=True)
    color = "#e74c3c" if error else "inherit"
    self._note.content = (f"<div style='font-size:0.85em;line-height:1.3;padding:0 1em 0.5em 1em;"
                          f"color:{color}'>{text}</div>") if text else ""

  # Stepping and the panel ------------------------------------------------------------
  def _execute_step(self) -> bool:
    try:
      with torch.no_grad():
        self.session.step()
      self._step_count += 1
      self._stats_steps += 1
      return True
    except Exception:
      self._last_error = traceback.format_exc()
      print(f"[ERROR] Exception during step:\n{self._last_error}", flush=True)
      self.pause()
      return False

  def _update_status_display(self) -> None:
    super()._update_status_display()
    now = time.perf_counter()
    if now - self._panel_time >= PANEL_S:
      self._panel_time = now
      self._refresh_panel()

  def _refresh_panel(self) -> None:
    session = self.session
    metrics, tables = session.metrics.snapshot()
    navigation = session.target.policy.navigation
    realtime = self.actual_realtime if not self._is_paused else None
    rows = panel_rows(metrics, tables, navigation, session.metrics.seconds, session.num_envs, realtime)
    self._perf.content = panel_html("<b>Since the last switch</b>", rows)
    self.panel = {"target": session.target.describe(), "seconds": session.metrics.seconds,
                  "steps": self._step_count, "realtime": realtime, "rows": dict(rows),
                  "collisions": metrics.get("collision/count"), "falls": metrics.get("fall/count"),
                  "arrivals": metrics.get("command/arrivals"), "robot_min": metrics.get("exposure/robot_min"),
                  "outcomes": outcome_counts(tables), "time": time.time()}

  def _handle_gui_reset(self, all_envs: bool) -> None:
    super()._handle_gui_reset(all_envs)
    self.session.metrics.restart()  # a reset from the command panel starts the episodes afresh

  # Switches -----------------------------------------------------------------------------
  def _handle_custom_action(self, action, payload) -> bool:
    if isinstance(payload, dict) and payload.get("type") == "live_switch":
      self._switch(payload)
      return True
    return super()._handle_custom_action(action, payload)

  def _switch(self, request: dict) -> None:
    """Apply a switch on the simulation thread: in place, or by ending this viewer for a rebuild."""
    session = self.session
    try:
      target = self.live.target(request["run"], request["checkpoint"], request["crowd"])
    except Exception as exc:  # noqa: BLE001 (shown to the user)
      return self._say(f"Refused: {exc}", error=True)
    refusal = incompatibility(target.policy, target.crowd.humans)
    if refusal:
      return self._say(f"Refused {target.describe()}: {refusal}", error=True)
    plan = plan_switch(session.target, target)
    if plan.rebuild:
      self.rebuild = (target, plan)
      return self._say("Rebuilding the env: " + "; ".join(plan.changes) + " …")
    try:
      if plan.load:
        session.check(target.checkpoint)
    except Refused as exc:
      return self._say(f"Refused: {exc}", error=True)
    if plan.fields:
      session.edit_crowd(plan.fields)
    if plan.load:
      session.load(target.checkpoint)
    session.target = target
    self.reset_environment()  # rewinds the episodes, restarts the counters; queued crowd edits apply
    if plan.fields and not session.crowd_result().startswith("Applied"):
      return self._say(f"Crowd edit failed: {session.crowd_result()}", error=True)
    changes = "; ".join(plan.changes) if plan.changes else "nothing changed"
    self._say(f"Switched in place ({changes}). Counters restarted.")
    self._show_target()
    self._sync_selection()

  def is_running(self) -> bool:
    return self.rebuild is None and not self.live.stopping


class Live:
  """The viewer's process: the Viser server, the runs and crowds on offer, the SONIC planner
  shared by every planning-arm env, and the loop that builds an env and a viewer for each
  target, keeping the server up between them."""

  def __init__(self, args):
    self.args = args
    self.choices = {c.name: c for c in crowd_choices(args.conditions)}
    self.scratch = Path(tempfile.mkdtemp(prefix="crowd_live_"))
    self.runs = discover_runs(args.runs)
    self.stopping = False
    self.viewer: LiveViewer | None = None
    self.history: list[dict] = []  # one entry per built env: target, build seconds, CUDA memory
    self._planner = None
    self.server = viser.ViserServer(host=args.host, port=args.port, label="crowd traversal live")

  def rescan(self) -> None:
    self.runs = discover_runs(self.args.runs)

  def run_options(self, current) -> dict:
    """Dropdown labels of the discovered runs, plus `current` when it lies elsewhere."""
    runs = list(self.runs)
    if all(r.path != current.path for r in runs):
      runs.insert(0, current)
    return run_labels(runs)

  def target(self, run, checkpoint, crowd: str) -> Target:
    if crowd not in self.choices:
      raise ValueError(f"unknown crowd {crowd!r}; choose from {', '.join(self.choices)}")
    policy = load_run(run)
    return Target(policy, resolve_checkpoint(policy, checkpoint), resolve_crowd(self.choices[crowd], policy,
                                                                               self.scratch))

  def planner(self, policy):
    if not policy.env.get("planner_rate"):
      return None
    if self._planner is None:
      from crowdmotion.backends import sonic

      self._planner = sonic.SonicPlanner(device=self.args.device)
    return self._planner

  def stop(self) -> None:
    self.stopping = True

  def _clear(self) -> None:
    """Remove every GUI element and scene node, for the next viewer's setup."""
    _remove_tabs(getattr(self.server.gui, "_container_handle_from_uuid", {}).get("root"))
    self.server.gui.reset()
    self.server.scene.reset()

  def _notice(self, text: str) -> None:
    self._clear()
    self.server.gui.add_markdown(text)

  def serve(self, first: Target) -> int:
    """Play `first`, then whatever the GUI switches to, until Ctrl+C (or `stop`)."""
    print(f"VISER host={socket.getfqdn()} port={self.server.get_port()} device={self.args.device}", flush=True)
    target, fallback, message = first, None, ("", False)
    callbacks = getattr(self.server, "_client_connect_cb", None)  # mjviser adds one per scene; viser has no remove
    try:
      while not self.stopping:
        self._notice(f"**Building the env** for {target.describe()} …")
        print(f"building {target.describe()}", flush=True)
        start = time.perf_counter()
        try:
          refusal = incompatibility(target.policy, target.crowd.humans)
          if refusal:
            raise Refused(refusal)
          session = Session(target, self.args, self.planner(target.policy))
        except Exception as exc:  # noqa: BLE001 (reported, then the previous target plays)
          traceback.print_exc()
          if fallback is None:
            raise
          message = (f"Could not build {target.describe()}: {exc}. Back to {fallback.describe()}.", True)
          target, fallback = fallback, None
          continue
        self.history.append({"target": target.describe(), "build_s": round(time.perf_counter() - start, 2),
                             **_cuda_memory()})
        print(f"built {target.describe()} in {self.history[-1]['build_s']} s; {self.history[-1]}", flush=True)
        self._clear()
        kept = len(callbacks) if callbacks is not None else 0
        viewer = LiveViewer(self, session, message)
        self.viewer = viewer
        try:
          viewer.run()
        finally:
          self.viewer = None
          if callbacks is not None:
            del callbacks[kept:]
        interrupted, rebuild = getattr(viewer, "_interrupted", False), viewer.rebuild
        del viewer
        session.close()
        del session
        self.history[-1]["after_close"] = _cuda_memory()
        if interrupted or rebuild is None:
          break
        fallback = target
        target, plan = rebuild
        message = ("Rebuilt the env (" + "; ".join(plan.changes) + "). Counters restarted.", False)
    finally:
      self.server.stop()
    return 0


def _remove_tabs(container) -> None:
  """Remove the tabs of every tab group below `container`, innermost first. viser 1.0's
  tab group `remove` (which `gui.reset` calls) marks the group removed before removing its
  tabs, and each tab then fails to update it; a group without tabs removes cleanly."""
  for child in tuple(getattr(container, "_children", {}).values()):
    for tab in tuple(getattr(child, "_tab_handles", ())):
      _remove_tabs(tab)
      tab.remove()
    _remove_tabs(child)


def _cuda_memory() -> dict:
  if not torch.cuda.is_available():
    return {}
  return {"cuda_allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 3),
          "cuda_reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 3)}


def serve(args, run: Path, checkpoint: Path) -> int:
  """`crowd play traversal`'s Viser path: play `run`'s `checkpoint` in `args.crowd`."""
  live = Live(args)
  if args.crowd not in live.choices:
    raise SystemExit(f"--crowd {args.crowd}: choose from {', '.join(live.choices)}")
  first = live.target(run, checkpoint, args.crowd)
  refusal = incompatibility(first.policy, first.crowd.humans)
  if refusal:
    raise SystemExit(f"--crowd {args.crowd}: {refusal}")
  return live.serve(first)


__all__ = ["TRAINED", "Live", "LiveMetrics", "LiveViewer", "Refused", "Session", "serve"]
