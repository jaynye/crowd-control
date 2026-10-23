"""Headless check of `crowd play traversal`'s live Viser viewer (not a crowd command).

Starts the viewer (no browser) and drives its GUI widgets from a second thread as a user
would: it sets the Run, Or-run-path, Checkpoint and Crowd widgets and presses Apply. A
navigation arm goes through every crowd (in-place queued crowd edits and env rebuilds), the
policy switches mid-run to a planning arm, which goes through every crowd too, a checkpoint
switch loads in place, an incompatible teacher/crowd pair must be refused, and the
navigation arm comes back through the path field. Every stage checks that the counters
restarted with the switch (simulated seconds equal the viewer's steps since then) and grew
while it played, and that no step raised.

  python scripts/smoke/live_viewer_check.py --navigation <run> --planning <run> [--teacher <run>]
      [--in-place <run>] [--play <run>=<crowd> ...] [--client] [--seconds 20] [--extra-seconds 10]
      [--speed 8] [--num-envs 1] [--port 8090]

`--in-place` names a navigation run whose env equals `--navigation`'s (they differ only in
rewards): switching to it must load in place. Each `--play` stage then plays one more
policy in one crowd (a 50 Hz policy, a teacher, a distilled student, a 2 Hz arm, b0...).
`--client` keeps a websocket client connected to the server through the rebuilds and
opens a fresh one after each, as a browser tab would, counting the messages it receives.

Writes `runs/<UTC>_live_viewer_check/check.json` (each stage's first and last panel,
build times and CUDA memory per env) and exits 1 if a stage fails.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from pathlib import Path

import torch

from crowdmotion.core.rundir import RunDir
from crowdmotion.sim.mjlab.live import TRAINED, crowd_choices
from crowdmotion.sim.mjlab.live_viewer import Live

DT = 0.02


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--navigation", type=Path, required=True, help="a --navigation traversal run")
  p.add_argument("--planning", type=Path, required=True, help="a velocity-goal planning-arm run")
  p.add_argument("--teacher", type=Path, help="a teacher run, refused in a 4-human crowd")
  p.add_argument("--in-place", type=Path, help="a navigation run that must switch in place from --navigation")
  p.add_argument("--play", action="append", default=[], metavar="RUN=CROWD", help="one more policy and crowd")
  p.add_argument("--client", action="store_true", help="connect websocket clients through the rebuilds")
  p.add_argument("--extra-seconds", type=float, default=10.0, help="simulated seconds per --in-place/--play stage")
  p.add_argument("--seconds", type=float, default=20.0, help="simulated seconds per stage")
  p.add_argument("--speed", type=int, default=8, choices=(1, 2, 4, 8), help="viewer speed target during stages")
  p.add_argument("--num-envs", type=int, default=1)
  p.add_argument("--port", type=int, default=8090)
  p.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for one stage")
  a = p.parse_args()
  args = argparse.Namespace(device="cuda:0", num_envs=a.num_envs, seed=0, end_on_collision=True, lidar_stride=8,
                            host="0.0.0.0", port=a.port, runs=a.navigation.parent, conditions=None, frame_rate=60.0)
  # In-place crowd edits (same humans and perception) between rebuilds; every crowd once.
  crowds = ["independent_n8", "coherent90_n8", "reactive_blind_n8", "reactive_aware_n8", "independent_n4",
            "independent_n16", "coherent90_n16"]
  assert sorted(crowds) == sorted(c.name for c in crowd_choices() if c.name != TRAINED)
  record = {"kind": "live_viewer_check", "navigation": str(a.navigation), "planning": str(a.planning),
            "teacher": str(a.teacher) if a.teacher else None, "seconds": a.seconds, "speed": a.speed,
            "num_envs": a.num_envs, "gpu": torch.cuda.get_device_name(0), "in_place": str(a.in_place),
            "play": a.play, "client": a.client}
  with RunDir.create("live_viewer_check", record=record) as run:
    live = Live(args)
    stages: list[dict] = []
    failures: list[str] = []

    def viewer_for(want: str, deadline: float):
      """The live viewer once it plays `want` (a Target.describe()) and has drawn a panel."""
      while time.time() < deadline:
        v = live.viewer
        if v is not None and v.ready and v.session.target.describe() == want and v.panel.get("target") == want:
          return v
        if v is not None and v._last_error:
          raise RuntimeError(v._last_error)
        time.sleep(0.1)
      raise TimeoutError(f"{want} not playing after {a.timeout} s")

    def label_of(v, path: Path) -> str:
      path = path.resolve()
      return next(k for k, r in v._labels.items() if r.path == path)

    def stage(name: str, crowd: str | None = None, run: Path | None = None, checkpoint: str | None = None,
              by_path: bool = False, refuse: bool = False, seconds: float | None = None,
              in_place: bool | None = None) -> None:
        """Select in the GUI, press Apply and watch the result."""
        v = live.viewer
        before = v.session.target
        if run is not None:
          if by_path:
            v._path.value = str(run)
          else:
            v._path.value = ""
            v._run.value = label_of(v, run)
        if checkpoint is not None:
          v._ckpt.value = checkpoint
        if crowd is not None:
          v._crowd.value = crowd
        policy_path = run.resolve() if run is not None else before.policy.path
        sent = time.time()
        v._apply.value = True  # a button press: viser calls on_click
        entry = {"stage": name, "crowd": crowd, "run": str(policy_path), "checkpoint": checkpoint}
        if refuse:
          time.sleep(3.0)
          now = live.viewer
          entry.update(refused=now is v and now.session.target == before and "Refused" in v._note.content,
                       note=v._note.content)
          if not entry["refused"]:
            failures.append(f"{name}: not refused")
          stages.append(entry)
          print(json.dumps(entry), flush=True)
          return
        deadline = sent + a.timeout
        while time.time() < deadline:  # wait for the switch to land (in place, or a new viewer)
          now = live.viewer
          if now is not None and now.ready and now.session.target != before and \
             (crowd is None or now.session.target.crowd.name == crowd) and now.session.target.policy.path == policy_path:
            break
          if now is v and v._last_error:
            raise RuntimeError(v._last_error)
          time.sleep(0.1)
        else:
          raise TimeoutError(f"{name}: switch did not land")
        want = now.session.target.describe()
        entry["rebuilt"] = now is not v
        entry["switch_s"] = round(time.time() - sent, 2)
        entry["note"] = now._note.content
        for _ in range({8: 3, 4: 2, 2: 1, 1: 0}[a.speed]):
          now.request_speed_up()
        v2 = viewer_for(want, deadline)
        first = dict(v2.panel)
        seconds = a.seconds if seconds is None else seconds
        while time.time() < deadline and (live.viewer is v2) and v2.panel.get("seconds", 0) < seconds:
          if v2._last_error:
            raise RuntimeError(v2._last_error)
          time.sleep(0.2)
        last = dict(v2.panel)
        entry.update(target=want, first=first, last=last)
        problems = []
        for panel in (first, last):  # counters restarted with the step count, then only grew
          if abs(panel["seconds"] - panel["steps"] * DT) > 1e-6 * max(panel["steps"], 1) + 1e-9:
            problems.append(f"seconds {panel['seconds']} != steps {panel['steps']} x dt")
        if last["seconds"] < seconds or last["seconds"] <= first["seconds"]:
          problems.append(f"played {first['seconds']} -> {last['seconds']} s")
        for key in ("collisions", "falls", "arrivals"):
          if (last.get(key) or 0) < (first.get(key) or 0):
            problems.append(f"{key} fell")
        if v2._last_error:
          problems.append("step error")
        if in_place is not None and entry["rebuilt"] == in_place:
          problems.append("rebuilt" if in_place else "switched in place, expected a rebuild")
        entry["problems"] = problems
        failures.extend(f"{name}: {x}" for x in problems)
        stages.append(entry)
        print(json.dumps({k: entry[k] for k in ("stage", "target", "rebuilt", "switch_s", "problems")}), flush=True)
        print("   ", json.dumps(last["rows"]), flush=True)

    def drive() -> None:
      try:
        v = viewer_for(live.viewer_target_hint, time.time() + a.timeout)
        while v.panel.get("seconds", 0) < 10 and live.viewer is v:  # real-time factor at 1x
          time.sleep(0.2)
        stages.append({"stage": "initial at 1x", "target": v.panel["target"], "last": dict(v.panel)})
        print(json.dumps({"stage": "initial at 1x", "realtime": v.panel["realtime"]}), flush=True)
        nav, plan = a.navigation, a.planning
        # The navigation arm through every crowd (its own first).
        for crowd in crowds + [TRAINED]:
          stage(f"navigation in {crowd}", crowd=crowd)
        # Policy switch mid-run: the planning arm, then every crowd.
        stage("planning arm (switch policy)", run=plan, crowd="reactive_aware_n8")
        for crowd in crowds + [TRAINED]:
          stage(f"planning in {crowd}", crowd=crowd)
        names = [c.name for c in live.viewer.session.target.policy.checkpoints]
        if len(names) > 1:
          stage("planning checkpoint in place", checkpoint=names[-2])
        if a.teacher is not None:
          stage("teacher among 4 humans", run=a.teacher, crowd="independent_n4", refuse=True)
        stage("navigation by path (switch policy)", run=nav, crowd="independent_n8", by_path=True)
        if a.in_place is not None:
          stage("navigation arm in place", run=a.in_place, seconds=a.extra_seconds, in_place=True)
        for item in a.play:
          path, crowd = item.rsplit("=", 1)
          stage(f"{Path(path).name} in {crowd}", run=Path(path), crowd=crowd, seconds=a.extra_seconds)
      except Exception:  # noqa: BLE001
        failures.append(traceback.format_exc())
        print(failures[-1], flush=True)
      finally:
        live.stop()

    clients = {"connections": 0, "messages": 0, "bytes": 0, "errors": []}

    def client() -> None:
      """A browser stand-in: one long-lived connection, and a fresh one after each rebuild."""
      import viser
      from websockets.sync.client import connect as ws_connect

      def connect(url, **kw):  # viser accepts only its own client version
        return ws_connect(url, subprotocols=[f"viser-v{viser.__version__}"], **kw)

      def listen(ws, until) -> None:
        while until():
          try:
            msg = ws.recv(timeout=0.5)
          except TimeoutError:
            continue
          clients["messages"] += 1
          clients["bytes"] += len(msg)

      url = f"ws://127.0.0.1:{a.port}"
      seen = None
      try:
        while live.viewer is None:
          time.sleep(0.2)
        with connect(url, max_size=None) as long_lived:
          clients["connections"] += 1
          while not live.stopping:
            v = live.viewer
            if v is not None and v.ready and v is not seen:
              seen = v
              with connect(url, max_size=None) as fresh:
                clients["connections"] += 1
                t = time.time()
                listen(fresh, lambda: time.time() < t + 2 and not live.stopping)
            try:
              msg = long_lived.recv(timeout=0.5)
              clients["messages"] += 1
              clients["bytes"] += len(msg)
            except TimeoutError:
              pass
      except Exception:  # noqa: BLE001
        clients["errors"].append(traceback.format_exc())

    first = live.target(a.navigation, None, TRAINED)
    live.viewer_target_hint = first.describe()
    driver = threading.Thread(target=drive, daemon=True)
    driver.start()
    if a.client:
      threading.Thread(target=client, daemon=True).start()
    live.serve(first)
    driver.join(timeout=30)
    if a.client:
      if clients["errors"] or clients["connections"] < 2 or not clients["messages"]:
        failures.append(f"websocket clients: {clients}")
    summary = {"stages": stages, "builds": live.history, "failures": failures, "clients": clients,
               "events": {k: sum((s.get("last") or {}).get(k) or 0 for s in stages)
                          for k in ("collisions", "falls", "arrivals")},
               "outcomes": {k: sum(((s.get("last") or {}).get("outcomes") or {}).get(k, 0) for s in stages)
                            for k in ("success", "collision", "fall", "deadline", "time_out")}}
    run.write_json("check.json", summary)
    run.update(failures=len(failures), stages=len(stages), builds=len(live.history))
    print(json.dumps({"out": str(run.path), "stages": len(stages), "builds": len(live.history), "clients": clients,
                      "failures": failures, "events": summary["events"], "outcomes": summary["outcomes"]}), flush=True)
  return 1 if failures else 0


if __name__ == "__main__":
  raise SystemExit(main())
