"""`crowd`: one command line for the project.

  crowd <group> <command> [args ...]
  crowd <group> <command> --help

Each command is the `main(argv)` of a package module, imported only when run, so
`crowd --help` works without mjlab, ARDY or Kimodo. Commands that write results
make a fresh `runs/<UTC>_<name>/` unless given `--out`.
"""

from __future__ import annotations

import importlib
import sys

# group -> command -> (module, help, argv prefix passed before the user's arguments)
COMMANDS: dict[str, dict[str, tuple[str, str, tuple[str, ...]]]] = {
  "library": {
    "generate": ("crowdmotion.library.generate", "generate an offline ARDY motion library", ()),
  },
  "graph": {
    "build": ("crowdmotion.graph.build", "build a motion graph from a library and save it", ()),
    "report": ("crowdmotion.viz.report", "audit a saved graph and write an HTML report", ()),
    "speed": ("crowdmotion.bench.graph_speed", "time graph stepping for a training population", ()),
  },
  "behavior": {
    "evaluate": ("crowdmotion.iterate.evaluate", "plaza rollouts on a saved graph: traces and metrics", ()),
    "render": ("crowdmotion.viz.replay", "replay a saved evaluation as a MuJoCo video", ()),
    "compare": ("crowdmotion.iterate.compare", "paired seed metrics and a review page", ()),
  },
  "iterate": {
    "start": ("crowdmotion.iterate.session", "open an iteration session (one run folder)", ("start",)),
    "run": ("crowdmotion.iterate.session", "queue or run a named request in a session", ("run",)),
    "status": ("crowdmotion.iterate.session", "show a session's requests and results", ("status",)),
    "note": ("crowdmotion.iterate.session", "add a plan, rule, observation or decision to the journal", ("note",)),
    "stop": ("crowdmotion.iterate.session", "stop a session", ("stop",)),
    "watch": ("crowdmotion.iterate.session", "enforce a Slurm session's limits", ("watch",)),
    "worker": ("crowdmotion.iterate.executors", "run queued requests (inside a Slurm allocation)", ("worker",)),
  },
  "bench": {
    "throughput": ("crowdmotion.bench.throughput", "online ARDY/Kimodo usable-motion throughput", ()),
    "quality": ("crowdmotion.bench.quality", "closed-loop continuation quality (online and graph)", ()),
    "tables": ("crowdmotion.bench.tables", "Markdown tables of throughput and quality runs", ()),
    "clips": ("crowdmotion.viz.stick", "stick-figure videos of quality clips", ()),
    "sim-speed": ("crowdmotion.bench.sim_speed", "simulator speed reference points (mjlab)", ()),
  },
  "demo": {
    "video": ("crowdmotion.sim.mjlab.demo", "render the mjlab plaza with LiDAR to MP4", ("video",)),
    "viser": ("crowdmotion.sim.mjlab.demo", "serve the mjlab plaza in a Viser browser viewer", ("viser",)),
    "corridor": ("crowdmotion.sim.mjlab.corridor.demo", "primitive-human corridor with LiDAR", ()),
  },
}


def usage() -> str:
  lines = [__doc__.strip().split("\n\n")[0], "", "commands:"]
  for group, cmds in COMMANDS.items():
    for name, (_, help_, _) in cmds.items():
      lines.append(f"  {group + ' ' + name:<22} {help_}")
  return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
  argv = list(sys.argv[1:] if argv is None else argv)
  if len(argv) < 2 or argv[0] not in COMMANDS or argv[1] not in COMMANDS[argv[0]]:
    asked = argv[:1] in (["-h"], ["--help"])
    print(usage(), file=sys.stdout if asked else sys.stderr)
    return 0 if asked else 2
  module, _, prefix = COMMANDS[argv[0]][argv[1]]
  sys.argv[0] = f"crowd {argv[0]} {argv[1]}"
  result = importlib.import_module(module).main([*prefix, *argv[2:]])
  return int(result or 0)


if __name__ == "__main__":
  sys.exit(main())
