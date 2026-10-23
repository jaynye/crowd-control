"""Former TensorBoard-to-W&B upload of `crowd train traversal` runs; now `crowd wandb sync`.

    .venv/bin/python scripts/wandb_sync.py runs/<UTC>_train_traversal_<name> ... [--entity <team> --project <name>]
    .venv/bin/python scripts/wandb_sync.py ... --dry-run     # print what would be uploaded

Kept for old command lines; it will be removed. If new lineages default to W&B
`disabled`, it refuses: it would turn W&B on for every run it is given. Otherwise it runs `crowd wandb sync --enable` on the given runs: one W&B run per training lineage, id = the directory name, rows from
TensorBoard past what W&B already holds (`published_through`, for runs this script
uploaded before W&B's last step). `--entity` and `--project` set `CROWD_WANDB_ENTITY` and
`CROWD_WANDB_PROJECT` for runs that do not yet record a W&B run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crowdmotion.tracking.lineage import DEFAULT_MODE  # noqa: E402
from crowdmotion.tracking.sync import main as wandb_main  # noqa: E402


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("runs", nargs="+", type=Path)
  p.add_argument("--entity", help="W&B team or user (default: $CROWD_WANDB_ENTITY or rl2-group)")
  p.add_argument("--project", help="W&B project (default: $CROWD_WANDB_PROJECT or crowd-control-traversal)")
  p.add_argument("--dry-run", action="store_true")
  args = p.parse_args()
  print("scripts/wandb_sync.py is retiring: use `crowd wandb sync`", file=sys.stderr)
  if DEFAULT_MODE == "disabled" and not args.dry_run:
    print("W&B is off by default (tracking.lineage.DEFAULT_MODE); nothing uploaded", file=sys.stderr)
    return 2
  if args.entity:
    os.environ["CROWD_WANDB_ENTITY"] = args.entity
  if args.project:
    os.environ["CROWD_WANDB_PROJECT"] = args.project
  return wandb_main(["sync", *map(str, args.runs), "--enable", *(["--dry-run"] if args.dry_run else [])])


if __name__ == "__main__":
  sys.exit(main())
