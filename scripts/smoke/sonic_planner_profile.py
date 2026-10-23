"""Where the batched planner port spends its time. Run on a GPU compute node:

    .venv/bin/python scripts/smoke/sonic_planner_profile.py --batch 1024

Times `backends.sonic.SonicPlanner` at --batch (synchronized, after warm-up), then profiles one
call with torch.profiler and prints the top operators by CUDA time, the ONNX operator mix of
the port's plan, and any vmap fallback warnings. Writes `runs/<UTC>_sonic_planner_profile/`.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crowdmotion.backends.sonic import SonicPlanner, example_planner_inputs  # noqa: E402
from crowdmotion.core.rundir import RunDir  # noqa: E402


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--batch", type=int, default=1024)
  p.add_argument("--repeats", type=int, default=3)
  args = p.parse_args()
  with RunDir.create("sonic_planner_profile", record={"kind": "sonic_planner_profile", **vars(args)}) as rd:
    planner = SonicPlanner(device="cuda")
    ex = example_planner_inputs()
    inputs = {k: torch.from_numpy(v).cuda()[None].expand(args.batch, *v.shape).contiguous() for k, v in ex.items()}
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      planner(inputs)
    fallbacks = sorted({str(w.message)[:200] for w in caught})
    times = []
    for _ in range(args.repeats):
      t = time.perf_counter()
      planner(inputs)
      times.append(time.perf_counter() - t)
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
      planner(inputs)
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
    shapes = prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=25)
    print(shapes, flush=True)
    (rd.path / "profile_shapes.txt").write_text(shapes)
    print(table, flush=True)
    (rd.path / "profile.txt").write_text(table)
    ops = Counter(op for op, *_ in planner.port.plan)
    result = {"batch": args.batch, "seconds": times, "plans_per_s": args.batch / min(times),
              "plan_ops": len(planner.port.plan), "op_mix": ops.most_common(25), "vmap_warnings": fallbacks,
              "gpu": torch.cuda.get_device_name(0)}
    rd.write_json("result.json", result)
    print(result, flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
