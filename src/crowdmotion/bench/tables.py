"""Summarize throughput and quality runs as Markdown tables.

  crowd bench tables --throughput runs/<a> runs/<b> \
      --quality runs/<c> runs/<d> > summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

POLICIES = ("refill", "P0.2", "P0.5", "P1")


def load(run: Path) -> list[dict]:
  return [json.loads(line) for line in (run / "rows.jsonl").read_text().splitlines() if line.strip()]


def short(model: str) -> str:
  return {"ARDY-Core-RP-20FPS-Horizon8": "ARDY H8", "ARDY-Core-RP-20FPS-Horizon40": "ARDY H40",
          "Kimodo-SOMA-RP-v1.1": "Kimodo"}.get(model, model)


def fmt(x, nd=3):
  if x is None:
    return "–"
  if isinstance(x, float):
    return f"{x:.{nd}g}" if abs(x) < 1000 else f"{x:,.0f}"
  return str(x)


def setting(r: dict) -> str:
  if r["backend"] == "ardy":
    return (f"{r.get('precision')} {r.get('cfg_type', 'separated')} s{r.get('steps', 10)} "
            f"h{r.get('history', 4)} f{r.get('future', 0)}")
  post = " post" if r.get("post_processing") else ""
  return f"{r.get('precision')} s{r.get('steps')} {r['new_frames']}fr{post}"


def throughput_tables(rows: list[dict]) -> str:
  ok = [r for r in rows if r.get("status") == "ok"]
  out = ["### Batch scaling (continuation)\n",
         "| Backend | Setting | Batch | p50 wall s | p95 wall s | R_new | Peak GB | denoise share |",
         "|---|---|---|---|---|---|---|---|"]
  for r in sorted(ok, key=lambda r: (r["model"], setting(r), r["batch"])):
    if r.get("name") not in ("batch",) or r.get("workload", "continue") != "continue":
      continue
    share = r.get("stage_denoise", 0) / max(1e-9, r.get("stage_sum", 1))
    out.append(f"| {short(r['model'])} | {setting(r)} | {r['batch']} | {fmt(r['wall_p50'])} | "
               f"{fmt(r['wall_p95'])} | {fmt(r['R_new'])} | {fmt(r['peak_gb'], 2)} | {share:.0%} |")
  out += ["\n### Best configurations: break-even simulator speed S* (isolated, one A40)\n",
          "`S* = u · R_new / (4096 m)`. Columns are replan policies; each cell is m = 1 / 4 / 8.\n",
          "| Backend | Setting | Batch | R_new | " + " | ".join(POLICIES) + " |",
          "|---|---|---|---|" + "---|" * len(POLICIES)]
  best: dict = {}
  for r in ok:
    if r.get("workload", "continue") != "continue":
      continue
    key = short(r["model"])
    if key not in best or r["R_new"] > best[key]["R_new"]:
      best[key] = r
  for key, r in sorted(best.items()):
    cells = []
    for p in POLICIES:
      vals = [r.get(f"S_star_{p}_m{m}") for m in (1, 4, 8)]
      cells.append(" / ".join(fmt(v, 2) for v in vals))
    out.append(f"| {key} | {setting(r)} | {r['batch']} | {fmt(r['R_new'])} | " + " | ".join(cells) + " |")
  others = [r for r in ok if r.get("name") not in ("batch",)]
  if others:
    out += ["\n### Other sweeps\n",
            "| Case | Backend | Setting | Batch | Workload | p50 wall s | R_new | conditions s | encode s | denoise s | decode s | post s |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(others, key=lambda r: (r["name"], r["model"], setting(r), r["batch"])):
      out.append(
        f"| {r['name']} | {short(r['model'])} | {setting(r)}{' lib' if r.get('conditions') == 'library' else ''} | "
        f"{r['batch']} | {r.get('workload', 'continue')} | {fmt(r['wall_p50'])} | {fmt(r['R_new'])} | "
        f"{fmt(r.get('stage_conditions'))} | {fmt(r.get('stage_encode_history'))} | {fmt(r.get('stage_denoise'))} | "
        f"{fmt(r.get('stage_decode'))} | {fmt(r.get('stage_postprocess'))} |")
  bad = [r for r in rows if r.get("status") != "ok"]
  if bad:
    out.append(f"\n{len(bad)} configuration(s) failed: " + ", ".join(f"{short(r['model'])} b{r['batch']} {r.get('status')}" for r in bad))
  return "\n".join(out)


QUALITY_COLS = [
  ("path_err_mean_m", "path err m"), ("speed_err_mean_mps", "speed err m/s"),
  ("foot_skate_mean_mps", "skate m/s"), ("foot_skate_frac_gt_0p1", "skate>0.1"),
  ("hgate_skate_mean_mps", "h-gated skate m/s"), ("hgate_skate_frac_gt_0p1", "h-gated >0.1"),
  ("frac_frames_penetration_gt_2cm", "pen>2cm"), ("frac_frames_no_contact", "no contact"),
  ("seam_to_interior_step_ratio", "seam/int step"), ("seam_root_step_change_p95_m", "seam Δstep p95"),
  ("interior_root_step_change_p95_m", "int Δstep p95"),
]
RESPONSE = ("replan_wait_s", "stop_response_s", "start_response_s", "turn_response_s", "speed_response_s")


def quality_tables(rows: list[dict]) -> str:
  out = ["| Case | Scenario | Period | " + " | ".join(c[1] for c in QUALITY_COLS) + " | response median / p95 s (never) | fails |",
         "|---|---|---|" + "---|" * len(QUALITY_COLS) + "---|---|"]
  branch = []
  for r in rows:
    if r["scenario"] == "branch_check":
      branch.append(r)
      continue
    resp = []
    for key in RESPONSE:
      if f"{key}_median" in r:
        label = "wait" if key.startswith("replan") else key.split("_")[0]
        resp.append(f"{label} {fmt(r[f'{key}_median'], 2)}/{fmt(r[f'{key}_p95'], 2)}"
                    f"{' (' + format(r[f'{key}_never_frac'], '.0%') + ')' if r[f'{key}_never_frac'] else ''}")
    fails = r.get("failed_humans_nonfinite", 0) + r.get("failed_humans_path_gt_1m", 0)
    out.append(f"| {r['name']} | {r['scenario']} | {r['period']} | "
               + " | ".join(fmt(r.get(k), 2) for k, _ in QUALITY_COLS)
               + f" | {'; '.join(resp) or '–'} | {fails} |")
  if branch:
    out += ["\nBranch checks (same executed history, different seeds):\n",
            "| Case | history unchanged | same-seed max diff | branch final-pose spread mean / min m |",
            "|---|---|---|---|"]
    for r in branch:
      out.append(f"| {r['name']} | {r['history_unchanged']} | {fmt(r['same_seed_max_abs_diff'])} | "
                 f"{fmt(r['branch_final_pose_dist_mean_m'], 2)} / {fmt(r['branch_final_pose_dist_min_m'], 2)} |")
  return "\n".join(out)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  parser.add_argument("--throughput", nargs="*", type=Path, default=[])
  parser.add_argument("--quality", nargs="*", type=Path, default=[])
  args = parser.parse_args(argv)
  for run in args.throughput:
    print(f"## Throughput: `{run.name}`\n")
    print(throughput_tables(load(run)) + "\n")
  for run in args.quality:
    print(f"## Quality: `{run.name}`\n")
    print(quality_tables(load(run)) + "\n")


if __name__ == "__main__":
  main()
