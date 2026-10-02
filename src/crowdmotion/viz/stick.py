"""Render quality-benchmark clips to MP4 (stick figures + planned root path).

  crowd bench clips runs/<UTC>_quality/clips_*.npz

Each panel follows one human: skeleton (Y-up data drawn Z-up), the executed root
trail, and the path in force for upcoming frames. The panel title turns red on
frames where a new generation call starts, so seams can be inspected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def render(path: Path, out: Path | None = None, humans: int = 6, stride: int = 1, max_seconds: float | None = None) -> Path:
  data = np.load(path)
  joints = data["joints"][:humans]  # [N, T, J, 3], Y-up
  planned = data["planned"][:humans]  # [N, T, 2] XZ
  contacts = data["contacts"][:humans]
  parents = data["parents"]
  fps = float(data["fps"])
  boundaries = set(int(b) for b in data["boundaries"])
  n, frames = joints.shape[:2]
  if max_seconds:
    frames = min(frames, int(max_seconds * fps))
  out = out or path.with_suffix(".mp4")
  cols = min(3, n)
  rows = int(np.ceil(n / cols))
  fig = plt.figure(figsize=(4 * cols, 4 * rows), dpi=80)
  axes, bones, trails, paths, titles, feet = [], [], [], [], [], []
  root = joints[:, :, 0]
  for i in range(n):
    ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
    ax.view_init(elev=15, azim=-60)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    bones.append([ax.plot([], [], [], color="tab:blue", lw=2)[0] for j in range(len(parents)) if parents[j] >= 0])
    trails.append(ax.plot([], [], [], color="tab:gray", lw=1)[0])
    paths.append(ax.plot([], [], [], color="tab:orange", lw=1, ls="--")[0])
    feet.append(ax.scatter([], [], [], color="tab:purple", s=12))
    titles.append(ax.set_title(f"human {i}", fontsize=9))
    axes.append(ax)
  child = [j for j in range(len(parents)) if parents[j] >= 0]
  foot_idx = None
  names = list(data["joint_names"])
  for cand in (["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"],
               ["left_ankle", "left_foot", "right_ankle", "right_foot"]):
    if all(c in names for c in cand):
      foot_idx = [names.index(c) for c in cand]
  frames_out = []
  for t in range(0, frames, stride):
    for i, ax in enumerate(axes):
      cx, cz = root[i, t, 0], root[i, t, 2]
      ax.set_xlim(cx - 1.5, cx + 1.5)
      ax.set_ylim(cz - 1.5, cz + 1.5)
      ax.set_zlim(0, 2.2)
      for line, j in zip(bones[i], child):
        p, c = joints[i, t, parents[j]], joints[i, t, j]
        line.set_data_3d([p[0], c[0]], [p[2], c[2]], [p[1], c[1]])
      lo = max(0, t - int(3 * fps))
      trails[i].set_data_3d(root[i, lo : t + 1, 0], root[i, lo : t + 1, 2], np.zeros(t + 1 - lo))
      hi = min(frames, t + int(1.5 * fps))
      paths[i].set_data_3d(planned[i, t:hi, 0], planned[i, t:hi, 1], np.zeros(hi - t))
      if foot_idx is not None:
        on = contacts[i, t]
        pts = joints[i, t, foot_idx][on[: len(foot_idx)]]
        feet[i]._offsets3d = (pts[:, 0], pts[:, 2], pts[:, 1])
      seam = t in boundaries
      titles[i].set_text(f"human {i}  t={t / fps:5.2f}s" + ("  NEW CALL" if seam else ""))
      titles[i].set_color("red" if seam else "black")
    fig.canvas.draw()
    frames_out.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
  plt.close(fig)
  imageio.mimwrite(out, frames_out, fps=fps / stride, macro_block_size=1)
  return out


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  parser.add_argument("clips", nargs="+", type=Path)
  parser.add_argument("--humans", type=int, default=6)
  parser.add_argument("--max-seconds", type=float, default=None)
  args = parser.parse_args(argv)
  for clip in args.clips:
    print(render(clip, humans=args.humans, max_seconds=args.max_seconds), flush=True)


if __name__ == "__main__":
  main()
