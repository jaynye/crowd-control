"""Plot a saved corridor scan without running the simulator or opening a window.

  crowd demo corridor plot --scan-file <dir>/scans.npz [--capture -1] [--output scan.png]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tyro
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def plot(scan_file: Path, capture: int = -1, output: Path | None = None) -> None:
  """Plot XY and XZ returns, current humans, and their captured trajectories."""
  with np.load(scan_file, allow_pickle=False) as data:
    index = np.arange(len(data["time"]))[capture]
    ids = data["geom_ids"][index]
    valid = ids >= 0
    points = data["hit_pos_w"][index, valid]
    names = data["geom_names"][ids[valid]]
    human_hit = np.char.startswith(names, "human_")
    people = data["human_xyyaw"][index]
    scanner = data["sensor_pos_w"][index]
    figure = Figure(figsize=(12, 3.6), layout="constrained")
    FigureCanvasAgg(figure)
    xy, xz = figure.subplots(1, 2)
    for axis, dimension, label in ((xy, 1, "Y"), (xz, 2, "Z")):
      for mask, color, name, size in (
        (~human_hit, "#9199a5", "Room", 4),
        (human_hit, "#008a89", "Human surface", 16),
      ):
        axis.scatter(
          points[mask, 0],
          points[mask, dimension],
          c=color,
          s=size,
          label=name,
          alpha=0.7,
        )
      axis.scatter(
        scanner[0], scanner[dimension], c="#df7315", marker="*", s=140, label="Scanner"
      )
      axis.set(xlabel="X (m)", ylabel=f"{label} (m)", aspect="equal")
      axis.grid(alpha=0.2)
    for i, (x, y, yaw) in enumerate(people):
      history = data["human_xyyaw"][: index + 1, i]
      xy.plot(history[:, 0], history[:, 1], color="#237bb0", alpha=0.6)
      xy.annotate(str(i), (x, y), xytext=(4, 5), textcoords="offset points")
      xy.arrow(
        x,
        y,
        0.3 * np.cos(yaw),
        0.3 * np.sin(yaw),
        head_width=0.1,
        color="#237bb0",
        length_includes_head=True,
      )
    xy.set_title("Top view · human trajectories and heading")
    xz.set_title("Side view · returned surface points")
    xy.legend(loc="upper right")
    figure.suptitle(
      f"Crowd LiDAR · t={float(data['time'][index]):.2f}s · "
      f"{valid.sum()}/{len(ids)} returns · {human_hit.sum()} human hits"
    )
    output = output or scan_file.with_name("scan.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    print(output)


if __name__ == "__main__":
  tyro.cli(plot)
