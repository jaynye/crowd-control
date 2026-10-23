"""Stochastic motion graph over an offline library (the ARDY-tabulated graph).

A library holds N streams of T frames, cut into edges of L = `edge_frames`
frames. Stream frames [0, L) come from a fresh start and are not an edge.

- **Boundary** (graph node) `(s, k)`, k = 1..K with K = T / L: the last frame
  before edge k, `p = k·L - 1`. Its descriptor is the root-relative,
  heading-aligned joint positions and velocities at `p`.
- **Edge** `(s, k)`, k = 1..K-1: frames `[k·L, k·L + L)`. Its id is the id of
  its start boundary, `s·K + k - 1`; it ends at boundary id + 1.
- **Candidates** of a boundary: the outgoing edges of its `neighbors` nearest
  boundaries in descriptor space. A boundary's own successor is a candidate at
  distance 0, so following a stream reproduces the library exactly.

Modules:
- `config`: `GraphConfig`;
- `tables`: `MotionGraph` (descriptors, edge tables, persistence of `graph.pt`);
- `build`: the kNN candidate build, `crowd graph build` and multi-library loading
  (`load_libraries`);
- `costs`: candidate cost terms (`CostTerm`, `TERMS`);
- `runtime`: `GraphState`, `GraphRuntime` (select, correct, inertialize);
- `audit`: what the saved tables contain (`audit_graph`);
- `turns`: turning-radius coverage under a mask (`crowd graph turns`).

Coordinates follow ARDY: Y-up, meters, heading 0 faces +Z, positive yaw turns
toward +X. "Local" vectors are (lateral, forward) in a heading-aligned frame.
"""

from crowdmotion.graph.config import GraphConfig
from crowdmotion.graph.runtime import GraphRuntime, GraphState
from crowdmotion.graph.tables import MotionGraph

__all__ = ["GraphConfig", "GraphRuntime", "GraphState", "MotionGraph"]
