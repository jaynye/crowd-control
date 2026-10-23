"""crowd-control: stochastic reactive human crowds for LiDAR-conditioned humanoid policies.

Layers (imports point down; `tests/unit/test_layering.py` enforces it):

  L4 cli                      the `crowd` command
  L3 iterate, library, bench, sim.mjlab, viz
  L2 behavior                 commands, planner, motion sources, CrowdController
  L1 graph, backends, metrics, sim.scene, sim.capsules, sim.body, sim.rays
  L0 core

Only `sim.mjlab` imports mjlab at module level, and only `backends` imports ARDY
or Kimodo. See docs/architecture.md.
"""
