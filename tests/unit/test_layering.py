"""Package layering: imports inside `crowdmotion` only point to the same or a lower layer.

L0 core; L1 graph, backends, metrics, sim.scene, sim.capsules; L2 behavior;
L3 iterate, library, bench, sim.mjlab, viz, tracking; L4 cli. Only `sim.mjlab` may import
mjlab at module level, and only `backends` may import ardy or kimodo.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import crowdmotion

# Report-only while the migration moves modules into their layers; strict afterwards.
REPORT_ONLY = False
PKG = Path(crowdmotion.__file__).resolve().parent
LAYERS = {
  "core": 0,
  "graph": 1, "backends": 1, "metrics": 1, "sim": 1, "sim.scene": 1, "sim.capsules": 1,
  "behavior": 2,
  "iterate": 3, "library": 3, "bench": 3, "sim.mjlab": 3, "viz": 3, "tracking": 3,
  "cli": 4,
}


def layer_of(module: str) -> int | None:
  """Layer of a dotted module path below `crowdmotion` (longest matching prefix)."""
  parts = module.split(".")
  for n in range(len(parts), 0, -1):
    key = ".".join(parts[:n])
    if key in LAYERS:
      return LAYERS[key]
  return None


def modules():
  for path in sorted(PKG.rglob("*.py")):
    rel = path.relative_to(PKG).with_suffix("")
    name = ".".join(p for p in rel.parts if p != "__init__")
    yield name, ast.parse(path.read_text())


def imports(tree: ast.Module, top_level_only: bool):
  nodes = tree.body if top_level_only else ast.walk(tree)
  for node in nodes:
    if isinstance(node, ast.Import):
      yield from (a.name for a in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
      yield node.module


def violations() -> list[str]:
  out = []
  for name, tree in modules():
    own = layer_of(name)
    for target in imports(tree, top_level_only=False):
      if target.startswith("crowdmotion."):
        other = layer_of(target.removeprefix("crowdmotion."))
        if own is not None and other is not None and other > own:
          out.append(f"{name} (L{own}) imports {target} (L{other})")
    for target in imports(tree, top_level_only=True):
      top = target.split(".")[0]
      if top == "mjlab" and not name.startswith("sim.mjlab"):
        out.append(f"{name} imports mjlab at module level")
      if top in ("ardy", "kimodo") and not name.startswith("backends"):
        out.append(f"{name} imports {top} at module level")
  return out


def unlayered() -> list[str]:
  return [name for name, _ in modules() if name and layer_of(name) is None]


def _check(found: list[str]) -> None:
  if found and REPORT_ONLY:
    pytest.skip("report-only: " + "; ".join(found))
  assert found == []


def test_imports_point_down():
  _check(violations())


def test_every_module_has_a_layer():
  _check(unlayered())
