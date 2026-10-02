"""What produced a result: hardware, software, source state and checkpoints.

With git, a run records the repository commit and, when the tree is dirty, the
diff (`RunDir` writes it next to `run.json`). Without git, `source_fingerprint`
still identifies the package source exactly.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import tomllib
from pathlib import Path

import torch

from crowdmotion.core.paths import ROOT


def file_sha256(path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _run(args: list[str], cwd: Path) -> str:
  try:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=30).stdout.strip()
  except (OSError, subprocess.TimeoutExpired):
    return ""


def git_state(path: Path) -> dict:
  """Commit and dirty-file count of the git work tree at `path` ("" outside git)."""
  if not path.is_dir():
    return {"commit": "", "dirty_files": 0}
  status = _run(["git", "-C", str(path), "status", "--porcelain"], path)
  return {
    "commit": _run(["git", "-C", str(path), "rev-parse", "HEAD"], path),
    "dirty_files": len([x for x in status.splitlines() if x]),
  }


def git_diff(path: Path = ROOT) -> str:
  """Uncommitted changes to tracked files (empty when clean or outside git)."""
  return _run(["git", "-C", str(path), "diff", "HEAD"], path)


def source_fingerprint() -> str:
  """sha256 over the package source (paths and contents), independent of git."""
  digest = hashlib.sha256()
  pkg = Path(__file__).resolve().parents[1]
  for path in sorted(p for p in pkg.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
    digest.update(str(path.relative_to(pkg.parents[1])).encode())
    digest.update(file_sha256(path).encode())
  return digest.hexdigest()


def third_party() -> dict:
  """Pinned and actual commits of the third-party clones, and mjlab's upstream base."""
  manifest = ROOT / "third_party.toml"
  if not manifest.exists():
    return {"error": f"{manifest} not found"}
  pins = tomllib.loads(manifest.read_text())
  out = {"mjlab": {"base_commit": pins["mjlab"]["base_commit"], "tracked_in": "repository"}}
  for name in ("kimodo", "ardy"):
    out[name] = {"pinned": pins[name]["commit"], **git_state(ROOT / pins[name]["path"])}
  out["soma-x"] = {"revision": pins["soma-x"]["revision"]}
  return out


def _hf_revision(model_dir: Path) -> str | None:
  meta = sorted((model_dir / ".cache" / "huggingface" / "download").glob("*.safetensors.metadata"))
  return meta[0].read_text().splitlines()[0] if meta else None


def checkpoint_revisions() -> dict:
  out = {}
  for env_var in ("CHECKPOINTS_DIR", "CHECKPOINT_DIR"):
    base = os.environ.get(env_var)
    if base and Path(base).is_dir():
      for child in sorted(Path(base).iterdir()):
        if child.is_dir() and not child.name.startswith("."):
          out[child.name] = _hf_revision(child.resolve())
  return out


def environment() -> dict:
  """Hardware, software and source provenance for a run record."""
  info: dict = {
    "host": platform.node(),
    "slurm_job": os.environ.get("SLURM_JOB_ID"),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
  }
  if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    info.update(
      gpu=props.name,
      gpu_count=torch.cuda.device_count(),
      gpu_mem_gb=round(props.total_memory / 1e9, 1),
      compute_capability=f"{props.major}.{props.minor}",
    )
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], ROOT).split()
    info["driver"] = driver[0] if driver else None
  info["repository"] = git_state(ROOT)
  info["source_fingerprint"] = source_fingerprint()
  info["third_party"] = third_party()
  info["checkpoint_revisions"] = checkpoint_revisions()
  return info
