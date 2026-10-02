"""Self-contained HTML pages from the templates in `viz/templates/`."""

from __future__ import annotations

from pathlib import Path

TEMPLATES = Path(__file__).resolve().parent / "templates"


def render_template(name: str, **slots: str) -> str:
  """Template text with each `__KEY__` replaced by `slots["key"]`, in argument order."""
  text = (TEMPLATES / name).read_text()
  for key, value in slots.items():
    token = f"__{key.upper()}__"
    if token not in text:
      raise KeyError(f"{name} has no {token}")
    text = text.replace(token, value)
  return text
