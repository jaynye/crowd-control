"""The `crowd` command table points at importable modules with a `main`."""

from __future__ import annotations

import importlib

import pytest

from crowdmotion import cli

OPTIONAL = ("mjlab",)


@pytest.mark.parametrize("group,name", [(g, n) for g, cmds in cli.COMMANDS.items() for n in cmds])
def test_every_command_resolves(group, name):
  module = cli.COMMANDS[group][name][0]
  try:
    mod = importlib.import_module(module)
  except ModuleNotFoundError as exc:
    if exc.name and exc.name.split(".")[0] in OPTIONAL:
      pytest.skip(f"{module} needs {exc.name}")
    raise
  assert callable(mod.main)


def test_usage_lists_all_groups(capsys):
  assert cli.main(["--help"]) == 0
  out = capsys.readouterr().out
  assert all(group in out for group in cli.COMMANDS)
  assert cli.main(["nope"]) == 2
