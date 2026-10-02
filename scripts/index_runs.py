"""Draft and check rows of the run log, `memory/log/runs.md`.

    .venv/bin/python scripts/index_runs.py            # draft rows for runs not yet logged
    .venv/bin/python scripts/index_runs.py --all      # draft rows for every run on disk
    .venv/bin/python scripts/index_runs.py --check    # exit 1 if a run on disk has no row
    .venv/bin/python scripts/index_runs.py --slurm    # Slurm job IDs named in run logs

`runs/` is local and not tracked in git; the run log is the tracked index. Each run
directory `runs/<UTC stamp>_<name>/` gets one row. Rows are drafted from the run's
`run.json` (and `session.json`, `summary.json`, file names); the outcome column is
written by hand from the experiment note or journal. `--check` passes when `runs/` is
absent, for example on a fresh clone.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = 'memory/log/runs.md'
RUN_ID = re.compile(r'^(\d{8}T\d{6}Z)_[A-Za-z0-9_.-]+$')
ROW_ID = re.compile(r'^\|\s*`(\d{8}T\d{6}Z_[A-Za-z0-9_.-]+)`\s*\|')
SLURM_LOG = re.compile(r'^(?P<name>.+)-(?P<job>\d{6,})\.out$')

# Kind and producing entry point, by run-name pattern (first match wins).
KINDS = (
    (r'pre_migration_snapshot', 'source snapshot', 'migration Phase 0 (tar, mjlab diff, goldens)'),
    (r'legacy_logs', 'legacy logs', 'former top-level `logs/` (smoke JSON, Slurm logs)'),
    (r'memory_archive', 'retired notes', 'memory retirement (pre-git)'),
    (r'behavior_iteration', 'behavior iteration session', 'behavior session tooling'),
    (r'behavior_report', 'graph audit report', 'graph behavior report'),
    (r'mjlab_graph_demo', 'mjlab demo video', 'mjlab graph demo'),
    (r'motion_graph', 'graph build', 'graph build'),
    (r'ardy_library', 'ARDY library', 'library generation'),
    (r'graph_speed', 'graph runtime speed', 'graph speed benchmark'),
    (r'graph_quality', 'graph quality rollouts', 'quality benchmark'),
    (r'quality', 'quality rollouts', 'quality benchmark'),
    (r'throughput|bench_smoke', 'throughput sweep', 'throughput benchmark'),
    (r'ardy_profile', 'ARDY profile', 'ARDY profiling'),
    (r'sim_speed', 'simulator speed', 'simulator speed benchmark'),
)


def run_dirs(root: Path) -> list[Path]:
    base = root / 'runs'
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir() and RUN_ID.match(p.name))


def logged_ids(root: Path) -> set[str]:
    path = root / LOG
    if not path.is_file():
        return set()
    ids = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        match = ROW_ID.match(line)
        if match:
            ids.add(match.group(1))
    return ids


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def _artifacts(root: Path) -> dict[str, str]:
    """Relative path -> 'name (sha256 prefix)' from artifacts.toml."""
    path = root / 'artifacts.toml'
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text(encoding='utf-8'))
    return {entry['path']: f"{name} (sha256 {entry['sha256'][:12]}…)"
            for name, entry in data.get('artifact', {}).items() if 'path' in entry}


def draft_row(root: Path, run: Path, artifacts: dict[str, str]) -> str:
    stamp = RUN_ID.match(run.name).group(1)
    date = datetime.strptime(stamp, '%Y%m%dT%H%M%SZ').strftime('%Y-%m-%d %H:%M')
    kind, command = 'unknown', ''
    for pattern, k, c in KINDS:
        if re.search(pattern, run.name):
            kind, command = k, c
            break
    record = _load(run / 'run.json') or _load(run / 'graph.json') or _load(run / 'session.json')
    env = record.get('environment') or {}
    config = record.get('config') if isinstance(record.get('config'), dict) else {}
    if config.get('name'):
        command += f"; config `{config['name']}`"
    job = env.get('slurm_job') or record.get('slurm_job') or '—'
    gpu = env.get('gpu') or ('CPU' if env else '—')
    if env.get('host') and env.get('gpu'):
        gpu += f" ({env['host'].split('.')[0]})"
    status = record.get('status')
    outcome = f'DRAFT: status {status}' if status else 'DRAFT'
    files = []
    for path, label in artifacts.items():
        if path.startswith(f'runs/{run.name}/'):
            files.append(label)
    media = sorted(p.name for p in run.iterdir() if p.suffix in ('.mp4', '.html'))
    if media:
        files.append(', '.join(media[:3]) + (' …' if len(media) > 3 else ''))
    return (f'| `{run.name}` | {date} | {kind} | {command} | {job} | {gpu} | {outcome} | '
            f'{"; ".join(files) or "—"} |')


def slurm_jobs(root: Path) -> list[tuple[str, str, str]]:
    """(job, name, path) for every `<name>-<job>.out` under a run's `slurm/` folder."""
    out = []
    for run in run_dirs(root):
        for path in sorted((run / 'slurm').glob('*.out')) if (run / 'slurm').is_dir() else []:
            match = SLURM_LOG.match(path.name)
            if match:
                out.append((match['job'], match['name'], path.relative_to(root).as_posix()))
    return sorted(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--check', action='store_true', help='exit 1 if a run on disk has no row')
    parser.add_argument('--all', action='store_true', help='draft rows for every run on disk')
    parser.add_argument('--slurm', action='store_true', help='list Slurm jobs named in run logs')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    runs = run_dirs(root)
    logged = logged_ids(root)
    if args.slurm:
        for job, name, path in slurm_jobs(root):
            print(f'{job}  {name}  {path}')
        return 0
    missing = [run for run in runs if run.name not in logged]
    if args.check:
        for run in missing:
            print(f'unlogged     runs/{run.name}')
        print(f'{len(runs)} run(s) on disk, {len(logged)} row(s) in {LOG}, {len(missing)} unlogged')
        return 1 if missing else 0
    artifacts = _artifacts(root)
    for run in runs if args.all else missing:
        print(draft_row(root, run, artifacts))
    return 0


if __name__ == '__main__':
    sys.exit(main())
