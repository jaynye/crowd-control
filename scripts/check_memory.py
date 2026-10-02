"""Check the boundaries between docs, code and project memory.

    .venv/bin/python scripts/check_memory.py            # report (default)
    .venv/bin/python scripts/check_memory.py --check    # exit 1 on anything to act on

Docs and code are self-contained; memory may cite them, never the reverse. Memory is
tracked in git; `runs/` is not, so memory cites runs by ID and the run log resolves them.

* **docs:** human-facing docs (`docs/`, the root `README.md`) and code (`src/`,
  `crowdmotion/`, `scripts/`, `tests/`, `configs/`, `mjlab/src/`) never cite `memory/`
  and never name a specific run (`runs/<UTC stamp>_<name>`). Generic output paths such
  as `runs/<UTC>_name` are fine. `artifacts.toml`, a data manifest, may name runs; the
  memory tools themselves (`scripts/check_memory.py`, `scripts/index_runs.py`) may cite
  `memory/`. In code, a timestamp counts as a run only if it matches a logged run or a
  run directory (tests may use synthetic stamps); in docs, any concrete stamp counts.
* **runs:** every run ID cited in `memory/` has a row in `memory/log/runs.md`
  (a cited prefix such as `<stamp>_quality_` matches any logged run it starts). When
  `runs/` exists, every run directory on disk has a row too (`scripts/index_runs.py`).
* **experiments:** a `memory/experiments/` note is live while it is reachable from a
  `memory/core/` or `memory/decisions/` note, directly or through other live
  experiment notes. Index files (`README.md`) are not roots, since they list
  everything. A note with a `Status: running` line near its top is in flight.
  - **unreachable:** nothing live cites the note. Retire it by moving it (`git mv`) to
    `memory/archive/`, or add a core conclusion that cites it if it still matters.
  - **outward:** a file outside memory's core, decisions, experiments, journal, log and
    archive names an experiment file. Link the core conclusion instead, so that
    retiring a note never breaks it.
* **journal links:** core and decision notes never link a journal entry; link the
  promoted note instead.
* **links:** relative Markdown links in `AGENTS.md`, `README.md`, `docs/` and `memory/`
  resolve to an existing file or directory. Never link into `runs/`: it is absent on a
  fresh clone. Cite run paths in backticks.

An experiment reference is any mention of its file name (links, backticked paths).
Nothing is modified by this script. Adapted from `~/flash/hrc/scripts/check_memory.py`.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = 'memory/experiments'
RUN_LOG = 'memory/log/runs.md'
ROOT_TREES = ('memory/core', 'memory/decisions')
MAY_CITE = ('memory/core', 'memory/decisions', 'memory/experiments', 'memory/journal', 'memory/log',
            'memory/archive')
TOP_LEVEL_SKIPS = {'.git', '.venv', '.uv-python', '.cache', 'runs', 'logs', 'mjlab', 'kimodo', 'ardy',
                   'soma-x', 'fpo-control', 'third_party', 'checkpoints'}
SKIP_DIRECTORIES = {'__pycache__', '.pytest_cache', 'node_modules', 'expected'}
TEXT_SUFFIXES = {'.md', '.py', '.sh', '.sbatch', '.json', '.toml', '.txt', '.html', '.yaml', '.yml',
                 '.cfg', '.rst'}
DOC_TREES = ('docs', 'README.md')
CODE_TREES = ('src', 'crowdmotion', 'scripts', 'tests', 'configs', 'mjlab/src')
MEMORY_TOOLS = ('scripts/check_memory.py', 'scripts/index_runs.py')
RUNNING = re.compile(r'^\W*status\W*running\b', re.IGNORECASE | re.MULTILINE)
MEMORY_PATH = re.compile(r'(?<![\w-])(?:\.\./)*memory/(?:core|decisions|experiments|knowledge|journal|log|archive|README\.md)')
RUN_ID = re.compile(r'(?<![\w])(\d{8}T\d{6}Z)(_[A-Za-z0-9_.-]*)?')
LOGGED_ROW = re.compile(r'^\|\s*`(\d{8}T\d{6}Z_[A-Za-z0-9_.-]+)`\s*\|')
JOURNAL_ENTRY = re.compile(r'\]\((?:\.\./)*(?:journal/|memory/journal/)?(\d{4}-\d{2}-\d{2}-[\w.-]+\.md)')
LINK = re.compile(r'\]\(([^)\s#]+)(?:#[^)]*)?\)')


def _within(relative: str, trees) -> bool:
    return any(relative == tree or relative.startswith(tree + '/') for tree in trees)


def _text_files(root: Path, top: str, suffixes=TEXT_SUFFIXES):
    base = root / top
    paths = [base] if base.is_file() else sorted(base.rglob('*')) if base.is_dir() else []
    for path in paths:
        relative = path.relative_to(root)
        if path.is_file() and path.suffix.lower() in suffixes \
                and not any(part in SKIP_DIRECTORIES for part in relative.parts):
            yield path


def _lines(path: Path):
    try:
        return path.read_text(encoding='utf-8').splitlines()
    except (UnicodeDecodeError, OSError):
        return []


def docs_violations(root: Path, known_runs: set[str] = frozenset()):
    """Lines in docs or code that cite memory or a specific run."""
    targets = [(path, True) for top in DOC_TREES for path in _text_files(root, top)]
    targets += [(path, False) for top in CODE_TREES for path in _text_files(root, top)]
    for path, is_doc in targets:
        relative = path.relative_to(root).as_posix()
        if relative in MEMORY_TOOLS:
            continue
        for number, line in enumerate(_lines(path), 1):
            where = f'{relative}:{number}'
            if MEMORY_PATH.search(line):
                yield where, 'cites memory'
            for match in RUN_ID.finditer(line):
                cited = match.group(0)
                # Tests may use synthetic stamps; elsewhere any concrete run ID is a citation.
                synthetic_ok = relative.startswith('tests/') and not any(
                    run.startswith(cited) or cited.startswith(run) for run in known_runs)
                if is_doc or not synthetic_ok:
                    yield where, 'names a specific run'
                    break


def logged_runs(root: Path) -> set[str]:
    return {m.group(1) for line in _lines(root / RUN_LOG) if (m := LOGGED_ROW.match(line))}


def unlogged_citations(root: Path, logged: set[str]):
    """Run IDs (or ID prefixes) cited in memory that match no row of the run log."""
    for path in _text_files(root, 'memory', {'.md'}):
        relative = path.relative_to(root).as_posix()
        if relative == RUN_LOG:
            continue
        for number, line in enumerate(_lines(path), 1):
            for match in RUN_ID.finditer(line):
                cited = match.group(0)
                if not any(run.startswith(cited) for run in logged):
                    yield f'{relative}:{number}', cited


def unlogged_runs(root: Path, logged: set[str]):
    """Run directories on disk without a run-log row (none when `runs/` is absent)."""
    base = root / 'runs'
    if not base.is_dir():
        return
    for path in sorted(base.iterdir()):
        if path.is_dir() and re.fullmatch(r'\d{8}T\d{6}Z_[A-Za-z0-9_.-]+', path.name) \
                and path.name not in logged:
            yield path.name


def journal_links(root: Path):
    """Core or decision notes that link a dated journal entry."""
    for tree in ROOT_TREES:
        for path in _text_files(root, tree, {'.md'}):
            for number, line in enumerate(_lines(path), 1):
                for target in LINK.findall(line):
                    if 'journal/' in target and JOURNAL_ENTRY.search(f']({target})'):
                        yield f'{path.relative_to(root).as_posix()}:{number}', target


def experiment_files(root: Path) -> dict[str, Path]:
    directory = root / EXPERIMENTS
    return {path.name: path for path in sorted(directory.iterdir())
            if path.is_file() and path.name != 'README.md'} if directory.is_dir() else {}


def cited(text: str, names, exclude: str | None = None) -> set[str]:
    return {name for name in names if name != exclude
            and re.search(r'(?<![\w.-])' + re.escape(name) + r'(?![\w.-])', text)}


def roots(root: Path):
    for tree in ROOT_TREES:
        for path in sorted((root / tree).rglob('*.md')):
            if path.name != 'README.md':
                yield path


def reachability(root: Path = ROOT):
    """Return (experiment files, live name -> first citing file, running names)."""
    files = experiment_files(root)
    live: dict[str, str] = {}
    queue = deque()
    for path in roots(root):
        for name in cited(path.read_text(encoding='utf-8'), files):
            if name not in live:
                live[name] = path.relative_to(root).as_posix()
                queue.append(name)
    while queue:
        source = files[queue.popleft()]
        if source.suffix != '.md':
            continue
        for name in cited(source.read_text(encoding='utf-8'), files, exclude=source.name):
            if name not in live:
                live[name] = source.relative_to(root).as_posix()
                queue.append(name)
    running = {name for name, path in files.items() if path.suffix == '.md'
               and RUNNING.search('\n'.join(path.read_text(encoding='utf-8').splitlines()[:8]))}
    return files, live, running


def outward_references(root: Path, names):
    """Files outside memory's core/decisions/experiments/journal/log/archive that name an experiment."""
    # Prune large top-level trees before traversal; filtering after rglob still
    # walks every run, cache and third-party checkout.
    for top in sorted(root.iterdir()):
        if top.name in TOP_LEVEL_SKIPS or top.name.startswith('.venv'):
            continue
        for path in _outward_paths(root, top):
            posix = path.relative_to(root).as_posix()
            if _within(posix, MAY_CITE) or path.resolve() == Path(__file__).resolve():
                continue
            for number, line in enumerate(_lines(path), 1):
                for name in cited(line, names):
                    yield posix, number, name


def _outward_paths(root: Path, top: Path):
    for path in ([top] if top.is_file() else top.rglob('*')):
        relative = path.relative_to(root)
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES \
                or any(part in SKIP_DIRECTORIES for part in relative.parts):
            continue
        yield path


def broken_links(root: Path):
    """Relative Markdown links that point at nothing, or into `runs/`."""
    files = [root / 'AGENTS.md', root / 'README.md', *_text_files(root, 'docs', {'.md'}),
             *_text_files(root, 'memory', {'.md'})]
    for path in files:
        if not path.is_file():
            continue
        for number, line in enumerate(_lines(path), 1):
            for target in LINK.findall(line):
                if '://' in target or target.startswith('mailto:'):
                    continue
                resolved = (path.parent / target).resolve()
                where = f'{path.relative_to(root).as_posix()}:{number}'
                if resolved == (root / 'runs').resolve() or (root / 'runs').resolve() in resolved.parents:
                    yield where, f'{target} (links into runs/)'
                elif not resolved.exists():
                    yield where, target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--check', action='store_true', help='exit 1 on any finding')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()

    files, live, running = reachability(root)
    unreachable = [name for name in files if name not in live and name not in running]
    outward = list(outward_references(root, files))
    logged = logged_runs(root)
    on_disk_names = {p.name for p in (root / 'runs').iterdir()} if (root / 'runs').is_dir() else set()
    docs = list(docs_violations(root, logged | on_disk_names))
    links = list(broken_links(root))
    citations = list(unlogged_citations(root, logged))
    on_disk = list(unlogged_runs(root, logged))
    journal = list(journal_links(root))
    for name in files:
        if name in live:
            print(f'live         {name}  <- {live[name]}')
        elif name in running:
            print(f'running      {name}')
        else:
            print(f'unreachable  {name}')
    for posix, number, name in outward:
        print(f'outward      {posix}:{number} names {name}')
    for where, problem in docs:
        print(f'docs         {where} {problem}')
    for where, cited_id in citations:
        print(f'run-id       {where} cites {cited_id}, which has no row in {RUN_LOG}')
    for name in on_disk:
        print(f'unlogged     runs/{name} has no row in {RUN_LOG} (scripts/index_runs.py drafts one)')
    for where, target in journal:
        print(f'journal      {where} links journal entry {target}')
    for where, target in links:
        print(f'broken       {where} -> {target}')
    print(f'{len(live)} live, {len(running - set(live))} running, {len(unreachable)} unreachable, '
          f'{len(outward)} outward reference(s), {len(docs)} docs/code boundary finding(s), '
          f'{len(citations)} unlogged run citation(s), {len(on_disk)} unlogged run dir(s), '
          f'{len(journal)} journal link(s), {len(links)} broken link(s); {len(logged)} logged runs')
    findings = unreachable or outward or docs or links or citations or on_disk or journal
    return 1 if args.check and findings else 0


if __name__ == '__main__':
    sys.exit(main())
