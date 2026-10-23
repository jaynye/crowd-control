# Codex behavior iteration in a batch allocation

User request: launch the Codex evaluation/render/refinement loop automatically from
a Slurm batch job, avoiding interactive QoS restrictions and counting iteration time
only after the allocation starts. Prepare the launcher and submission instructions;
do not submit a job during this setup task.

Plan:
- [x] Check installed `codex exec`, authentication, existing session execution and
  official noninteractive-mode documentation.
- [x] Add a bounded `crowd iterate agent` command, a batch wrapper and a reusable
  behavior-refinement prompt. Keep all requests local to the allocation.
- [x] Verify lifecycle, timeout and stop handling with a fake Codex executable;
  update usage, core status/design, and retire this journal.
- [x] Run the memory check. No behavior rules, dependencies or goldens change.

Installed Codex CLI is 0.160.0 and reports saved ChatGPT authentication. Its exec
command supports `--approve-for-me`, workspace-write, JSON events and a final-message
file. No GPU allocation, agent inference or behavior experiment has been run.

Implemented the outer RunDir with a nested local session, a 2.75-hour default agent
deadline and three-hour nonpreemptible batch wrapper. The runner preserves prompt,
JSON events, stderr, final response and tracked diffs; handles STOP/signals/deadline;
and gives active request runners time to reap their separate process groups.

Validation: the initial launcher/session/CLI/layering selection passed 45 CPU tests;
after strengthening STOP cleanup, all 10 launcher tests passed, including a running
request in a separate process group. Bash syntax, Ruff, CLI help, diff whitespace and
the memory check passed. `sbatch --test-only scripts/slurm/iterate_codex.sbatch`
accepted the three-hour A40 request on `rl2-lab` with `short` QoS; no job was submitted.
The test-only output printed a tentative job number (4013098), not an active allocation.
No real inference, EGL rendering or behavior tuning has been verified by this task.

Durable setup and limits are in `docs/slurm.md`, `docs/usage/behavior-iteration.md`,
core status and the motion-library design note. The default refinement task is in
`configs/behavior/iterate_codex.txt`. The user can submit the prepared wrapper; no
commit or push was made. Retire this completed implementation journal.

**Correction (2026-10-04, later):** the validation above never ran the real
`codex exec` argument parser. The first submitted job (4013182,
`runs/20261004T171209Z_behavior_iteration`) exited in 0.5 s because Codex 0.160.0
rejects `--sandbox workspace-write` together with `--approve-for-me`. The fake Codex
accepted any argv and the preflight only searched `exec --help`. The launcher was
fixed, given a real-CLI preflight and a model/effort default (`gpt-6-astra`,
`xhigh`); see `docs/slurm.md` and the run log.
