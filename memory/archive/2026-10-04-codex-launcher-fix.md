# Codex launcher fix and model selection

User request (2026-10-04): find why the batch Codex iteration (job 4013182,
`runs/20261004T171209Z_behavior_iteration`) did no work, apply the fixes, and let the
model variant be chosen per launch with GPT-6 Astra on `xhigh` as the default. The user
relaunches manually after verification; do not submit jobs, commit or push.

Diagnosis: `codex exec` 0.160.0 rejects `--sandbox` together with `--approve-for-me`
(clap usage error, exit 2, 0.5 s, before any model request). The fake Codex accepted
any argv, and the preflight only searched `exec --help` for `--approve-for-me`.

Plan:
- [x] `crowd iterate agent`: drop `--sandbox`; one argv builder; `--model`
  (default `gpt-6-astra`) and `--effort` (default `xhigh`, passed as
  `-c model_reasoning_effort=...`); preflight the exact argv with empty stdin
  (expects "No prompt provided") and the model/effort against `codex debug models`;
  `--check` runs only the preflight.
- [x] Fake Codex enforces the flag conflict, empty-stdin exit and the catalog; tests for
  defaults, overrides, rejected model/effort/argv and `--check`.
- [x] Real-Codex `--check` on sky1 (no inference): defaults, an override, bad model
  and bad effort.
- [x] Docs (`slurm.md`, `cli.md`, sbatch header), status, motion-library note, dated
  correction on the archived setup journal, run-log row for the failed run.
- [x] Ruff, unit suite (one thread, 112 passed), memory check. Retired.

Sandbox observation (sky1, Ubuntu 24.04, `kernel.apparmor_restrict_unprivileged_userns=1`):
Codex's bubblewrap sandbox cannot start (`bwrap: loopback: Failed RTM_NEWADDR`,
`setting up uid map: Permission denied`). Past 0.160.0 `codex exec` sessions with
`approvals_reviewer = "auto_review"` ran commands by escalation (41 of 45 completed
in one hrc session), so commands run unsandboxed after automatic review. Voltron is
untested.

Results: the launcher tests (17) pass with the fake Codex. The deadline test now bounds
the run's `wall_s`: its launch-to-exit bound had included three slow CLI startups.
Real Codex 0.160.0 on sky1, `crowd iterate agent --check` (no inference, no run
directory): defaults (`gpt-6-astra`, `xhigh`) and `--model gpt-6-sol --effort max`
pass; `--model nope`, `--effort bogus` and `gpt-5.5` at `ultra` are refused with the
catalog's options; the old argv with `--sandbox` is refused (exit 2). Promoted to
`docs/slurm.md`, status, the motion-library note, the archived setup journal and the
run log. Not tested: a real agent turn, the compute node's sandbox behavior, EGL.
