# Data and checkpoints

Nothing large is tracked in git. Three kinds of data live outside it:

| Kind | Where | Fetched by |
|---|---|---|
| Third-party sources and assets (Kimodo, ARDY, SOMA-X) | `third_party/` | `scripts/bootstrap.sh` ([third-party sources](third-party.md)) |
| Model checkpoints (ARDY, Kimodo) | `checkpoints/` (or `$CROWD_CHECKPOINTS`) | `scripts/fetch_artifacts.sh --checkpoints` |
| Run artifacts (the saved motion graph, the motion library) | `runs/` | `scripts/fetch_artifacts.sh` |

## Checkpoints

`source scripts/env.sh` sets two directories under `$CROWD_CHECKPOINTS` (default
`checkpoints/` in the repository; on the cluster that is a symlink to the shared
checkpoint folder):

| Variable | Default | Read by |
|---|---|---|
| `CHECKPOINTS_DIR` | `$CROWD_CHECKPOINTS/ARDY` | ARDY's `load_model` |
| `CHECKPOINT_DIR` | `$CROWD_CHECKPOINTS/kimodo` | Kimodo's `load_model` |

Both loaders resolve `<dir>/<model name>`. The `[checkpoint.*]` tables of
[`artifacts.toml`](../artifacts.toml) list each model, its Hugging Face repository and
the revision used on the cluster:

| Name (directory) | Hugging Face repository | Size | Used for |
|---|---|---|---|
| `ARDY-Core-RP-20FPS-Horizon40` | `nvidia/ARDY-Core-RP-20FPS-Horizon40` | 0.73 GB | Library generation (default), benchmarks |
| `ARDY-Core-RP-20FPS-Horizon8` | `nvidia/ARDY-Core-RP-20FPS-Horizon8` | 0.73 GB | Benchmarks (short-horizon comparison) |
| `Kimodo-SOMA-RP-v1.1` | `nvidia/Kimodo-SOMA-RP-v1.1` | 1.1 GB | Benchmarks; a possible future library generator |

```bash
scripts/fetch_artifacts.sh --checkpoints     # downloads any that are missing
```

`env.sh` sets `HF_HUB_OFFLINE=1` so that nothing downloads at run time; the fetch script
lifts it for its own downloads. If a download is refused, open the model page, accept
its license, and log in with `.venv/bin/hf auth login`.

**Copying from the cluster instead.** The cluster keeps the same files under display-name
symlinks; copy them with `-L` so the links are followed:

```bash
mkdir -p checkpoints/ARDY checkpoints/kimodo
rsync -aL <user>@<cluster>:/coc/flash7/jni66/checkpoints/ARDY/ARDY-Core-RP-20FPS-Horizon40 checkpoints/ARDY/
rsync -aL <user>@<cluster>:/coc/flash7/jni66/checkpoints/kimodo/Kimodo-SOMA-RP-v1.1 checkpoints/kimodo/
```

Checkpoints are needed only to generate motion: `crowd library generate`,
`crowd bench throughput` and `crowd bench quality` for online cases, the GPU adapter
tests and the slow CPU goldens. Behavior work on a saved graph needs none.

Do not load the ARDY `soma` nickname: no SOMA checkpoint of ARDY has been released.

## SOMA-X assets

`third_party/soma-x/` holds the SOMA-X v0.3.2 assets from the Hugging Face repository
`nvidia/soma-x` (body models, meshes at several levels of detail, rig, correctives),
for `py-soma-x==0.3.2`. `SOMA-X-HF-MANIFEST.json` lists each file with its sha256. They
are used for SOMA meshes (Kimodo's skeleton), not by the motion graph, whose humans are
capsules posed from ARDY Core joints.

## Run artifacts

[`artifacts.toml`](../artifacts.toml) lists outputs that other machines need. Each entry
records the path relative to the repository (inside `runs/`), the sha256 and the size.
`fetch_artifacts.sh` copies an entry with rsync to the same relative path and checks its
hash. A file that is missing or fails its hash (for example an interrupted copy) is
fetched again, resuming where it stopped; `--verify` only re-checks files already present.
`--checkpoints` always reruns `hf download`, which skips complete files and resumes the rest.

```bash
scripts/fetch_artifacts.sh --list
scripts/fetch_artifacts.sh --from <user>@<cluster>:/coc/flash7/jni66/crowd-control                   # defaults
scripts/fetch_artifacts.sh --from <user>@<cluster>:/coc/flash7/jni66/crowd-control library-ardy-h40  # by name
```

`--from` can also come from `$CROWD_ARTIFACT_SOURCE`. Small provenance files next to an
artifact (`run.json`, `graph.json`) are copied too when they exist.

| Artifact | Default | Size | What |
|---|---|---|---|
| `graph-default` | yes | 0.44 GB | The saved motion graph: k = 128 candidates, `configs/graph/default.json`, built from `library-ardy-h40`; 134,820 valid 0.4 s edges. Everything in behavior iteration and the mjlab demos reads it |
| `graph-rotations` | yes | 0.32 GB | Joint rotations for every frame of `graph-default` (`crowd graph rotations`, from `library-ardy-h40`). The mjlab plaza's rigid mesh humans need it; a load-time check rejects a file that does not match the graph |
| `library-ardy-h40` | no | 1.96 GB | The offline ARDY H40 library it was built from: 2,048 streams × 32 s (18.2 h) at 20 fps, Core skeleton. Needed only to rebuild the graph or audit library commands |
| `library-ardy-turns` | no | 0.98 GB | Supplementary ARDY H40 turning library (`configs/library/ardy_turns_a40.json`): 1,024 streams × 32 s (9.1 h) of in-place turns, arcs, walking and standing generated with body-heading targets ([library and graph](usage/library-and-graph.md#turning-commands-and-body-heading-targets)). Needed only to rebuild `graph-turns` |
| `graph-turns` | no | 0.66 GB | Opt-in merged graph over `library-ardy-h40` then `library-ardy-turns` (`configs/graph/default.json`; 213,608 valid edges). Select it with `CROWD_GRAPH` (or `--graph`) |
| `graph-turns-rotations` | no | 0.48 GB | Joint rotations for `graph-turns` (`crowd graph rotations` over both libraries, same order). Select with `CROWD_GRAPH_ROTATIONS` together with `graph-turns` |
| `pre-migration-snapshot` | no | 65 MB | Source tree before the 2026-10 reorganization, kept as a reference for bit-exact checks of generation on a GPU |

Commands that read the saved graph (`crowd behavior evaluate`, `crowd graph report`,
`crowd demo ...`) default to `$CROWD_GRAPH`, else the `graph-default` path from the
manifest (`crowdmotion.core.paths.default_graph`). Rigid mesh humans also read
`$CROWD_GRAPH_ROTATIONS`, else the `graph-rotations` path; a different graph needs its
own `crowd graph rotations --graph <graph.pt> --library <library.pt>`. Formats: `graph.pt` and `library.pt`
are described in [library and graph](usage/library-and-graph.md).

To publish a new artifact: add a `[artifact.<name>]` table with `path`, `sha256`
(`sha256sum <file>`), `bytes`, `default` and a one-line `description`.

## Text encoder

ARDY and Kimodo both condition on text through LLM2Vec built on the gated
`meta-llama/Meta-Llama-3-8B-Instruct`, which currently refuses this project's Hugging
Face account. Both adapters therefore run with empty prompts: zero features of the
pooled `[B, 1, 4096]` shape (`crowdmotion.backends.text`). Throughput does not depend on
prompt content, but walking style is uncontrolled. The project does not substitute an
ungated mirror of the encoder.
