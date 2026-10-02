# Third-party sources

[`third_party.toml`](../third_party.toml) is the single source of URLs and pinned
revisions. [`scripts/bootstrap.sh`](../scripts/bootstrap.sh) fetches everything it
lists into `third_party/`, which git ignores. mjlab is the exception: it is our own
copy, tracked in `mjlab/`.

| Entry | What | Pin | How it is used |
|---|---|---|---|
| `kimodo` | [nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo): full-window diffusion motion model (SOMA skeleton, 30 fps) | commit `58e7818` | Installed into `.venv` as a non-editable path dependency |
| `ardy` | [nv-tlabs/ardy](https://github.com/nv-tlabs/ardy): autoregressive diffusion motion model (Core skeleton, 20 fps) | commit `693f74d` | Imported from source via `PYTHONPATH`; never installed |
| `soma-x` | [nvidia/soma-x](https://huggingface.co/nvidia/soma-x) on Hugging Face: SOMA body-model assets | tag `v0.3.2` (commit `c8ed728`) | Data directory for `py-soma-x==0.3.2` |
| `mjlab` | [mujocolab/mjlab](https://github.com/mujocolab/mjlab): GPU MuJoCo environments | base commit `c2e1e06` | **First-party**: our copy in `mjlab/`, a uv workspace member, installed editable |

Model weights are not third-party sources; they are listed in `artifacts.toml` and
fetched separately ([data](data.md)).

## kimodo

- `pyproject.toml` declares `kimodo = { path = "third_party/kimodo" }`. uv builds it
  non-editable, which compiles the `MotionCorrection` C++ extension (CMake and a C++
  compiler) and installs it as the `motion_correction` package.
- After changing anything in `third_party/kimodo`, rebuild with
  `scripts/bootstrap.sh --reinstall-kimodo` (it runs `uv sync --reinstall-package kimodo`
  with the CMake fallback; a bare `uv sync` needs `cmake` on `PATH`).
- **Never install `kimodo[demo]`.** It pulls a viser fork that conflicts with mjlab's
  viser.
- Its LLM2Vec text encoder needs the gated `meta-llama/Meta-Llama-3-8B-Instruct`. The
  adapter loads Kimodo with `crowdmotion.backends.text.ZeroTextEncoder` instead, which
  returns empty-prompt features ([data](data.md#text-encoder)).
- Wrapped by `crowdmotion.backends.kimodo.KimodoBackend`; do not edit the clone to
  change behavior. Put adaptations in the adapter.

## ardy

- ARDY pins `numpy<2` and `transformers==5.8.1`, which conflict with this environment,
  so it is **not** a dependency and is not in `uv.lock`. `scripts/env.sh` puts
  `third_party/ardy` on `PYTHONPATH`.
- The only runtime dependency it adds is `vector-quantize-pytorch`, declared in the root
  `pyproject.toml`.
- Its `motion_correction` extension is the same code that Kimodo installs, so ARDY
  reuses Kimodo's build. Do not build ARDY's extension into this environment.
- Wrapped by `crowdmotion.backends.ardy.ArdyBackend`. Do not load the `soma` model
  nickname: no SOMA checkpoint of ARDY has been released.

## soma-x

- `py-soma-x==0.3.2` comes from PyPI with the lock. Its assets come from Hugging Face:
  `bootstrap.sh` runs
  `hf download nvidia/soma-x --revision v0.3.2 --local-dir third_party/soma-x`.
- Pass the directory explicitly, for example
  `SOMALayer(data_root="third_party/soma-x", identity_model_type="soma", lod="xlo")`.
- `third_party/soma-x/SOMA-X-HF-MANIFEST.json` lists every file with its sha256.

## Upgrading a pin

1. Edit the commit (or revision) in `third_party.toml`.
2. Move the existing checkout:
   ```bash
   git -C third_party/kimodo fetch origin
   git -C third_party/kimodo checkout --detach <new commit>
   ```
   (or delete `third_party/<name>` and rerun `scripts/bootstrap.sh`). For SOMA-X,
   delete `third_party/soma-x` and rerun `bootstrap.sh`.
3. For kimodo: `scripts/bootstrap.sh --reinstall-kimodo`. If its dependencies changed, run
   `uv lock` first and read the `uv.lock` diff. Then confirm torch 2.9.0 and warp 1.14.0
   ([install](install.md#2-environment)).
4. Run the GPU adapter parity tests (`pytest tests/gpu`), which compare our adapters with
   upstream's own calls on the same GPU, and the real-model CPU goldens
   (`pytest -m slow tests/golden`). A new upstream revision may legitimately change
   generated motion; re-capturing goldens is a deliberate decision
   ([development](development.md#golden-outputs)).

## mjlab: our copy

`mjlab/` holds mjlab's source at upstream commit `c2e1e0640` plus our changes, tracked
as ordinary files in this repository. It is a uv workspace member (`[tool.uv.workspace]`
in `pyproject.toml`), installed editable, so edits take effect immediately.

- **Never run `uv sync` inside `mjlab/`.** It does an exact sync of the member and
  uninstalls Kimodo and py-soma-x from the shared `.venv`. Run it from the root.
  `uv run` inside `mjlab/` is fine.
- **Keep mjlab changes general** (simulator features) and record each in
  `mjlab/docs/source/changelog.rst` under "Upcoming version". Crowd-specific code (the
  plaza and corridor scenes, the crowd command terms, primitive humans) lives in
  `crowdmotion.sim.mjlab`, which keeps our diff small enough to merge upstream releases.
- **Our changes** (sensor features, documented in
  `mjlab/docs/source/sensors/raycast_sensor.rst`, tested in
  `mjlab/tests/test_lidar_pattern.py`):
  - `LidarPatternCfg`: an angular azimuth × elevation raster from one origin;
  - `RayCastSensorCfg.min_distance`: nearer returns report −1 and still occlude;
  - `RayCastData.geom_ids`: hit geom IDs for debugging (never a policy observation);
  - `RayCastSensorCfg.VizCfg.stride`: draw every Nth ray.
- Run mjlab's own tests from inside `mjlab/` with the shared environment:
  `cd mjlab && ../.venv/bin/python -m pytest tests/test_lidar_pattern.py`.

On the cluster, the clone's original git history is kept at
`/coc/flash7/jni66/mjlab-upstream.git`. To list our changes against the base:
`git --git-dir=/coc/flash7/jni66/mjlab-upstream.git --work-tree=mjlab diff --stat c2e1e0640`.

### Merging an upstream release

Apply upstream's changes between our base and the new release as a three-way patch:

```bash
git remote add mjlab-upstream https://github.com/mujocolab/mjlab.git   # once
git fetch mjlab-upstream --tags
base=c2e1e06400e309b6897a536693fe5d2aa772c5b2      # [mjlab] base_commit in third_party.toml
new=<release tag or commit>
git diff --binary "$base" "$new" > /tmp/mjlab-upstream.patch
git apply --3way --directory=mjlab /tmp/mjlab-upstream.patch
git status                                          # resolve any conflicts in mjlab/
```

Fetching the upstream remote brings the base blobs into this repository, so `--3way`
can merge files we changed. Then:

1. set `base_commit` in `third_party.toml` to the new commit;
2. if mjlab's dependencies changed, run `uv lock` and `uv sync` from the root, and
   confirm torch 2.9.0 and warp 1.14.0 (the root depends on plain `mjlab`, never
   `mjlab[cu128]`);
3. run mjlab's LiDAR tests, our unit and golden tests, and a demo
   ([development](development.md)). The `mjlab_plaza` and `corridor` goldens hash the
   compiled MuJoCo model, so an upstream change that alters it shows up there first.

If merges become frequent, `git subtree` (with `--prefix=mjlab`) can replace the patch
step; the recorded base commit is the starting point either way.
