# RoboCasa-GR1-Plus v0.3

A controlled robustness extension for the **24 original RoboCasa-GR1 tabletop tasks**,
usable with **original DiT4DiT** and **LoTTA**. Robot, task predicates, native ego-image
crop, language prefix and 29 continuous absolute joint actions remain GR1-specific.

This branch adds a pinned CC0 photo-texture pack, disjoint task/asset splits, fixed
compositions, complete preflight gating, an identified DiT4DiT websocket service/client,
and matched-episode reporting. See [design and research references](docs/robust-v3.md)
and [validation status](docs/validation.md). Official GR1 assets and trained checkpoint/H200
rollouts have not been executed in the authoring session; the tests use small MuJoCo
fixtures and the real upstream socket transport with a synthetic policy.

## Evaluation matrix

| Profile | Single-shift subtypes | Fixed compositions | Cases per task/trial including clean |
| --- | ---: | ---: | ---: |
| `core` (default) | 16: camera, background, light, noise, controlled language | 4 | 21 |
| `full` | 20: core plus commonsense language, 14-joint initialization, target pose, household clutter | 4 | 25 |

Core excludes physical initial-state interventions so the first evaluation focuses on
observation robustness with the original mechanics. Full is a separate stress profile,
requiring mechanics and semantic inspection. `--no-compositions` keeps only single shifts.
Five **strength presets** are supported; they are not calibrated difficulty levels.

The four compositions are camera orbit + tabletop texture; scene texture + diffuse light;
camera orientation + Gaussian blur; tabletop texture + fog. Each reuses exactly the same
component parameters as its matching single-shift case. Report their rates separately.

All original v0.2 numeric recipes remain documented in [protocol-v2.md](docs/protocol-v2.md).
The CLI defaults to v0.3; `--protocol v1` and `--protocol v2` retain legacy recipes.
Use a separate clone/environment/output from any active evaluation.

## Install in a working GR1 environment

```bash
git clone --branch refine/robocasa-gr1-v3 https://github.com/islexu/robocasa-gr1-plus.git robocasa-gr1-plus-v3
cd robocasa-gr1-plus-v3
python -m pip install -e '.[sim,client]'
export GR1_PLUS_UPSTREAM=/home/jovyan/robocasa-gr1-tabletop-tasks
export GR1_PLUS_CHECKPOINT=/home/jovyan/checkpoints/dit4dit-model/dit4dit_robocasa_gr1/checkpoints/YOUR_CHECKPOINT.pt
export LOTTA_UPSTREAM_ROOT=/home/jovyan/LoTTA/third_party/DiT4DiT
export GR1_PLUS_ASSET_CACHE=/home/jovyan/.cache/robocasa-gr1-plus/assets-v1
gr1-plus assets fetch
gr1-plus assets verify
```

For a fresh simulator environment, use Python 3.10/3.11, run `bash scripts/install_sim.sh`,
then download the official assets with
`python external/robocasa-gr1-tabletop-tasks/robocasa/scripts/download_tabletop_assets.py -y`.
The pinned GR1 simulator is `4840e671596f93ca03651524b9f72ffb1aadfeff`; keep its original
assets. Do not substitute RoboCasa365 or RoboTwin's simulator/embodiment.

Use the RoboCasa-GR1 DiT4DiT checkpoint with its `config.yaml` and
`dataset_statistics.json`, not LIBERO weights. Set the config's Cosmos base path to your
local Cosmos-Predict2.5-2B directory, as required by the original DiT4DiT loader.
The independent client needs only the simulator/client extras; the GPU server needs the
already-working DiT4DiT model environment. The [official DiT4DiT setup](https://github.com/Mondo-Robotics/DiT4DiT)
contains those model dependencies.

## Assets and reproducibility

The eight real 1K photo textures total **4,424,108 bytes**. Calibration uses wood/fabric;
development uses brick/concrete; confirmation uses marble/ceramic/plaster/leather.
Every file is pinned by SHA-256, byte count, source URL and CC0 license in
[the asset lock](src/robocasa_gr1_plus/data/textures.lock.json). **Powered by Poly Haven.**
See [asset licenses](ASSET_LICENSES.md).

Fetch once, then evaluations read the cache without downloading assets. `assets verify`
fails on missing/corrupt bytes; no procedural or network fallback silently replaces a photo.
A cache may be prepared elsewhere and copied to the same configured cache root. Runtime
materials are derived deterministically and their RGB hashes enter environment provenance.
Household clutter uses six original collidable geometry models, not scanned object meshes.

## Original DiT4DiT, without LoTTA

Start with a one-trial mechanics/connection pilot in a new output directory:

```bash
export GR1_PLUS_OUTPUT=$PWD/runs/dit4dit-v3-pilot
gr1-plus plan --mode checkpoint --profile core --trials 1 --output "$GR1_PLUS_OUTPUT/plan"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl gr1-plus validate \
  --root "$GR1_PLUS_UPSTREAM" --manifest "$GR1_PLUS_OUTPUT/plan/confirmation.jsonl" \
  --output "$GR1_PLUS_OUTPUT/validation"
```

Inspect `validation.json` and saved images. Start the model server in its working model
environment, explicitly selecting a free GPU:

```bash
CUDA_VISIBLE_DEVICES=2 gr1-plus serve-dit4dit \
  --dit4dit-root "$LOTTA_UPSTREAM_ROOT" --checkpoint "$GR1_PLUS_CHECKPOINT" --port 6398
```

Then evaluate from the simulator/client environment:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=2 gr1-plus eval-dit4dit \
  --dit4dit-root "$LOTTA_UPSTREAM_ROOT" --checkpoint "$GR1_PLUS_CHECKPOINT" --port 6398 \
  --root "$GR1_PLUS_UPSTREAM" --manifest "$GR1_PLUS_OUTPUT/plan/confirmation.jsonl" \
  --validation "$GR1_PLUS_OUTPUT/validation/validation.json" \
  --output "$GR1_PLUS_OUTPUT/evaluation" --resume
```

The service identifies checkpoint/config/statistics hashes, enforces one client, and resets
model caches and RNG per episode. It preserves model parameter dtypes by default, matching
LoTTA's loading convention; `--bf16` is an explicit full-model cast recorded in metadata.
Action sampling uses the checkpoint's action-model configuration. The pinned upstream ignores
client DDIM kwargs, so this client does not advertise an ineffective sampling override.
With multiple normalization keys, pass the same `--unnorm-key` to server and client.

For the fixed checkpoint audit, use a **new** output and `--trials 50`: core has
24 × 21 × 50 = **25,200 episodes** at S1. The full profile has 30,000. Run preflight on
that exact manifest; a small pilot report cannot authorize the larger evaluation.

## LoTTA pipeline and multi-GPU evaluation

Install the GR1 interface from your LoTTA checkout; the tested integration is the
[`robocasa-gr1-plus` branch](https://github.com/islexu/LoTTA/tree/robocasa-gr1-plus).
The existing `environment.kind: robocasa_gr1` factory imports this package automatically.
No LoTTA algorithm, intervention location or adapter site is changed in this benchmark repo.

```bash
cp scripts/paths.example.sh scripts/paths.local.sh
# Edit checkpoint/base/source paths, output and genuinely free physical GPU IDs.
bash scripts/run_lotta.sh plan
bash scripts/run_lotta.sh pilot
# After fixing all choices on development tasks:
bash scripts/run_lotta.sh confirmation
```

This retains collection → actual one-update smoke → activation patching → layer selection
→ Frozen / Random-LoRA / LoTTA evaluation. Defaults: rank 4, width 2, one update, LR 1e-5,
12 actions per inference, horizon 720 at the official 20 Hz. The process-per-GPU launcher
is retained; no DDP or interruption of other jobs is introduced.

V0.3 calibrates on **all four visual dimensions**, covering 14 subtypes on six base tasks.
Calibration and evaluation trial budgets are separate:

```bash
export GR1_PLUS_TRIALS=50
export GR1_PLUS_CALIBRATION_TRIALS=2
export GR1_PLUS_OUTPUT=$PWD/runs/lotta-v3-core-50
bash scripts/run_lotta.sh pilot
```

At S1/one realization, this gives 168 calibration episodes, 6,300 development episodes and
12,600 confirmation episodes per policy method. Confirmation shares neither base tasks nor
texture IDs/families with calibration/development. Use `GR1_PLUS_PROFILE=full` for a separate
physical stress plan. The selected band remains the result of LoTTA's existing aggregate
patching rule; broader calibration does not prove a universally optimal band.

## Validation and reporting

`validate` checks every requested reset twice, hash-identical initial images/states, visible
visual effects, finite hold states, passive task completion and free-object drift. Invalid
poses are rolled back; invalid/error/pending cases are never converted to policy failures
or silently dropped. Full-profile target-pose cases retain positive support/receptacle
relations. These checks do not prove reachability or language equivalence.

Both entry points require complete preflight coverage bound to the exact manifest, source
and simulator/image-library versions. `verify-validation` can check the report explicitly.
The direct runner resumes completed episodes and retains recovered error records.

```bash
gr1-plus summarize --manifest "$GR1_PLUS_OUTPUT/plan/confirmation.jsonl" \
  --results "$GR1_PLUS_OUTPUT/confirmation/evaluation" --output "$GR1_PLUS_OUTPUT/confirmation/coverage.json"
```

Reports separate clean, single-dimension and composition rates; subtype/strength/asset
coverage; matched clean-to-shift drops; and paired task-cluster bootstrap gains versus Frozen.
Mixed checkpoints, duplicate records, altered manifests or different initial states for the
same episode are rejected. Pooled rates use completed valid cases and always show missing
coverage. Compare complete matching episodes, not partial headline percentages.

```bash
python -m pip install -e '.[test,sim,client]'
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
DIT4DIT_TEST_ROOT="$LOTTA_UPSTREAM_ROOT" python -m pytest
```

The test suite downloads nothing implicitly. Prepare the asset pack first to execute all
photo-texture fixtures; set `DIT4DIT_TEST_ROOT` to execute the actual upstream websocket test.
GitHub CI prepares both explicitly. [Historical v0.2 README](docs/README-v2.md).
