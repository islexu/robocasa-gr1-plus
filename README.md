# RoboCasa-GR1-Plus

Robustness shifts for the **24 official RoboCasa GR1 tabletop tasks**, with the same
**calibration → activation patching → layer selection → online adaptation → reporting**
pipeline used by [LoTTA](https://github.com/islexu/LoTTA) on LIBERO-Plus.

This branch defaults to the **v0.2 protocol: seven dimensions and 20 explicit subtypes**.
This is an independent
extension of [RoboCasa GR1](https://github.com/robocasa/robocasa-gr1-tabletop-tasks),
inspired by [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus).
**The real GR1 task suite, trained checkpoint and H200 policy rollouts have not been run
in the authoring session. Tests of a small MuJoCo renderer fixture are not GR1 benchmark results.**

## Protocol v0.2

| Dimension | Subtypes | Five GR1 strength presets, S1 → S5 |
| --- | --- | --- |
| Camera | Distance, orbit, orientation | Distance ×1.05/1.10/1.20/1.35/1.50; orbit parameter norm 5/10/15/25/35°; orientation 2/4/6/8/10° |
| Background | Scene texture, tabletop texture | Authored bitmap patterns, contrast .2/.4/.6/.8/1.0 |
| Light | Diffuse RGB, direction, specular, shadows | Separate renderer controls; shadows are categorical |
| Noise | Motion blur, Gaussian blur, zoom blur, fog, glass blur | Five explicit parameter recipes per subtype |
| Language | Action paraphrase, irrelevant context, commonsense descriptors | Categorical authored transformations; keep entity nouns and action order |
| Robot initial state | Joint vector over both seven-joint arms | Total 14-joint L2 radius .05/.10/.15/.20/.25 rad |
| Object layout | Target-only pose, physical clutter | Target XY 1/2/3/4/5 cm and yaw 3/6/10/15/20°; clutter count 1/1/2/3/4 |

See [the exact parameter table and design differences](docs/protocol-v2.md).
Run `gr1-plus catalogue` to print all machine-readable recipes. These are **strength presets,
not empirically calibrated difficulty levels**. Bitmap patterns are generated locally;
clutter uses collidable primitives. Neither reproduces LIBERO-Plus's external asset diversity.
Target pose shifts must retain positive source-receptacle/support predicates. Robot/layout
poses are checked for joint limits, new deep penetration, finite state and initial non-success.
Invalid cases roll back and are reported separately; reachability remains unverified.

Pure visual calibration uses nominal and shifted observations from **one simulation state**.
No clean policy rollout, expert action, reward, or future frame is used by online adaptation.
Paired observations are unavailable in development/confirmation episodes.

The default pilot splits base tasks by seed 42 into **6 calibration / 6 development /
12 confirmation** tasks. Different splits share no base task. Each development and
confirmation task includes clean plus **every subtype**, one realization and one trial:

| Split | Environment episodes | Policy rollouts |
| --- | --- | --- |
| Calibration | 48 (3 camera + 5 noise per task) | 48 Frozen |
| Development | 126 | 378: Frozen / Random-LoRA / LoTTA |
| Confirmation | 252 | 756: Frozen / Random-LoRA / LoTTA |

These are pilot counts, not full-benchmark confidence estimates. Default strength is S1.
For a larger fixed plan use `GR1_PLUS_SEVERITIES=1,3,5 GR1_PLUS_TRIALS=10` and a new output
directory. `--variants` requests additional realizations where meaningful; fixed recipes
are emitted once per strength and authored language templates are capped at three.
Never tune on confirmation tasks.

The v0.1 recipes remain accessible with `gr1-plus plan --protocol v1`. Keep any running
v0.1 evaluation on its original checkout and output directory: **clone this branch separately**
and generate a fresh v0.2 plan. Source fingerprints intentionally reject a stale manifest.

## Installation

Use a **separate conda environment** because LIBERO and GR1 have different simulator dependencies.
Keep the existing LIBERO environment and jobs in place.

```bash
conda create -n lotta-gr1-v2 python=3.10 -y
conda activate lotta-gr1-v2
git clone --branch refine/robocasa-gr1-v2 https://github.com/islexu/robocasa-gr1-plus.git robocasa-gr1-plus-v2
cd robocasa-gr1-plus-v2
bash scripts/install_sim.sh
python external/robocasa-gr1-tabletop-tasks/robocasa/scripts/download_tabletop_assets.py -y
```

Install the LoTTA branch containing the GR1 interface into this environment (an existing
LoTTA checkout can supply its source without changing the environment of an active process):

```bash
git clone --branch robocasa-gr1-plus https://github.com/islexu/LoTTA.git ../LoTTA-gr1
cd ../LoTTA-gr1
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[model,test]'
cd ../robocasa-gr1-plus-v2
```

The complete Isaac-GR00T training package and FlashAttention are not needed by this DiT4DiT
evaluation path; the official GR1 simulator, its assets, and LoTTA's model dependencies are.

## Checkpoint

Use the **RoboCasa-GR1** checkpoint, its `config.yaml` and normalization statistics;
the LIBERO checkpoint has the wrong action space.

```bash
hf download mondo-robotics/dit4dit-model --include 'dit4dit_robocasa_gr1/*' \
  --local-dir /home/jovyan/checkpoints/dit4dit-model
```

Reuse your local Cosmos-Predict2.5-2B base. Set the exact `.pt` path in
`scripts/paths.local.sh`; if normalization has multiple keys, explicitly set the GR1 key in
`configs/lotta_gr1_plus.yaml`. The backend rejects statistics with other than 29 actions.
Checkpoint/source hashes are stored with the result rather than copying weights into Git.

## Run the same pipeline

```bash
cp scripts/paths.example.sh scripts/paths.local.sh
# Edit paths, the GR1 checkpoint and free GPU IDs.
bash scripts/run_lotta.sh plan
bash scripts/run_lotta.sh doctor
bash scripts/run_lotta.sh pilot
# After fixing all settings on development tasks:
bash scripts/run_lotta.sh confirmation
```

`pilot` validates calibration resets and image pairs, collects observed trajectories, tests
one actual adapter update and exact reset, runs activation patching, selects a two-layer band,
validates development resets, and evaluates all three methods. `confirmation` uses the fixed
selection. `collect`, `validate`, and `development` can also be run separately.

For a **frozen checkpoint audit on all 24 tasks**, use a different output directory:

```bash
export GR1_PLUS_OUTPUT=/home/jovyan/robocasa-gr1-plus-v2/runs/checkpoint-v2
export GR1_PLUS_TRIALS=50
bash scripts/run_lotta.sh checkpoint
```

This uses `configs/gr1_checkpoint_v2.yaml`, supplies no calibration pairs/selection, and
evaluates 24 × 21 × 50 = **25,200 episodes** at the default S1. It is separate from the
task-disjoint LoTTA experiment. Set `GR1_PLUS_OUTPUT` back to the pilot directory to run LoTTA.

The launcher starts one process per selected GPU, uses no DDP/NCCL, and resumes completed
streams. GPU IDs are explicit. Busy GPUs fail the preflight check unless you deliberately set
`GR1_PLUS_ALLOW_BUSY=1`; the launcher never terminates existing jobs. For concurrent LIBERO
work, select genuinely free GPUs, or wait until a GPU becomes free before launching.

Defaults: horizon 720 control steps at the simulator's 20 Hz, execute 12 predicted actions
per inference, rank 4, alpha 4, learning rate 1e-5, one video-loss update, width 2, episodic
reset. All three methods use the same task/initial-state/corruption seeds. The official GR1
reset settles the scene internally; there is no LIBERO dummy-gripper settling sequence.

## Outputs and validation

`plan/` fixes episode identities and task splits; existing files cannot be replaced by a
different plan. `validation/` saves reset checks, pair proofs and images. `calibration/`
contains trajectories, patch scores, `smoke.json` and `selected.json`. Development and
confirmation store per-episode success, updates, timing, memory and the existing LoTTA reports.
Runtime errors and `InvalidPerturbation` cases are separate from policy failures.
`coverage.json` reports planned/completed/success/error/invalid/pending counts per method,
dimension, subtype and strength. Rates use completed valid episodes; partial coverage
remains explicit. Fixed-dimension macro rates accompany pooled rates because the dimensions
contain different numbers of subtypes. A recovered retry is counted once.

```bash
gr1-plus summarize --manifest runs/pilot-v2/plan/development.jsonl \
  --results runs/pilot-v2/development/evaluation --output runs/pilot-v2/development/coverage.json
```

```bash
python -m pytest
# Optional real MuJoCo renderer fixture checks:
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m pytest tests/test_mujoco_pair.py
```

See [docs/validation.md](docs/validation.md) for the measured authoring-session status and
[upstream.lock.json](upstream.lock.json) for pinned versions. Simulator assets remain under
their upstream licenses and are downloaded separately.
