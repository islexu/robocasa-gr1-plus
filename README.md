# RoboCasa-GR1-Plus

Robustness shifts for the **24 official RoboCasa GR1 tabletop tasks**, with the same
**calibration → activation patching → layer selection → online adaptation → reporting**
pipeline used by [LoTTA](https://github.com/islexu/LoTTA) on LIBERO-Plus.

The benchmark implementation supports seven shift dimensions. This is an independent
extension of [RoboCasa GR1](https://github.com/robocasa/robocasa-gr1-tabletop-tasks),
inspired by [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus).
**The real GR1 task suite, trained checkpoint and H200 policy rollouts have not been run
in the authoring session. Tests of a small MuJoCo renderer fixture are not GR1 benchmark results.**

## Protocol

| Dimension | Six variants | Severity |
| --- | --- | --- |
| Camera | ±x / ±y / ±z in the ego camera's parent frame | 1.5 / 3 / 4.5 cm |
| Background | Six material/color palettes on table, wall and floor geoms | Blend 0.35 / 0.65 / 1.0 |
| Light | Warm / cool / green / magenta / dim / bright | Three intensity levels |
| Noise | Gaussian / impulse / Gaussian blur / motion blur / JPEG / pixelation | Three strengths |
| Language | Six instruction wrappers, retaining the original instruction and GR1 prefix | One categorical level |
| Robot initial state | Left / right / both elbow offsets, positive or negative | 0.03 / 0.06 / 0.09 rad |
| Object layout | Six planar translations of all free-joint objects as a group | 1 / 2 / 3 cm |

Background changes are material/color shifts, **not** LIBERO-Plus texture assets.
Layout shifts preserve object-to-object relative placements, while changing their placement
relative to the robot and fixed fixtures. Robot/layout poses are checked for joint limits,
new deep penetrations, finite state and initial non-success; these checks do not establish
reachability. Invalid variants fail explicitly and remain visible as errors.

Pure visual calibration uses nominal and shifted observations from **one simulation state**.
No clean policy rollout, expert action, reward, or future frame is used by online adaptation.
Paired observations are unavailable in development/confirmation episodes.

The default pilot splits base tasks by seed 42 into **6 calibration / 6 development /
12 confirmation** tasks. Different splits share no base task. Each development and
confirmation task includes clean plus all seven dimensions, one variant and one trial:

| Split | Environment episodes | Policy rollouts |
| --- | --- | --- |
| Calibration | 12 (camera + noise) | 12 Frozen |
| Development | 48 | 144: Frozen / Random-LoRA / LoTTA |
| Confirmation | 96 | 288: Frozen / Random-LoRA / LoTTA |

These are pilot counts, not full-benchmark confidence estimates. Language has six categorical
variants and one severity; other dimensions support six variants × three severities.
For a larger fixed plan use `GR1_PLUS_VARIANTS=6 GR1_PLUS_SEVERITIES=1,2,3 GR1_PLUS_TRIALS=10`
and a new output directory. Never tune on the confirmation tasks.

## Installation

Use a **separate conda environment** because LIBERO and GR1 have different simulator dependencies.
Keep the existing LIBERO environment and jobs in place.

```bash
conda create -n lotta-gr1 python=3.10 -y
conda activate lotta-gr1
git clone https://github.com/islexu/robocasa-gr1-plus.git
cd robocasa-gr1-plus
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
cd ../robocasa-gr1-plus
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
Failures are errors, not counted as ordinary unsuccessful episodes.

```bash
python -m pytest
# Optional real MuJoCo renderer fixture checks:
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m pytest tests/test_mujoco_pair.py
```

See [docs/validation.md](docs/validation.md) for the measured authoring-session status and
[upstream.lock.json](upstream.lock.json) for pinned versions. Simulator assets remain under
their upstream licenses and are downloaded separately.
