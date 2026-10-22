# Validation status (2026-10-22, refine/libero-plus-v2)

v0.2 implements 20 named subtypes, five numeric strength presets (categorical language
and shadows use one), actual bitmap texture materials, five image degradations, all-14-arm-joint
vectors, target-only pose changes with source/support guards, and physical collidable clutter.
Plans serialize concrete parameters and separate strength from unmeasured difficulty.
The existing LoTTA environment interface and calibration/patching/selection/adaptation pipeline
remain in use. A separate frozen-only checkpoint plan covers all 24 tasks.

Measured locally: **114 tests passed** (64 benchmark + 50 LoTTA). This includes all original
77 regressions, 14 v0.2 visual subtype cases rendered with MuJoCo 3.2.6/EGL, a rotated
camera-parent transform, actual collidable clutter and 14-joint interventions, rollback on
lost relations/limits, isolated hard-reset XML hooks, and complete subtype/strength accounting.
CLI smoke checks cover default v0.2, explicit legacy v0.1, checkpoint planning, catalogue,
and pending-coverage summaries. Shell syntax checks pass. CI now installs MuJoCo and runs
the renderer fixtures, rather than skipping them when the dependency is absent.

These simulator tests use a small authored fixture, **not official GR1 task assets**.
No GR1 checkpoint/H200 rollout, success rate, full-manifest reset validation, semantic review
or empirical difficulty calibration was performed in this authoring session. Run validation
and inspect images on the target machine before a new v0.2 evaluation. Existing v0.1 jobs
must keep their original checkout and output. See [protocol-v2.md](protocol-v2.md).

## Historical v0.1 validation (2026-10-21)

Implemented: fixed 24-task catalogue; seven dimensions; six variants per dimension;
three strengths for all but categorical language; task-disjoint plans; replayable seeds;
source/manifest/checkpoint hashes; same-state visual pairing; GR1-specific sensor/state/action
contract; existing LoTTA collection, patching, selection, adaptation, reporting and resume.

Measured locally: **77 tests passed** (50 LoTTA + 27 benchmark tests). This includes
six tests using MuJoCo 3.2.6 and its EGL renderer with a small box/hinge scene. The tests
verify visible camera/background/light/noise changes, unchanged physical state and calibration-only
pair access. CPU tests additionally verify the official GR1 normalization/partwise sin-cos
formula, continuous hands, split separation, plan immutability, reset-proof requirements,
perturbation determinism, rollback, and all original LoTTA regressions.

Not measured: loading all official RoboCasa assets; GR1 task reset and reachability across
the manifest; the RoboCasa-GR1 DiT4DiT checkpoint; Cosmos video loss with those weights;
real LoRA gradients on H200; GPU memory/throughput; task success or LoTTA improvement.

Run `run_lotta.sh doctor` and `pilot` on the target machine. The pilot performs actual
simulator reset checks and a real one-update adapter smoke test before patching. An invalid
initial pose is an explicit validation error; it is not resampled or counted as policy failure.
Inspect saved validation images, particularly the layout/robot-state cases, before treating
this new protocol as a validated robustness benchmark. Confirmation remains task-disjoint.

There is no active job or remote launch in the authoring environment. Server access and
available GPU IDs were not connected to this session.
