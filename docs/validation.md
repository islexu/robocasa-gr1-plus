# Validation status (2026-10-21)

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
