# Robustness design and implementation references

Target: the original 24-task **RoboCasa-GR1** checkpoint domain. Protocol:
`robocasa-gr1-plus-v0.3`; branch `refine/robocasa-gr1-v3`. All referenced project pages
and source entry points below were checked on 2026-10-22. This is an independent
GR1 extension, not an official score on any referenced benchmark.

## What was adopted, and what was deliberately left separate

| Primary reference | Useful practice | GR1 implementation |
| --- | --- | --- |
| [RoboTwin 2.0 domain randomization](https://robotwin-platform.github.io/doc/usage/domain-randomization.html) and [code](https://github.com/RoboTwin-Platform/RoboTwin) | Explicit clutter, lighting, textures, camera and table-height controls; broad texture/object diversity | Real photo-texture pack; explicit renderer controls; six household clutter geometries. No SAPIEN assets, table-height mutation or robot replacement is imported. |
| [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO), [paper](https://arxiv.org/abs/2510.03827) | Feasible position perturbations; explicit object/language/task/environment interventions | Positive support/receptacle guards; separate physical stress profile. Task swaps, removed targets and contradictory instructions are excluded from same-task TTA. |
| [SimplerEnv](https://github.com/simpler-env/SimplerEnv) | Visual matching and aggregation across background/light/distractor variants | Report single shifts, preset compositions and asset strata separately. No claim that an unmatched kitchen variant predicts real-robot performance. |
| [RoboCasa365 overview](https://robocasa.ai/docs/build/html/introduction/overview.html) | Diverse assets/scenes and explicit benchmarking structure | Keep the pinned original GR1 simulator and task predicates; borrow the principle of separate data/asset splits. Do not replace the pretrained GR1 benchmark with the newer 365-task API. |
| [LIBERO-Plus](https://arxiv.org/html/2510.13626v2) | Controlled seven-dimension robustness analysis | Retain documented v0.2 parameter recipes, but do not transfer model-consensus difficulty labels to GR1. |
| [Poly Haven license](https://polyhaven.com/license), [API](https://polyhaven.com/our-api) | CC0 reusable textures with source metadata | Eight exact diffuse images, approximately 4.4 MB; SHA-256-locked, split by asset ID and declared material family, fetched once before evaluation. |

These are design adaptations, not claims of implementation equivalence. The RoboTwin
library is much larger than this compact photo pack. Household primitives are not its
scanned distractor corpus. No third-party benchmark source was copied into a new robot
interface and presented as GR1 validation.

## Matrix and asset isolation

`core` uses camera distance/orbit/orientation; scene/tabletop photo textures;
diffuse/direction/specular/shadow lighting; motion/Gaussian/zoom/fog/glass image degradation;
action paraphrase and irrelevant-context language. It preserves original physical
parameters and excludes all physical initial-state edits.

`full` adds commonsense descriptors, both-arm 14-joint vectors, target-only pose and
household clutter. It is a physical/semantic stress profile and must be inspected
separately. Source/goal containers and the task success predicate remain original.
Primitive household geometry IDs are disjoint between development and confirmation.

Four predeclared combinations extend both profiles. Unlike independently randomized
combinations, each component has the **same sampled parameters, nominal reset seed and
asset ID as its corresponding single-shift case**. This supports matched comparisons
between individual and composed changes. Compositions are never calibration pairs.

LoTTA splits six base tasks for calibration, six for development and twelve for fixed
confirmation. Texture IDs and declared material-family labels are also disjoint. Every
confirmation base task cycles all four held-out texture assets when evaluation trials
are at least four. Default one-trial plans are smoke pilots, not full asset coverage.
Variant count is capped for finite texture pools to avoid duplicate same-state renders.
These are holdouts from this extension's calibration/development data; they do not
establish that the assets were absent from Cosmos or DiT4DiT pretraining.

Calibration covers all four visual dimensions by default, not camera/noise alone.
`--calibration-trials` is separate from evaluation `--trials`. This changes the benchmark
calibration set; it does not modify LoTTA's aggregate patching/selection algorithm.
Any different calibration weighting or phase-specific band selection remains a method
experiment, outside this benchmark update.

## Direct policy contract

Both paths use the same `GR1PlusEnvironment`, `plan_id`, episode identity, 29-dimensional
absolute actions, image crop/padding and original `unlocked_waist: ` prefix. The standalone
client encodes each body part as `[sin(part), cos(part)]`, concatenates parts and pads to
64; action decoding uses the GR1 checkpoint's exact min/max/mask and preserves continuous
hands. Input images are resized with OpenCV `INTER_AREA` to 224 pixels.

The independent entry point uses the actual upstream websocket client/server protocol at
DiT4DiT commit `66a6f3a12e2c8740157c1e478795952c040d31dd`. Client/server source hashes and
checkpoint/config/statistics hashes are saved. The server accepts one client, resets RNGs
and VideoDiT caches per episode, and freezes all model weights. It preserves model dtypes
by default; explicit full-model bf16 casts get separate identities.

The pinned framework's `predict_action(examples, **kwargs)` does not pass client DDIM
kwargs into the action model. The extension therefore uses checkpoint sampling, just like
the existing LoTTA backend. It does not expose an ineffective DDIM-step knob. Actual
cross-path GPU prediction parity remains to be measured on the trained checkpoint.

Standalone frozen results and LoTTA runs should have separate output directories because
their complete backend fingerprints have different definitions. Use Frozen from the same
LoTTA run for the primary paired TTA comparison. The standalone path checks the original
model interface independently; it is not a second adaptation implementation.

## Preflight, validity and uncertainty

Preflight attempts every requested row twice with fresh native wrappers, compares initial
state and image hashes, checks visible effects, finite hold states, passive completion and
free-body drift beyond 3 cm. Source/asset hashes and simulator/image-library versions bind
the report to the actual manifest. Partial `--limit` smoke reports cannot pass the gate.

Mechanically invalid shifts roll back. No policy-success filtering, silent resampling,
case dropping or replacement of a failed asset download is performed. A failed full
profile must be investigated, not pooled with a cleaner subset under the same plan ID.
These checks do not prove reachability, target visibility under every trajectory, or
natural-language equivalence; inspect saved images and full-profile instructions.

Calibration clean/shifted visual samples share one physical state. No reference pairs,
expert actions, success signals or future frames are supplied to online adaptation in
development/confirmation. Validation diagnostic images are offline artifacts only.

Reports distinguish completed policy failures from inference/environment errors, invalid
perturbations and pending coverage. Headline single-dimension macro rates exclude
compositions, which get their own rate. Texture/subtype/strength tables and matched clean
drops expose concentration of failures. Paired gains require full matching coverage,
consistent reset/physics proofs and one backend fingerprint; confidence intervals resample
base-task clusters. Few clusters remain explicitly marked. Parameter strength S1–S5 is
not an empirically calibrated difficulty label.

## Remaining target-machine measurements

Official GR1 task/asset loading, reset coverage of the selected large manifest, trained
DiT4DiT server inference, cross-path prediction parity, real LoRA update gradients on GR1,
H200 memory/throughput and task success must be measured on the target machine. The
authoring-session fixture/socket tests do not establish any of those results.
