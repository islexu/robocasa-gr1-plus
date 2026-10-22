# LIBERO-Plus-inspired GR1 refinement

Protocol ID: `robocasa-gr1-plus-v0.2`. This is an experimental GR1 extension, not
the official LIBERO-Plus or RoboCasa evaluation protocol. Implementation reference:
[LIBERO-Plus paper, Appendix A](https://arxiv.org/html/2510.13626v2) and
[LIBERO-Plus source](https://github.com/sylvestf/LIBERO-plus/tree/4976dc30028e805ff8094b55501d532c48fec182).

The reference separates camera geometry, background textures, illumination controls,
image degradations, language variation, joint initialization and object layout. v0.1
mostly used a coarse representative per dimension. v0.2 exposes 20 named subtypes
with concrete parameters in every manifest row. It adapts the design to GR1's ego camera,
14 arm joints, 29 continuous actions, native instruction prefix and task predicates.

## Exact presets

Values are ordered S1/S2/S3/S4/S5. A sampled direction is shared across strengths for
the same task/trial/subtype/realization. Directions and per-frame corruptions use local
RNGs; they do not consume simulator/global NumPy RNG state.

| Dimension / subtype | Operation and presets |
| --- | --- |
| camera / distance | Scale world-space camera displacement about the counter-top center by 1.05/1.10/1.20/1.35/1.50. Keep orientation and FOV. |
| camera / orbit | Sample a unit 2-vector, scale by 5/10/15/25/35 degrees for yaw/elevation parameters; apply world-Z yaw then world-X elevation about the same pivot. Rotate camera orientation with the orbit. The actual net rotation is not exactly the parameter-vector norm. |
| camera / orientation | Sample a unit 3-vector, axis-angle norm 2/4/6/8/10 degrees. Apply in camera-local axes; position stays fixed. Parent-body transforms are respected for all camera subtypes. |
| background / scene_texture | Apply one generated 256×256 brick/wood/tile/fabric/marble/checker bitmap to `wall_` and `floor_` geoms. Contrast .2/.4/.6/.8/1.0 about mid-gray. |
| background / surface_texture | Same texture recipe, only `table_main_` geoms. Task objects and robot materials are preserved. |
| light / diffuse | Sample RGB tint in [.35,1], normalize its maximum to 1, multiply nominal diffuse by tint × .84/.68/.52/.36/.20. Ambient/specular stay fixed. |
| light / direction | Rotate each light's local direction by sampled axis-angle norm 10/20/30/45/60 degrees. |
| light / specular | Set `max(nominal specular, .05) × gain`, gain 1.5/2/3/4/6. The .05 floor prevents a zero-specular no-op. |
| light / shadow | Toggle each nominal `castshadow` bit. One categorical preset. |
| noise / motion_blur | Line kernel with radius 5/8/15/25/35 px, Gaussian line weights σ=2/3/6/12/20, seeded angle in [−45°,45°]. |
| noise / gaussian_blur | Gaussian σ=1/2/4/6/10 px. |
| noise / zoom_blur | Average original with central zoom crops; maximum factor 1.11/1.21/1.31/1.46/1.56, scale step .02. |
| noise / fog | Original five-octave interpolated noise field; density .5/1.5/2.5/3.5/5; octave decay 3/2.5/2/1.6/1.3. Alpha = clip(field×density/6 + density/12, 0, .95), blend toward white. |
| noise / glass_blur | Gaussian σ=.5/.9/1.3/1.8/2.5; local remap displacement ±1/2/3/4/5 px; iterations 3/3/2/2/1; final Gaussian pass. |
| language / paraphrase | Replace action verbs in a single pass: e.g. `pick → take`, `place → put`; preserve entity text, spatial prepositions and action order. Three authored recipes. |
| language / distraction | Add one irrelevant room/calendar/context sentence before the unchanged task. Three authored recipes. |
| language / commonsense | Enrich one entity noun with a bounded descriptor, e.g. `microwave → microwave used for heating food`. Keep the noun so object identity is explicit. One recipe, not a reasoning benchmark. |
| robot_state / joint_vector | Sample a unit vector over left/right shoulder pitch/roll/yaw, elbow pitch, wrist yaw/roll/pitch. Add vector × .05/.10/.15/.20/.25 rad. This is total 14-joint L2 radius, not a per-joint bound. Zero touched joint velocities; reset controllers after validation. |
| layout / target_pose | Move only `objects['obj']`: sampled XY direction × .01/.02/.03/.04/.05 m; world-Z yaw ±3/6/10/15/20 degrees. Source/goal containers stay fixed. |
| layout / clutter | Compile 1/1/2/3/4 extra collidable primitives, half-size .016/.022/.022/.022/.022 m, density 150 kg/m³; box/cylinder/sphere by realization index. Place on counter at fixed seeded grid candidates, ≥.10 m from other object centers. Group 1 makes them visible with the native wrapper's collision-mesh rendering disabled. No policy-dependent sampling. |

Full details, including every sampled angle, RGB multiplier, joint delta and texture seed,
are serialized under `variant.parameters`. Texture RGB hashes and actual clutter placements
are saved in environment provenance. `gr1-plus catalogue` prints recipes at sample seed 42;
real plans have separate sample seeds per task/trial/subtype/realization.

## Deliberate limits

Camera and joint ranges are conservative GR1 presets, not a literal copy of LIBERO's
geometry or single-arm joint magnitudes. Language is deterministic authored text, not the
reference's generated language corpus or reasoning instructions. The six original bitmap
patterns do not reproduce the reference's 950-texture pool; primitives do not reproduce its
416-class distractor pool. Fog/glass/motion implementations are independent approximations,
not bit-identical reproductions. Asset diversity and language ambiguity need separate study.

S1–S5 mean parameter strength only. The reference benchmark's model-consensus difficulty
labels cannot be transferred to GR1. Rows therefore set `difficulty: null` and
`difficulty_kind: not_empirically_calibrated`. No claim of calibrated difficulty or
improved policy robustness follows from these implementations.

## Validity and accounting

Visual interventions restore all changed renderer arrays in a `finally` block, including
light direction/shadows. Textures are compiled as additional materials; physics is unchanged.
Calibration renders clean/shifted observations at one simulation state. Development and
confirmation expose no paired reference to adaptation. Validation may save side-by-side
diagnostic images; those diagnostics are not policy inputs.

Robot/target changes check limits, finite state, new/worsened penetration, and initial
non-success. Target changes additionally retain every positive initial source-receptacle
and counter-contact predicate. Clutter uses fixed geometry rules and collision validation.
Invalid states restore the original simulation state and raise `InvalidPerturbation`.
There is no silent resampling, clipping, difficulty relabeling or counting invalid cases
as policy failures. Hold/reset checks do not establish reachability or semantic equivalence.

`gr1-plus validate` attempts each requested row, saves images and a reproducibility report,
and exits nonzero on any invalid case or runtime error. Missing named surfaces, unsupported
instructions and invisible visual changes are explicit failures. Inspect the full report
and images before launching checkpoint rollouts. `--limit` is only a smoke check, not full
validation coverage.

`gr1-plus summarize` rejects mixed plans, different model fingerprints, modified manifest provenance and duplicate
episode results. A completed retry supersedes its stale error artifact, with a recovery
count. Error, invalid and pending counts remain separate. Pooled rates and dimension-macro
rates are descriptive until `complete_valid_coverage` is true. Comparing methods also
requires the same episode set, model fingerprint and valid full coverage.

## Isolation from existing runs

Use a separate clone/worktree and output directory for this branch. Keep existing v0.1
jobs on their original source. `--protocol v1` retains legacy recipes, but regenerating
their manifests on new source produces new source hashes and episode IDs. Never append
v0.2 results to the ongoing 9,600-episode run or compare its partial headline rate directly
with the new subtype-weighted sample.

The LoTTA mode keeps disjoint 6/6/12 base-task splits. The checkpoint mode evaluates all
24 tasks, freezes the policy, and provides no calibration/selection; it must not be used
as held-out confirmation after tuning on those same tasks.
