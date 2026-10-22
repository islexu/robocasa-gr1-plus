#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$repo_root/scripts/paths.local.sh" ]]; then source "$repo_root/scripts/paths.local.sh"; fi
stage="${1:-pilot}"
case "$stage" in plan|doctor|validate|collect|pilot|development|confirmation|checkpoint) ;; *)
  printf '%s\n' 'Usage: bash scripts/run_lotta.sh [plan|doctor|validate|collect|pilot|development|confirmation|checkpoint]' >&2; exit 2;;
esac
protocol="${GR1_PLUS_PROTOCOL:-v3}"
default_run=pilot
if [[ "$stage" == checkpoint ]]; then default_run=checkpoint; fi
export GR1_PLUS_OUTPUT="${GR1_PLUS_OUTPUT:-$repo_root/runs/$default_run-$protocol}"
export GR1_PLUS_SELECTION="$GR1_PLUS_OUTPUT/calibration/selected.json"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
config="${GR1_PLUS_CONFIG:-$repo_root/configs/lotta_gr1_plus.yaml}"
if [[ "$protocol" == v3 ]]; then config="${GR1_PLUS_CONFIG:-$repo_root/configs/lotta_gr1_plus_v3.yaml}"; fi
mode=lotta
if [[ "$stage" == checkpoint ]]; then
  mode=checkpoint
  config="${GR1_PLUS_CONFIG:-$repo_root/configs/gr1_checkpoint_v2.yaml}"
  if [[ "$protocol" == v3 ]]; then config="${GR1_PLUS_CONFIG:-$repo_root/configs/gr1_checkpoint_v3.yaml}"; fi
fi
if [[ "$stage" == plan || "$stage" == pilot || "$stage" == collect || "$stage" == checkpoint ]]; then
  plan_options=()
  if [[ "$protocol" == v3 ]]; then
    plan_options+=(--profile "${GR1_PLUS_PROFILE:-core}" --calibration-trials "${GR1_PLUS_CALIBRATION_TRIALS:-1}")
  fi
  if [[ "${GR1_PLUS_COMPOSITIONS:-1}" == 0 ]]; then plan_options+=(--no-compositions); fi
  if [[ -n "${GR1_PLUS_CALIBRATION_DIMENSIONS:-}" ]]; then plan_options+=(--calibration-dimensions "$GR1_PLUS_CALIBRATION_DIMENSIONS"); fi
  gr1-plus plan --protocol "$protocol" --mode "$mode" --output "$GR1_PLUS_OUTPUT/plan" --variants "${GR1_PLUS_VARIANTS:-1}" \
    --severities "${GR1_PLUS_SEVERITIES:-1}" --trials "${GR1_PLUS_TRIALS:-1}" "${plan_options[@]}"
  if [[ "$stage" == plan ]]; then exit 0; fi
fi
: "${GR1_PLUS_UPSTREAM:?Set GR1_PLUS_UPSTREAM}"
if [[ "$protocol" == v3 ]]; then gr1-plus assets verify; fi
gr1-plus doctor --root "$GR1_PLUS_UPSTREAM"
: "${GR1_PLUS_DEVICES:?Explicitly select GPUs that are free from existing LIBERO jobs}"
IFS=',' read -ra devices <<< "$GR1_PLUS_DEVICES"
export MUJOCO_EGL_DEVICE_ID="${devices[0]}"
# Do not silently contend with existing training/evaluation processes.
python - "$GR1_PLUS_DEVICES" <<'PY'
import os, subprocess, sys
devices=sys.argv[1].split(',')
if any(not x.isdigit() for x in devices) or len(devices)!=len(set(devices)):
    raise SystemExit('GR1_PLUS_DEVICES must be distinct numeric physical GPU IDs')
for device in devices:
    result=subprocess.run(['nvidia-smi','-i',device,'--query-compute-apps=pid','--format=csv,noheader,nounits'],
                          check=True,text=True,capture_output=True)
    if result.stdout.strip() and os.getenv('GR1_PLUS_ALLOW_BUSY')!='1':
        raise SystemExit(f'GPU {device} already has compute processes; choose free GPUs or set GR1_PLUS_ALLOW_BUSY=1')
PY
if [[ "$stage" == validate ]]; then
  for split in calibration development confirmation; do
    gr1-plus validate --root "$GR1_PLUS_UPSTREAM" --manifest "$GR1_PLUS_OUTPUT/plan/$split.jsonl" \
      --output "$GR1_PLUS_OUTPUT/validation/$split"
    gr1-plus verify-validation --manifest "$GR1_PLUS_OUTPUT/plan/$split.jsonl" \
      --validation "$GR1_PLUS_OUTPUT/validation/$split/validation.json"
  done
  exit 0
fi
gr1-plus doctor --root "$GR1_PLUS_UPSTREAM" --require-model
CUDA_VISIBLE_DEVICES="${devices[0]}" LOTTA_DEVICE=cuda:0 lotta doctor --config "$config" --load-model
if [[ "$stage" == doctor ]]; then exit 0; fi
if [[ "$stage" == collect || "$stage" == pilot ]]; then
  gr1-plus validate --root "$GR1_PLUS_UPSTREAM" --manifest "$GR1_PLUS_OUTPUT/plan/calibration.jsonl" \
    --output "$GR1_PLUS_OUTPUT/validation/calibration"
  gr1-plus verify-validation --manifest "$GR1_PLUS_OUTPUT/plan/calibration.jsonl" \
    --validation "$GR1_PLUS_OUTPUT/validation/calibration/validation.json"
  lotta launch --config "$config" --manifest "$GR1_PLUS_OUTPUT/plan/calibration.jsonl" \
    --stage collect --devices "$GR1_PLUS_DEVICES" --output "$GR1_PLUS_OUTPUT/calibration/collection" --resume
  if [[ "$stage" == collect ]]; then exit 0; fi
  trajectory="$(python - "$GR1_PLUS_OUTPUT/calibration/collection" <<'PY'
from pathlib import Path
import sys
paths=sorted(Path(sys.argv[1]).glob('rank-*/trajectories/*.npz'))
if not paths: raise SystemExit('No calibration trajectories collected')
print(paths[0])
PY
)"
  CUDA_VISIBLE_DEVICES="${devices[0]}" LOTTA_DEVICE=cuda:0 lotta smoke --config "$config" \
    --trajectory "$trajectory" --output "$GR1_PLUS_OUTPUT/calibration/smoke.json"
  python "$repo_root/scripts/patch_devices.py" --config "$config" \
    --pairs "$GR1_PLUS_OUTPUT/calibration/collection" --output "$GR1_PLUS_OUTPUT/calibration/scores" \
    --devices "$GR1_PLUS_DEVICES"
  lotta select --scores "$GR1_PLUS_OUTPUT/calibration/scores" --output "$GR1_PLUS_SELECTION"
  split=development
elif [[ "$stage" == development || "$stage" == confirmation || "$stage" == checkpoint ]]; then
  split="$stage"
  if [[ "$stage" == checkpoint ]]; then split=confirmation; fi
else
  exit 2
fi
gr1-plus validate --root "$GR1_PLUS_UPSTREAM" --manifest "$GR1_PLUS_OUTPUT/plan/$split.jsonl" \
  --output "$GR1_PLUS_OUTPUT/validation/$split"
gr1-plus verify-validation --manifest "$GR1_PLUS_OUTPUT/plan/$split.jsonl" \
  --validation "$GR1_PLUS_OUTPUT/validation/$split/validation.json"
lotta launch --config "$config" --manifest "$GR1_PLUS_OUTPUT/plan/$split.jsonl" \
  --devices "$GR1_PLUS_DEVICES" --output "$GR1_PLUS_OUTPUT/$split/evaluation" --resume
lotta summarize --output "$GR1_PLUS_OUTPUT/$split/evaluation"
methods="$(python - "$config" <<'PY'
import sys,yaml
print(','.join(yaml.safe_load(open(sys.argv[1]))['methods']))
PY
)"
gr1-plus summarize --manifest "$GR1_PLUS_OUTPUT/plan/$split.jsonl" --methods "$methods" \
  --results "$GR1_PLUS_OUTPUT/$split/evaluation" --output "$GR1_PLUS_OUTPUT/$split/coverage.json"
