#!/usr/bin/env bash
set -euo pipefail
# Run in a separate activated Python 3.10/3.11 environment, not the active LIBERO environment.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
external="${GR1_PLUS_EXTERNAL:-$repo_root/external}"
mkdir -p "$external"
upstream="$external/robocasa-gr1-tabletop-tasks"
if [[ ! -d "$upstream" ]]; then
  git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git "$upstream"
  git -C "$upstream" checkout --detach 4840e671596f93ca03651524b9f72ffb1aadfeff
fi
actual="$(git -C "$upstream" rev-parse HEAD)"
if [[ "$actual" != 4840e671596f93ca03651524b9f72ffb1aadfeff ]]; then
  printf '%s\n' 'Existing upstream checkout has a different revision; use a separate GR1_PLUS_EXTERNAL directory.' >&2
  exit 2
fi
python -m pip install 'numpy==1.26.4' 'mujoco==3.2.6' 'gymnasium==0.29.1' \
  'opencv-python==4.11.0.86' \
  'robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@51cc01785bab80ffeed20da15e67d7dd4140e76a'
python -m pip install -e "$upstream" -e "$repo_root[test,sim]"
printf '%s\n' "Simulation dependencies installed. Download official assets with:"
printf 'python %q/robocasa/scripts/download_tabletop_assets.py -y\n' "$upstream"
printf '%s\n' 'Install LoTTA model dependencies separately in this environment; see README.md.'
