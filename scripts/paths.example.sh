# Copy to paths.local.sh; absolute paths only. This file contains no credentials.
export GR1_PLUS_UPSTREAM=/home/jovyan/robocasa-gr1-tabletop-tasks
export GR1_PLUS_LOTTA_ROOT=/home/jovyan/LoTTA
export GR1_PLUS_CHECKPOINT=/home/jovyan/checkpoints/dit4dit-model/dit4dit_robocasa_gr1/checkpoints/YOUR_CHECKPOINT.pt
export LOTTA_BASE_MODEL=/home/jovyan/checkpoints/Cosmos-Predict2.5-2B
export LOTTA_UPSTREAM_ROOT="$GR1_PLUS_LOTTA_ROOT/third_party/DiT4DiT"
# Choose GPUs free from the existing LIBERO jobs; the launcher never stops other jobs.
export GR1_PLUS_DEVICES=2,3
export GR1_PLUS_OUTPUT=/home/jovyan/robocasa-gr1-plus-v2/runs/pilot-v2
export GR1_PLUS_PROTOCOL=v2
