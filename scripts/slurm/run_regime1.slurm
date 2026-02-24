#!/bin/bash
#SBATCH --job-name=lineagevi_regime1
#SBATCH --partition=gpu-normal
#SBATCH --mem=500G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# LineageVI regime 1 only. Edit the parameters below. Run after run_preprocessing.
# sbatch run_regime1.slurm

set -e
# Use submission directory when running under SLURM so paths resolve correctly on compute nodes
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$( cd "$SCRIPT_DIR/../.." && pwd )}"

# ============ Parameters (edit these) ============
OUTPUT_DIR="/home/lgolinelli/git/lineageVI/outputs/pancreas_20260222_230813/"
CONDA_ENV="lineagevi"
# Paths (default: under OUTPUT_DIR from preprocessing)
ADATA_PATH=""        # leave empty to use $OUTPUT_DIR/adata_preprocessed.h5ad
#
EPOCHS1=400
K=10
BATCH_SIZE=256
LR_REGIME1=1e-2
TRAIN_SIZE=0.9
KL_WEIGHT_SCHEDULE="linear"
KL_WEIGHT_MIN=0.0
KL_WEIGHT_MAX=0.1
KL_CYCLE_RAMP_FRAC=0.2
N_HIDDEN=300
N_LAYERS=1
DROPOUT=0.0
CLUSTER_EMBEDDING_DIM=8
CLUSTER_KEY="leiden"
SEED=0

# ================================================
# When OUTPUT_DIR.latest.txt exists (from preprocessing), use that path and --no_timestamp
# so regime1 writes into the same timestamped run folder.

OUTPUT_BASE="$OUTPUT_DIR"
if [ -f "${OUTPUT_BASE}.latest.txt" ]; then
  OUTPUT_DIR=$(tr -d '\n' < "${OUTPUT_BASE}.latest.txt")
  NO_TIMESTAMP=true
fi
[ -z "$ADATA_PATH" ] && ADATA_PATH="$OUTPUT_DIR/adata_preprocessed.h5ad"
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR/logs"

if command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" || true
fi

ARGS=(
  --adata_path "$ADATA_PATH"
  --output_dir "$OUTPUT_DIR"
  --epochs1 "$EPOCHS1"
  --K "$K"
  --batch_size "$BATCH_SIZE"
  --lr_regime1 "$LR_REGIME1"
  --train_size "$TRAIN_SIZE"
  --kl_weight_schedule "$KL_WEIGHT_SCHEDULE"
  --kl_weight_min "$KL_WEIGHT_MIN"
  --kl_weight_max "$KL_WEIGHT_MAX"
  --kl_cycle_ramp_frac "$KL_CYCLE_RAMP_FRAC"
  --n_hidden "$N_HIDDEN"
  --n_layers "$N_LAYERS"
  --dropout "$DROPOUT"
  --cluster_embedding_dim "$CLUSTER_EMBEDDING_DIM"
  --cluster_key "$CLUSTER_KEY"
  --seed "$SEED"
)
[ "$NO_TIMESTAMP" = true ] && ARGS+=(--no_timestamp)

python scripts/run_regime1.py "${ARGS[@]}"
echo "Regime 1 done. Checkpoint: $OUTPUT_DIR/regime1/pretrained_vae_regime1.pt"
