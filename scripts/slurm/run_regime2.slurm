#!/bin/bash
#SBATCH --job-name=lineagevi_regime2
#SBATCH --partition=gpu-normal
#SBATCH --mem=500G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# LineageVI regime 2 only. Edit the parameters below. Run after run_regime1.
# sbatch run_regime2.slurm

set -e
# Use submission directory when running under SLURM so paths resolve correctly on compute nodes
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$( cd "$SCRIPT_DIR/../.." && pwd )}"

# ============ Parameters (edit these) ============
OUTPUT_DIR="/home/lgolinelli/git/lineageVI/outputs/pancreas_20260222_230813_20260222_230956"
CONDA_ENV="lineagevi"
# Paths (default: preprocessed adata and regime1 checkpoint under OUTPUT_DIR)
ADATA_PATH="/home/lgolinelli/git/lineageVI/outputs/pancreas_20260222_230813/adata_preprocessed.h5ad"          # leave empty for $OUTPUT_DIR/adata_preprocessed.h5ad
CHECKPOINT_PATH=""     # leave empty for $OUTPUT_DIR/regime1/pretrained_vae_regime1.pt
#
EPOCHS2=150
K=10
BATCH_SIZE=256
LR_REGIME2=5e-3
TRAIN_SIZE=0.9
VELOCITY_LOSS_WEIGHT_GENE=1.0
VELOCITY_LOSS_WEIGHT_GP=0
SEED_REGIME2=2
MONITOR_EVERY_EPOCHS=25

# MONITOR_GENES="Gnas Ins1 Pdx1"   # space-separated, or leave empty
# ================================================
# When OUTPUT_DIR.latest.txt exists, use that path and --no_timestamp for same run folder.

OUTPUT_BASE="$OUTPUT_DIR"
if [ -f "${OUTPUT_BASE}.latest.txt" ]; then
  OUTPUT_DIR=$(tr -d '\n' < "${OUTPUT_BASE}.latest.txt")
  NO_TIMESTAMP=true
fi
[ -z "$ADATA_PATH" ] && ADATA_PATH="$OUTPUT_DIR/adata_preprocessed.h5ad"
[ -z "$CHECKPOINT_PATH" ] && CHECKPOINT_PATH="$OUTPUT_DIR/regime1/pretrained_vae_regime1.pt"
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR/logs"

if command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" || true
fi

ARGS=(
  --adata_path "$ADATA_PATH"
  --checkpoint_path "$CHECKPOINT_PATH"
  --output_dir "$OUTPUT_DIR"
  --epochs2 "$EPOCHS2"
  --K "$K"
  --batch_size "$BATCH_SIZE"
  --lr_regime2 "$LR_REGIME2"
  --train_size "$TRAIN_SIZE"
  --velocity_loss_weight_gene "$VELOCITY_LOSS_WEIGHT_GENE"
  --velocity_loss_weight_gp "$VELOCITY_LOSS_WEIGHT_GP"
  --seed_regime2 "$SEED_REGIME2"
  --monitor_every_epochs "$MONITOR_EVERY_EPOCHS"
)
[ "$NO_TIMESTAMP" = true ] && ARGS+=(--no_timestamp)
[ -n "${MONITOR_GENES:-}" ] && ARGS+=(--monitor_genes $MONITOR_GENES)

python scripts/run_regime2.py "${ARGS[@]}"
echo "Regime 2 done. Model: $OUTPUT_DIR/pretrained_vae.pt, AnnData: $OUTPUT_DIR/adata_with_latent.h5ad"
