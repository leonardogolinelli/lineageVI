#!/bin/bash
#SBATCH --job-name=lineagevi_downstream
#SBATCH --partition=normal
#SBATCH --mem=500G
#SBATCH --cpus-per-task=4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# LineageVI downstream (differential, plots, velocity UMAP, etc.). Run after run_regime2.
# Set OUTPUT_DIR to the regime2 output folder; adata and model are read from there.
# sbatch run_downstream.slurm

set -e
# Use submission directory when running under SLURM so paths resolve correctly on compute nodes
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$( cd "$SCRIPT_DIR/../.." && pwd )}"

# ============ Parameters (edit these) ============
OUTPUT_DIR="/home/lgolinelli/git/lineageVI/outputs/pancreas_20260222_230813_20260222_230956_20260222_231733"   # regime2 output folder
CONDA_ENV="lineagevi"
# Paths: leave empty to use OUTPUT_DIR/adata_with_latent.h5ad and OUTPUT_DIR/pretrained_vae.pt
ADATA_PATH=""        # leave empty for $OUTPUT_DIR/adata_with_latent.h5ad (from regime2)
MODEL_PATH=""       # leave empty for $OUTPUT_DIR/pretrained_vae.pt (from regime2)
GROUPBY_KEY="leiden"   # tutorial uses "clusters" for pancreas
# Optional: run perturbation for this group (e.g. Alpha, Beta); leave empty to skip
PERTURB_GROUP=""
# ================================================
# When OUTPUT_DIR.latest.txt exists, use that path and --no_timestamp for same run folder.

OUTPUT_BASE="$OUTPUT_DIR"
if [ -f "${OUTPUT_BASE}.latest.txt" ]; then
  OUTPUT_DIR=$(tr -d '\n' < "${OUTPUT_BASE}.latest.txt")
  NO_TIMESTAMP=true
fi
[ -z "$ADATA_PATH" ] && ADATA_PATH="$OUTPUT_DIR/adata_with_latent.h5ad"
[ -z "$MODEL_PATH" ] && MODEL_PATH="$OUTPUT_DIR/pretrained_vae.pt"
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR/logs"

if command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" || true
fi

ARGS=(
  --adata_path "$ADATA_PATH"
  --model_path "$MODEL_PATH"
  --output_dir "$OUTPUT_DIR"
  --groupby_key "$GROUPBY_KEY"
)
[ "$NO_TIMESTAMP" = true ] && ARGS+=(--no_timestamp)
[ -n "$PERTURB_GROUP" ] && ARGS+=(--run_perturbation --perturb_group "$PERTURB_GROUP")

python scripts/run_downstream.py "${ARGS[@]}"
echo "Downstream done. Figures: $OUTPUT_DIR/figures/"
