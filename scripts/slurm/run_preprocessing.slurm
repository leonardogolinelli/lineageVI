#!/bin/bash
#SBATCH --job-name=lineagevi_preprocess
#SBATCH --partition=normal
#SBATCH --time=2:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# LineageVI preprocessing. Edit the parameters below and run: sbatch run_preprocessing.slurm

set -e
# Use submission directory when running under SLURM so paths resolve correctly on compute nodes
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$( cd "$SCRIPT_DIR/../.." && pwd )}"

# ============ Parameters (edit these) ============
OUTPUT_DIR="./outputs/pancreas"
CONDA_ENV="lineagevi"
DATASET="pancreas"
GENE_SETS_DIR="./gene_sets"
ANNOTATION_LIBRARY="KEGG_2026"   # Enrichr library; downloaded to GENE_SETS_DIR if not present
MIN_SHARED_COUNTS=20
N_TOP_GENES=3000
MIN_GENES_PER_TERM=12
N_PCS=100
N_NEIGHBORS=200
K_NEIGHBORS=20
CLUSTER_KEY=""       # leave empty for default leiden
CLEAN_TERMS=true
# ================================================
# Preprocessing appends a timestamp to OUTPUT_DIR (e.g. outputs/pancreas_20250207_143022)
# and writes that path to OUTPUT_DIR.latest.txt for chained jobs (single output folder).

cd "$PROJECT_ROOT"
# Ensure parent dir exists (e.g. outputs); only the timestamped run dir is created by the script
mkdir -p "$(dirname "$OUTPUT_DIR")"

if command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" || true
fi

ARGS=(
  --output_dir "$OUTPUT_DIR"
  --dataset "$DATASET"
  --min_shared_counts "$MIN_SHARED_COUNTS"
  --n_top_genes "$N_TOP_GENES"
  --min_genes_per_term "$MIN_GENES_PER_TERM"
  --n_pcs "$N_PCS"
  --n_neighbors "$N_NEIGHBORS"
  --K_neighbors "$K_NEIGHBORS"
  --clean_terms "$CLEAN_TERMS"
  --gene_sets_dir "$GENE_SETS_DIR"
  --annotation_library "$ANNOTATION_LIBRARY"
)
[ -n "$CLUSTER_KEY" ] && ARGS+=(--cluster_key "$CLUSTER_KEY")

python scripts/run_preprocessing.py "${ARGS[@]}"
# Actual output is in OUTPUT_DIR_timestamp; path written to OUTPUT_DIR.latest.txt
[ -f "${OUTPUT_DIR}.latest.txt" ] && echo "Preprocessing done. Output: $(cat "${OUTPUT_DIR}.latest.txt")" || echo "Preprocessing done. Output: $OUTPUT_DIR"
