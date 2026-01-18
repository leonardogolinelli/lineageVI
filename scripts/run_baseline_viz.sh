#!/bin/bash
# Bash script to run baseline trajectory visualization (no agent, just velocity flow)

set -e  # Exit on error

# Default values
CONDA_ENV="test3"
LINEAGEVI_OUTPUT_DIR="/Users/lgolinelli/git/lineageVI/test_outputs/lineagevi_20260117_201810"
LINEAGE_KEY="leiden"
TARGET_GOAL="1"
START_CELL_IDX=""
START_LINEAGE=""
GOAL_MODE="centroid"
T="512"
T_MAX="512"
EMBEDDING="pca"
Z_KEY="mean"
OUTPUT_DIR_BASE="./test_outputs/baseline_viz"
SEED=42
DEVICE="auto"
DT=0.1
USE_NEGATIVE_VELOCITY=""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lineagevi_output_dir)
            LINEAGEVI_OUTPUT_DIR="$2"
            shift 2
            ;;
        --lineage_key)
            LINEAGE_KEY="$2"
            shift 2
            ;;
        --target_goal)
            TARGET_GOAL="$2"
            shift 2
            ;;
        --start_cell_idx)
            START_CELL_IDX="$2"
            shift 2
            ;;
        --start_lineage)
            START_LINEAGE="$2"
            shift 2
            ;;
        --goal_mode)
            GOAL_MODE="$2"
            shift 2
            ;;
        --T)
            T="$2"
            shift 2
            ;;
        --T_max)
            T_MAX="$2"
            shift 2
            ;;
        --embedding)
            EMBEDDING="$2"
            shift 2
            ;;
        --z_key)
            Z_KEY="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR_BASE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --dt)
            DT="$2"
            shift 2
            ;;
        --use_negative_velocity)
            USE_NEGATIVE_VELOCITY="--use_negative_velocity"
            shift
            ;;
        --conda_env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --lineagevi_output_dir DIR --target_goal LABEL [OPTIONS]"
            echo ""
            echo "REQUIRED ARGUMENTS:"
            echo "  --lineagevi_output_dir DIR  Path to LineageVI output folder (contains model and adata)"
            echo "  --target_goal LABEL         Target goal label for visualization"
            echo ""
            echo "OPTIONAL ARGUMENTS:"
            echo "  --lineage_key KEY           Key in adata.obs for lineage labels (default: leiden)"
            echo "  --start_cell_idx N          Start cell index (mutually exclusive with --start_lineage)"
            echo "  --start_lineage LABEL       Start lineage label (mutually exclusive with --start_cell_idx, default: random)"
            echo "  --goal_mode MODE            Goal mode: 'centroid' or 'goal_cell' (default: centroid)"
            echo "  --T N                       Rollout horizon (default: 256)"
            echo "  --T_max N                   Maximum episode length (default: same as T)"
            echo "  --embedding METHOD          Embedding method: 'pca' or 'umap' (default: pca)"
            echo "  --z_key KEY                 Key in adata.obsm for latent states (default: mean)"
            echo "  --output_dir DIR            Output directory base (default: ./test_outputs/baseline_viz)"
            echo "                             Timestamp will be appended automatically"
            echo "  --seed N                    Random seed (default: 42)"
            echo "  --device DEV                Device: auto, cpu, or cuda (default: auto)"
            echo "  --dt FLOAT                  Time step size (default: 0.1)"
            echo "  --use_negative_velocity     Use negative velocity instead of normal velocity"
            echo "  --conda_env ENV             Conda environment name (default: test3)"
            echo ""
            echo "EXAMPLES:"
            echo "  $0 --lineagevi_output_dir ./test_outputs/lineagevi_20260117_201810 --target_goal 1"
            echo "  $0 --lineagevi_output_dir ./test_outputs/lineagevi_20260117_201810 --target_goal Beta --start_lineage Alpha --T 512 --T_max 512"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$LINEAGEVI_OUTPUT_DIR" ]]; then
    echo "Error: --lineagevi_output_dir is required"
    exit 1
fi

if [[ -z "$TARGET_GOAL" ]]; then
    echo "Error: --target_goal is required"
    exit 1
fi

# Convert to absolute paths
LINEAGEVI_OUTPUT_DIR="$(cd "$LINEAGEVI_OUTPUT_DIR" && pwd)"

# Find model and adata paths
MODEL_PATH="$LINEAGEVI_OUTPUT_DIR/vae_velocity_model.pt"
if [[ ! -f "$MODEL_PATH" ]]; then
    # Try alternative name
    MODEL_PATH="$LINEAGEVI_OUTPUT_DIR/test_model.pt"
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo "Error: Model file not found in $LINEAGEVI_OUTPUT_DIR"
        echo "Expected: vae_velocity_model.pt or test_model.pt"
        exit 1
    fi
fi

ADATA_PATH="$LINEAGEVI_OUTPUT_DIR/test_outputs.h5ad"
if [[ ! -f "$ADATA_PATH" ]]; then
    echo "Error: AnnData file not found: $ADATA_PATH"
    exit 1
fi

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$OUTPUT_DIR_BASE/baseline_viz_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Baseline Trajectory Visualization"
echo "=========================================="
echo "LineageVI output dir: $LINEAGEVI_OUTPUT_DIR"
echo "Model: $MODEL_PATH"
echo "AnnData: $ADATA_PATH"
echo "Target goal: $TARGET_GOAL"
echo "T: $T"
if [[ -n "$T_MAX" ]]; then
    echo "T_max: $T_MAX"
fi
echo "Output dir: $OUTPUT_DIR"
echo "=========================================="

# Activate conda environment
if command -v conda &> /dev/null; then
    echo "Activating conda environment: $CONDA_ENV"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
else
    echo "Warning: conda not found. Assuming environment is already active."
fi

# Build Python command
cd "$PROJECT_ROOT"

PYTHON_ARGS=(
    --model_path "$MODEL_PATH"
    --adata_path "$ADATA_PATH"
    --lineage_key "$LINEAGE_KEY"
    --target_goal "$TARGET_GOAL"
    --goal_mode "$GOAL_MODE"
    --T "$T"
    --embedding "$EMBEDDING"
    --z_key "$Z_KEY"
    --outdir "$OUTPUT_DIR"
    --seed "$SEED"
    --device "$DEVICE"
    --dt "$DT"
)

if [[ -n "$START_CELL_IDX" ]]; then
    PYTHON_ARGS+=(--start_cell_idx "$START_CELL_IDX")
fi

if [[ -n "$START_LINEAGE" ]]; then
    PYTHON_ARGS+=(--start_lineage "$START_LINEAGE")
fi

if [[ -n "$T_MAX" ]]; then
    PYTHON_ARGS+=(--T_max "$T_MAX")
fi

if [[ -n "$USE_NEGATIVE_VELOCITY" ]]; then
    PYTHON_ARGS+=(--use_negative_velocity)
fi

# Run the visualization script
echo "Running baseline visualization script..."
python -m lineagevi.rl.baseline_viz "${PYTHON_ARGS[@]}"

echo ""
echo "=========================================="
echo "Visualization complete!"
echo "Outputs saved to: $OUTPUT_DIR"
echo "=========================================="
