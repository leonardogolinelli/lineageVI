#!/bin/bash
# Bash script to run baseline trajectory visualization (no agent, just velocity flow)

set -e  # Exit on error

# Default values
CONDA_ENV="test3"
LINEAGEVI_OUTPUT_DIR="/Users/lgolinelli/git/lineageVI/test_outputs/lineagevi_20260117_201810"
LINEAGE_KEY="leiden"
TARGET_LINEAGE="1"
SOURCE_LINEAGE=""
SOURCE_MODE="sample"  # "centroid" or "sample"
TARGET_MODE="centroid"  # "centroid" or "goal_cell"
T="512"
T_MAX="512"
EMBEDDING="pca"
Z_KEY="mean"
OUTPUT_DIR_BASE="./test_outputs/baseline_viz"
SEED=42
DEVICE="auto"
DT=0.04
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
        --target_lineage)
            TARGET_LINEAGE="$2"
            shift 2
            ;;
        --source_lineage)
            SOURCE_LINEAGE="$2"
            shift 2
            ;;
        --source_mode)
            SOURCE_MODE="$2"
            shift 2
            ;;
        --target_mode)
            TARGET_MODE="$2"
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
            echo "  --start_cell_idx N          Start cell index (mutually exclusive with --start_lineage/--source_lineage)"
            echo "  --start_lineage LABEL       Start lineage label (deprecated, use --source_lineage)"
            echo "  --source_lineage LABEL      Source lineage label (mutually exclusive with --start_cell_idx, default: random)"
            echo "  --source_mode MODE          Source mode: 'centroid' (use source lineage centroid) or 'sample' (sample a cell from source lineage, default)"
            echo "  --target_goal LABEL         Target goal label (deprecated, use --target_lineage)"
            echo "  --target_lineage LABEL      Target lineage label (required if --target_goal not provided)"
            echo "  --goal_mode MODE            Goal mode: 'centroid' (use target lineage centroid, default) or 'goal_cell' (sample a cell from target lineage)"
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
            echo "  $0 --lineagevi_output_dir ./test_outputs/lineagevi_20260117_201810 --target_lineage 1"
            echo "  $0 --lineagevi_output_dir ./test_outputs/lineagevi_20260117_201810 --target_lineage Beta --source_lineage Alpha --source_mode centroid --target_mode goal_cell --T 512 --T_max 512"
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

if [[ -z "$TARGET_LINEAGE" ]]; then
    echo "Error: --target_lineage is required"
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
echo "Target lineage: $TARGET_LINEAGE"
if [[ -n "$SOURCE_LINEAGE" ]]; then
    echo "Source lineage: $SOURCE_LINEAGE (mode: $SOURCE_MODE)"
fi
echo "Goal mode: $GOAL_MODE"
echo "T: $T"
if [[ -n "$T_MAX" ]]; then
    echo "T_max: $T_MAX"
fi
if [[ -n "$USE_NEGATIVE_VELOCITY" ]]; then
    echo "Using negative velocity"
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
    --target_lineage "$TARGET_LINEAGE"
    --target_mode "$TARGET_MODE"
    --T "$T"
    --embedding "$EMBEDDING"
    --z_key "$Z_KEY"
    --outdir "$OUTPUT_DIR"
    --seed "$SEED"
    --device "$DEVICE"
    --dt "$DT"
)

if [[ -n "$SOURCE_LINEAGE" ]]; then
    PYTHON_ARGS+=(--source_lineage "$SOURCE_LINEAGE")
fi

if [[ -n "$SOURCE_MODE" ]]; then
    PYTHON_ARGS+=(--source_mode "$SOURCE_MODE")
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
