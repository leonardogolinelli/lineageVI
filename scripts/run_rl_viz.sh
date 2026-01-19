#!/bin/bash
# Bash script to run RL agent trajectory visualization in conda environment 'test3'

set -e  # Exit on error

# Default values
CONDA_ENV="test3"
RL_OUTPUT_DIR="/Users/lgolinelli/git/lineageVI/test_outputs/rl_20260117_231344"
LINEAGEVI_OUTPUT_DIR="/Users/lgolinelli/git/lineageVI/test_outputs/lineagevi_20260117_201810"
LINEAGE_KEY="leiden"
CHECKPOINT=""
TARGET_GOAL="1"
START_CELL_IDX=""
START_LINEAGE=""
GOAL_MODE="centroid"
T=512
EMBEDDING="pca"
Z_KEY="mean"
OUTPUT_DIR_BASE="./test_outputs/viz"
SEED=42
DEVICE="auto"
DETERMINISTIC=""
INTERVENTION_METHOD="heatmap"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --rl_output_dir)
            RL_OUTPUT_DIR="$2"
            shift 2
            ;;
        --lineagevi_output_dir)
            LINEAGEVI_OUTPUT_DIR="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
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
        --deterministic)
            DETERMINISTIC="--deterministic"
            shift
            ;;
        --intervention_method)
            INTERVENTION_METHOD="$2"
            shift 2
            ;;
        --conda_env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --rl_output_dir DIR --target_goal LABEL [OPTIONS]"
            echo ""
            echo "REQUIRED ARGUMENTS:"
            echo "  --rl_output_dir DIR         Path to RL training output folder (contains checkpoints)"
            echo "  --target_goal LABEL         Target goal label for visualization"
            echo ""
            echo "OPTIONAL ARGUMENTS:"
            echo "  --lineagevi_output_dir DIR  Path to LineageVI output folder (default: auto-detect from RL output)"
            echo "  --checkpoint PATH           Path to specific checkpoint file (default: latest policy_iter_*.pt)"
            echo "  --lineage_key KEY           Key in adata.obs for lineage labels (default: leiden)"
            echo "  --start_cell_idx N          Start cell index (mutually exclusive with --start_lineage)"
            echo "  --start_lineage LABEL       Start lineage label (mutually exclusive with --start_cell_idx, default: random)"
            echo "  --goal_mode MODE            Goal mode: 'centroid' (use lineage centroid, default) or 'goal_cell' (sample a cell from target lineage)"
            echo "  --T N                       Rollout horizon (default: 64)"
            echo "  --embedding METHOD          Embedding method: 'pca' or 'umap' (default: pca)"
            echo "  --z_key KEY                 Key in adata.obsm for latent states (default: mean)"
            echo "  --output_dir DIR            Output directory base (default: ./test_outputs/viz)"
            echo "                             Timestamp will be appended automatically"
            echo "  --seed N                    Random seed (default: 42)"
            echo "  --device DEV                Device: auto, cpu, or cuda (default: auto)"
            echo "  --deterministic             Use deterministic policy (default: True)"
            echo "  --intervention_method M     Intervention plot method: 'stem' or 'heatmap' (default: heatmap)"
            echo "  --conda_env ENV             Conda environment name (default: test3)"
            echo ""
            echo "EXAMPLES:"
            echo "  $0 --rl_output_dir ./test_outputs/rl_20260117_205544 --target_goal 1"
            echo "  $0 --rl_output_dir ./test_outputs/rl_20260117_205544 --target_goal Beta --start_lineage Alpha --goal_mode goal_cell"
            echo "  $0 --rl_output_dir ./test_outputs/rl_20260117_205544 --target_goal Beta --start_cell_idx 123 --goal_mode goal_cell"
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
if [[ -z "$RL_OUTPUT_DIR" ]]; then
    echo "Error: --rl_output_dir is required"
    exit 1
fi

if [[ -z "$TARGET_GOAL" ]]; then
    echo "Error: --target_goal is required"
    exit 1
fi

# Convert to absolute paths
RL_OUTPUT_DIR="$(cd "$RL_OUTPUT_DIR" && pwd)"

# Auto-detect LineageVI output directory if not provided
if [[ -z "$LINEAGEVI_OUTPUT_DIR" ]]; then
    # Try to find it from RL output directory name or config
    # Look for a common pattern or check config file
    if [[ -f "$RL_OUTPUT_DIR/config_iter_0.json" ]]; then
        # Try to extract from config (if stored)
        echo "Attempting to auto-detect LineageVI output directory..."
        # Default to a common location
        LINEAGEVI_OUTPUT_DIR="$PROJECT_ROOT/test_outputs/lineagevi"
        if [[ ! -d "$LINEAGEVI_OUTPUT_DIR" ]]; then
            echo "Warning: Could not auto-detect LineageVI output directory."
            echo "Please provide --lineagevi_output_dir explicitly."
            exit 1
        fi
    else
        echo "Error: Could not auto-detect LineageVI output directory."
        echo "Please provide --lineagevi_output_dir explicitly."
        exit 1
    fi
fi

LINEAGEVI_OUTPUT_DIR="$(cd "$LINEAGEVI_OUTPUT_DIR" && pwd)"

# Find checkpoint if not provided
if [[ -z "$CHECKPOINT" ]]; then
    # Find latest policy checkpoint
    CHECKPOINT=$(find "$RL_OUTPUT_DIR" -name "policy_iter_*.pt" | sort -V | tail -1)
    if [[ -z "$CHECKPOINT" ]]; then
        echo "Error: No checkpoint found in $RL_OUTPUT_DIR"
        echo "Please provide --checkpoint explicitly or ensure policy_iter_*.pt files exist"
        exit 1
    fi
    echo "Using checkpoint: $CHECKPOINT"
else
    # Convert to absolute path if relative
    if [[ ! "$CHECKPOINT" = /* ]]; then
        CHECKPOINT="$RL_OUTPUT_DIR/$CHECKPOINT"
    fi
fi

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
OUTPUT_DIR="$OUTPUT_DIR_BASE/viz_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "RL Agent Trajectory Visualization"
echo "=========================================="
echo "RL output dir: $RL_OUTPUT_DIR"
echo "LineageVI output dir: $LINEAGEVI_OUTPUT_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "Model: $MODEL_PATH"
echo "AnnData: $ADATA_PATH"
echo "Target goal: $TARGET_GOAL"
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
    --checkpoint "$CHECKPOINT"
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
    --intervention_method "$INTERVENTION_METHOD"
)

if [[ -n "$START_CELL_IDX" ]]; then
    PYTHON_ARGS+=(--start_cell_idx "$START_CELL_IDX")
fi

if [[ -n "$START_LINEAGE" ]]; then
    PYTHON_ARGS+=(--start_lineage "$START_LINEAGE")
fi

if [[ -n "$DETERMINISTIC" ]]; then
    PYTHON_ARGS+=(--deterministic)
fi

# Run the visualization script
echo "Running visualization script..."
python -m lineagevi.rl.viz "${PYTHON_ARGS[@]}"

echo ""
echo "=========================================="
echo "Visualization complete!"
echo "Outputs saved to: $OUTPUT_DIR"
echo "=========================================="
