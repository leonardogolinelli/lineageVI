#!/bin/bash
# Bash script to run RL agent training in conda environment 'test3'

set -e  # Exit on error

# Default values
CONDA_ENV="test3"
LINEAGEVI_OUTPUT_DIR="/Users/lgolinelli/git/lineageVI/test_outputs/lineagevi_20260117_201810"
LINEAGE_KEY="leiden"
OUTPUT_DIR_BASE="./test_outputs/rl"
CONFIG_FILE=""
SEED=42
DEVICE="auto"
Z_KEY="mean"
GOAL_ALLOWED=()
GOAL_EXCLUDE=()
GOAL_MIN_CELLS=1
FIXED_GOAL=""
USE_NEGATIVE_VELOCITY=""
N_ITERATIONS="200"
EPOCHS="3"
BATCH_SIZE="128"
T_ROLLOUT="256"
T_MAX="256"
MINIBATCH_SIZE="4096"
SAVE_FREQ="25"
DT="" # default is 0.1
LAMBDA_PROGRESS="2.0" # default is 1.0
LAMBDA_ACT="" # default is 0.02
LAMBDA_MAG="" # default is 0.15
R_SUCC="" # default is 20.0
GAMMA="" # default is 0.99
N_VIZ_TRAJECTORIES="3"
VIZ_EMBEDDING="pca"
SKIP_VIZ=""


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
        --output_dir)
            OUTPUT_DIR_BASE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
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
        --z_key)
            Z_KEY="$2"
            shift 2
            ;;
        --goal_allowed)
            # Collect all allowed goals
            GOAL_ALLOWED=()
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                GOAL_ALLOWED+=("$1")
                shift
            done
            ;;
        --goal_exclude)
            # Collect all excluded goals
            GOAL_EXCLUDE=()
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                GOAL_EXCLUDE+=("$1")
                shift
            done
            ;;
        --goal_min_cells)
            GOAL_MIN_CELLS="$2"
            shift 2
            ;;
        --fixed_goal)
            FIXED_GOAL="$2"
            shift 2
            ;;
        --n_iterations)
            N_ITERATIONS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --T_rollout)
            T_ROLLOUT="$2"
            shift 2
            ;;
        --T_max)
            T_MAX="$2"
            shift 2
            ;;
        --minibatch_size)
            MINIBATCH_SIZE="$2"
            shift 2
            ;;
        --save_freq)
            SAVE_FREQ="$2"
            shift 2
            ;;
        --use_negative_velocity)
            USE_NEGATIVE_VELOCITY="--use_negative_velocity"
            shift
            ;;
        --dt)
            DT="$2"
            shift 2
            ;;
        --lambda_progress)
            LAMBDA_PROGRESS="$2"
            shift 2
            ;;
        --lambda_act)
            LAMBDA_ACT="$2"
            shift 2
            ;;
        --lambda_mag)
            LAMBDA_MAG="$2"
            shift 2
            ;;
        --R_succ)
            R_SUCC="$2"
            shift 2
            ;;
        --n_viz_trajectories)
            N_VIZ_TRAJECTORIES="$2"
            shift 2
            ;;
        --viz_embedding)
            VIZ_EMBEDDING="$2"
            shift 2
            ;;
        --skip_viz)
            SKIP_VIZ="--skip_viz"
            shift
            ;;
        --conda_env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --lineagevi_output_dir DIR --lineage_key KEY [OPTIONS]"
            echo ""
            echo "REQUIRED ARGUMENTS:"
            echo "  --lineagevi_output_dir DIR  Path to LineageVI output folder (contains model and adata)"
            echo "  --lineage_key KEY           Key in adata.obs for lineage labels"
            echo ""
            echo "OPTIONAL ARGUMENTS:"
            echo "  --output_dir DIR            Output directory base for RL checkpoints (default: ./test_outputs/rl)"
            echo "                             Timestamp will be appended automatically"
            echo "  --config PATH               Path to config YAML file (default: use defaults)"
            echo "  --seed N                   Random seed (default: 42)"
            echo "  --device DEV               Device: auto, cpu, or cuda (default: auto)"
            echo "  --z_key KEY                Key in adata.obsm for latent states (default: mean)"
            echo "  --goal_allowed LABEL ...   Allowed goal labels (default: all)"
            echo "  --goal_exclude LABEL ...   Excluded goal labels"
            echo "  --goal_min_cells N         Minimum cells per goal lineage (default: 1)"
            echo "  --fixed_goal LABEL         Fixed goal label for all episodes (optional)"
            echo "  --use_negative_velocity    Use negative velocity instead of normal velocity"
            echo ""
            echo "ENVIRONMENT PARAMETERS (override config):"
            echo "  --dt FLOAT                Time step size (overrides config)"
            echo "  --lambda_progress FLOAT   Progress reward scaling factor (overrides config, default: 1.0)"
            echo "  --lambda_act FLOAT        Action penalty coefficient (overrides config)"
            echo "  --lambda_mag FLOAT         Magnitude penalty coefficient (overrides config)"
            echo "  --R_succ FLOAT            Success reward bonus (overrides config)"
            echo "  --gamma FLOAT             Discount factor for future rewards (overrides config, default: 0.99)"
            echo ""
            echo "VISUALIZATION PARAMETERS:"
            echo "  --n_viz_trajectories N     Number of example trajectories to visualize (default: 3)"
            echo "  --viz_embedding METHOD     Embedding method: 'pca' or 'umap' (default: pca)"
            echo "  --skip_viz                Skip trajectory visualization after training"
            echo ""
            echo "TRAINING PARAMETERS (override config):"
            echo "  --n_iterations N         Total training iterations (overrides config)"
            echo "  --epochs N               PPO inner epochs per iteration (overrides config)"
            echo "  --batch_size N           Environment batch size (overrides config)"
            echo "  --T_rollout N            Rollout horizon (overrides config)"
            echo "  --minibatch_size N       Minibatch size for PPO updates (overrides config)"
            echo "  --save_freq N            Checkpoint save frequency (overrides config)"
            echo ""
            echo "OTHER OPTIONS:"
            echo "  --conda_env NAME           Conda environment (default: test3)"
            echo "  -h, --help                 Show this help message"
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
if [[ -z "$LINEAGE_KEY" ]]; then
    echo "Error: --lineage_key is required"
    exit 1
fi

# Check if LineageVI output directory exists
if [[ ! -d "$LINEAGEVI_OUTPUT_DIR" ]]; then
    echo "Error: LineageVI output directory not found: $LINEAGEVI_OUTPUT_DIR"
    exit 1
fi

# Find model file (prefer vae_velocity_model.pt, fallback to test_model.pt)
MODEL_PATH=""
if [[ -f "$LINEAGEVI_OUTPUT_DIR/vae_velocity_model.pt" ]]; then
    MODEL_PATH="$LINEAGEVI_OUTPUT_DIR/vae_velocity_model.pt"
elif [[ -f "$LINEAGEVI_OUTPUT_DIR/test_model.pt" ]]; then
    MODEL_PATH="$LINEAGEVI_OUTPUT_DIR/test_model.pt"
else
    echo "Error: Model file not found in $LINEAGEVI_OUTPUT_DIR"
    echo "Expected: vae_velocity_model.pt or test_model.pt"
    exit 1
fi

# Find adata file
ADATA_PATH=""
if [[ -f "$LINEAGEVI_OUTPUT_DIR/test_outputs.h5ad" ]]; then
    ADATA_PATH="$LINEAGEVI_OUTPUT_DIR/test_outputs.h5ad"
else
    echo "Error: AnnData file not found in $LINEAGEVI_OUTPUT_DIR"
    echo "Expected: test_outputs.h5ad"
    exit 1
fi

# Get the directory of this script and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Generate timestamp and create output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_DIR_BASE}_${TIMESTAMP}"

echo "=========================================="
echo "RL Agent Training Script"
echo "=========================================="
echo "Conda environment: $CONDA_ENV"
echo "LineageVI output: $LINEAGEVI_OUTPUT_DIR"
echo "Model path: $MODEL_PATH"
echo "AnnData path: $ADATA_PATH"
echo "Lineage key: $LINEAGE_KEY"
echo "Output directory: $OUTPUT_DIR"
echo "Seed: $SEED"
echo "Device: $DEVICE"
if [[ -n "$CONFIG_FILE" ]]; then
    echo "Config file: $CONFIG_FILE"
fi
echo "Project root: $PROJECT_ROOT"
echo "=========================================="

# Initialize conda (if not already initialized)
if ! command -v conda &> /dev/null; then
    echo "Error: conda command not found. Please initialize conda first."
    echo "Try: source $(conda info --base)/etc/profile.d/conda.sh"
    exit 1
fi

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Check if activation was successful
if [[ $? -ne 0 ]]; then
    echo "Error: Failed to activate conda environment '$CONDA_ENV'"
    echo "Available environments:"
    conda env list
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT"

# Build Python command with all arguments
PYTHON_ARGS=(
    --model_path "$MODEL_PATH"
    --adata_path "$ADATA_PATH"
    --lineage_key "$LINEAGE_KEY"
    --output_dir "$OUTPUT_DIR"
    --seed "$SEED"
    --device "$DEVICE"
    --z_key "$Z_KEY"
)

if [[ -n "$CONFIG_FILE" ]]; then
    PYTHON_ARGS+=(--config "$CONFIG_FILE")
fi

# Add goal filtering arguments
if [[ ${#GOAL_ALLOWED[@]} -gt 0 ]]; then
    PYTHON_ARGS+=(--goal_allowed "${GOAL_ALLOWED[@]}")
fi

if [[ ${#GOAL_EXCLUDE[@]} -gt 0 ]]; then
    PYTHON_ARGS+=(--goal_exclude "${GOAL_EXCLUDE[@]}")
fi

PYTHON_ARGS+=(--goal_min_cells "$GOAL_MIN_CELLS")

if [[ -n "$FIXED_GOAL" ]]; then
    PYTHON_ARGS+=(--fixed_goal "$FIXED_GOAL")
fi

# Add training parameters (override config if provided)
if [[ -n "$N_ITERATIONS" ]]; then
    PYTHON_ARGS+=(--n_iterations "$N_ITERATIONS")
fi
if [[ -n "$EPOCHS" ]]; then
    PYTHON_ARGS+=(--epochs "$EPOCHS")
fi
if [[ -n "$BATCH_SIZE" ]]; then
    PYTHON_ARGS+=(--batch_size "$BATCH_SIZE")
fi
if [[ -n "$T_ROLLOUT" ]]; then
    PYTHON_ARGS+=(--T_rollout "$T_ROLLOUT")
fi
if [[ -n "$T_MAX" ]]; then
    PYTHON_ARGS+=(--T_max "$T_MAX")
fi
if [[ -n "$MINIBATCH_SIZE" ]]; then
    PYTHON_ARGS+=(--minibatch_size "$MINIBATCH_SIZE")
fi
if [[ -n "$SAVE_FREQ" ]]; then
    PYTHON_ARGS+=(--save_freq "$SAVE_FREQ")
fi
if [[ -n "$USE_NEGATIVE_VELOCITY" ]]; then
    PYTHON_ARGS+=(--use_negative_velocity)
fi
if [[ -n "$DT" ]]; then
    PYTHON_ARGS+=(--dt "$DT")
fi
if [[ -n "$LAMBDA_PROGRESS" ]]; then
    PYTHON_ARGS+=(--lambda_progress "$LAMBDA_PROGRESS")
fi
if [[ -n "$LAMBDA_ACT" ]]; then
    PYTHON_ARGS+=(--lambda_act "$LAMBDA_ACT")
fi
if [[ -n "$LAMBDA_MAG" ]]; then
    PYTHON_ARGS+=(--lambda_mag "$LAMBDA_MAG")
fi
if [[ -n "$R_SUCC" ]]; then
    PYTHON_ARGS+=(--R_succ "$R_SUCC")
fi
if [[ -n "$GAMMA" ]]; then
    PYTHON_ARGS+=(--gamma "$GAMMA")
fi
if [[ -n "$N_VIZ_TRAJECTORIES" ]]; then
    PYTHON_ARGS+=(--n_viz_trajectories "$N_VIZ_TRAJECTORIES")
fi
if [[ -n "$VIZ_EMBEDDING" ]]; then
    PYTHON_ARGS+=(--viz_embedding "$VIZ_EMBEDDING")
fi
if [[ -n "$SKIP_VIZ" ]]; then
    PYTHON_ARGS+=(--skip_viz)
fi

# Run the RL training script
echo "Running RL training script..."
python -m lineagevi.rl.train "${PYTHON_ARGS[@]}"

echo ""
echo "=========================================="
echo "RL training completed successfully!"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
