#!/bin/bash
# Bash script to run RL agent training in conda environment 'test3'

set -e  # Exit on error


# Architecture parameters
HIDDEN_SIZES="128, 128" # default is 128,128
ACTIVATION="" # default is relu
DELTA_CLIP="" # default is none

# Default values
CONDA_ENV="test3"
LINEAGEVI_OUTPUT_DIR="/Users/lgolinelli/git/lineageVI/test_outputs/lineagevi_20260117_201810"
LINEAGE_KEY="leiden"
OUTPUT_DIR_BASE="./test_outputs/rl"
CONFIG_FILE=""
SEED=42
DEVICE="auto"
Z_KEY="mean"
SOURCE_LINEAGE="4"
TARGET_LINEAGE="0"
SOURCE_MODE="centroid"  # "centroid" or "sample"
TARGET_MODE="centroid"  # "centroid" or "sample"
USE_NEGATIVE_VELOCITY=""
DETERMINISTIC=""
DEACTIVATE_VELOCITY="--deactivate_velocity"
N_ITERATIONS="200"
EPOCHS="2"
BATCH_SIZE="256"
T_ROLLOUT="1000"
T_MAX="1000"
MINIBATCH_SIZE="2048"
SAVE_FREQ="25"
DT="" # default is 0.1
LAMBDA_PROGRESS="1" # default is 1.0
LAMBDA_ACT="1e-3" # default is 0.02
LAMBDA_MAG="1e-3" # default is 0.15
R_SUCC="10" # default is 20.0 # INCREASE IT BASED ON EPS_SUCCESS_PCT AND EPS_SUCCESS_DECAY_FACTOR
ALPHA_STAY="0" # default is 0.0 (state cost for staying near goal)
EPS_SUCCESS_PCT="0.99"
EPS_SUCCESS_DECAY_ON_SUCCESS="--eps_success_decay_on_success"
EPS_SUCCESS_SUCCESS_RATE_THRESHOLD="0.55"
EPS_SUCCESS_DECAY_FACTOR="0.99"
EPS_SUCCESS_DECAY_REWARD_PCT="0.0" # fraction of reward bonus to apply when eps_success decays
EPS_SUCCESS_REWARD_MATCH_DECAY="--eps_success_reward_match_decay"
PERTURB_CLIP="1" # env-side perturbation clip (default: none)
GAMMA=".995" # default is 0.99     1 IS USUALLY TOO NOISY
ENT_COEF="1e-3"
KL_STOP_THRESHOLD="0.02" # default is 0.02
KL_STOP_IMMEDIATE_THRESHOLD="0.03" # default is 0.03
LR="3e-4" # default is 3e-4
ACTOR_LR="3e-5" # default is LR
CRITIC_LR="" # default is LR
GOAL_COND_DIM="32" # default is 32
DISABLE_NOOP_ACTION=""
USE_T_NORM=""
GMM_PATH=""
GMM_COMPONENTS="32"
LAMBDA_OFF="0"
N_VIZ_TRAJECTORIES="10"
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
        --source_lineage)
            SOURCE_LINEAGE="$2"
            shift 2
            ;;
        --target_lineage)
            TARGET_LINEAGE="$2"
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
        --deterministic)
            DETERMINISTIC="--deterministic"
            shift
            ;;
        --deactivate_velocity)
            DEACTIVATE_VELOCITY="--deactivate_velocity"
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
        --alpha_stay)
            ALPHA_STAY="$2"
            shift 2
            ;;
        --eps_success_decay_on_success)
            EPS_SUCCESS_DECAY_ON_SUCCESS="--eps_success_decay_on_success"
            shift
            ;;
        --eps_success_pct)
            EPS_SUCCESS_PCT="$2"
            shift 2
            ;;
        --eps_success_success_rate_threshold)
            EPS_SUCCESS_SUCCESS_RATE_THRESHOLD="$2"
            shift 2
            ;;
        --eps_success_decay_factor)
            EPS_SUCCESS_DECAY_FACTOR="$2"
            shift 2
            ;;
        --eps_success_decay_reward_pct)
            EPS_SUCCESS_DECAY_REWARD_PCT="$2"
            shift 2
            ;;
        --eps_success_reward_match_decay)
            EPS_SUCCESS_REWARD_MATCH_DECAY="--eps_success_reward_match_decay"
            shift
            ;;
        --perturb_clip)
            PERTURB_CLIP="$2"
            shift 2
            ;;
        --n_viz_trajectories)
            N_VIZ_TRAJECTORIES="$2"
            shift 2
            ;;
        --ent_coef)
            ENT_COEF="$2"
            shift 2
            ;;
        --kl_stop_threshold)
            KL_STOP_THRESHOLD="$2"
            shift 2
            ;;
        --kl_stop_immediate_threshold)
            KL_STOP_IMMEDIATE_THRESHOLD="$2"
            shift 2
            ;;
        --goal_cond_dim)
            GOAL_COND_DIM="$2"
            shift 2
            ;;
        --disable_noop_action)
            DISABLE_NOOP_ACTION="--disable_noop_action"
            shift
            ;;
        --use_t_norm)
            USE_T_NORM="--use_t_norm"
            shift
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --actor_lr)
            ACTOR_LR="$2"
            shift 2
            ;;
        --critic_lr)
            CRITIC_LR="$2"
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
        --hidden_sizes)
            HIDDEN_SIZES="$2"
            shift 2
            ;;
        --activation)
            ACTIVATION="$2"
            shift 2
            ;;
        --delta_clip)
            DELTA_CLIP="$2"
            shift 2
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
            echo "  --source_lineage LABEL     Source lineage label (cells will be sampled from this lineage as starting points)"
            echo "  --target_lineage LABEL     Target lineage label (goal for all episodes)"
            echo "  --source_mode MODE         Source mode: 'centroid' (use source lineage centroid) or 'sample' (sample a cell from source lineage, default)"
            echo "  --target_mode MODE         Target mode: 'centroid' (use target lineage centroid, default) or 'sample' (sample a cell from target lineage)"
            echo "  --use_negative_velocity    Use negative velocity instead of normal velocity"
            echo "  --deterministic            Use deterministic policy for visualization (default: False, uses stochastic sampling)"
            echo "  --deactivate_velocity       Deactivate velocity effect on next state (default: velocity affects state)"
            echo ""
            echo "ENVIRONMENT PARAMETERS (override config):"
            echo "  --dt FLOAT                Time step size (overrides config)"
            echo "  --lambda_progress FLOAT   Progress reward scaling factor (overrides config, default: 1.0)"
            echo "  --lambda_act FLOAT        Action penalty coefficient (overrides config)"
            echo "  --lambda_mag FLOAT         Magnitude penalty coefficient (overrides config)"
            echo "  --R_succ FLOAT            Success reward bonus (overrides config)"
            echo "  --alpha_stay FLOAT        State cost coefficient for staying near goal (overrides config, default: 0.0)"
            echo "  --eps_success_decay_on_success  Decay eps_success percentage when success rate exceeds threshold"
            echo "  --eps_success_pct FLOAT   Success radius as fraction of initial distance (default: 0.1)"
            echo "  --eps_success_success_rate_threshold FLOAT  Success-rate threshold to decay eps_success (default: 0.2)"
            echo "  --eps_success_decay_factor FLOAT  Multiplicative decay factor (default: 0.95)"
            echo "  --eps_success_decay_reward_pct FLOAT  Reward bonus percent when eps_success decays (default: 0.0)"
            echo "  --eps_success_reward_match_decay  Match success reward increase to eps_success decay"
            echo "  --perturb_clip FLOAT      Clip applied perturbation magnitude (env-side, default: none)"
            echo "  --gamma FLOAT             Discount factor for future rewards (overrides config, default: 0.99)"
            echo "  --ent_coef FLOAT          Entropy coefficient for exploration bonus (overrides config, default: 0.01)"
            echo "  --kl_stop_threshold FLOAT  Stop PPO epoch if KL exceeds this twice (default: 0.02)"
            echo "  --kl_stop_immediate_threshold FLOAT  Stop PPO epoch if KL exceeds this once (default: 0.03)"
            echo "  --goal_cond_dim INT       Goal conditioning projection dim (default: 32)"
            echo "  --disable_noop_action     Disallow no-op action (force perturbation each step)"
            echo "  --use_t_norm              Include normalized time in policy conditioning"
            echo "  --lr FLOAT                Learning rate (overrides config, default: 3e-4)"
            echo "  --actor_lr FLOAT          Actor learning rate (default: LR)"
            echo "  --critic_lr FLOAT         Critic learning rate (default: LR)"
            echo ""
            echo "OFF-MANIFOLD PENALTY PARAMETERS:"
            echo "  --gmm_path PATH           Path to saved GMM (.pkl). If not provided and lambda_off > 0, will fit automatically"
            echo "  --gmm_components N        Number of GMM components (default: 32)"
            echo "  --lambda_off FLOAT        Off-manifold penalty coefficient (default: 0.0, disabled)"
            echo ""
            echo "VISUALIZATION PARAMETERS:"
            echo "  --n_viz_trajectories N     Number of example trajectories to visualize (default: 3)"
            echo "  --viz_embedding METHOD     Embedding method: 'pca' or 'umap' (default: pca)"
            echo "  --skip_viz                Skip trajectory visualization after training"
            echo ""
            echo "ARCHITECTURE PARAMETERS:"
            echo "  --hidden_sizes LIST        Comma-separated hidden sizes (default: 128,128)"
            echo "  --activation NAME          Activation function: relu or tanh (default: relu)"
            echo "  --delta_clip FLOAT         Clip magnitude to [-x, x] (default: none)"
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
if [[ -n "$SOURCE_LINEAGE" ]]; then
    PYTHON_ARGS+=(--source_lineage "$SOURCE_LINEAGE")
fi
if [[ -n "$TARGET_LINEAGE" ]]; then
    PYTHON_ARGS+=(--target_lineage "$TARGET_LINEAGE")
fi
if [[ -n "$SOURCE_MODE" ]]; then
    PYTHON_ARGS+=(--source_mode "$SOURCE_MODE")
fi
if [[ -n "$TARGET_MODE" ]]; then
    PYTHON_ARGS+=(--target_mode "$TARGET_MODE")
fi

if [[ -n "$DETERMINISTIC" ]]; then
    PYTHON_ARGS+=("$DETERMINISTIC")
fi
if [[ -n "$DEACTIVATE_VELOCITY" ]]; then
    PYTHON_ARGS+=("$DEACTIVATE_VELOCITY")
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
if [[ -n "$ALPHA_STAY" ]]; then
    PYTHON_ARGS+=(--alpha_stay "$ALPHA_STAY")
fi
if [[ -n "$EPS_SUCCESS_DECAY_ON_SUCCESS" ]]; then
    PYTHON_ARGS+=(--eps_success_decay_on_success)
fi
if [[ -n "$EPS_SUCCESS_PCT" ]]; then
    PYTHON_ARGS+=(--eps_success_pct "$EPS_SUCCESS_PCT")
fi
if [[ -n "$EPS_SUCCESS_SUCCESS_RATE_THRESHOLD" ]]; then
    PYTHON_ARGS+=(--eps_success_success_rate_threshold "$EPS_SUCCESS_SUCCESS_RATE_THRESHOLD")
fi
if [[ -n "$EPS_SUCCESS_DECAY_FACTOR" ]]; then
    PYTHON_ARGS+=(--eps_success_decay_factor "$EPS_SUCCESS_DECAY_FACTOR")
fi
if [[ -n "$EPS_SUCCESS_DECAY_REWARD_PCT" ]]; then
    PYTHON_ARGS+=(--eps_success_decay_reward_pct "$EPS_SUCCESS_DECAY_REWARD_PCT")
fi
if [[ -n "$EPS_SUCCESS_REWARD_MATCH_DECAY" ]]; then
    PYTHON_ARGS+=(--eps_success_reward_match_decay)
fi
if [[ -n "$PERTURB_CLIP" ]]; then
    PYTHON_ARGS+=(--perturb_clip "$PERTURB_CLIP")
fi
if [[ -n "$GAMMA" ]]; then
    PYTHON_ARGS+=(--gamma "$GAMMA")
fi
if [[ -n "$ENT_COEF" ]]; then
    PYTHON_ARGS+=(--ent_coef "$ENT_COEF")
fi
if [[ -n "$KL_STOP_THRESHOLD" ]]; then
    PYTHON_ARGS+=(--kl_stop_threshold "$KL_STOP_THRESHOLD")
fi
if [[ -n "$KL_STOP_IMMEDIATE_THRESHOLD" ]]; then
    PYTHON_ARGS+=(--kl_stop_immediate_threshold "$KL_STOP_IMMEDIATE_THRESHOLD")
fi
if [[ -n "$GOAL_COND_DIM" ]]; then
    PYTHON_ARGS+=(--goal_cond_dim "$GOAL_COND_DIM")
fi
if [[ -n "$DISABLE_NOOP_ACTION" ]]; then
    PYTHON_ARGS+=(--disable_noop_action)
fi
if [[ -n "$USE_T_NORM" ]]; then
    PYTHON_ARGS+=(--use_t_norm)
fi
if [[ -n "$LR" ]]; then
    PYTHON_ARGS+=(--lr "$LR")
fi
if [[ -n "$ACTOR_LR" ]]; then
    PYTHON_ARGS+=(--actor_lr "$ACTOR_LR")
fi
if [[ -n "$CRITIC_LR" ]]; then
    PYTHON_ARGS+=(--critic_lr "$CRITIC_LR")
fi
if [[ -n "$GMM_PATH" ]]; then
    PYTHON_ARGS+=(--gmm_path "$GMM_PATH")
fi
if [[ -n "$GMM_COMPONENTS" ]]; then
    PYTHON_ARGS+=(--gmm_components "$GMM_COMPONENTS")
fi
if [[ -n "$LAMBDA_OFF" ]]; then
    PYTHON_ARGS+=(--lambda_off "$LAMBDA_OFF")
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
if [[ -n "$HIDDEN_SIZES" ]]; then
    PYTHON_ARGS+=(--hidden_sizes "$HIDDEN_SIZES")
fi
if [[ -n "$ACTIVATION" ]]; then
    PYTHON_ARGS+=(--activation "$ACTIVATION")
fi
if [[ -n "$DELTA_CLIP" ]]; then
    PYTHON_ARGS+=(--delta_clip "$DELTA_CLIP")
fi

# Run the RL training script
echo "Running RL training script..."
python -m lineagevi.rl.train "${PYTHON_ARGS[@]}"

echo ""
echo "=========================================="
echo "RL training completed successfully!"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="
