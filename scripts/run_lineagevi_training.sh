#!/bin/bash
# Bash script to run LineageVI test training in conda environment 'test3'

set -e  # Exit on error

# Default values
CONDA_ENV="test3"

# Data loading
DATASET_NAME="pancreas"
ADATA_PATH=""
ANNOTATION_FILE="/Users/lgolinelli/git/lineageVI/gene_sets/Mm_development_or_pancreas_msigdb.gmt"

# Preprocessing
MIN_SHARED_COUNTS=20
N_TOP_GENES=2000
MIN_GENES_PER_TERM=12
N_PCS=100
N_NEIGHBORS=200
K_NEIGHBORS=20
SKIP_IF_PREPROCESSED=true
CLUSTER_KEY="leiden"

# Model initialization
N_HIDDEN=128
MASK_KEY="I"
UNSPLICED_KEY="Mu"
SPLICED_KEY="Ms"
NN_KEY="indices"
CLUSTER_EMBEDDING_DIM=8
SEED=1

# Training
K=10
BATCH_SIZE=256
LR=0.001
EPOCHS1=50
EPOCHS2=50
SEEDS="0,1,2"
OUTPUT_DIR_BASE="./test_outputs/lineagevi"
VERBOSE=1
MONITOR_GENES="Gnas,Rbfox3"
MONITOR_NEGATIVE_VELO=true
MONITOR_EVERY_EPOCHS=5
PLOT_CLUSTER_KEY=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        # Data loading
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --adata_path)
            ADATA_PATH="$2"
            shift 2
            ;;
        --annotation_file)
            ANNOTATION_FILE="$2"
            shift 2
            ;;
        # Preprocessing
        --min_shared_counts)
            MIN_SHARED_COUNTS="$2"
            shift 2
            ;;
        --n_top_genes)
            N_TOP_GENES="$2"
            shift 2
            ;;
        --min_genes_per_term)
            MIN_GENES_PER_TERM="$2"
            shift 2
            ;;
        --n_pcs)
            N_PCS="$2"
            shift 2
            ;;
        --n_neighbors)
            N_NEIGHBORS="$2"
            shift 2
            ;;
        --K_neighbors)
            K_NEIGHBORS="$2"
            shift 2
            ;;
        --skip_if_preprocessed)
            SKIP_IF_PREPROCESSED="$2"
            shift 2
            ;;
        --cluster_key)
            CLUSTER_KEY="$2"
            shift 2
            ;;
        # Model initialization
        --n_hidden)
            N_HIDDEN="$2"
            shift 2
            ;;
        --mask_key)
            MASK_KEY="$2"
            shift 2
            ;;
        --unspliced_key)
            UNSPLICED_KEY="$2"
            shift 2
            ;;
        --spliced_key)
            SPLICED_KEY="$2"
            shift 2
            ;;
        --nn_key)
            NN_KEY="$2"
            shift 2
            ;;
        --cluster_embedding_dim)
            CLUSTER_EMBEDDING_DIM="$2"
            shift 2
            ;;
        # Training
        --K)
            K="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --epochs1)
            EPOCHS1="$2"
            shift 2
            ;;
        --epochs2)
            EPOCHS2="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR_BASE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="$2"
            shift 2
            ;;
        --monitor_genes)
            MONITOR_GENES="$2"
            shift 2
            ;;
        --monitor_negative_velo)
            MONITOR_NEGATIVE_VELO="$2"
            shift 2
            ;;
        --monitor_every_epochs)
            MONITOR_EVERY_EPOCHS="$2"
            shift 2
            ;;
        --plot_cluster_key)
            PLOT_CLUSTER_KEY="$2"
            shift 2
            ;;
        # Other
        --seed)
            SEED="$2"
            shift 2
            ;;
        --conda_env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [DATA_OPTIONS] [PREPROCESSING_OPTIONS] [MODEL_OPTIONS] [TRAINING_OPTIONS]"
            echo ""
            echo "DATA LOADING (one of --dataset_name or --adata_path required):"
            echo "  --dataset_name NAME      Load from scvelo datasets (e.g., 'pancreas')"
            echo "  --adata_path PATH        Path to AnnData file (.h5ad)"
            echo "  --annotation_file PATH   Path to annotation file (.gmt)"
            echo ""
            echo "PREPROCESSING OPTIONS:"
            echo "  --min_shared_counts N     Minimum shared counts (default: 20)"
            echo "  --n_top_genes N           Number of top HVGs (default: 2000)"
            echo "  --min_genes_per_term N   Min genes per annotation term (default: 12)"
            echo "  --n_pcs N                Number of PCs for moments (default: 100)"
            echo "  --n_neighbors N          Neighbors for moments smoothing (default: 200)"
            echo "  --K_neighbors N          Neighbors for model (default: 20)"
            echo "  --skip_if_preprocessed   Skip if already preprocessed (default: true)"
            echo "  --cluster_key KEY        Key for cluster labels in adata.obs"
            echo ""
            echo "MODEL INITIALIZATION OPTIONS:"
            echo "  --n_hidden N             Hidden units (default: 128)"
            echo "  --mask_key KEY           Gene program mask key (default: I)"
            echo "  --unspliced_key KEY      Unspliced layer key (default: unspliced)"
            echo "  --spliced_key KEY        Spliced layer key (default: spliced)"
            echo "  --nn_key KEY             Neighbor indices key (default: indices)"
            echo "  --cluster_embedding_dim N Cluster embedding dim (default: 8)"
            echo ""
            echo "TRAINING OPTIONS:"
            echo "  --K N                    Number of neighbors for loss (default: 10)"
            echo "  --batch_size N           Batch size (default: 256)"
            echo "  --lr FLOAT               Learning rate (default: 0.001)"
            echo "  --epochs1 N              Epochs for regime 1 (default: 50)"
            echo "  --epochs2 N              Epochs for regime 2 (default: 50)"
            echo "  --seeds STR              Comma-separated seeds (default: 0,1,2)"
            echo "  --output_dir PATH        Output directory base (default: ./test_outputs/lineagevi)"
            echo "                           Timestamp will be appended automatically"
            echo "  --verbose N              Verbosity level (default: 1)"
            echo "  --monitor_genes STR      Comma-separated gene names to monitor"
            echo "  --monitor_negative_velo  Monitor negative velocities (default: true)"
            echo "  --monitor_every_epochs N Monitor frequency (default: 5)"
            echo "  --plot_cluster_key KEY   Cluster key for velocity embedding plots (default: cluster_key or 'leiden')"
            echo ""
            echo "OTHER OPTIONS:"
            echo "  --seed N                 Random seed (default: None for non-deterministic)"
            echo "  --conda_env NAME         Conda environment (default: test3)"
            echo "  -h, --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if adata file exists (if provided)
if [[ -n "$ADATA_PATH" && ! -f "$ADATA_PATH" ]]; then
    echo "Error: AnnData file not found: $ADATA_PATH"
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Generate timestamp and create output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_DIR_BASE}_${TIMESTAMP}"

echo "=========================================="
echo "LineageVI Test Training Script"
echo "=========================================="
echo "Conda environment: $CONDA_ENV"
if [[ -n "$DATASET_NAME" ]]; then
    echo "Dataset name: $DATASET_NAME"
else
    echo "AnnData path: $ADATA_PATH"
fi
if [[ -n "$ANNOTATION_FILE" ]]; then
    echo "Annotation file: $ANNOTATION_FILE"
fi
echo "Output directory: $OUTPUT_DIR"
echo "Epochs (regime 1): $EPOCHS1"
echo "Epochs (regime 2): $EPOCHS2"
if [[ -n "$SEED" ]]; then
    echo "Seed: $SEED"
else
    echo "Seed: None (non-deterministic)"
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

# Validate data loading arguments
if [[ -n "$DATASET_NAME" && -n "$ADATA_PATH" ]]; then
    echo "Error: Both --dataset_name and --adata_path are specified. Please provide only one."
    exit 1
fi
if [[ -z "$DATASET_NAME" && -z "$ADATA_PATH" ]]; then
    echo "Error: Either --dataset_name or --adata_path must be specified."
    exit 1
fi
# Check if adata file exists (if provided)
if [[ -n "$ADATA_PATH" && ! -f "$ADATA_PATH" ]]; then
    echo "Error: AnnData file not found: $ADATA_PATH"
    exit 1
fi

# Build Python command with all arguments
PYTHON_ARGS=()

# Data loading
if [[ -n "$DATASET_NAME" ]]; then
    PYTHON_ARGS+=(--dataset_name "$DATASET_NAME")
fi
if [[ -n "$ADATA_PATH" ]]; then
    PYTHON_ARGS+=(--adata_path "$ADATA_PATH")
fi
if [[ -n "$ANNOTATION_FILE" ]]; then
    PYTHON_ARGS+=(--annotation_file "$ANNOTATION_FILE")
fi

# Preprocessing
PYTHON_ARGS+=(--min_shared_counts "$MIN_SHARED_COUNTS")
PYTHON_ARGS+=(--n_top_genes "$N_TOP_GENES")
PYTHON_ARGS+=(--min_genes_per_term "$MIN_GENES_PER_TERM")
PYTHON_ARGS+=(--n_pcs "$N_PCS")
PYTHON_ARGS+=(--n_neighbors "$N_NEIGHBORS")
PYTHON_ARGS+=(--K_neighbors "$K_NEIGHBORS")
PYTHON_ARGS+=(--skip_if_preprocessed "$SKIP_IF_PREPROCESSED")
if [[ -n "$CLUSTER_KEY" ]]; then
    PYTHON_ARGS+=(--cluster_key "$CLUSTER_KEY")
fi

# Model initialization
PYTHON_ARGS+=(--n_hidden "$N_HIDDEN")
PYTHON_ARGS+=(--mask_key "$MASK_KEY")
PYTHON_ARGS+=(--unspliced_key "$UNSPLICED_KEY")
PYTHON_ARGS+=(--spliced_key "$SPLICED_KEY")
PYTHON_ARGS+=(--nn_key "$NN_KEY")
PYTHON_ARGS+=(--cluster_embedding_dim "$CLUSTER_EMBEDDING_DIM")
if [[ -n "$SEED" ]]; then
    PYTHON_ARGS+=(--seed "$SEED")
fi

# Training
PYTHON_ARGS+=(--K "$K")
PYTHON_ARGS+=(--batch_size "$BATCH_SIZE")
PYTHON_ARGS+=(--lr "$LR")
PYTHON_ARGS+=(--epochs1 "$EPOCHS1")
PYTHON_ARGS+=(--epochs2 "$EPOCHS2")
PYTHON_ARGS+=(--seeds "$SEEDS")
PYTHON_ARGS+=(--output_dir "$OUTPUT_DIR")
PYTHON_ARGS+=(--verbose "$VERBOSE")
if [[ -n "$MONITOR_GENES" ]]; then
    PYTHON_ARGS+=(--monitor_genes "$MONITOR_GENES")
fi
PYTHON_ARGS+=(--monitor_negative_velo "$MONITOR_NEGATIVE_VELO")
PYTHON_ARGS+=(--monitor_every_epochs "$MONITOR_EVERY_EPOCHS")
if [[ -n "$PLOT_CLUSTER_KEY" ]]; then
    PYTHON_ARGS+=(--plot_cluster_key "$PLOT_CLUSTER_KEY")
fi

# Run the test script
echo "Running test training script..."
python -m tests.lineagevi.test_training "${PYTHON_ARGS[@]}"

echo ""
echo "=========================================="
echo "Test completed successfully!"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
