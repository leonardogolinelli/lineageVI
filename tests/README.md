# Test Scripts

This directory contains test scripts for both LineageVI model training and RL agent training.

## Structure

```
tests/
├── lineagevi/       # Tests for LineageVI (VAE) model training
│   └── test_training.py
├── rl/              # Tests for RL agent (PPO) training
│   └── test_training.py
└── README.md        # This file
```

## LineageVI Training Tests

Test the basic VAE model training workflow:

```bash
python -m tests.lineagevi.test_training \
    --adata_path <path_to_data.h5ad> \
    --output_dir ./test_outputs/lineagevi \
    --epochs1 10 \
    --epochs2 10
```

This will:
1. Load AnnData
2. Initialize and train LineageVI model
3. Compute model outputs
4. Save model checkpoint and outputs

## RL Agent Training Tests

Test the PPO agent training workflow:

```bash
python -m tests.rl.test_training \
    --model_path <path_to_pretrained_model.pt> \
    --adata_path <path_to_data.h5ad> \
    --lineage_key clusters \
    --output_dir ./test_outputs/rl \
    --n_iterations 10 \
    --batch_size 8 \
    --T_rollout 10
```

This will:
1. Load pretrained VAE model
2. Load AnnData and compute centroids
3. Initialize PPO environment and policy
4. Train agent for a few iterations
5. Save policy checkpoint

## Notes

- Test scripts use smaller hyperparameters (fewer epochs, smaller batches) for quick testing
- Outputs are saved to `test_outputs/` directory
- All scripts support `--seed` for reproducibility
