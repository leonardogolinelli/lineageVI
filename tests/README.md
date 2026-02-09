# Test Scripts

This directory contains test scripts for LineageVI model training.

## Structure

```
tests/
├── lineagevi/       # Tests for LineageVI (VAE) model training
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

## Notes

- Test scripts use smaller hyperparameters (fewer epochs, smaller batches) for quick testing
- Outputs are saved to `test_outputs/` directory
- All scripts support `--seed` for reproducibility
