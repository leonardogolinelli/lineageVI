# LineageVI

Deep learning-based RNA velocity with gene programs.

## Install

```bash
pip install -e .
```

Requires Python ≥3.9, PyTorch, scanpy, scvelo, anndata, and dependencies in `pyproject.toml`.

## Pipeline

1. **Preprocessing** — filter, normalize, annotate with gene sets (Enrichr), neighbor indices.  
   `python scripts/run_preprocessing.py --dataset pancreas --annotation_library GO_Biological_Process_2025 --output_dir ./outputs/pancreas`

2. **Regime 1** — expression reconstruction (VAE).  
   `python scripts/run_regime1.py --adata_path <run_dir>/adata_preprocessed.h5ad --output_dir <run_dir> --no_timestamp`

3. **Regime 2** — velocity training. Writes `adata_with_latent.h5ad` and `pretrained_vae.pt`.  
   `python scripts/run_regime2.py --adata_path <run_dir>/adata_preprocessed.h5ad --checkpoint_path <run_dir>/regime1/pretrained_vae_regime1.pt --output_dir <run_dir> --no_timestamp`

4. **Downstream** — velocities, UMAP, differential, heatmaps (follows the tutorial).  
   `python scripts/run_downstream.py --adata_path <run_dir>/adata_with_latent.h5ad --model_path <run_dir>/pretrained_vae.pt --output_dir <run_dir> --no_timestamp`

SLURM scripts: `scripts/slurm/run_preprocessing.slurm`, `run_regime1.slurm`, `run_regime2.slurm`, `run_downstream.slurm`. Set `OUTPUT_DIR` (and optionally use `.latest.txt` for chained jobs).

## Tutorial

`notebooks/Tutorial_Pancreas_Clean_V2.ipynb` — full workflow and downstream analysis.
