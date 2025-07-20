import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from model import VAEModel

# ──────── USER SETTINGS ────────────────────────────────────────────────────────
# Data files
ADATAPATH     = "path/to/adata.h5ad"
MASKPATH      = "path/to/mask.npy"
UNSPLICED_KEY = "unspliced"
SPLICED_KEY   = "spliced"

# Model dimensions
N_LATENT   = 32
N_HIDDEN   = 128

# Training regime 1 (ELBO)
EPOCHS1     = 50
LR1         = 1e-3
BATCH_SIZE  = 64

# Training regime 2 (velocity)
EPOCHS2     = 30
LR2         = 5e-4
K_NEIGHBORS = 10

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ────────────────────────────────────────────────────────────────────────────────

class SingleCellDataset(Dataset):
    """Yields (spliced_counts, cell_index)."""
    def __init__(self, adata, layer_key):
        X = adata.layers[layer_key]
        if sp.issparse(X):
            X = X.toarray()
        self.X = torch.from_numpy(np.asarray(X)).float()
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], idx

def train_elbo(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x_spliced, _ in loader:
        x_spliced = x_spliced.to(device)
        optimizer.zero_grad()
        recon, vel, vel_gp, mu, logvar = model(x_spliced)
        loss_recon = model.reconstruction_loss(recon, x_spliced)
        loss_kld   = model.kl_divergence(mu, logvar)
        loss       = loss_recon + loss_kld
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_spliced.size(0)
    return total_loss / len(loader.dataset)

def train_velocity(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x_spliced, idx in loader:
        x_spliced = x_spliced.to(device)
        idx        = idx.to(device)
        optimizer.zero_grad()
        recon, vel, vel_gp, mu, logvar = model(x_spliced)
        loss_vel = model.velocity_loss(vel, x_spliced, idx)
        loss_vel.backward()
        optimizer.step()
        total_loss += loss_vel.item() * x_spliced.size(0)
    return total_loss / len(loader.dataset)

def main():
    # 1) Load data & mask
    adata   = sc.read_h5ad(ADATAPATH)
    mask_np = np.load(MASKPATH)
    mask    = torch.from_numpy(mask_np.astype(np.float32))

    # 2) Build model
    model = VAEModel(
        adata,
        unspliced_key=UNSPLICED_KEY,
        spliced_key=SPLICED_KEY,
        n_input=adata.n_vars,
        n_latent=N_LATENT,
        n_hidden=N_HIDDEN,
        mask=mask
    ).to(DEVICE)

    # 3) Precompute KNN on full unspliced+spliced
    full_data = model.full_data.cpu().numpy()
    knn       = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric="cosine")
    knn.fit(full_data)
    neigh_idx = knn.kneighbors(return_distance=False)  # (n_cells, K)
    model.K = K_NEIGHBORS
    model.nn_indices = torch.from_numpy(neigh_idx).long().to(DEVICE)

    # 4) DataLoader
    dataset = SingleCellDataset(adata, layer_key=SPLICED_KEY)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ─────────── Regime 1: ELBO ───────────────────────────────────────────────
    # freeze velocity decoder
    for p in model.velocity_decoder.parameters():
        p.requires_grad = False

    opt1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR1
    )

    print("Starting Regime 1 (ELBO)…")
    for epoch in range(1, EPOCHS1+1):
        avg_loss = train_elbo(model, loader, opt1, DEVICE)
        print(f"Epoch {epoch}/{EPOCHS1} — ELBO loss: {avg_loss:.4f}")

    # ─────────── Regime 2: Velocity ────────────────────────────────────────────
    # freeze encoder + gene decoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.gene_decoder.parameters():
        p.requires_grad = False
    # unfreeze velocity decoder
    for p in model.velocity_decoder.parameters():
        p.requires_grad = True

    opt2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR2
    )

    print("\nStarting Regime 2 (Velocity)…")
    for epoch in range(1, EPOCHS2+1):
        avg_loss = train_velocity(model, loader, opt2, DEVICE)
        print(f"Epoch {epoch}/{EPOCHS2} — Velocity loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
