import os
import torch
import torch.optim as optim
import numpy as np
import scanpy as sc
from datetime import datetime
from dataloader import make_dataloader
from model import VAEModel


class VAETrainer:
    def __init__(
        self,
        adata,
        n_hidden: int = 128,
        mask_key: str = 'I',
        gene_prior: bool = True,
        seeds: tuple = (0, 1, 2),
        K: int = 10,
        batch_size: int = 1024,
        lr: float = 1e-3,
        epochs1: int = 50,
        epochs2: int = 50,
        device: torch.device = None,
        save_path: str = None,           # <-- new
    ):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adata = adata
        self.n_hidden = n_hidden
        self.mask_key = mask_key
        self.gene_prior = gene_prior
        self.seeds = seeds
        self.K = K
        self.batch_size = batch_size
        self.lr = lr
        self.epochs1 = epochs1
        self.epochs2 = epochs2
        self.save_path = save_path       # <-- store it

        # initialize model & optimizer
        self.model = VAEModel(
            adata,
            n_hidden=self.n_hidden,
            mask_key=self.mask_key,
            gene_prior=self.gene_prior,
            seed=self.seeds[0],
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train_regime1(self):
        for p in self.model.velocity_decoder.parameters():
            p.requires_grad = False
        for module in (self.model.encoder, self.model.gene_decoder):
            for p in module.parameters():
                p.requires_grad = True

        self.model.first_regime = True
        self.model.train()
        loader = make_dataloader(
            self.adata,
            first_regime=True,
            K=self.K,
            batch_size=self.batch_size,
            shuffle=True,
            seed=self.seeds[1],
        )
        for epoch in range(1, self.epochs1 + 1):
            running_loss = 0.0
            for x, idx, x_neigh in loader:
                x, x_neigh = x.to(self.device), x_neigh.to(self.device)
                recon, _, _, mean, logvar = self.model(x)
                loss = (
                    self.model.reconstruction_loss(recon, x)
                    + 1e-5 * self.model.kl_divergence(mean, logvar)
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * x.size(0)
            epoch_loss = running_loss / len(loader.dataset)
            print(f"[Regime1] Epoch {epoch}/{self.epochs1} – Loss: {epoch_loss:.4f}")

    def compute_latent(self):
        self.model.eval()
        loader = make_dataloader(
            self.adata,
            first_regime=True,
            K=self.K,
            batch_size=self.batch_size,
            shuffle=True,
            seed=self.seeds[1],
        )
        n_cells = loader.dataset.adata.n_obs
        zs, idxs = [], []
        with torch.no_grad():
            for x, idx, _ in loader:
                x = x.to(self.device)
                z, _, _ = self.model.encoder(x)
                zs.append(z.cpu())
                idxs.extend(idx.numpy().tolist())
        z_cat = torch.cat(zs, dim=0)
        latent = torch.zeros((n_cells, z_cat.size(1)), dtype=z_cat.dtype)
        latent[idxs] = z_cat
        return latent.numpy()

    def train_regime2(self):
        for module in (self.model.encoder, self.model.gene_decoder):
            for p in module.parameters():
                p.requires_grad = False
        for p in self.model.velocity_decoder.parameters():
            p.requires_grad = True

        self.model.first_regime = False
        self.model.train()
        loader = make_dataloader(
            self.adata,
            first_regime=False,
            K=self.K,
            batch_size=self.batch_size,
            shuffle=False,
            seed=self.seeds[2],
        )
        for epoch in range(1, self.epochs2 + 1):
            running_loss = 0.0
            for x, idx, x_neigh, z, z_neigh in loader:
                x, x_neigh, z, z_neigh = [
                    t.to(self.device) for t in (x, x_neigh, z, z_neigh)
                ]
                _, v_pred, v_gp, _, _ = self.model(x)
                xz = torch.cat([x, z], dim=1)
                xz_neigh = torch.cat([x_neigh, z_neigh], dim=2)
                v_comb = torch.cat([v_pred, v_gp], dim=1)
                loss = self.model.velocity_loss(v_comb, xz, xz_neigh)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * x.size(0)
            epoch_loss = running_loss / len(loader.dataset)
            print(f"[Regime2] Epoch {epoch}/{self.epochs2} – Velocity Loss: {epoch_loss:.4f}")

    def annotate_adata(self, spliced_key: str = 'spliced'):
        self.model.eval()
        n_cells = self.adata.n_obs
        u_layer = self.adata.layers[spliced_key]
        G = u_layer.shape[1]
        recon_all = np.zeros((n_cells, G * 2), dtype=np.float32)
        vel_all = np.zeros((n_cells, G * 2), dtype=np.float32)
        gp_all = None

        loader = make_dataloader(
            self.adata,
            first_regime=False,
            K=self.K,
            batch_size=self.batch_size,
            shuffle=False,
            seed=self.seeds[2],
        )
        with torch.no_grad():
            for x, idx, x_neigh, z, z_neigh in loader:
                x = x.to(self.device)
                recon_b, vel_b, gp_b, _, _ = self.model(x)
                recon_np, vel_np, gp_np = (
                    recon_b.cpu().numpy(),
                    vel_b.cpu().numpy(),
                    gp_b.cpu().numpy(),
                )
                recon_all[idx] = recon_np
                vel_all[idx] = vel_np
                if gp_all is None:
                    gp_all = np.zeros((n_cells, gp_np.shape[1]), dtype=np.float32)
                gp_all[idx] = gp_np

        recon_u, recon_s = np.split(recon_all, 2, axis=1)
        vel_u, vel_s = np.split(vel_all, 2, axis=1)
        self.adata.layers['recon_u'] = recon_u
        self.adata.layers['recon']   = recon_s
        self.adata.layers['velocity_u'] = vel_u
        self.adata.layers['velocity']   = vel_s
        self.adata.obsm['velocity_gp']   = gp_all

    def train(self):
        # Regime 1
        self.train_regime1()

        # Compute & store latent embeddings
        z_emb = self.compute_latent()
        self.adata.obsm['z'] = z_emb

        # Regime 2
        self.train_regime2()

        # Annotate AnnData
        self.annotate_adata()

        # If save_path is set, make timestamped folder and dump outputs
        if self.save_path is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(self.save_path, f"outputs_{timestamp}")
            os.makedirs(out_dir, exist_ok=True)
            adata_file = os.path.join(out_dir, "adata_with_velocity.h5ad")
            model_file = os.path.join(out_dir, "vae_velocity_model.pt")
            self.adata.write(adata_file)
            torch.save(self.model.state_dict(), model_file)
            print(f"Saved AnnData to {adata_file}")
            print(f"Saved model weights to {model_file}")

        return self.adata, self.model

if __name__ == "__main__":
    adata = sc.read_h5ad(
        "/home/lgolinelli/git/lineageVI/input_processed_anndata/"
        "pancreas_PCA_on_Ms_200moments.h5ad"
    )
    # Pass a base directory where you want everything saved:
    trainer = VAETrainer(adata, save_path=".")
    annotated = trainer.train()
