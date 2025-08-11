import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from typing import List, Tuple, Optional, Dict

import torch
import torch.optim as optim
import numpy as np
import scanpy as sc

from dataloader import make_dataloader
from model import lineageVIModel  # refactored impl below


# -------------------- Trainer engine --------------------

class LineageVITrainer:
    def __init__(
        self,
        model: lineageVIModel,
        adata: sc.AnnData,
        device: Optional[torch.device] = None,
        verbose: int = 1,
    ):
        self.model = model
        self.adata = adata
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = int(verbose)
        self.model = self.model.to(self.device)

    # High-level procedure
    def fit(
        self,
        K: int,
        batch_size: int,
        lr: float,
        epochs1: int,
        epochs2: int,
        shuffle_regime1: bool,
        shuffle_regime2: bool,
        seeds: Tuple[int, int, int],
        output_dir: str,
    ) -> Dict[str, List[float]]:
        os.makedirs(output_dir, exist_ok=True)
        self._set_seeds(seeds[0])

        history = {"regime1_loss": [], "regime2_velocity_loss": []}

        # ------- Regime 1 -------
        loader1 = make_dataloader(
            self.adata,
            first_regime=True,
            K=K,
            batch_size=batch_size,
            shuffle=shuffle_regime1,
            seed=seeds[1],
        )
        r1_losses = self._train_regime1(loader1, lr=lr, epochs=epochs1)
        history["regime1_loss"] = r1_losses

        # Latent for all cells (used by regime 2 dataloader)
        z = self._compute_latent(loader1)
        self.adata.obsm["z"] = z

        # ------- Regime 2 -------
        loader2 = make_dataloader(
            self.adata,
            first_regime=False,
            K=K,
            batch_size=batch_size,
            shuffle=shuffle_regime2,
            seed=seeds[2],
        )

        r2_losses = self._train_regime2(loader2, lr=lr, epochs=epochs2)
        history["regime2_velocity_loss"] = r2_losses

        # Annotate & save
        self._annotate_adata(loader2)
        self.adata.write(f"{output_dir}/adata_with_velocity.h5ad")
        torch.save(self.model.state_dict(), f"{output_dir}/vae_velocity_model.pt")
        if self.verbose:
            print(f"Saved AnnData → {output_dir}/adata_with_velocity.h5ad")
            print(f"Saved model  → {output_dir}/vae_velocity_model.pt")

        return history

    # ------- Pieces -------

    def _train_regime1(self, loader, lr: float, epochs: int) -> List[float]:
        # Freeze velocity_decoder; unfreeze encoder & gene_decoder
        for p in self.model.velocity_decoder.parameters():
            p.requires_grad = False
        for group in (self.model.encoder, self.model.gene_decoder):
            for p in group.parameters():
                p.requires_grad = True

        self.model.first_regime = True
        self.model.train()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

        losses = []
        for epoch in range(1, epochs + 1):
            running = 0.0
            for x, idx, x_neigh in loader:
                x = x.to(self.device)
                x_neigh = x_neigh.to(self.device)

                recon, v_pred, v_gp, mean, logvar = self.model(x)
                loss_recon = self.model.reconstruction_loss(recon, x)
                loss_kl = self.model.kl_divergence(mean, logvar)
                loss = loss_recon + 1e-5 * loss_kl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running += loss.item() * x.size(0)

            epoch_loss = running / len(loader.dataset)
            losses.append(float(epoch_loss))
            if self.verbose:
                print(f"[Regime1] Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}")
        return losses

    def _compute_latent(self, loader) -> np.ndarray:
        self.model.eval()
        n_cells = loader.dataset.adata.n_obs
        latent_list, idx_all = [], []

        with torch.no_grad():
            for x, idx, x_neigh in loader:
                x = x.to(self.device)
                z, _, _ = self.model.encoder(x)
                latent_list.append(z.cpu())
                idx_all.extend(idx.numpy().tolist())

        z_concat = torch.cat(latent_list, dim=0)
        latent_dim = z_concat.size(1)
        z_all = torch.zeros((n_cells, latent_dim), dtype=z_concat.dtype)
        z_all[idx_all] = z_concat
        return z_all.numpy()

    def _train_regime2(self, loader, lr: float, epochs: int) -> List[float]:
        # Freeze encoder & gene_decoder; unfreeze velocity_decoder
        for group in (self.model.encoder, self.model.gene_decoder):
            for p in group.parameters():
                p.requires_grad = False
        for p in self.model.velocity_decoder.parameters():
            p.requires_grad = True

        self.model.first_regime = False
        self.model.train()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

        losses = []
        for epoch in range(1, epochs + 1):
            running = 0.0
            for x, idx, x_neigh, z, z_neigh in loader:
                x, x_neigh, z, z_neigh = [t.to(self.device) for t in (x, x_neigh, z, z_neigh)]

                _, v_pred, v_gp, _, _ = self.model(x)

                xz = torch.cat([x, z], dim=1)
                xz_neigh = torch.cat([x_neigh, z_neigh], dim=2)
                v_comb = torch.cat([v_pred, v_gp], dim=1)
                loss_vel = self.model.velocity_loss(v_comb, xz, xz_neigh)

                optimizer.zero_grad()
                loss_vel.backward()
                optimizer.step()

                running += loss_vel.item() * x.size(0)

            epoch_loss = running / len(loader.dataset)
            losses.append(float(epoch_loss))
            if self.verbose:
                print(f"[Regime2] Epoch {epoch}/{epochs} - Velocity Loss: {epoch_loss:.4f}")
        return losses

    def _annotate_adata(self, loader) -> None:
        """Write recon, velocity (u/s), and velocity_gp into AnnData."""
        self.model.eval()
        adata = self.adata
        n_cells = adata.n_obs

        # layer keys live on the model
        unspliced_key = getattr(self.model, "unspliced_key", "unspliced")
        spliced_key = getattr(self.model, "spliced_key", "spliced")

        # Infer G from the unspliced layer
        u = adata.layers[unspliced_key]
        G = u.shape[1]

        recon_all = np.zeros((n_cells, G), dtype=np.float32)
        vel_all = np.zeros((n_cells, 2 * G), dtype=np.float32)
        gp_all = None
        gp_dim = None

        with torch.no_grad():
            for x, idx, x_neigh, z, z_neigh in loader:
                x = x.to(self.device)
                recon_batch, vel_batch, gp_batch, _, _ = self.model(x)

                recon_np = recon_batch.cpu().numpy()
                vel_np = vel_batch.cpu().numpy()
                gp_np = gp_batch.cpu().numpy()
                batch_idx = idx.numpy()

                recon_all[batch_idx] = recon_np
                vel_all[batch_idx] = vel_np

                if gp_dim is None:
                    gp_dim = gp_np.shape[1]
                    gp_all = np.zeros((n_cells, gp_dim), dtype=np.float32)
                gp_all[batch_idx] = gp_np

        vel_u = vel_all[:, :G]
        vel_s = vel_all[:, G:]

        adata.layers["recon"] = recon_all
        adata.layers["velocity_u"] = vel_u
        adata.layers["velocity"] = vel_s
        adata.obsm["velocity_gp"] = gp_all

    @staticmethod
    def _set_seeds(seed: int):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# -------------------- User-facing model wrapper --------------------

class lineageVI(lineageVIModel):
    """
    Initialize once with data + config; call .fit(...) to train.
    Users specify unspliced_key/spliced_key ONLY here.
    """

    def __init__(
        self,
        adata: sc.AnnData,
        n_hidden: int = 128,
        mask_key: str = "I",
        gene_prior: bool = True,
        seed: int = 0,
        device: Optional[torch.device] = None,
        *,
        unspliced_key: str = "unspliced",
        spliced_key: str = "spliced",
    ):
        super().__init__(adata, n_hidden=n_hidden, mask_key=mask_key, gene_prior=gene_prior, seed=seed)
        self._adata_ref = adata
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unspliced_key = unspliced_key
        self.spliced_key = spliced_key
        self.to(self.device)

    def fit(
        self,
        K: int = 10,
        batch_size: int = 1024,
        lr: float = 1e-3,
        epochs1: int = 50,
        epochs2: int = 50,
        shuffle_regime1: bool = True,
        shuffle_regime2: bool = False,
        seeds: Tuple[int, int, int] = (0, 1, 2),
        output_dir: str = "output_model",
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """
        Two-regime training + annotation + saving. Returns dict of epoch losses.
        """
        engine = LineageVITrainer(self, self._adata_ref, device=self.device, verbose=verbose)
        history = engine.fit(
            K=K,
            batch_size=batch_size,
            lr=lr,
            epochs1=epochs1,
            epochs2=epochs2,
            shuffle_regime1=shuffle_regime1,
            shuffle_regime2=shuffle_regime2,
            seeds=seeds,
            output_dir=output_dir,
        )
        self._adata_ref = engine.adata
        return history

    # Convenience accessors
    def get_latent(self) -> np.ndarray:
        if "z" not in self._adata_ref.obsm:
            raise RuntimeError("Latent not computed yet. Call model.fit(...) first.")
        return self._adata_ref.obsm["z"]

    def get_adata(self) -> sc.AnnData:
        return self._adata_ref

# -------------------- Example usage --------------------
if __name__ == "__main__":
    adata = sc.read_h5ad("/home/lgolinelli/git/lineageVI/input_processed_anndata/pancreas_PCA_on_Ms_200moments.h5ad")

    model = lineageVI(
        adata,
        n_hidden=128,
        mask_key="I",
        gene_prior=True,
        seed=0,
        unspliced_key="unspliced",
        spliced_key="spliced",
    )

    history = model.fit(
        K=10,
        batch_size=1024,
        lr=1e-3,
        epochs1=50,
        epochs2=50,
        shuffle_regime1=True,
        shuffle_regime2=False,
        seeds=(0, 1, 2),
        output_dir="output_model_test2",
        verbose=1,
    )

    z = model.get_latent()
    adata_out = model.get_adata()