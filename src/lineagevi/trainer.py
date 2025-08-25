import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from typing import List, Tuple, Optional, Dict

import torch
import torch.optim as optim
import numpy as np
import scanpy as sc

from .dataloader import make_dataloader
from .model import LineageVIModel  # refactored impl below

# -------------------- Trainer engine --------------------

class LineageVITrainer:
    def __init__(
        self,
        model: LineageVIModel,
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

                out = self.model(x)                 # <-- dict
                recon  = out["recon"]
                mean   = out["mean"]
                logvar = out["logvar"]

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
                _, mu, _ = self.model.encoder(x)
                latent_list.append(mu.cpu())
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

                out = self.model(x)                      # <-- dict
                v_pred = out["velocity"]                 # (B, G) spliced velocities
                v_gp   = out["velocity_gp"]              # (B, L)

                # concatenate inputs/targets the same way as before
                xz       = torch.cat([x, z], dim=1)
                xz_neigh = torch.cat([x_neigh, z_neigh], dim=2)
                v_comb   = torch.cat([v_pred, v_gp], dim=1)

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
        """Write recon, velocity_u (u), velocity (s), velocity_gp, z, mean, logvar, alpha/beta/gamma into AnnData."""
        self.model.eval()
        # Force mean if multiple samples; write directly into adata
        self.model.get_model_outputs(
            adata=self.adata,
            n_samples=1,            # or >1 if you want sample-averaged annotations
            return_mean=True,
            return_negative_velo=True,
            save_to_adata=True,
        )


    @staticmethod
    def _set_seeds(seed: int):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

# -------------------- User-facing model wrapper --------------------

class LineageVI(LineageVIModel):
    """
    Initialize once with data + config; call .fit(...) to train.
    Users specify unspliced_key/spliced_key ONLY here.
    """

    def __init__(
        self,
        adata: sc.AnnData,
        n_hidden: int = 128,
        mask_key: str = "I",
        device: Optional[torch.device] = None,
        *,
        unspliced_key: str = "unspliced",
        spliced_key: str = "spliced",
    ):
        super().__init__(adata, n_hidden=n_hidden, mask_key=mask_key, seed=None)
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

if __name__ == "__main__":
    import datetime
    import lineagevi as linvi
    processed_path = '/Users/lgolinelli/git/lineageVI/notebooks/data/inputs/anndata/processed'
    output_base_path = '/Users/lgolinelli/git/lineageVI/notebooks/data/outputs'
    dataset_name = 'pancreas'
    time = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    output_dir_name =f'{dataset_name}_{time}'
    output_dir_path = os.path.join(output_base_path, output_dir_name)
    input_adata_path = os.path.join(processed_path, dataset_name+ '.h5ad')
    adata = sc.read_h5ad(input_adata_path)

    model = linvi.trainer.LineageVI(
        adata,
        n_hidden=128,
        mask_key="I",
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
        output_dir=output_dir_path,
        verbose=1,
    )

    z = model.get_latent()
    adata_out = model.get_adata()