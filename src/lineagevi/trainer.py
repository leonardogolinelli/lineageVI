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

class _Trainer:
    """
    Internal trainer for LineageVI model using two-regime training.
    
    This class handles the two-regime training process:
    1. **Regime 1**: Expression reconstruction - trains encoder and gene decoder
    2. **Regime 2**: Velocity prediction - trains velocity decoder
    
    The trainer uses different data loaders and loss functions for each regime
    to optimize the model components in the correct order.
    
    Parameters
    ----------
    model : LineageVIModel
        The LineageVI neural network model to train.
    adata : AnnData
        Single-cell data for training.
    device : torch.device, optional
        Device to run training on. Defaults to CUDA if available.
    verbose : int, default 1
        Verbosity level for training progress.
    unspliced_key : str, default "Mu"
        Key for unspliced counts in adata.layers.
    spliced_key : str, default "Ms"
        Key for spliced counts in adata.layers.
    latent_key : str, default "z"
        Key for latent representations in adata.obsm.
    nn_key : str, default "indices"
        Key for nearest neighbor indices in adata.uns.
    
    Attributes
    ----------
    model : LineageVIModel
        The neural network model being trained.
    adata : AnnData
        Training data.
    device : torch.device
        Device used for computations.
    verbose : int
        Verbosity level.
    
    Notes
    -----
    This is an internal class used by the LineageVI API. Users should not
    instantiate this class directly.
    """
    
    def __init__(
        self,
        model: LineageVIModel,
        adata: sc.AnnData,
        device: Optional[torch.device] = None,
        verbose: int = 1,
        unspliced_key: str = "Mu",
        spliced_key: str = "Ms",
        latent_key: str = "z",
        nn_key: str = "indices",

        ):

        self.model = model
        self.adata = adata
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = int(verbose)
        self.model = self.model.to(self.device)
    
        self.unspliced_key = unspliced_key
        self.spliced_key = spliced_key
        self.latent_key = latent_key
        self.nn_key = nn_key



    # High-level procedure
    def fit(
        self,
        K: int,
        batch_size: int,
        lr: float,
        epochs1: int,
        epochs2: int,
        seeds: Tuple[int, int, int],
        output_dir: str,
        monitor_genes: Optional[List[str]] = None,
        monitor_negative_velo: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the LineageVI model using two-regime training.
        
        This method implements the two-regime training strategy:
        1. **Regime 1**: Expression reconstruction - trains encoder and gene decoder
        2. **Regime 2**: Velocity prediction - trains velocity decoder
        
        Parameters
        ----------
        K : int
            Number of nearest neighbors for velocity computation.
        batch_size : int
            Batch size for training.
        lr : float
            Learning rate for optimization.
        epochs1 : int
            Number of epochs for regime 1 (expression reconstruction).
        epochs2 : int
            Number of epochs for regime 2 (velocity prediction).
        seeds : Tuple[int, int, int]
            Random seeds for (model initialization, regime 1, regime 2).
        output_dir : str
            Directory to save model weights.
        monitor_genes : List[str], optional
            List of gene names to monitor during training. Phase plane plots will be
            generated for these genes at each epoch during regime 2 and saved to
            output_dir/training_plots/ with filenames like {gene_name}_epoch_{epoch:03d}.png.
        
        Returns
        -------
        Dict[str, List[float]]
            Training history with keys:
            - 'regime1_loss': List of reconstruction losses for regime 1
            - 'regime2_velocity_loss': List of velocity losses for regime 2
        
        Notes
        -----
        The model weights are saved to {output_dir}/vae_velocity_model.pt.
        After training, call model.get_model_outputs() to annotate adata with velocities.
        """
        os.makedirs(output_dir, exist_ok=True)
        self._set_seeds(seeds[0])

        history = {"regime1_loss": [], "regime2_velocity_loss": []}

        # ------- Regime 1 -------
        loader1 = make_dataloader(
            self.adata,
            first_regime=True,
            K=K,
            batch_size=batch_size,
            shuffle=True,  # Always shuffle for better training
            seed=seeds[1],
            unspliced_key=self.unspliced_key,
            spliced_key=self.spliced_key,
            latent_key=self.latent_key,
            nn_key=self.nn_key,

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
            shuffle=True,  # Always shuffle for better training
            seed=seeds[2],
            unspliced_key=self.unspliced_key,
            spliced_key=self.spliced_key,
            latent_key=self.latent_key,
            nn_key=self.nn_key
        )

        r2_losses = self._train_regime2(loader2, lr=lr, epochs=epochs2, monitor_genes=monitor_genes, output_dir=output_dir, monitor_negative_velo=monitor_negative_velo)
        history["regime2_velocity_loss"] = r2_losses

        # Save model weights only (no automatic annotation)
        torch.save(self.model.state_dict(), f"{output_dir}/vae_velocity_model.pt")
        if self.verbose:
            print(f"Saved model  → {output_dir}/vae_velocity_model.pt")
            print("Note: Call model.get_model_outputs() to annotate adata with velocities")

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

    def _train_regime2(self, loader, lr: float, epochs: int, monitor_genes: Optional[List[str]] = None, output_dir: str = ".", monitor_negative_velo: bool = True) -> List[float]:
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
            
            # Generate monitoring plots if requested (only during regime 2)
            if monitor_genes is not None and len(monitor_genes) > 0:
                self._generate_monitoring_plots(epoch, monitor_genes, output_dir, monitor_negative_velo)
        return losses

    def _generate_monitoring_plots(self, epoch: int, monitor_genes: List[str], output_dir: str, monitor_negative_velo: bool = True):
        """
        Generate phase plane plots for monitoring genes during regime 2 training.
        
        Parameters
        ----------
        epoch : int
            Current regime 2 epoch number.
        monitor_genes : List[str]
            List of gene names to plot.
        output_dir : str
            Output directory for saving plots.
        monitor_negative_velo : bool, default True
            Whether to use negative velocities in monitoring plots. If True, shows negative
            velocities (matches scVelo convention). If False, shows positive velocities.
        """
        import os
        import matplotlib.pyplot as plt
        from .plots import plot_phase_plane
        
        # Create training plots directory
        plots_dir = os.path.join(output_dir, "training_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Get current model outputs for plotting
        with torch.no_grad():
            # Create a temporary adata with current velocities
            temp_adata = self.adata.copy()
            
            # Get model outputs for all cells
            outputs = self.model._get_model_outputs(
                temp_adata,
                n_samples=1,
                return_mean=True,
                return_negative_velo=monitor_negative_velo,
                save_to_adata=False,
                unspliced_key=self.unspliced_key,
                spliced_key=self.spliced_key,
                latent_key=self.latent_key,
                nn_key=self.nn_key,
                batch_size=256,
            )
            
            # Add velocities to temporary adata
            temp_adata.layers["velocity_u"] = outputs["velocity_u"]
            temp_adata.layers["velocity"] = outputs["velocity"]
        
        # Generate plots for each monitoring gene
        for gene_name in monitor_genes:
            if gene_name not in temp_adata.var_names:
                if self.verbose:
                    print(f"Warning: Gene '{gene_name}' not found in adata.var_names, skipping...")
                continue
            
            try:
                # Generate phase plane plot
                fig, ax = plot_phase_plane(
                    temp_adata,
                    gene_name,
                    u_scale=0.1,
                    s_scale=0.1,
                    alpha=1,
                    head_width=0.02,
                    head_length=0.03,
                    length_includes_head=False,
                    show_plot=False,  # Don't display, just save
                    save_plot=True,
                    save_path=os.path.join(plots_dir, f"{gene_name}_epoch_{epoch:03d}.png"),
                    unspliced_key=self.unspliced_key,
                    spliced_key=self.spliced_key,
                    velocity_u_key="velocity_u",
                    velocity_s_key="velocity",
                )
                plt.close(fig)  # Close to free memory
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to generate plot for gene '{gene_name}' at epoch {epoch}: {e}")
        
        if self.verbose and epoch % 10 == 0:  # Print every 10 epochs to avoid spam
            print(f"Generated monitoring plots for epoch {epoch} → {plots_dir}")




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