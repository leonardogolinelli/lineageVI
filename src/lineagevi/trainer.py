import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from typing import List, Tuple, Optional, Dict

import torch
import torch.optim as optim
import numpy as np
import scanpy as sc

from .dataloader import make_dataloader
from .model import LineageVIModel  # refactored impl below


def _kl_weight_from_schedule(
    epoch: int,
    total_epochs: int,
    schedule: str,
    kl_weight: float,
    kl_weight_min: float,
    kl_weight_max: float,
    n_cycles: int,
    cyclical_style: str = "triangle",
    cycle_ramp_frac: float = 0.5,
) -> float:
    """
    Compute KL weight for a given epoch under the chosen schedule.
    epoch is 1-based (1 .. total_epochs).
    - "linear": ramp min->max over first cycle_ramp_frac of total epochs, then hold at max.
      (cycle_ramp_frac=1 gives ramp over all epochs.)
    When schedule is "cyclical", cyclical_style can be:
    - "triangle": weight goes min->max->min within each cycle (symmetric triangle).
    - "fu": ramp min->max over first cycle_ramp_frac of each cycle, then hold at max.
    """
    if schedule == "none":
        return kl_weight
    if schedule == "linear":
        if total_epochs <= 1:
            return kl_weight_max
        # Ramp over first cycle_ramp_frac of epochs, then hold at max (if ramp_frac < 1)
        R = max(1e-9, min(1.0, cycle_ramp_frac))
        ramp_epochs = max(1, int(total_epochs * R))
        if epoch <= ramp_epochs:
            if ramp_epochs <= 1:
                frac = 1.0
            else:
                frac = (epoch - 1) / (ramp_epochs - 1)
        else:
            frac = 1.0
        return kl_weight_min + (kl_weight_max - kl_weight_min) * frac
    if schedule == "cyclical":
        if total_epochs <= 1 or n_cycles <= 0:
            return kl_weight_max
        period = total_epochs / n_cycles
        t = epoch - 1  # 0-based
        pos_in_period = (t % period) / period  # in [0, 1)
        if cyclical_style == "fu":
            # Ramp over first cycle_ramp_frac of period, then hold at max
            R = max(1e-9, min(1.0, cycle_ramp_frac))
            if pos_in_period <= R:
                frac = pos_in_period / R
            else:
                frac = 1.0
            return kl_weight_min + (kl_weight_max - kl_weight_min) * frac
        # default: "triangle"
        if pos_in_period < 0.5:
            frac = 2.0 * pos_in_period
        else:
            frac = 2.0 * (1.0 - pos_in_period)
        return kl_weight_min + (kl_weight_max - kl_weight_min) * frac
    return kl_weight


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
        train_size: Optional[float] = None,
        lr_regime1: Optional[float] = None,
        lr_regime2: Optional[float] = None,
        velocity_loss_weight_gene: float = 1.0,
        velocity_loss_weight_gp: float = 1.0,
        kl_weight_schedule: str = "none",
        kl_weight: float = 1e-5,
        kl_weight_min: float = 0.0,
        kl_weight_max: float = 1e-5,
        kl_weight_n_cycles: int = 2,
        kl_cyclical_style: str = "triangle",
        kl_cycle_ramp_frac: float = 0.5,
        monitor_genes: Optional[List[str]] = None,
        monitor_negative_velo: bool = True,
        monitor_every_epochs: int = 1,
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
            Default learning rate for both regimes (used when lr_regime1/lr_regime2 are not set).
        lr_regime1 : float, optional
            Learning rate for regime 1 (expression reconstruction). If None, uses lr.
        lr_regime2 : float, optional
            Learning rate for regime 2 (velocity prediction). If None, uses lr.
        epochs1 : int
            Number of epochs for regime 1 (expression reconstruction).
        epochs2 : int
            Number of epochs for regime 2 (velocity prediction).
        seeds : Tuple[int, int, int]
            Random seeds for (model initialization, regime 1, regime 2).
        output_dir : str
            Directory to save model weights.
        train_size : float, optional
            Fraction of cells to use for training (0 < train_size < 1). The rest is used for
            validation. If None, all cells are used for training and no validation loss is computed.
        velocity_loss_weight_gene : float, default 1.0
            Weight for the gene-level velocity loss (expression space).
        velocity_loss_weight_gp : float, default 1.0
            Weight for the gene program velocity loss (latent space).
        kl_weight_schedule : str, default "none"
            Schedule for KL weight in regime 1: "none" (constant), "linear" (anneal from
            kl_weight_min to kl_weight_max), or "cyclical" (see kl_cyclical_style: "triangle" or "fu").
        kl_weight : float, default 1e-5
            Constant KL weight when kl_weight_schedule is "none".
        kl_weight_min : float, default 0.0
            Minimum KL weight for "linear" and "cyclical" schedules.
        kl_weight_max : float, default 1e-5
            Maximum KL weight for "linear" and "cyclical" schedules.
        kl_weight_n_cycles : int, default 2
            Number of cycles for "cyclical" schedule (ignored otherwise).
        kl_cyclical_style : str, default "triangle"
            When kl_weight_schedule is "cyclical": "triangle" (min->max->min per cycle)
            or "fu" (ramp min->max over first kl_cycle_ramp_frac of each cycle, then hold at max).
        kl_cycle_ramp_frac : float, default 0.5
            For "linear": fraction of total epochs used for ramp min->max, then hold at max (1 = ramp all epochs).
            For "fu" cyclical: fraction of each cycle used for ramp; remainder hold at max. In (0, 1].
        monitor_genes : List[str], optional
            List of gene names to monitor during training. Phase plane plots will be
            generated for these genes during both regimes and saved to
            output_dir/training_plots/ with filenames like {gene_name}_epoch_{epoch:03d}.png.
        monitor_negative_velo : bool, default True
            Whether to use negative velocities in monitoring plots.
        monitor_every_epochs : int, default 1
            Generate monitoring plots every N epochs. Plots are always generated at
            epoch 0 and the last epoch of each regime if monitor_genes is provided.
        
        Returns
        -------
        Dict[str, List[float]]
            Training history with keys:
            - 'regime1_loss': List of training total losses for regime 1 (recon + kl_weight*kl)
            - 'regime1_recon_loss': List of training reconstruction losses for regime 1
            - 'regime1_kl_loss': List of training KL losses for regime 1
            - 'regime1_kl_weight': List of KL weight used each epoch (schedule)
            - 'regime2_velocity_loss': List of training weighted velocity losses for regime 2
            - 'regime2_velocity_loss_gene': List of training gene-level velocity losses for regime 2
            - 'regime2_velocity_loss_gp': List of training gene-program velocity losses for regime 2
            - 'regime1_val_loss', 'regime1_val_recon_loss', 'regime1_val_kl_loss': validation (if train_size is set)
            - 'regime2_velocity_val_loss', 'regime2_velocity_val_loss_gene', 'regime2_velocity_val_loss_gp': validation (if train_size is set)
        
        Notes
        -----
        The model weights are saved to {output_dir}/vae_velocity_model.pt.
        After training, call model.get_model_outputs() to annotate adata with velocities.
        """
        os.makedirs(output_dir, exist_ok=True)
        self._set_seeds(seeds[0])

        history = {"regime1_loss": [], "regime2_velocity_loss": []}
        if train_size is not None:
            history["regime1_val_loss"] = []
            history["regime2_velocity_val_loss"] = []

        # Train/validation split
        n_cells = self.adata.n_obs
        train_indices = None
        val_indices = None
        if train_size is not None:
            if not 0 < train_size < 1:
                raise ValueError("train_size must be in (0, 1).")
            rng = np.random.default_rng(seeds[1])
            perm = rng.permutation(n_cells)
            n_train = int(n_cells * train_size)
            n_train = max(1, min(n_train, n_cells - 1))
            train_indices = perm[:n_train]
            val_indices = perm[n_train:]
            if self.verbose:
                print(f"Train/val split: {len(train_indices)} train, {len(val_indices)} val (train_size={train_size})")

        def _make_loader(regime1: bool, indices, shuffle: bool, seed):
            return make_dataloader(
                self.adata,
                first_regime=regime1,
                K=K,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                unspliced_key=self.unspliced_key,
                spliced_key=self.spliced_key,
                latent_key=self.latent_key,
                nn_key=self.nn_key,
            cluster_key=self.model.cluster_key,
            cluster_to_idx=cluster_to_idx,
            indices=indices,
        )

        cluster_to_idx = self.model.cluster_to_idx if self.model.cluster_key is not None else None

        # ------- Regime 1 -------
        loader1 = _make_loader(True, train_indices, shuffle=True, seed=seeds[1])
        val_loader1 = _make_loader(True, val_indices, shuffle=False, seed=None) if val_indices is not None else None
        if kl_weight_schedule not in ("none", "linear", "cyclical"):
            raise ValueError(
                f"kl_weight_schedule must be 'none', 'linear', or 'cyclical', got {kl_weight_schedule!r}"
            )
        if kl_cyclical_style not in ("triangle", "fu"):
            raise ValueError(
                f"kl_cyclical_style must be 'triangle' or 'fu', got {kl_cyclical_style!r}"
            )
        lr1 = lr_regime1 if lr_regime1 is not None else lr
        lr2 = lr_regime2 if lr_regime2 is not None else lr
        r1_losses, r1_val, r1_recon, r1_kl, r1_val_recon, r1_val_kl, r1_kl_weight = self._train_regime1(
            loader1, lr=lr1, epochs=epochs1, val_loader=val_loader1,
            kl_weight_schedule=kl_weight_schedule,
            kl_weight=kl_weight,
            kl_weight_min=kl_weight_min,
            kl_weight_max=kl_weight_max,
            kl_weight_n_cycles=kl_weight_n_cycles,
            kl_cyclical_style=kl_cyclical_style,
            kl_cycle_ramp_frac=kl_cycle_ramp_frac,
            monitor_genes=monitor_genes, output_dir=output_dir, monitor_negative_velo=monitor_negative_velo, monitor_every_epochs=monitor_every_epochs,
        )
        history["regime1_loss"] = r1_losses
        history["regime1_recon_loss"] = r1_recon
        history["regime1_kl_loss"] = r1_kl
        history["regime1_kl_weight"] = r1_kl_weight
        if r1_val is not None:
            history["regime1_val_loss"] = r1_val
            history["regime1_val_recon_loss"] = r1_val_recon
            history["regime1_val_kl_loss"] = r1_val_kl

        # Latent for all cells (used by regime 2 dataloader)
        full_loader1 = _make_loader(True, None, shuffle=False, seed=None)
        z = self._compute_latent(full_loader1)
        self.adata.obsm[self.latent_key] = z

        # ------- Regime 2 -------
        loader2 = _make_loader(False, train_indices, shuffle=True, seed=seeds[2])
        val_loader2 = _make_loader(False, val_indices, shuffle=False, seed=None) if val_indices is not None else None
        r2_losses, r2_val, r2_gene, r2_gp, r2_val_gene, r2_val_gp = self._train_regime2(
            loader2, lr=lr2, epochs=epochs2, val_loader=val_loader2,
            velocity_loss_weight_gene=velocity_loss_weight_gene,
            velocity_loss_weight_gp=velocity_loss_weight_gp,
            monitor_genes=monitor_genes, output_dir=output_dir, monitor_negative_velo=monitor_negative_velo, monitor_every_epochs=monitor_every_epochs,
        )
        history["regime2_velocity_loss"] = r2_losses
        history["regime2_velocity_loss_gene"] = r2_gene
        history["regime2_velocity_loss_gp"] = r2_gp
        if r2_val is not None:
            history["regime2_velocity_val_loss"] = r2_val
            history["regime2_velocity_val_loss_gene"] = r2_val_gene
            history["regime2_velocity_val_loss_gp"] = r2_val_gp

        # Plot loss curves over epochs
        self._plot_loss_curves(history, output_dir)

        # Save model weights and configuration
        import json
        from pathlib import Path
        
        # Save state dict
        model_path = f"{output_dir}/vae_velocity_model.pt"
        torch.save(self.model.state_dict(), model_path)
        
        # Save model configuration for easy loading later
        # Infer n_hidden from encoder layer sizes
        n_hidden = None
        if hasattr(self.model.encoder, 'encoder') and len(self.model.encoder.encoder) > 0:
            first_layer = self.model.encoder.encoder[0]
            if isinstance(first_layer, torch.nn.Linear):
                n_hidden = first_layer.out_features
        
        config = {
            "n_hidden": n_hidden,
            "mask_key": "I",  # Default, could be stored if made configurable
            "cluster_key": self.model.cluster_key,
            "cluster_embedding_dim": self.model.cluster_embedding_dim if self.model.cluster_embedding is not None else None,
            "n_latent": self.model.n_latent,
            "n_genes": self.model.n_genes,
            "lr_regime1": lr1,
            "lr_regime2": lr2,
        }
        config_path = f"{output_dir}/model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        if self.verbose:
            print(f"Saved model  → {model_path}")
            print(f"Saved config → {config_path}")
            print("Note: Call model.get_model_outputs() to annotate adata with velocities")

        return history

    # ------- Pieces -------

    def _train_regime1(
        self,
        loader,
        lr: float,
        epochs: int,
        val_loader: Optional[object] = None,
        kl_weight_schedule: str = "none",
        kl_weight: float = 1e-5,
        kl_weight_min: float = 0.0,
        kl_weight_max: float = 1e-5,
        kl_weight_n_cycles: int = 2,
        kl_cyclical_style: str = "triangle",
        kl_cycle_ramp_frac: float = 0.5,
        monitor_genes: Optional[List[str]] = None,
        output_dir: str = ".",
        monitor_negative_velo: bool = True,
        monitor_every_epochs: int = 1,
    ) -> Tuple[List[float], Optional[List[float]], List[float], List[float], Optional[List[float]], Optional[List[float]], List[float]]:
        # Freeze velocity_decoder, cluster_embedding, and CLS embedding; unfreeze encoder & gene_decoder
        for p in self.model.velocity_decoder.parameters():
            p.requires_grad = False
        if self.model.cluster_embedding is not None:
            for p in self.model.cluster_embedding.parameters():
                p.requires_grad = False
        for group in (self.model.encoder, self.model.gene_decoder):
            for p in group.parameters():
                p.requires_grad = True

        self.model.first_regime = True
        self.model.train()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

        losses = []
        losses_recon = []
        losses_kl = []
        kl_weights = []
        val_losses = [] if val_loader is not None else None
        val_recon = [] if val_loader is not None else None
        val_kl = [] if val_loader is not None else None
        n_train = len(loader.dataset)
        for epoch in range(1, epochs + 1):
            current_kl_weight = _kl_weight_from_schedule(
                epoch, epochs, kl_weight_schedule, kl_weight,
                kl_weight_min, kl_weight_max, kl_weight_n_cycles,
                kl_cyclical_style, kl_cycle_ramp_frac,
            )
            self.model.train()
            running = 0.0
            running_recon = 0.0
            running_kl = 0.0
            for batch in loader:
                if len(batch) == 4:  # Has cluster indices
                    x, idx, x_neigh, cluster_idx = batch
                    cluster_idx = cluster_idx.to(self.device)
                elif len(batch) == 3:
                    x, idx, x_neigh = batch
                    cluster_idx = None
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")
                x = x.to(self.device)
                x_neigh = x_neigh.to(self.device)
                out = self.model(x, cluster_indices=cluster_idx)
                recon  = out["recon"]
                mean   = out["mean"]
                logvar = out["logvar"]

                loss_recon = self.model.reconstruction_loss(recon, x)
                loss_kl = self.model.kl_divergence(mean, logvar)
                loss = loss_recon + current_kl_weight * loss_kl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                b = x.size(0)
                running += loss.item() * b
                running_recon += loss_recon.item() * b
                running_kl += loss_kl.item() * b

            epoch_loss = running / n_train
            epoch_recon = running_recon / n_train
            epoch_kl = running_kl / n_train
            losses.append(float(epoch_loss))
            losses_recon.append(float(epoch_recon))
            losses_kl.append(float(epoch_kl))
            kl_weights.append(float(current_kl_weight))

            # Validation loss (same KL weight as training for this epoch)
            if val_loader is not None:
                val_total, val_recon_epoch, val_kl_epoch = self._eval_loss_regime1(val_loader, kl_weight=current_kl_weight)
                val_losses.append(float(val_total))
                val_recon.append(float(val_recon_epoch))
                val_kl.append(float(val_kl_epoch))
                if self.verbose:
                    print(f"[Regime1] Epoch {epoch}/{epochs} - train loss: {epoch_loss:.4f} (recon: {epoch_recon:.4f}, kl: {epoch_kl:.4f}) - val loss: {val_total:.4f} (recon: {val_recon_epoch:.4f}, kl: {val_kl_epoch:.4f})")
            elif self.verbose:
                print(f"[Regime1] Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} (recon: {epoch_recon:.4f}, kl: {epoch_kl:.4f})")
        return losses, val_losses, losses_recon, losses_kl, val_recon, val_kl, kl_weights

    def _eval_loss_regime1(self, val_loader, kl_weight: float = 1e-5) -> Tuple[float, float, float]:
        """Compute mean regime-1 loss on validation set (no grad). Returns (total, recon, kl)."""
        self.model.eval()
        running = 0.0
        running_recon = 0.0
        running_kl = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    x, idx, x_neigh, cluster_idx = batch
                    cluster_idx = cluster_idx.to(self.device)
                elif len(batch) == 3:
                    x, idx, x_neigh = batch
                    cluster_idx = None
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")
                x = x.to(self.device)
                out = self.model(x, cluster_indices=cluster_idx)
                loss_recon = self.model.reconstruction_loss(out["recon"], x)
                loss_kl = self.model.kl_divergence(out["mean"], out["logvar"])
                loss = loss_recon + kl_weight * loss_kl
                b = x.size(0)
                running += loss.item() * b
                running_recon += loss_recon.item() * b
                running_kl += loss_kl.item() * b
                n += b
        if n == 0:
            return 0.0, 0.0, 0.0
        return running / n, running_recon / n, running_kl / n

    def _eval_loss_regime2(
        self,
        val_loader,
        velocity_loss_weight_gene: float = 1.0,
        velocity_loss_weight_gp: float = 1.0,
    ) -> Tuple[float, float, float]:
        """Compute mean velocity loss on validation set (no grad). Returns (total_weighted, loss_gene, loss_gp)."""
        self.model.eval()
        running = 0.0
        running_gene = 0.0
        running_gp = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 6:
                    x, idx, x_neigh, z, z_neigh, cluster_idx = batch
                    cluster_idx = cluster_idx.to(self.device)
                elif len(batch) == 5:
                    x, idx, x_neigh, z, z_neigh = batch
                    cluster_idx = None
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")
                x, x_neigh, z, z_neigh = [t.to(self.device) for t in (x, x_neigh, z, z_neigh)]
                out = self.model(x, cluster_indices=cluster_idx)
                v_pred = out["velocity"]
                v_gp = out["velocity_gp"]
                loss_gene = self.model.velocity_loss(v_pred, x, x_neigh)
                loss_gp = self.model.velocity_loss(v_gp, z, z_neigh)
                loss_vel = velocity_loss_weight_gene * loss_gene + velocity_loss_weight_gp * loss_gp
                b = x.size(0)
                running += loss_vel.item() * b
                running_gene += loss_gene.item() * b
                running_gp += loss_gp.item() * b
                n += b
        if n == 0:
            return 0.0, 0.0, 0.0
        return running / n, running_gene / n, running_gp / n

    def _compute_latent(self, loader) -> np.ndarray:
        self.model.eval()
        n_cells = loader.dataset.adata.n_obs
        latent_list, idx_all = [], []

        with torch.no_grad():
            for batch in loader:
                if len(batch) == 4:
                    x, idx, x_neigh, cluster_idx = batch
                elif len(batch) == 3:
                    x, idx, x_neigh = batch
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")
                x = x.to(self.device)
                _, mu, _ = self.model.encoder(x)
                latent_list.append(mu.cpu())
                idx_all.extend(idx.numpy().tolist())

        z_concat = torch.cat(latent_list, dim=0)
        latent_dim = z_concat.size(1)
        z_all = torch.zeros((n_cells, latent_dim), dtype=z_concat.dtype)
        z_all[idx_all] = z_concat
        return z_all.numpy()

    def _train_regime2(
        self,
        loader,
        lr: float,
        epochs: int,
        val_loader: Optional[object] = None,
        velocity_loss_weight_gene: float = 1.0,
        velocity_loss_weight_gp: float = 1.0,
        monitor_genes: Optional[List[str]] = None,
        output_dir: str = ".",
        monitor_negative_velo: bool = True,
        monitor_every_epochs: int = 1,
    ) -> Tuple[List[float], Optional[List[float]], List[float], List[float], Optional[List[float]], Optional[List[float]]]:
        # Freeze encoder & gene_decoder; unfreeze velocity_decoder, cluster_embedding, and CLS embedding
        for group in (self.model.encoder, self.model.gene_decoder):
            for p in group.parameters():
                p.requires_grad = False
        for p in self.model.velocity_decoder.parameters():
            p.requires_grad = True
        if self.model.cluster_embedding is not None:
            for p in self.model.cluster_embedding.parameters():
                p.requires_grad = True

        self.model.first_regime = False
        self.model.train()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

        # Generate epoch 0 plots if monitoring is enabled
        if monitor_genes is not None and len(monitor_genes) > 0:
            self._generate_monitoring_plots(0, monitor_genes, output_dir, monitor_negative_velo, regime=2, total_epochs=epochs)

        losses = []
        losses_gene = []
        losses_gp = []
        val_losses = [] if val_loader is not None else None
        val_gene = [] if val_loader is not None else None
        val_gp = [] if val_loader is not None else None
        n_train = len(loader.dataset)
        for epoch in range(1, epochs + 1):
            self.model.train()
            running = 0.0
            running_gene = 0.0
            running_gp = 0.0
            for batch in loader:
                if len(batch) == 6:  # Has cluster indices
                    x, idx, x_neigh, z, z_neigh, cluster_idx = batch
                    cluster_idx = cluster_idx.to(self.device)
                elif len(batch) == 5:
                    x, idx, x_neigh, z, z_neigh = batch
                    cluster_idx = None
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")
                x, x_neigh, z, z_neigh = [t.to(self.device) for t in (x, x_neigh, z, z_neigh)]
                out = self.model(x, cluster_indices=cluster_idx)
                v_pred = out["velocity"]   # (B, 2*G) gene-level velocity
                v_gp   = out["velocity_gp"]  # (B, L) gene program velocity

                # Weighted sum of gene-level and gene program velocity losses
                loss_gene = self.model.velocity_loss(v_pred, x, x_neigh)
                loss_gp = self.model.velocity_loss(v_gp, z, z_neigh)
                loss_vel = velocity_loss_weight_gene * loss_gene + velocity_loss_weight_gp * loss_gp

                optimizer.zero_grad()
                loss_vel.backward()
                optimizer.step()

                b = x.size(0)
                running += loss_vel.item() * b
                running_gene += loss_gene.item() * b
                running_gp += loss_gp.item() * b

            epoch_loss = running / n_train
            epoch_gene = running_gene / n_train
            epoch_gp = running_gp / n_train
            losses.append(float(epoch_loss))
            losses_gene.append(float(epoch_gene))
            losses_gp.append(float(epoch_gp))

            # Validation loss
            if val_loader is not None:
                val_total, val_gene_epoch, val_gp_epoch = self._eval_loss_regime2(
                    val_loader,
                    velocity_loss_weight_gene=velocity_loss_weight_gene,
                    velocity_loss_weight_gp=velocity_loss_weight_gp,
                )
                val_losses.append(float(val_total))
                val_gene.append(float(val_gene_epoch))
                val_gp.append(float(val_gp_epoch))
                if self.verbose:
                    print(f"[Regime2] Epoch {epoch}/{epochs} - train loss: {epoch_loss:.4f} (gene: {epoch_gene:.4f}, gp: {epoch_gp:.4f}) - val loss: {val_total:.4f} (gene: {val_gene_epoch:.4f}, gp: {val_gp_epoch:.4f})")
            elif self.verbose:
                print(f"[Regime2] Epoch {epoch}/{epochs} - Velocity Loss: {epoch_loss:.4f} (gene: {epoch_gene:.4f}, gp: {epoch_gp:.4f})")
            
            # Generate monitoring plots if requested
            if monitor_genes is not None and len(monitor_genes) > 0:
                should_plot = (epoch == epochs) or (epoch % monitor_every_epochs == 0)
                if should_plot:
                    self._generate_monitoring_plots(epoch, monitor_genes, output_dir, monitor_negative_velo, regime=2, total_epochs=epochs)
        return losses, val_losses, losses_gene, losses_gp, val_gene, val_gp

    def _generate_monitoring_plots(self, epoch: int, monitor_genes: List[str], output_dir: str, monitor_negative_velo: bool = True, regime: int = 2, total_epochs: int = 50):
        """
        Generate phase plane plots for monitoring genes during training.
        
        Parameters
        ----------
        epoch : int
            Current epoch number (0-indexed: 0 means before training, 1-N means after epoch).
        monitor_genes : List[str]
            List of gene names to plot.
        output_dir : str
            Output directory for saving plots.
        monitor_negative_velo : bool, default True
            Whether to use negative velocities in monitoring plots. If True, shows negative
            velocities (matches scVelo convention). If False, shows positive velocities.
        regime : int, default 2
            Training regime (1 or 2). Used to organize plots into subdirectories.
        total_epochs : int, default 50
            Total number of epochs for this regime. Used for determining if this is the last epoch.
        """
        import os
        import matplotlib.pyplot as plt
        from .plots import plot_phase_plane
        
        # Create base training plots directory
        base_plots_dir = os.path.join(output_dir, "training_plots")
        os.makedirs(base_plots_dir, exist_ok=True)
        
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
        
        # Get cluster key from model - use the exact cluster_key used in initialization
        # This ensures colors match the cluster_key specified in LineageVI initialization
        cluster_key = self.model.cluster_key if self.model.cluster_key is not None else "clusters"
        
        # Generate plots for each monitoring gene
        for gene_name in monitor_genes:
            if gene_name not in temp_adata.var_names:
                if self.verbose:
                    print(f"Warning: Gene '{gene_name}' not found in adata.var_names, skipping...")
                continue
            
            # Create a folder for each monitored gene
            gene_plots_dir = os.path.join(base_plots_dir, gene_name)
            os.makedirs(gene_plots_dir, exist_ok=True)
            
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
                    save_path=os.path.join(gene_plots_dir, f"{gene_name}_regime{regime}_epoch_{epoch:03d}.png"),
                    cluster_key=cluster_key,  # Use cluster_key from model
                    unspliced_key=self.unspliced_key,
                    spliced_key=self.spliced_key,
                    velocity_u_key="velocity_u",
                    velocity_s_key="velocity",
                )
                plt.close(fig)  # Close to free memory
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to generate plot for gene '{gene_name}' at epoch {epoch}: {e}")
        
        if self.verbose:
            print(f"Generated monitoring plots for regime {regime}, epoch {epoch} → {base_plots_dir}")

    def _plot_loss_curves(self, history: Dict[str, List[float]], output_dir: str) -> None:
        """Plot and save training/validation loss curves to output_dir/training_plots/loss_curves.png."""
        import os
        import matplotlib.pyplot as plt

        base_plots_dir = os.path.join(output_dir, "training_plots")
        os.makedirs(base_plots_dir, exist_ok=True)
        has_val = "regime1_val_loss" in history

        fig, axes = plt.subplots(2, 4, figsize=(14, 8))
        epochs_r1 = list(range(1, len(history["regime1_loss"]) + 1))
        epochs_r2 = list(range(1, len(history["regime2_velocity_loss"]) + 1))

        # Regime 1: total, recon, kl, KL weight
        ax = axes[0, 0]
        ax.plot(epochs_r1, history["regime1_loss"], label="train", color="C0")
        if has_val:
            ax.plot(epochs_r1, history["regime1_val_loss"], label="val", color="C1", linestyle="--")
        ax.set_title("Regime 1: total loss")
        ax.set_xlabel("Epoch")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(epochs_r1, history["regime1_recon_loss"], label="train", color="C0")
        if has_val:
            ax.plot(epochs_r1, history["regime1_val_recon_loss"], label="val", color="C1", linestyle="--")
        ax.set_title("Regime 1: recon loss")
        ax.set_xlabel("Epoch")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        ax.plot(epochs_r1, history["regime1_kl_loss"], label="train", color="C0")
        if has_val:
            ax.plot(epochs_r1, history["regime1_val_kl_loss"], label="val", color="C1", linestyle="--")
        ax.set_title("Regime 1: KL loss")
        ax.set_xlabel("Epoch")
        kl_vals = list(history["regime1_kl_loss"])
        if has_val:
            kl_vals.extend(history["regime1_val_kl_loss"])
        if kl_vals:
            y_max = float(np.percentile(kl_vals, 99))
            ax.set_ylim(bottom=0, top=y_max)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 3]
        ax.plot(epochs_r1, history["regime1_kl_weight"], color="C2")
        ax.set_title("Regime 1: KL weight")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

        # Regime 2: total, gene, gp
        ax = axes[1, 0]
        ax.plot(epochs_r2, history["regime2_velocity_loss"], label="train", color="C0")
        if has_val:
            ax.plot(epochs_r2, history["regime2_velocity_val_loss"], label="val", color="C1", linestyle="--")
        ax.set_title("Regime 2: velocity loss (total)")
        ax.set_xlabel("Epoch")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(epochs_r2, history["regime2_velocity_loss_gene"], label="train", color="C0")
        if has_val:
            ax.plot(epochs_r2, history["regime2_velocity_val_loss_gene"], label="val", color="C1", linestyle="--")
        ax.set_title("Regime 2: velocity loss (gene)")
        ax.set_xlabel("Epoch")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        ax.plot(epochs_r2, history["regime2_velocity_loss_gp"], label="train", color="C0")
        if has_val:
            ax.plot(epochs_r2, history["regime2_velocity_val_loss_gp"], label="val", color="C1", linestyle="--")
        ax.set_title("Regime 2: velocity loss (GP)")
        ax.set_xlabel("Epoch")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        axes[1, 3].set_visible(False)

        plt.tight_layout()
        save_path = os.path.join(base_plots_dir, "loss_curves.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        if self.verbose:
            print(f"Saved loss curves → {save_path}")

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