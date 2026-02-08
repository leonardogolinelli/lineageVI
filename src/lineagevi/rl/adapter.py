"""Adapter for pretrained VAE model to provide velocity computation interface."""

from typing import Optional, Literal
import torch
import torch.nn as nn
import numpy as np
import scanpy as sc

from ..model import LineageVIModel


class VelocityVAEAdapter:
    """
    Wrapper around LineageVIModel for RL environment.
    
    Provides clean interface for velocity computation with different modes
    to handle gene expression x during rollouts.
    
    Parameters
    ----------
    model : LineageVIModel
        Pretrained VAE model (will be frozen).
    device : torch.device
        Device to run computations on.
    velocity_mode : {'latent_only', 'fixed_x', 'decode_x'}, default 'decode_x'
        How to handle gene expression x for velocity computation:
        - 'latent_only': Use zero/dummy x (may not work well with kinetic model)
        - 'fixed_x': Use fixed initial x throughout rollout
        - 'decode_x': Decode x from z using gene decoder (recommended)
    """
    
    def __init__(
        self,
        model: LineageVIModel,
        device: torch.device,
        velocity_mode: Literal["latent_only", "fixed_x", "decode_x"] = "decode_x",
    ):
        self.model = model
        self.device = device
        self.velocity_mode = velocity_mode
        
        # Freeze model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Store model dimensions
        self.n_latent = model.n_latent
        self.n_genes = model.n_genes
        
        # For fixed_x mode, we'll store initial x when provided
        self._fixed_x: Optional[torch.Tensor] = None
    
    def encode(self, adata: sc.AnnData) -> torch.Tensor:
        """
        Extract latent states from AnnData.
        
        Parameters
        ----------
        adata : AnnData
            AnnData with latent states in obsm['mean'] or obsm['z'].
            If not present, will encode from gene expression.
        
        Returns
        -------
        torch.Tensor
            Latent states of shape (n_cells, n_latent).
        """
        if "mean" in adata.obsm:
            z = torch.from_numpy(adata.obsm["mean"]).to(self.device).float()
        elif "z" in adata.obsm:
            z = torch.from_numpy(adata.obsm["z"]).to(self.device).float()
        else:
            # Encode from gene expression
            # This requires unspliced/spliced layers
            raise ValueError(
                "No latent states found in adata.obsm. "
                "Please run get_model_outputs() first or provide 'mean' or 'z' in obsm."
            )
        return z
    
    def get_gene_expression(self, adata: sc.AnnData) -> torch.Tensor:
        """
        Extract gene expression from AnnData.
        
        Parameters
        ----------
        adata : AnnData
            AnnData with unspliced/spliced layers.
        
        Returns
        -------
        torch.Tensor
            Gene expression of shape (n_cells, 2*n_genes) [unspliced, spliced].
        """
        unspliced_key = "unspliced" if "unspliced" in adata.layers else "Mu"
        spliced_key = "spliced" if "spliced" in adata.layers else "Ms"
        
        u = torch.from_numpy(np.asarray(adata.layers[unspliced_key])).to(self.device).float()
        s = torch.from_numpy(np.asarray(adata.layers[spliced_key])).to(self.device).float()
        x = torch.cat([u, s], dim=1)  # (n_cells, 2*n_genes)
        return x
    
    def set_fixed_x(self, x: torch.Tensor):
        """
        Set fixed gene expression for 'fixed_x' mode.
        
        Parameters
        ----------
        x : torch.Tensor
            Gene expression of shape (batch_size, 2*n_genes) or (2*n_genes,).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        self._fixed_x = x.to(self.device)
    
    def velocity(
        self,
        z: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        cluster_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute gene program velocities from latent states.

        Parameters
        ----------
        z : torch.Tensor
            Latent states of shape (batch_size, n_latent).
        x : torch.Tensor, optional
            Gene expression of shape (batch_size, 2*n_genes).
            If None, will be determined by velocity_mode.
        cluster_indices : torch.Tensor, optional
            Cluster indices of shape (batch_size,).
            Required if model uses cluster embeddings.

        Returns
        -------
        torch.Tensor
            Gene program velocities of shape (batch_size, n_latent).
        """
        batch_size = z.shape[0]
        
        # Handle x based on velocity_mode
        if self.velocity_mode == "latent_only":
            # Use zero x (may not work well with kinetic model)
            if x is None:
                x = torch.zeros(batch_size, 2 * self.n_genes, device=self.device)
        elif self.velocity_mode == "fixed_x":
            # Use fixed initial x
            if self._fixed_x is None:
                if x is None:
                    raise ValueError(
                        "fixed_x mode requires x to be provided initially. "
                        "Call set_fixed_x(x) first or provide x in first call."
                    )
                self.set_fixed_x(x)
            # Broadcast fixed x to batch size
            if self._fixed_x.shape[0] == 1:
                x = self._fixed_x.expand(batch_size, -1)
            else:
                x = self._fixed_x[:batch_size]
        elif self.velocity_mode == "decode_x":
            # Decode x from z
            with torch.no_grad():
                x = self.model._forward_gene_decoder(z)  # (B, n_genes)
                # Velocity decoder expects [unspliced, spliced], so duplicate for now
                # In practice, we might want to split this differently
                # For now, use x for both unspliced and spliced (approximation)
                x = torch.cat([x, x], dim=1)  # (B, 2*n_genes)
        else:
            if x is None:
                raise ValueError(f"x must be provided for velocity_mode='{self.velocity_mode}'")
        
        # Ensure x is on correct device
        x = x.to(self.device)
        
        # Compute velocity with no gradients
        with torch.no_grad():
            _, velocity_gp, _, _, _ = self.model._forward_velocity_decoder(
                z, x, cluster_indices
            )
        
        return velocity_gp  # (batch_size, n_latent)
