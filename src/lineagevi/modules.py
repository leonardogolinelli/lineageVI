import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional


class Encoder(nn.Module):
    """
    Variational encoder that maps gene expression to latent space.
    
    This encoder implements a variational autoencoder (VAE) encoder that:
    1. Takes concatenated unspliced+spliced gene expression as input
    2. Projects to a shared hidden representation
    3. Outputs mean and log-variance for latent space sampling
    4. Uses reparameterization trick for differentiable sampling
    
    Parameters
    ----------
    n_input : int
        Input dimension (2 * number of genes for unspliced+spliced).
    n_hidden : int
        Number of hidden units in the encoder network.
    n_latent : int
        Dimension of the latent space (number of gene programs).
    
    Attributes
    ----------
    encoder : nn.Sequential
        Shared encoder network with LayerNorm and ReLU activation.
    mean_layer : nn.Linear
        Linear layer that outputs latent mean.
    logvar_layer : nn.Linear
        Linear layer that outputs latent log-variance.
    
    Examples
    --------
    >>> # Create encoder for 2000 genes and 50 gene programs
    >>> encoder = Encoder(n_input=4000, n_hidden=128, n_latent=50)
    >>> 
    >>> # Forward pass
    >>> z, mean, logvar = encoder(x)  # x shape: (batch, 4000)
    >>> # z, mean, logvar shape: (batch, 50)
    """
    
    def __init__(self, n_input: int, n_hidden: int, n_latent: int):
        super().__init__()
        # shared encoder MLP
        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
        )

        # project to mean and log-variance
        self.mean_layer = nn.Linear(n_hidden, n_latent)
        self.logvar_layer = nn.Linear(n_hidden, n_latent)

    def forward(self, x: torch.Tensor, *, generator: Optional[torch.Generator] = None):
        """
        Forward pass through the encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 2*n_genes) containing
            concatenated unspliced and spliced gene expression.
        generator : torch.Generator, optional
            Random number generator for reproducible sampling.
        
        Returns
        -------
        z : torch.Tensor
            Sampled latent representations (batch_size, n_latent).
        mean : torch.Tensor
            Latent mean (batch_size, n_latent).
        logvar : torch.Tensor
            Latent log-variance (batch_size, n_latent).
        """
        # x is concatenated [u, s] so split then sum for the encoder signal
        u, s = torch.split(x, x.shape[1] // 2, dim=1)
        xs = u + s
        h = self.encoder(xs)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        z = self.reparametrize(mean, logvar, generator=generator)
        return z, mean, logvar

    @staticmethod
    def reparametrize(mean, logvar, *, generator=None):
        """
        Reparameterization trick for differentiable sampling from Gaussian.
        
        This method enables backpropagation through the sampling process by
        expressing the random sample as a deterministic function of the mean
        and variance plus a standard Gaussian noise term.
        
        Parameters
        ----------
        mean : torch.Tensor
            Mean of the Gaussian distribution.
        logvar : torch.Tensor
            Log-variance of the Gaussian distribution.
        generator : torch.Generator, optional
            Random number generator for reproducible sampling.
        
        Returns
        -------
        torch.Tensor
            Sampled values from the Gaussian distribution.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.empty_like(std).normal_(mean=0.0, std=1.0, generator=generator)
        return mean + eps * std

class MaskedLinearDecoder(nn.Module):
    """
    Linear decoder with hard mask on regression weights for gene program reconstruction.
    
    This decoder reconstructs gene expression from latent gene program representations
    using a masked linear layer. The mask ensures that each gene is only reconstructed
    from the gene programs that contain it, enforcing biological interpretability.
    
    Parameters
    ----------
    n_latent : int
        Number of gene programs (latent dimensions).
    n_output : int
        Number of genes to reconstruct.
    mask : torch.Tensor
        Binary mask of shape (n_output, n_latent) where mask[i,j] = 1 if
        gene i is in gene program j, 0 otherwise.
    
    Attributes
    ----------
    mask : torch.Tensor
        Binary mask stored as a buffer (non-trainable).
    linear : nn.Linear
        Linear layer for reconstruction.
    
    Examples
    --------
    >>> # Create decoder for 2000 genes and 50 gene programs
    >>> mask = torch.randint(0, 2, (2000, 50))  # Random binary mask
    >>> decoder = MaskedLinearDecoder(n_latent=50, n_output=2000, mask=mask)
    >>> 
    >>> # Forward pass
    >>> reconstructed = decoder(z)  # z shape: (batch, 50)
    >>> # reconstructed shape: (batch, 2000)
    """
    
    def __init__(self, n_latent: int, n_output: int, mask: torch.Tensor):
        super().__init__()
        # mask shape must be [n_output, n_latent]
        assert mask.shape == (n_output, n_latent), \
            f"Mask shape {tuple(mask.shape)} must be (n_output={n_output}, n_latent={n_latent})"
        self.register_buffer("mask", mask)

        self.linear = nn.Linear(n_latent, n_output)

        # zero out masked positions at init
        with torch.no_grad():
            self.linear.weight.mul_(self.mask)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the masked linear decoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input latent representations of shape (batch_size, n_latent).
        
        Returns
        -------
        torch.Tensor
            Reconstructed gene expression of shape (batch_size, n_output).
        
        Notes
        -----
        The mask is reapplied on every forward pass to ensure that weights
        remain masked even after gradient updates during training.
        """
        # reapply mask every forward (in case of weight updates)
        masked_w = self.linear.weight * self.mask
        return F.linear(x, masked_w, self.linear.bias)

class VelocityDecoder(nn.Module):
    """
    Velocity decoder that predicts RNA velocities in gene and gene program spaces.
    
    This decoder takes latent representations and predicts:
    1. Gene program velocities in latent space
    2. Gene-level velocities using kinetic parameters (α, β, γ)
    3. Kinetic parameters for RNA velocity modeling
    
    The gene-level velocity is computed using the kinetic model:
    - velocity_u = α - β * unspliced
    - velocity_s = β * unspliced - γ * spliced
    
    Parameters
    ----------
    n_input : int
        Input dimension (n_latent or n_latent + cluster_embedding_dim when cluster embeddings are used).
    n_hidden : int
        Number of hidden units in the shared decoder.
    n_output : int
        Output dimension (2 * number of genes for unspliced+spliced velocities).
    n_latent : int
        Number of gene programs (latent dimensions) for gp_velocity_decoder output.
    
    Attributes
    ----------
    shared_decoder : nn.Sequential
        Shared decoder network with LayerNorm and ReLU activation.
    gp_velocity_decoder : nn.Linear
        Linear layer for gene program velocities.
    gene_velocity_decoder : nn.Sequential
        Network for kinetic parameters with Softplus activation.
    _G : int
        Number of genes (n_output // 2).
    
    Examples
    --------
    >>> # Create velocity decoder for 2000 genes and 50 gene programs
    >>> decoder = VelocityDecoder(n_input=50, n_hidden=128, n_output=4000, n_latent=50)
    >>> 
    >>> # Forward pass
    >>> vel, vel_gp, α, β, γ = decoder(z, x)
    >>> # vel shape: (batch, 4000), vel_gp shape: (batch, 50)
    >>> # α, β, γ shape: (batch, 2000)
    """

    def __init__(
        self, 
        n_input: int, 
        n_hidden: int, 
        n_output: int, 
        n_latent: int,
        cluster_embedding_dim: Optional[int] = None,
        gp_embedding_dim: int = 32,
        attention_dim: Optional[int] = None,
    ):
        """
        Initialize velocity decoder.
        
        Parameters
        ----------
        n_input : int
            Input dimension (n_latent).
        n_hidden : int
            Number of hidden units in the shared decoder.
        n_output : int
            Output dimension (2 * number of genes for unspliced+spliced velocities).
        n_latent : int
            Number of gene programs (latent dimensions) for gp_velocity_decoder output.
        cluster_embedding_dim : int, optional
            Dimension of cluster embeddings. If provided, attention mechanism is used.
        gp_embedding_dim : int, default 32
            Dimension of gene program-specific embeddings used in attention.
        attention_dim : int, optional
            Dimension of attention space. If None, uses n_latent.
        """
        super().__init__()
        
        # Initialize attention mechanism if cluster embeddings are used
        self.use_attention = cluster_embedding_dim is not None
        if self.use_attention:
            self.attention = ClusterAttention(
                cluster_embedding_dim=cluster_embedding_dim,
                n_latent=n_latent,
                gp_embedding_dim=gp_embedding_dim,
                attention_dim=attention_dim
            )
            # Input to decoder is attention_dim (output of attention mechanism)
            decoder_input_dim = self.attention.attention_dim
        else:
            self.attention = None
            # Input to decoder is n_latent (no cluster embeddings)
            decoder_input_dim = n_latent
        
        self.shared_decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
        )

        # latent GP-ish velocity head (L-dim)
        self.gp_velocity_decoder = nn.Linear(n_hidden, n_latent)

        # outputs [alpha, beta, gamma] each in R^G; n_output here is 2G, so G = n_output//2
        G2 = n_output
        G = G2 // 2
        self.gene_velocity_decoder = nn.Sequential(
            nn.Linear(n_hidden, 3 * G),
            nn.Softplus()
        )
        self._G = G

    def forward(self, z: torch.Tensor, x: torch.Tensor, cluster_emb: Optional[torch.Tensor] = None):
        """
        Forward pass through the velocity decoder.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent representations of shape (batch_size, n_latent).
        x : torch.Tensor
            Input gene expression of shape (batch_size, 2*n_genes) containing
            concatenated unspliced and spliced counts.
        cluster_emb : torch.Tensor, optional
            Cluster embeddings of shape (batch_size, cluster_embedding_dim).
            Required if attention mechanism is enabled.
        
        Returns
        -------
        velocity : torch.Tensor
            Gene-level velocities of shape (batch_size, 2*n_genes) with
            concatenated unspliced and spliced velocities.
        velocity_gp : torch.Tensor
            Gene program velocities of shape (batch_size, n_latent).
        alpha : torch.Tensor
            Transcription rate parameters of shape (batch_size, n_genes).
        beta : torch.Tensor
            Splicing rate parameters of shape (batch_size, n_genes).
        gamma : torch.Tensor
            Degradation rate parameters of shape (batch_size, n_genes).
        
        Notes
        -----
        The gene-level velocity is computed using the kinetic model:
        - velocity_u = α - β * unspliced
        - velocity_s = β * unspliced - γ * spliced
        
        If cluster embeddings are provided, they are integrated with z via attention
        mechanism using GP-specific embeddings before being passed to the decoder.
        The attention mechanism computes L attention weights (one per gene program),
        allowing selective attention to different gene programs based on cluster identity.
        """
        if self.use_attention:
            if cluster_emb is None:
                raise ValueError("cluster_emb is required when attention mechanism is enabled")
            # Apply attention: Q from cluster_emb, K and V from enriched GP embeddings
            attended_output = self.attention(cluster_emb, z)  # (batch_size, attention_dim)
            decoder_input = attended_output
        else:
            # No cluster embeddings, use z directly
            decoder_input = z
        
        h = self.shared_decoder(decoder_input)
        velocity_gp = self.gp_velocity_decoder(h)

        kinetic_params = self.gene_velocity_decoder(h)  # (B, 3G)
        alpha, beta, gamma = torch.split(kinetic_params, self._G, dim=1)
        unspliced, spliced = torch.split(x, x.size(1) // 2, dim=1)
        velocity_u = alpha - beta * unspliced
        velocity_s = beta * unspliced - gamma * spliced
        velocity = torch.cat([velocity_u, velocity_s], dim=1)  # (B, 2G)
        
        return velocity, velocity_gp, alpha, beta, gamma


class ClusterEmbedding(nn.Module):
    """
    Cluster embedding module that learns embeddings for each cluster.
    
    This module creates a lookup table of embeddings for each unique cluster label.
    All cells in the same cluster share the same embedding, which can be concatenated
    to the latent representation to provide lineage-specific information to the velocity decoder.
    
    Parameters
    ----------
    n_clusters : int
        Number of unique clusters.
    embedding_dim : int, default 32
        Dimension of cluster embeddings.
    
    Attributes
    ----------
    embeddings : nn.Embedding
        Embedding table with shape (n_clusters, embedding_dim).
    
    Examples
    --------
    >>> # Create cluster embeddings for 10 clusters with 32-dimensional embeddings
    >>> cluster_emb = ClusterEmbedding(n_clusters=10, embedding_dim=32)
    >>> 
    >>> # Forward pass with cluster indices
    >>> cluster_idx = torch.tensor([0, 1, 2, 0, 1])  # 5 cells, 3 clusters
    >>> emb = cluster_emb(cluster_idx)  # shape: (5, 32)
    """
    
    def __init__(self, n_clusters: int, embedding_dim: int = 32):
        super().__init__()
        self.embeddings = nn.Embedding(n_clusters, embedding_dim)
        
        # Initialize with small random values
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)
    
    def forward(self, cluster_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get cluster embeddings.
        
        Parameters
        ----------
        cluster_indices : torch.Tensor
            Cluster indices of shape (batch_size,) with integer values in [0, n_clusters).
        
        Returns
        -------
        torch.Tensor
            Cluster embeddings of shape (batch_size, embedding_dim).
        """
        return self.embeddings(cluster_indices)
    
    def get_all_embeddings(self) -> torch.Tensor:
        """
        Get all cluster embeddings as a lookup table.
        
        Returns
        -------
        torch.Tensor
            All cluster embeddings of shape (n_clusters, embedding_dim).
        """
        return self.embeddings.weight


class ClusterAttention(nn.Module):
    """
    Attention mechanism to integrate cluster embeddings with latent representations.
    
    This module uses gene program-specific embeddings to create keys and values:
    1. Each gene program has its own learnable embedding
    2. Each GP embedding is enriched by concatenating with its corresponding z_i value
    3. The enriched GP embeddings are transformed using a shared transformation
    4. Query (Q) comes from cluster embeddings
    5. Keys (K) and Values (V) come from enriched GP embeddings (one per GP)
    6. Attention computes L weights (one per GP), allowing selective attention
    
    Parameters
    ----------
    cluster_embedding_dim : int
        Dimension of cluster embeddings (E).
    n_latent : int
        Dimension of latent representations (L) - number of gene programs.
    gp_embedding_dim : int, default 32
        Dimension of gene program-specific embeddings.
    attention_dim : int, optional
        Dimension of the attention space (d_k). If None, uses n_latent.
    
    Attributes
    ----------
    gp_embeddings : nn.Embedding
        Gene program-specific embeddings of shape (n_latent, gp_embedding_dim).
    gp_enricher : nn.Linear
        Shared transformation from [gp_emb, z_i] to enriched GP embedding.
    key_proj : nn.Linear
        Key projection from enriched GP embeddings to attention_dim.
    value_proj : nn.Linear
        Value projection from enriched GP embeddings to attention_dim.
    query_proj : nn.Linear
        Query projection from cluster embedding to attention_dim.
    attention_dim : int
        Dimension of the attention space.
    scale : float
        Scaling factor for attention scores (1 / sqrt(attention_dim)).
    
    Examples
    --------
    >>> # Create attention module for 32-dim cluster embeddings and 50 gene programs
    >>> attn = ClusterAttention(cluster_embedding_dim=32, n_latent=50)
    >>> 
    >>> # Forward pass
    >>> cluster_emb = torch.randn(10, 32)  # (batch_size, cluster_embedding_dim)
    >>> z = torch.randn(10, 50)  # (batch_size, n_latent)
    >>> out = attn(cluster_emb, z)  # shape: (batch_size, attention_dim)
    """
    
    def __init__(
        self, 
        cluster_embedding_dim: int, 
        n_latent: int, 
        gp_embedding_dim: int = 32,
        attention_dim: Optional[int] = None
    ):
        super().__init__()
        self.cluster_embedding_dim = cluster_embedding_dim
        self.n_latent = n_latent
        self.gp_embedding_dim = gp_embedding_dim
        self.attention_dim = attention_dim if attention_dim is not None else n_latent
        
        # Step 1: GP-specific embeddings (one per gene program)
        self.gp_embeddings = nn.Embedding(n_latent, gp_embedding_dim)
        
        # Step 2: Shared transformation from [gp_emb, z_i] to enriched embedding
        self.gp_enricher = nn.Linear(gp_embedding_dim + 1, self.attention_dim)
        
        # Step 3: Key and value projections from enriched GP embeddings
        self.key_proj = nn.Linear(self.attention_dim, self.attention_dim)
        self.value_proj = nn.Linear(self.attention_dim, self.attention_dim)
        
        # Step 4: Query projection from cluster embedding
        self.query_proj = nn.Linear(cluster_embedding_dim, self.attention_dim)
        
        # Scaling factor for attention scores
        self.scale = 1.0 / (self.attention_dim ** 0.5)
        
        # Initialize weights
        nn.init.normal_(self.gp_embeddings.weight, mean=0.0, std=0.01)
        nn.init.xavier_uniform_(self.gp_enricher.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.query_proj.weight)
    
    def forward(self, cluster_emb: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention mechanism.
        
        Parameters
        ----------
        cluster_emb : torch.Tensor
            Cluster embeddings of shape (batch_size, cluster_embedding_dim).
        z : torch.Tensor
            Latent representations of shape (batch_size, n_latent).
            Each dimension z_i corresponds to one gene program.
        
        Returns
        -------
        torch.Tensor
            Attention output of shape (batch_size, attention_dim).
            This is a weighted combination of enriched GP embeddings, where weights
            are computed via attention between cluster embeddings and GP embeddings.
        """
        B, L = z.shape
        
        # Step 1: Get GP embeddings for all gene programs
        gp_indices = torch.arange(L, device=z.device)  # (L,)
        gp_emb = self.gp_embeddings(gp_indices)  # (L, gp_embedding_dim)
        
        # Step 2: Enrich each GP embedding with its corresponding z_i
        # Expand GP embeddings to batch dimension
        gp_emb_expanded = gp_emb.unsqueeze(0).expand(B, -1, -1)  # (B, L, gp_embedding_dim)
        
        # Get z_i for each GP (reshape to (B, L, 1))
        z_expanded = z.unsqueeze(-1)  # (B, L, 1)
        
        # Concatenate: [gp_emb_i, z_i] for each GP
        enriched_input = torch.cat([gp_emb_expanded, z_expanded], dim=-1)  # (B, L, gp_embedding_dim + 1)
        
        # Transform enriched GP embeddings using shared transformation
        enriched_gp = self.gp_enricher(enriched_input)  # (B, L, attention_dim)
        
        # Step 3: Create keys and values from enriched GP embeddings
        K = self.key_proj(enriched_gp)  # (B, L, attention_dim)
        V = self.value_proj(enriched_gp)  # (B, L, attention_dim)
        
        # Step 4: Query from cluster embedding
        Q = self.query_proj(cluster_emb)  # (B, attention_dim)
        
        # Attention: Q attends to L keys (one per gene program)
        scores = Q.unsqueeze(1) @ K.transpose(-1, -2) * self.scale  # (B, 1, L)
        attn_weights = F.softmax(scores, dim=-1)  # (B, 1, L) - L weights!
        
        # Weighted sum of values
        out = attn_weights @ V  # (B, 1, L) @ (B, L, attention_dim) -> (B, 1, attention_dim)
        
        return out.squeeze(1)  # (B, attention_dim)
