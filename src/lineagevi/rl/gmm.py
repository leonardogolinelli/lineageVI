"""GMM scorer for off-manifold penalty."""

from pathlib import Path
import numpy as np
import torch
try:
    from sklearn.mixture import GaussianMixture
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SklearnGMMScorer:
    """
    Wrapper for sklearn GMM to compute negative log-likelihood.
    
    Parameters
    ----------
    path : str
        Path to saved GMM (.pkl file).
    """
    
    def __init__(self, path: str):
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "sklearn is required for GMM scorer. Install with: pip install scikit-learn"
            )
        
        if not Path(path).exists():
            raise FileNotFoundError(f"GMM file not found: {path}")
        
        self.gmm = joblib.load(path)
        if not isinstance(self.gmm, GaussianMixture):
            raise ValueError(f"Loaded object is not a GaussianMixture: {type(self.gmm)}")
    
    def nll(self, z: np.ndarray) -> np.ndarray:
        """
        Compute negative log-likelihood for latent states.
        
        Parameters
        ----------
        z : np.ndarray
            Latent states of shape (B, d) or (d,).
        
        Returns
        -------
        nll : np.ndarray
            Negative log-likelihood of shape (B,) or scalar.
        """
        # Ensure 2D
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        
        # Compute negative log-likelihood
        # score_samples returns log-likelihood, so negate it
        log_likelihood = self.gmm.score_samples(z)  # (B,)
        nll = -log_likelihood  # (B,)
        
        return nll
    
    def nll_torch(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood for PyTorch tensors.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent states of shape (B, d) or (d,).
        
        Returns
        -------
        nll : torch.Tensor
            Negative log-likelihood of shape (B,) or scalar.
        """
        # Convert to numpy on CPU
        z_np = z.detach().cpu().numpy()
        nll_np = self.nll(z_np)
        
        # Convert back to torch on same device/dtype
        nll_torch = torch.from_numpy(nll_np).to(z.device).to(z.dtype)
        
        return nll_torch
