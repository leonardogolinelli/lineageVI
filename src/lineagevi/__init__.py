from .model import LineageVIModel   # main nn.Module (renamed)
from .trainer import LineageVI, LineageVITrainer
from .dataloader import RegimeDataset, make_dataloader
from . import plotting, utils
__all__ = [
    "LineageVIModel", "LineageVI", "LineageVITrainer",
    "RegimeDataset", "make_dataloader", "plotting", "utils", "__version__"
]
__version__ = "0.1.0"
