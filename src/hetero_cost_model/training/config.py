"""Training hyper-parameter configuration."""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 50
    batch_size: int = 32
    ranking_lambda: float = 0.1
    device: str = "cpu"


__all__ = ["TrainConfig"]
