# MDPCR — core/point.py
# A single point in N-dimensional space with a density scalar.
# This is the atomic unit of the MDPCR architecture.

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional
import config


@dataclass
class Point:
    """
    A single token represented as a position in N-dimensional space.

    Attributes:
        position  : torch.Tensor of shape (N,) — the point's location
        density   : float — how strongly this point influences neighbours
        token_id  : int — which vocabulary token this point represents
        token_str : str — the surface form of the token
    """
    position  : torch.Tensor
    density   : float = config.DENSITY_INIT
    token_id  : int   = -1
    token_str : str   = ""

    def __post_init__(self):
        assert self.position.shape == (config.POINT_DIMS,), (
            f"Point position must have shape ({config.POINT_DIMS},), "
            f"got {self.position.shape}"
        )
        assert 0.0 < self.density <= 2.0, (
            f"Density must be in (0, 2], got {self.density}"
        )

    def detach(self) -> "Point":
        """Return a detached copy (for convergence checking)."""
        return Point(
            position  = self.position.detach().clone(),
            density   = self.density,
            token_id  = self.token_id,
            token_str = self.token_str,
        )

    def to(self, device: str) -> "Point":
        """Move point to device."""
        return Point(
            position  = self.position.to(device),
            density   = self.density,
            token_id  = self.token_id,
            token_str = self.token_str,
        )

    def __repr__(self) -> str:
        pos_preview = self.position[:4].tolist()
        return (
            f"Point(token='{self.token_str}', "
            f"density={self.density:.3f}, "
            f"pos={[f'{v:.3f}' for v in pos_preview]}...)"
        )


def make_point(
    position  : torch.Tensor,
    token_id  : int,
    token_str : str,
    density   : Optional[float] = None,
) -> Point:
    """Convenience constructor."""
    return Point(
        position  = position.to(config.DEVICE),
        density   = density if density is not None else config.DENSITY_INIT,
        token_id  = token_id,
        token_str = token_str,
    )
