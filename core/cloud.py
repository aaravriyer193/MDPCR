# MDPCR — core/cloud.py
# The Cloud: a collection of points that iterates to convergence.
# This is the reasoning substrate — not a sequence, a formation.

import torch
import torch.nn as nn
from typing import Optional
from core.point import Point
from core.influence import InfluenceLayer
from core.convergence import ConvergenceTracker
import config


class Cloud(nn.Module):
    """
    A point cloud that iterates under mutual influence until convergence.

    The cloud is both the input representation and the reasoning medium.
    It is initialised from token embeddings, then allowed to evolve.
    The settled state is the output.

    The InfluenceLayer (with its learned weights) is what gets trained.
    The Cloud itself is the container and iteration engine.
    """

    def __init__(self, influence_layer: InfluenceLayer):
        super().__init__()
        self.influence = influence_layer

    def forward(
        self,
        positions  : torch.Tensor,          # (M, N) — initial point positions
        densities  : torch.Tensor,          # (M,)   — point densities
        tracker    : Optional[ConvergenceTracker] = None,
        training   : bool = True,
    ) -> tuple[torch.Tensor, ConvergenceTracker]:
        """
        Run the cloud from its initial state to convergence.

        During training: runs for a fixed number of steps and
        backpropagates through all of them.

        During inference: runs until convergence criterion is met.

        Args:
            positions : initial positions, shape (M, N)
            densities : density per point, shape (M,)
            tracker   : optional pre-existing tracker (pass None to create fresh)
            training  : if True, unroll fixed steps; if False, run to convergence

        Returns:
            (final_positions, tracker)
            final_positions: shape (M, N)
            tracker: contains convergence metadata
        """
        if tracker is None:
            tracker = ConvergenceTracker()
        else:
            tracker.reset()

        current = positions

        if training:
            # During training: fixed number of unrolled steps
            # so gradients flow cleanly through every iteration
            for _ in range(config.MIN_ITERATIONS):
                current = self.influence(current, densities)
                tracker.step(current)

            # Additional steps with gradient checkpointing for memory efficiency
            for _ in range(config.MAX_ITERATIONS - config.MIN_ITERATIONS):
                next_pos = self.influence(current, densities)
                tracker.step(next_pos)
                if tracker.converged:
                    current = next_pos
                    break
                current = next_pos

        else:
            # During inference: run until true convergence
            with torch.no_grad():
                while True:
                    next_pos = self.influence(current, densities)
                    if tracker.step(next_pos):
                        current = next_pos
                        break
                    current = next_pos

        return current, tracker

    def run_to_convergence(
        self,
        positions : torch.Tensor,
        densities : torch.Tensor,
    ) -> tuple[torch.Tensor, ConvergenceTracker]:
        """Convenience method for inference — always runs to true convergence."""
        return self.forward(positions, densities, training=False)


def cloud_distance(
    cloud_a : torch.Tensor,  # (M, N)
    cloud_b : torch.Tensor,  # (M, N)
) -> torch.Tensor:
    """
    Compute the mean squared distance between two clouds of the same size.
    Used as the training loss signal.

    For clouds of different sizes, we use the centroid distance as a
    fallback — though same-size comparison is preferred.
    """
    if cloud_a.shape == cloud_b.shape:
        return ((cloud_a - cloud_b) ** 2).mean()
    else:
        # Centroid fallback for variable-length clouds
        centroid_a = cloud_a.mean(dim=0)
        centroid_b = cloud_b.mean(dim=0)
        return ((centroid_a - centroid_b) ** 2).mean()


def cloud_centroid(positions: torch.Tensor) -> torch.Tensor:
    """Return the mean position of all points — the cloud's centre of mass."""
    return positions.mean(dim=0)  # (N,)
