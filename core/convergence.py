# MDPCR — core/convergence.py
# Convergence checking: has the cloud stopped moving meaningfully?
# The termination condition: cloud(t) ≈ cloud(t-1)

import torch
import config


class ConvergenceTracker:
    """
    Tracks whether the cloud has converged between iterations.

    Convergence is defined as:
        mean( |positions(t) - positions(t-1)| ) < threshold

    Also tracks iteration count and convergence history for logging.
    """

    def __init__(
        self,
        threshold  : float = config.CONVERGENCE_THRESHOLD,
        max_iters  : int   = config.MAX_ITERATIONS,
        min_iters  : int   = config.MIN_ITERATIONS,
    ):
        self.threshold = threshold
        self.max_iters = max_iters
        self.min_iters = min_iters

        self.iteration      : int         = 0
        self.last_positions : torch.Tensor | None = None
        self.delta_history  : list[float] = []
        self.converged      : bool        = False

    def reset(self):
        """Reset tracker for a new forward pass."""
        self.iteration      = 0
        self.last_positions = None
        self.delta_history  = []
        self.converged      = False

    def step(self, positions: torch.Tensor) -> bool:
        """
        Check if the cloud has converged.

        Args:
            positions: current cloud positions, shape (M, N)

        Returns:
            True if converged (should stop iterating), False otherwise.
        """
        self.iteration += 1

        if self.last_positions is None:
            # First iteration — no previous state to compare
            self.last_positions = positions.detach().clone()
            return False

        # Mean absolute change across all points and dimensions
        delta = (positions.detach() - self.last_positions).abs().mean().item()
        self.delta_history.append(delta)
        self.last_positions = positions.detach().clone()

        # Must complete minimum iterations regardless
        if self.iteration < self.min_iters:
            return False

        # Hard cap
        if self.iteration >= self.max_iters:
            self.converged = True
            return True

        # Convergence condition
        if delta < self.threshold:
            self.converged = True
            return True

        return False

    @property
    def summary(self) -> dict:
        return {
            "iterations"    : self.iteration,
            "converged"     : self.converged,
            "final_delta"   : self.delta_history[-1] if self.delta_history else None,
            "delta_history" : self.delta_history,
        }

    def __repr__(self) -> str:
        status = "converged" if self.converged else "running"
        return (
            f"ConvergenceTracker({status}, "
            f"iter={self.iteration}/{self.max_iters}, "
            f"delta={self.delta_history[-1]:.6f if self.delta_history else 'N/A'})"
        )
