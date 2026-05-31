# MDPCR — utils/visualise.py
# Print cloud state, convergence info, and point positions for inspection.

import torch
from core.convergence import ConvergenceTracker
import config


def print_cloud(
    positions  : torch.Tensor,   # (M, N)
    densities  : torch.Tensor,   # (M,)
    tokens     : list[str],
    label      : str = "Cloud",
    max_dims   : int = 6,
) -> None:
    """Pretty-print a cloud's current state."""
    M = positions.shape[0]
    print(f"\n{'─'*50}")
    print(f"  {label}  ({M} points, {config.POINT_DIMS} dims)")
    print(f"{'─'*50}")
    for i in range(min(M, len(tokens))):
        pos_preview = positions[i, :max_dims].tolist()
        print(
            f"  [{i:3d}] '{tokens[i]:15s}' "
            f"density={densities[i].item():.3f}  "
            f"pos=[{', '.join(f'{v:6.3f}' for v in pos_preview)}...]"
        )
    print(f"{'─'*50}")
    centroid = positions.mean(dim=0)
    print(f"  Centroid: [{', '.join(f'{v:.3f}' for v in centroid[:max_dims].tolist())}...]")
    print()


def print_convergence(tracker: ConvergenceTracker) -> None:
    """Print convergence summary."""
    s = tracker.summary
    status = "✓ converged" if s["converged"] else "✗ hit max iterations"
    print(f"\n  Convergence: {status}")
    print(f"  Iterations : {s['iterations']}")
    if s["final_delta"] is not None:
        print(f"  Final Δ    : {s['final_delta']:.8f}  (threshold: {config.CONVERGENCE_THRESHOLD})")
    if len(s["delta_history"]) > 1:
        print(f"  Δ history  : {[f'{d:.5f}' for d in s['delta_history']]}")
    print()
