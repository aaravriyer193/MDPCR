# MDPCR — core/influence.py
# The update rule: how points influence each other each iteration.
# This is the heart of MDPCR — replaces attention with spatial physics.
#
# Every point influences every other point simultaneously,
# weighted by density / distance².
# No sequence. No left-to-right. Pure geometry.

import torch
import torch.nn as nn
import config


class InfluenceLayer(nn.Module):
    """
    Learned influence update rule.

    Given a cloud of M points each with N dimensions, computes
    the new positions after one iteration of mutual influence.

    The neural network learns HOW dimensions influence each other —
    analogous to attention weights but operating geometrically.

    Input:  positions tensor of shape (M, N)
            densities tensor of shape (M,)
    Output: updated positions tensor of shape (M, N)
    """

    def __init__(self):
        super().__init__()
        N = config.POINT_DIMS

        # Learned projection: how each dimension affects every other dimension
        # during the influence step. This is the core learned parameter.
        self.influence_proj = nn.Linear(N, N, bias=True)

        # Learned density modulation — adjusts how much each point's
        # density contributes to its influence on others
        self.density_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Residual scale — learned scalar controlling update step size
        self.step_scale = nn.Parameter(torch.tensor(config.INFLUENCE_LR))

        self._init_weights()

    def _init_weights(self):
        # Start near identity so early training is stable
        nn.init.eye_(self.influence_proj.weight)
        nn.init.zeros_(self.influence_proj.bias)
        # Clamp step_scale so it can't explode during training
        with torch.no_grad():
            self.step_scale.clamp_(0.001, 0.5)

    def forward(
        self,
        positions : torch.Tensor,   # (M, N)
        densities : torch.Tensor,   # (M,)
    ) -> torch.Tensor:              # (M, N)
        """
        One iteration of the influence update.

        For each point p, compute the weighted sum of all other points,
        where weight = density(q) / distance(p, q)².
        Apply learned projection, then add as residual to current position.
        """
        M, N = positions.shape

        # Keep step_scale bounded — it's learnable but must not explode
        self.step_scale.data.clamp_(0.001, 0.5)

        # ── Normalise positions for stable distance computation ───────────────
        # Early in training embeddings cluster near zero → distances near zero
        # → 1/d² explodes. Normalising to unit sphere prevents this.
        pos_norm = positions / (positions.norm(dim=-1, keepdim=True).clamp(min=1e-6))

        # ── Distance matrix ───────────────────────────────────────────────────
        dist = torch.cdist(pos_norm, pos_norm, p=2)  # (M, M)

        # Clamp ALL distances to a minimum — not just diagonal.
        # Off-diagonal near-zero distances are the real NaN source.
        dist = dist.clamp(min=0.01)

        # ── Influence weights ─────────────────────────────────────────────────
        # Apply learned density gating
        density_gated = self.density_gate(
            densities.unsqueeze(-1)         # (M, 1)
        ).squeeze(-1).clamp(1e-4, 2.0)      # (M,)  — clamp for safety

        inv_dist_sq = 1.0 / (dist ** 2)                              # (M, M)
        weights = inv_dist_sq * density_gated.unsqueeze(0)           # (M, M)

        # Zero out self-influence
        eye_mask = torch.eye(M, device=positions.device, dtype=torch.bool)
        weights = weights.masked_fill(eye_mask, 0.0)

        # Optional: apply influence radius cutoff
        if config.INFLUENCE_RADIUS is not None:
            mask = dist > config.INFLUENCE_RADIUS
            weights = weights.masked_fill(mask, 0.0)

        # Normalise weights — clamp denominator to prevent 0/0
        weights = weights / (weights.sum(dim=1, keepdim=True).clamp(min=1e-8))

        # ── Aggregate influence ───────────────────────────────────────────────
        aggregated = weights @ positions     # (M, N)  — use original positions

        # ── Learned projection ────────────────────────────────────────────────
        projected = self.influence_proj(aggregated)  # (M, N)

        # ── Residual update ───────────────────────────────────────────────────
        delta = projected - positions
        # Clamp delta magnitude per-point to prevent runaway updates
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        delta = delta / delta_norm.clamp(min=1.0) * delta_norm  # scale if >1

        updated = positions + self.step_scale * delta

        # Final NaN guard — replace any NaN with the original position
        nan_mask = torch.isnan(updated)
        if nan_mask.any():
            updated = torch.where(nan_mask, positions, updated)

        return updated  # (M, N)


def compute_influence_matrix(
    positions : torch.Tensor,  # (M, N)
    densities : torch.Tensor,  # (M,)
) -> torch.Tensor:             # (M, M)
    """
    Utility: return the raw influence weight matrix for visualisation.
    Not used in training — useful for inspecting what the cloud is doing.
    """
    M = positions.shape[0]
    dist = torch.cdist(positions, positions, p=2)
    dist = dist + torch.eye(M, device=positions.device) * 1e-8
    inv_dist_sq = 1.0 / (dist ** 2)
    weights = inv_dist_sq * densities.unsqueeze(0)
    weights = weights * (1 - torch.eye(M, device=positions.device))
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    return weights