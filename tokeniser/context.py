# MDPCR — tokeniser/context.py
# Window-of-3 contextual encoder.
#
# Core rule: a token's identity is defined by its context, not just itself.
# Same surface form + different neighbours = different starting position.
# This is what keeps the cloud geometrically meaningful.

import torch
import torch.nn as nn
from tokeniser.vocab import Vocabulary
import config


class ContextualEncoder(nn.Module):
    """
    Maps (left_token, token, right_token) triplets to N-dimensional positions.

    Architecture:
        1. Base embedding: token → N dims (seeded from co-occurrence)
        2. Context offset: (left, right) → N dims (learned from scratch)
        3. Final position: base + context_offset

    This means every unique context produces a unique starting position,
    even for the same surface-form token.
    """

    def __init__(self, vocab: Vocabulary):
        super().__init__()
        V = len(vocab)
        N = config.POINT_DIMS

        self.vocab = vocab

        # Base embedding: one vector per token in vocabulary
        # Initialised using co-occurrence frequencies (meaningful seed)
        self.base_embed = nn.Embedding(V, N, padding_idx=0)
        self._seed_from_cooccurrence()

        # Context offset network: takes left + right token embeddings,
        # outputs a shift to apply to the base position
        # This is what makes the same token context-sensitive
        self.context_net = nn.Sequential(
            nn.Linear(N * 2, N * 2),
            nn.LayerNorm(N * 2),
            nn.GELU(),
            nn.Linear(N * 2, N),
            nn.Tanh(),  # Bound the offset so it doesn't dominate base position
        )

        # Learned density predictor: how "loaded" is this token in this context?
        self.density_net = nn.Sequential(
            nn.Linear(N * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _seed_from_cooccurrence(self):
        """
        Seed base embeddings using co-occurrence statistics.
        Tokens that frequently co-occur start closer together.
        This gives the cloud a meaningful geometric head start.
        """
        V, N = self.base_embed.weight.shape
        with torch.no_grad():
            # Build a dense co-occurrence matrix (normalised)
            comat = torch.zeros(V, V)
            for tok_id, neighbours in self.vocab.cooccur.items():
                if tok_id < V:
                    for nbr_id, count in neighbours.items():
                        if nbr_id < V:
                            comat[tok_id, nbr_id] = float(count)

            # Row-normalise
            row_sums = comat.sum(dim=1, keepdim=True).clamp(min=1e-8)
            comat = comat / row_sums

            # Use top-N co-occurrence counts as the seed dimensions
            # (take only N dims worth of signal)
            if V >= N:
                # SVD to get N-dimensional projection of co-occurrence space
                try:
                    U, S, Vh = torch.linalg.svd(comat, full_matrices=False)
                    seed = U[:, :N] * S[:N].unsqueeze(0)
                    # Normalise to unit scale
                    seed = seed / (seed.norm(dim=1, keepdim=True).clamp(min=1e-8))
                    self.base_embed.weight.data = seed * 0.1
                except Exception:
                    # Fall back to random if SVD fails (small vocab)
                    nn.init.normal_(self.base_embed.weight, std=0.01)
            else:
                nn.init.normal_(self.base_embed.weight, std=0.01)

    def forward(
        self,
        token_ids  : torch.Tensor,   # (M,) — token IDs for each position
        left_ids   : torch.Tensor,   # (M,) — left neighbour IDs (PAD if none)
        right_ids  : torch.Tensor,   # (M,) — right neighbour IDs (PAD if none)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a sequence of tokens with their contexts into cloud positions.

        Returns:
            positions : (M, N) — starting positions in N-dimensional space
            densities : (M,)   — density scalar per point
        """
        # Base positions from vocabulary
        base    = self.base_embed(token_ids)    # (M, N)
        left_e  = self.base_embed(left_ids)     # (M, N)
        right_e = self.base_embed(right_ids)    # (M, N)

        # Context offset from left + right neighbours
        context_input  = torch.cat([left_e, right_e], dim=-1)  # (M, 2N)
        context_offset = self.context_net(context_input)        # (M, N)

        # Final position: base + context shift
        positions = base + context_offset                       # (M, N)

        # Density: how informationally loaded is this token in this context?
        density_input = torch.cat([base, left_e, right_e], dim=-1)  # (M, 3N)
        densities = self.density_net(density_input).squeeze(-1)      # (M,)

        # Rescale densities to (0, 2] range
        densities = densities * 2.0 + 1e-4

        return positions, densities

    def encode_text(
        self,
        text : str,
        device : str = config.DEVICE,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """
        Convenience: encode a raw text string into (positions, densities, tokens).
        Handles BOS/EOS padding and window-of-3 context automatically.
        """
        token_ids  = self.vocab.encode(text)
        tokens_str = self.vocab.tokenize(text)

        pad_id = self.vocab.token2id[config.PAD_TOKEN]
        M = len(token_ids)

        # Build left and right context arrays
        left_ids  = [pad_id] + token_ids[:-1]
        right_ids = token_ids[1:] + [pad_id]

        t  = torch.tensor(token_ids, dtype=torch.long, device=device)
        l  = torch.tensor(left_ids,  dtype=torch.long, device=device)
        r  = torch.tensor(right_ids, dtype=torch.long, device=device)

        positions, densities = self.forward(t, l, r)
        return positions, densities, tokens_str
