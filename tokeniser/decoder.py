# MDPCR — tokeniser/decoder.py
# Decodes a converged output cloud back into text.
#
# The inverse of the contextual encoder:
# for each point in the output cloud, find the nearest vocabulary token
# by comparing the point's position to all base embeddings.

import torch
import torch.nn.functional as F
from tokeniser.vocab import Vocabulary
from tokeniser.context import ContextualEncoder
import config


class CloudDecoder:
    """
    Decodes a converged point cloud back into a token sequence.

    Strategy: for each point in the output cloud, find the vocabulary
    token whose base embedding is closest in N-dimensional space.
    The sequence of nearest tokens is the decoded text.

    This is the inverse of the base_embed lookup in ContextualEncoder.
    """

    def __init__(self, vocab: Vocabulary, encoder: ContextualEncoder):
        self.vocab   = vocab
        self.encoder = encoder

    def decode_positions(
        self,
        positions    : torch.Tensor,   # (M, N) — converged cloud positions
        temperature  : float = config.TEMPERATURE,
        skip_special : bool  = True,
    ) -> list[str]:
        """
        Map each point to its nearest vocabulary token.

        Args:
            positions    : converged cloud positions (M, N)
            temperature  : softness of nearest-token selection
                           1.0 = argmax (nearest), higher = softer
            skip_special : if True, skip PAD/BOS/EOS tokens in output

        Returns:
            list of decoded token strings, one per point
        """
        # All base embeddings: (V, N)
        all_embeddings = self.encoder.base_embed.weight.detach()  # (V, N)

        # Cosine similarity between each output point and all base embeddings
        # positions: (M, N), all_embeddings: (V, N)
        pos_norm   = F.normalize(positions, dim=-1)          # (M, N)
        emb_norm   = F.normalize(all_embeddings, dim=-1)     # (V, N)
        similarity = pos_norm @ emb_norm.T                   # (M, V)

        if temperature == 1.0:
            # Hard nearest-neighbour
            token_ids = similarity.argmax(dim=-1).tolist()   # (M,)
        else:
            # Soft sampling over similarity scores
            probs     = F.softmax(similarity / temperature, dim=-1)  # (M, V)
            token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1).tolist()

        # Convert IDs to strings
        special = {config.PAD_TOKEN, config.BOS_TOKEN, config.EOS_TOKEN}
        tokens  = []
        for tid in token_ids:
            tok = self.vocab.id2token.get(tid, config.UNK_TOKEN)
            if skip_special and tok in special:
                continue
            tokens.append(tok)

        return tokens

    def decode_to_text(
        self,
        positions   : torch.Tensor,
        temperature : float = config.TEMPERATURE,
    ) -> str:
        """Decode positions to a joined text string."""
        tokens = self.decode_positions(positions, temperature)
        return " ".join(tokens)
