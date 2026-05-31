# MDPCR — model/mdpcr.py
# Top-level model: ties tokeniser, cloud, and decoder together.
# This is what training and inference both interact with.

import torch
import torch.nn as nn
from tokeniser.vocab import Vocabulary
from tokeniser.context import ContextualEncoder
from tokeniser.decoder import CloudDecoder
from core.influence import InfluenceLayer
from core.cloud import Cloud, cloud_distance
from core.convergence import ConvergenceTracker
import config


class MDPCRModel(nn.Module):
    """
    Multi Dimensional Point Cloud Reasoning model.

    Full pipeline:
        text → contextual tokeniser → prompt cloud
             → iterative influence  → converged cloud
             → decoder              → response text

    The model is trained on (prompt, response) text pairs.
    The learning signal is the distance between the model's
    converged cloud and the target response cloud.
    """

    def __init__(self, vocab: Vocabulary):
        super().__init__()
        self.vocab    = vocab
        self.encoder  = ContextualEncoder(vocab)
        self.influence = InfluenceLayer()
        self.cloud    = Cloud(self.influence)
        self.decoder  = CloudDecoder(vocab, self.encoder)

    def forward(
        self,
        prompt_positions   : torch.Tensor,  # (M_p, N)
        prompt_densities   : torch.Tensor,  # (M_p,)
        training           : bool = True,
    ) -> tuple[torch.Tensor, ConvergenceTracker]:
        """
        Run a prompt cloud through to convergence.

        Returns the converged positions and convergence metadata.
        """
        return self.cloud(
            prompt_positions,
            prompt_densities,
            training=training,
        )

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Encode raw text into (positions, densities, tokens)."""
        return self.encoder.encode_text(text, device=config.DEVICE)

    def decode(self, positions: torch.Tensor) -> str:
        """Decode converged positions into text."""
        return self.decoder.decode_to_text(positions)

    def loss(
        self,
        converged_positions : torch.Tensor,  # (M_out, N)
        target_positions    : torch.Tensor,  # (M_tgt, N)
    ) -> torch.Tensor:
        """
        Cloud-to-cloud distance loss.
        Trains the model to transform prompt clouds into response clouds.
        """
        return cloud_distance(converged_positions, target_positions)

    def predict(self, prompt_text: str) -> tuple[str, ConvergenceTracker]:
        """
        Full inference: text in, text out.
        Returns (response_text, convergence_tracker).
        """
        self.eval()
        with torch.no_grad():
            positions, densities, _ = self.encode(prompt_text)
            converged, tracker = self.cloud.run_to_convergence(positions, densities)
            response = self.decode(converged)
        return response, tracker

    def parameter_count(self) -> dict:
        total  = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trained}

    def save(self, path: str) -> None:
        torch.save({
            "model_state" : self.state_dict(),
            "config"      : {
                "POINT_DIMS"       : config.POINT_DIMS,
                "POINTS_PER_TOKEN" : config.POINTS_PER_TOKEN,
                "INFLUENCE_RADIUS" : config.INFLUENCE_RADIUS,
            },
        }, path)

    @classmethod
    def load(cls, path: str, vocab: Vocabulary) -> "MDPCRModel":
        checkpoint = torch.load(path, map_location=config.DEVICE)
        model = cls(vocab)
        model.load_state_dict(checkpoint["model_state"])
        return model
