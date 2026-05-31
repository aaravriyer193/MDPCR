# MDPCR — demo.py
# Interactive demo: type a prompt, watch the cloud converge, get a response.
#
# Usage:
#   python demo.py                              # uses best checkpoint
#   python demo.py --checkpoint checkpoints/epoch_20.pt
#   python demo.py --cloud                      # show cloud state each turn

import argparse
import os
import torch
import config

from tokeniser.vocab import Vocabulary
from model.mdpcr import MDPCRModel
from utils.visualise import print_cloud, print_convergence


def main(checkpoint: str, show_cloud: bool):
    print("\n" + "═" * 60)
    print("  MDPCR — Multi Dimensional Point Cloud Reasoning")
    print("  Interactive Demo")
    print("═" * 60)

    # ── Load vocab and model ───────────────────────────────────────
    if not os.path.exists(config.VOCAB_PATH):
        print(f"No vocabulary found at {config.VOCAB_PATH}. Run train.py first.")
        return

    if not os.path.exists(checkpoint):
        print(f"No checkpoint found at {checkpoint}. Run train.py first.")
        return

    vocab = Vocabulary.load(config.VOCAB_PATH)
    model = MDPCRModel.load(checkpoint, vocab)
    model.eval()

    params = model.parameter_count()
    print(f"\nModel loaded  |  {params['trainable']:,} parameters  |  device={config.DEVICE}")
    print(f"Vocabulary    |  {len(vocab)} tokens")
    print(f"Cloud dims    |  {config.POINT_DIMS}D")
    print(f"\nType a question. ':cloud' to toggle cloud view. ':quit' to exit.\n")

    show = show_cloud

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not prompt:
            continue

        if prompt == ":quit":
            print("Goodbye.")
            break

        if prompt == ":cloud":
            show = not show
            print(f"  Cloud view {'ON' if show else 'OFF'}\n")
            continue

        # ── Encode prompt ──────────────────────────────────────────
        positions, densities, tokens = model.encode(prompt)

        if show:
            print_cloud(positions, densities, tokens, label="Prompt Cloud")

        # ── Run to convergence ─────────────────────────────────────
        converged, tracker = model.cloud.run_to_convergence(positions, densities)

        if show:
            print_cloud(converged, densities, tokens, label="Converged Cloud")
            print_convergence(tracker)
        else:
            iters = tracker.summary["iterations"]
            print(f"  [{iters} iteration{'s' if iters != 1 else ''}]")

        # ── Decode response ────────────────────────────────────────
        response = model.decode(converged)
        print(f"MDPCR: {response}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type    = str,
        default = os.path.join(config.CHECKPOINT_DIR, "best.pt"),
        help    = "Path to model checkpoint",
    )
    parser.add_argument(
        "--cloud",
        action  = "store_true",
        help    = "Show cloud state on each turn",
    )
    args = parser.parse_args()
    main(checkpoint=args.checkpoint, show_cloud=args.cloud)
