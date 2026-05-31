# MDPCR — Multi Dimensional Point Cloud Reasoning
# config.py — single source of truth for all hyperparameters

import torch

# ── Model dimensions ──────────────────────────────────────────────────────────
POINT_DIMS        = 64        # N: number of dimensions per point
POINTS_PER_TOKEN  = 1         # how many points represent a single token
                               # (1 = small, 4 = medium, 16 = large)

# ── Tokeniser ─────────────────────────────────────────────────────────────────
CONTEXT_WINDOW    = 3         # window size for contextual tokenisation (fixed)
UNK_TOKEN         = "<UNK>"   # unknown token string
PAD_TOKEN         = "<PAD>"   # padding token string
BOS_TOKEN         = "<BOS>"   # beginning of sequence
EOS_TOKEN         = "<EOS>"   # end of sequence

# ── Influence / update rule ───────────────────────────────────────────────────
INFLUENCE_RADIUS  = None      # None = global (all points), float = local cutoff
DENSITY_INIT      = 1.0       # default starting density for all points
INFLUENCE_LR      = 0.01      # step size for point position updates per iteration

# ── Convergence ───────────────────────────────────────────────────────────────
CONVERGENCE_THRESHOLD = 1e-4  # mean absolute change below this → stop
MAX_ITERATIONS        = 50    # hard cap to prevent infinite loops
MIN_ITERATIONS        = 1     # always do at least this many iterations

# ── Training ──────────────────────────────────────────────────────────────────
LEARNING_RATE     = 3e-4
BATCH_SIZE        = 16
EPOCHS            = 20
TRAIN_SPLIT       = 0.9       # 90% train, 10% validation
SEED              = 42

# ── Embeddings ────────────────────────────────────────────────────────────────
COOCCURRENCE_WINDOW = 5       # window for co-occurrence matrix seeding
CONTEXT_OFFSET_DIM  = POINT_DIMS  # dimensionality of context offset network

# ── Inference ─────────────────────────────────────────────────────────────────
MAX_RESPONSE_TOKENS = 64      # maximum tokens to decode in a response
TEMPERATURE         = 1.0     # softness of nearest-cloud matching

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH        = "data/pairs.json"
CHECKPOINT_DIR   = "checkpoints/"
VOCAB_PATH       = "checkpoints/vocab.json"
LOG_PATH         = "logs/training.log"