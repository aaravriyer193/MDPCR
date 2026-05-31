# MDPCR — Multi Dimensional Point Cloud Reasoning

> Language doesn't flow. It converges.

MDPCR is an experimental neural architecture for language modelling that replaces sequential transformer attention with a spatial point cloud that undergoes iterative convergence.

## Core Idea

In a transformer, tokens are processed left-to-right in a sequence. Meaning is extracted by attending over positions. This is a useful approximation, but it isn't how meaning actually works.

In MDPCR:
- Every token is a **point in N-dimensional space**, positioned by its local context (not its sequence index)
- All points **influence all other points simultaneously**, weighted by density and distance
- The cloud **iterates until it converges** — simple inputs settle fast, hard inputs take longer
- The model learns **transformations between clouds** — not what word comes next, but how a prompt cloud becomes a response cloud

## Architecture

```
prompt text
    ↓
contextual tokeniser (window of 3)
    ↓
prompt cloud  (M points × N dimensions)
    ↓
iterative influence update  (until cloud(t) ≈ cloud(t-1))
    ↓
converged cloud
    ↓
decoder
    ↓
response text
```

### The Contextual Tokeniser

A token's starting position is determined by a window of 3 tokens (itself + left + right neighbour). Same surface form, different neighbours → different starting position in the cloud. "bank" near "river" ≠ "bank" near "money".

### The Influence Update Rule

```python
for each point p:
    influence(p) = Σ over all other points q:
        (attributes of q) × density(q) / distance(p, q)²
    p_new = p + step_scale × learned_projection(influence(p))
```

No sequence. No positional encoding. No left-to-right. Pure geometry.

### Convergence

```
stop when: mean( |cloud(t) - cloud(t-1)| ) < threshold
```

The cloud thinks as long as the problem requires. Variable depth reasoning emerges naturally.

## Model Sizes

| Size   | Dimensions | Points/token | Influence |
|--------|------------|--------------|-----------|
| Small  | 16         | 1            | Local     |
| Medium | 64         | 4            | Regional  |
| Large  | 256        | 16           | Global    |

## File Structure

```
mdpcr/
├── core/
│   ├── point.py          # N-dimensional point with density
│   ├── cloud.py          # Cloud container and iteration engine
│   ├── influence.py      # Learned influence update rule
│   └── convergence.py    # Convergence checking
├── tokeniser/
│   ├── vocab.py          # Vocabulary + co-occurrence seeding
│   ├── context.py        # Window-of-3 contextual encoder
│   └── decoder.py        # Cloud → text decoder
├── model/
│   └── mdpcr.py          # Top-level model class
├── training/
│   ├── dataset.py        # Q&A pair dataset
│   ├── trainer.py        # Training loop
│   └── checkpoint.py     # Save/load
├── utils/
│   ├── logging.py        # Training logger
│   └── visualise.py      # Cloud state printer
├── config.py             # All hyperparameters
├── train.py              # Training entry point
└── demo.py               # Interactive demo
```

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Add your training data
# data/pairs.json should be: [{"q": "...", "a": "..."}, ...]

# Train
python train.py

# Run interactive demo
python demo.py

# Show cloud state during inference
python demo.py --cloud
```

## Training Data Format

```json
[
  {"q": "What is fire?", "a": "Fire is a chemical reaction that releases heat and light."},
  {"q": "How are you?",  "a": "I am doing well, thank you for asking."}
]
```

## What's Different from a Transformer

| | Transformer | MDPCR |
|---|---|---|
| Token representation | 1D vector | N-D point in space |
| Attention | Sequential Q/K/V | Spatial density/distance |
| Processing order | Left to right | Simultaneous, all points |
| Depth | Fixed layers | Variable (until convergence) |
| Training signal | Next token prediction | Cloud-to-cloud distance |
| Positional encoding | Required | Not needed (position IS the encoding) |

## Status

This is early-stage research. The architecture is theoretically grounded and the core mechanics are implemented. The model is not yet competitive with transformers at scale — that isn't the point yet. The point is to establish whether cloud-based reasoning is a viable substrate for language at all.

## Contributing

This project is looking for researchers, engineers, and curious minds. If you find this interesting:

- Open an issue with thoughts, questions, or critiques
- Submit a PR for improvements to the core mechanics
- Try training on your own data and report what you find

## License

MIT
