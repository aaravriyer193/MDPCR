#!/bin/bash
# MDPCR — setup.sh
# Run this once after unzipping to install dependencies.

echo "=================================="
echo "  MDPCR Setup"
echo "=================================="

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install torch numpy --break-system-packages 2>/dev/null || pip install torch numpy

echo ""
echo "Checking data..."
python -c "
import json, sys, os

path = 'data/pairs.json'
if not os.path.exists(path):
    print('ERROR: data/pairs.json not found.')
    print('Please add your Q&A pairs file before training.')
    sys.exit(1)

with open(path) as f:
    pairs = json.load(f)

if len(pairs) < 10:
    print(f'WARNING: Only {len(pairs)} pairs found. Recommend at least 1000 for meaningful training.')
else:
    print(f'Found {len(pairs)} Q&A pairs. Ready to train.')

# Validate format
sample = pairs[0]
assert 'q' in sample and 'a' in sample, 'Each pair must have a q and a field.'
print(f'Format OK. Sample: q=\"{sample[\"q\"][:50]}\"')
"

echo ""
echo "=================================="
echo "  Setup complete."
echo ""
echo "  To train:        python train.py"
echo "  To run demo:     python demo.py"
echo "  To see cloud:    python demo.py --cloud"
echo "=================================="
