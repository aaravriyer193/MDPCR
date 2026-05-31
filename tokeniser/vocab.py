# MDPCR — tokeniser/vocab.py
# Builds and manages the vocabulary from the training corpus.
# Also computes the co-occurrence matrix used to seed embeddings.

import json
import re
from collections import Counter, defaultdict
from typing import Optional
import config


class Vocabulary:
    """
    Vocabulary built from the training corpus.

    Maps tokens ↔ integer IDs.
    Computes co-occurrence frequencies for embedding seeding.
    """

    SPECIAL_TOKENS = [config.PAD_TOKEN, config.UNK_TOKEN,
                      config.BOS_TOKEN, config.EOS_TOKEN]

    def __init__(self):
        self.token2id : dict[str, int] = {}
        self.id2token : dict[int, str] = {}
        self.freq     : Counter        = Counter()

        # Co-occurrence matrix: cooccur[a][b] = times a and b appear within
        # COOCCURRENCE_WINDOW tokens of each other in the corpus
        self.cooccur  : dict[int, Counter] = defaultdict(Counter)

        # Add special tokens first
        for tok in self.SPECIAL_TOKENS:
            self._add(tok)

    def _add(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx]   = token
        return self.token2id[token]

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        Simple whitespace + punctuation tokeniser.
        Lowercases everything. Splits on punctuation as separate tokens.
        """
        text = text.lower().strip()
        # Split on whitespace and common punctuation, keeping punctuation as tokens
        tokens = re.findall(r"[a-z']+|[.,!?;:]", text)
        return tokens

    def build(self, pairs: list[dict], min_freq: int = 1) -> None:
        """
        Build vocabulary from a list of {"q": ..., "a": ...} pairs.
        Also computes the co-occurrence matrix.
        """
        all_tokens: list[list[str]] = []

        # Count frequencies
        for pair in pairs:
            for text in [pair["q"], pair["a"]]:
                tokens = self.tokenize(text)
                all_tokens.append(tokens)
                self.freq.update(tokens)

        # Add tokens that meet minimum frequency
        for token, count in self.freq.most_common():
            if count >= min_freq:
                self._add(token)

        # Build co-occurrence matrix
        w = config.COOCCURRENCE_WINDOW
        for tokens in all_tokens:
            ids = [self.token2id.get(t, self.token2id[config.UNK_TOKEN])
                   for t in tokens]
            for i, id_a in enumerate(ids):
                for j in range(max(0, i - w), min(len(ids), i + w + 1)):
                    if i != j:
                        self.cooccur[id_a][ids[j]] += 1

    def encode(self, text: str) -> list[int]:
        """Convert text to a list of token IDs."""
        tokens = self.tokenize(text)
        unk_id = self.token2id[config.UNK_TOKEN]
        return [self.token2id.get(t, unk_id) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back to text."""
        tokens = [self.id2token.get(i, config.UNK_TOKEN) for i in ids]
        return " ".join(tokens)

    def __len__(self) -> int:
        return len(self.token2id)

    def save(self, path: str) -> None:
        data = {
            "token2id" : self.token2id,
            "freq"     : dict(self.freq),
            "cooccur"  : {str(k): dict(v) for k, v in self.cooccur.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path) as f:
            data = json.load(f)
        vocab = cls()
        vocab.token2id = data["token2id"]
        vocab.id2token = {int(k): v for k, v in
                          {v: k for k, v in data["token2id"].items()}.items()}
        vocab.id2token = {v: k for k, v in vocab.token2id.items()}
        vocab.freq     = Counter(data["freq"])
        vocab.cooccur  = defaultdict(
            Counter,
            {int(k): Counter({int(kk): vv for kk, vv in v.items()})
             for k, v in data["cooccur"].items()}
        )
        return vocab
