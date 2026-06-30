# Probability — Cheatsheet

**Payoff chapter** for softmax and cross-entropy previews.

## Core

- \(P(\text{event}) \in [0,1]\); probabilities sum to 1
- Softmax: scores → distribution
- Cross-entropy: \(-\log \hat{p}_{\text{true class}}\) — punishes confident wrong guesses

## Python

```python
import torch.nn.functional as F
loss = F.cross_entropy(logits, target)  # expects raw logits
```

## Stuck?

Reread softmax and cross-entropy §3 in the [full chapter](06-probability.md).

→ [Full chapter](06-probability.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
