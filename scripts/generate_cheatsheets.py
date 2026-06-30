"""Generate minimal chapter cheatsheets for handbook review.

Only writes files that do not exist or are shorter than generated content.
Hand-maintained cheatsheets (PyTorch, NN, transformers) are not overwritten.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


CHEATSHEETS: dict[str, str] = {
    "book/00-intro/05-math-basics-cheatsheet.md": """# Math Basics — Cheatsheet

Quick review of [Math Basics](05-math-basics.md).

## Core ideas

- PEMDAS; `**` for exponents; slope = rise/run
- x-axis = input, y-axis = output
- `np.exp`, `np.log`, `sin`/`cos` are previews for later chapters

## Stuck?

Reread readiness checklist in [Math Basics](05-math-basics.md).

→ [Full chapter](05-math-basics.md) · [Vocabulary Roadmap](04-vocabulary-roadmap.md)
""",
    "book/01-math/02-derivatives-cheatsheet.md": """# Derivatives — Cheatsheet

Quick review of [Derivatives](02-derivatives.md).

## Core

- Derivative = rate of change = slope of tangent at a point
- Power rule: \\(d/dx\\, x^n = n x^{n-1}\\)
- Chain rule (preview): \\(\\frac{dy}{dx} = \\frac{dy}{du}\\frac{du}{dx}\\) → [Backpropagation](../03-neural-networks/03-backpropagation.md)

## Python

```python
def numerical_derivative(f, x, h=0.001):
    return (f(x + h) - f(x - h)) / (2 * h)
```

## Preview terms

| Term | Learn in |
|------|----------|
| ReLU derivative | [Single Neuron](../03-neural-networks/01-single-neuron.md) |
| `loss.backward()` | [Backpropagation](../03-neural-networks/03-backpropagation.md) |

→ [Full chapter](02-derivatives.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
""",
    "book/01-math/03-vectors-cheatsheet.md": """# Vectors — Cheatsheet

Quick review of [Vectors](03-vectors.md).

## Core

- Vector = ordered list of numbers; magnitude \\(\\|v\\|\\), addition, scalar multiply
- Dot product: \\(u \\cdot v = \\sum u_i v_i\\) — alignment (preview for attention)

## Python

```python
import numpy as np
v = np.array([3, 4])
np.linalg.norm(v)  # magnitude
u @ v              # dot product
```

→ [Full chapter](03-vectors.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
""",
    "book/01-math/04-gradients-cheatsheet.md": """# Gradients — Cheatsheet

Quick review of [Gradients](04-gradients.md).

## Core

- Gradient = vector of partial derivatives \\(\\nabla f\\)
- Points uphill; gradient descent steps opposite: \\(\\theta \\leftarrow \\theta - \\eta \\nabla L\\)
- MSE gradient preview connects loss to training

## Preview

| Term | Learn in |
|------|----------|
| Adam, momentum | [Backpropagation](../03-neural-networks/03-backpropagation.md) |

→ [Full chapter](04-gradients.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
""",
    "book/01-math/05-matrices-cheatsheet.md": """# Matrices — Cheatsheet

Quick review of [Matrices](05-matrices.md).

## Core

- Matrix = grid of numbers; shape \\((m, n)\\)
- Matmul: \\((m,n)(n,p) \\rightarrow (m,p)\\); inner dims must match
- Linear layer: `output = input @ weight.T + bias`

## Preview

| Term | Learn in |
|------|----------|
| Full attention formula | [Attention Mechanism](../04-transformers/01-attention-mechanism.md) |

→ [Full chapter](05-matrices.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
""",
    "book/01-math/06-probability-cheatsheet.md": """# Probability — Cheatsheet

**Payoff chapter** for softmax and cross-entropy previews.

## Core

- \\(P(\\text{event}) \\in [0,1]\\); probabilities sum to 1
- Softmax: scores → distribution
- Cross-entropy: \\(-\\log \\hat{p}_{\\text{true class}}\\) — punishes confident wrong guesses

## Python

```python
import torch.nn.functional as F
loss = F.cross_entropy(logits, target)  # expects raw logits
```

→ [Full chapter](06-probability.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
""",
}


def main() -> None:
    for rel, content in CHEATSHEETS.items():
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        new_wc = word_count(content)
        if path.exists():
            existing_wc = word_count(path.read_text(encoding="utf-8"))
            if existing_wc >= new_wc:
                print("skip (richer exists)", rel)
                continue
        path.write_text(content.strip() + "\n", encoding="utf-8")
        print("wrote", rel)


if __name__ == "__main__":
    main()
