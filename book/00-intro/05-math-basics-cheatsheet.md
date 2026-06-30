# Math Basics — Cheatsheet

Quick review of [Math Basics](05-math-basics.md).

---

## Core ideas

- **PEMDAS:** parentheses → `**` → `*`/`/` → `+`/`-`
- **Negatives:** `(-3) ** 2` is 9; slope can be negative (downhill)
- **Exponents:** `x ** 2` means `x * x`; \(x^2\) on paper
- **Fractions / slope:** `rise / run` — vertical change per horizontal step
- **Axes:** x = input (horizontal), y = output (vertical) — read labels
- **`np.exp(x)`** — \(e^x\); preview for softmax/sigmoid
- **`np.log(x)`** — preview for cross-entropy
- **`np.sin` / `np.cos`** — wavy functions; preview for positional encoding

## Python

```python
import numpy as np

print(2 + 3 * 4)       # 14
print((-2) ** 2)       # 4
print(np.exp(1))       # ~2.718
print(np.log(0.5))     # negative
```

## Preview terms

| Term | Learn in |
|------|----------|
| \(e\), `exp` | [Probability](../01-math/06-probability.md) |
| `log` | [Probability](../01-math/06-probability.md) |
| sin, cos | [Special Tensors](../02-pytorch/07-special-tensors.md) |

## Stuck?

Reread the **Readiness for Functions** list in the [full chapter](05-math-basics.md) and run [`app/math/00_math_basics.ipynb`](../../app/math/00_math_basics.ipynb).

---

→ [Full chapter](05-math-basics.md) · [Vocabulary Roadmap](04-vocabulary-roadmap.md)
