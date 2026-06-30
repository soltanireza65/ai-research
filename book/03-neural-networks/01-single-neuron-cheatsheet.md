# Single Neuron — Cheatsheet

Quick review of [Single Neuron](01-single-neuron.md).

> **Payoff chapter:** [Functions](../01-math/01-functions.md) and [Derivatives](../01-math/02-derivatives.md) previewed **ReLU** and **sigmoid**. This chapter teaches them properly.

---

## Core ideas

- A **neuron** = weighted sum + bias + activation
- **Pre-activation** \(z\) = raw score before nonlinearity
- **Weights** scale each input; **bias** shifts the threshold
- **ReLU** — default hidden activation: passes positive values, zeros negatives
- **Sigmoid** — squashes to \((0, 1)\); used for binary classification / probabilities
- One neuron = logistic regression (sigmoid) or one unit in a layer (ReLU)

---

## Key formulas

| Concept | Formula |
|---------|---------|
| Pre-activation | \(z = \mathbf{w} \cdot \mathbf{x} + b = \sum_i w_i x_i + b\) |
| ReLU | \(y = \max(0, z)\) |
| Sigmoid | \(\sigma(z) = \frac{1}{1 + e^{-z}}\) |
| Neuron output | \(y = \sigma(z)\) |

```python
import torch

z = (x * w).sum() + b
relu_out = torch.maximum(z, torch.tensor(0.0))
sigmoid_out = torch.sigmoid(z)
```

| Activation | When to use |
|------------|-------------|
| **ReLU** | Hidden layers — sparse, fast, avoids vanishing gradients |
| **Sigmoid** | Binary output — probability in \((0, 1)\) |

---

## Common mistakes

| Wrong | Right |
|-------|-------|
| ReLU and sigmoid interchangeable | ReLU for hidden; sigmoid for binary probability output |
| Forget bias | Bias shifts decision boundary without changing input weights |
| Deep nets with only sigmoid | ReLU became standard because sigmoid saturates (vanishing gradients) |

## Stuck?

Reread §6 worked examples in the [full chapter](01-single-neuron.md).

→ [Full chapter](01-single-neuron.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
