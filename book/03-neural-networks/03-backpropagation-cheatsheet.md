# Backpropagation — Cheatsheet

Quick review of [Backpropagation](03-backpropagation.md).

> **Payoff:** The **chain rule** from [Derivatives](../01-math/02-derivatives.md) — \(\frac{dy}{dx} = \frac{dy}{du}\frac{du}{dx}\) — is what `loss.backward()` automates through the computational graph.

---

## Core ideas

- **Forward pass** builds predictions; **backward pass** computes gradients
- **Chain rule** multiplies local derivatives along the graph path
- **Autograd** (`loss.backward()`) does this automatically in PyTorch
- **SGD update:** \(w \leftarrow w - \eta \frac{\partial L}{\partial w}\)
- Call `optimizer.zero_grad()` before each backward to reset accumulated grads
- **Dead ReLU:** neuron with \(z \leq 0\) gets zero gradient permanently

---

## Key formulas

| Concept | Formula |
|---------|---------|
| Chain rule | \(\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}\) |
| MSE gradient | \(\frac{\partial}{\partial \hat{y}}\frac{1}{2}(\hat{y}-y)^2 = \hat{y} - y\) |
| ReLU derivative | \(1\) if \(z > 0\), else \(0\) |
| SGD update | \(w \leftarrow w - \eta \frac{\partial L}{\partial w}\) |

```python
loss = criterion(pred, target)
optimizer.zero_grad()
loss.backward()              # chain rule through graph
optimizer.step()               # w -= lr * grad
```

---

## Common mistakes

| Wrong | Right |
|-------|-------|
| Skip `zero_grad()` | Gradients accumulate — reset each step |
| Backward without `requires_grad` | Parameters need `requires_grad=True` |
| Huge learning rate | Start small; \(\eta\) controls step size |

## Stuck?

Rerun [`03_backpropagation.ipynb`](../../app/neural_networks/03_backpropagation.ipynb) and reread §6.

→ [Full chapter](03-backpropagation.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
