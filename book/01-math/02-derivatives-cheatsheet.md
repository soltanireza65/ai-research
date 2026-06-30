# Derivatives — Cheatsheet

Quick review of [Derivatives](02-derivatives.md). Read this when you want the essentials without the full chapter.

---

## One-liner

The **derivative** measures sensitivity: nudge the input a tiny bit — how much does the output move, and in which direction?

```
x  ──►  f(x)  ──►  y
nudge x     f changes by ≈ f'(x) × nudge
```

---

## Notation

| Symbol | Meaning |
|--------|---------|
| \(f'(x)\) | Derivative of \(f\) at \(x\) — slope of tangent |
| \(\frac{df}{dx}\) | Same as \(f'(x)\) — "dee f dee x" |
| \(\frac{d}{dx}[x^n]\) | Derivative of \(x^n\) with respect to \(x\) |
| \(h\) | Tiny step in input (for numerical estimates) |

---

## Key formulas

| Rule | Math | Python |
|------|------|--------|
| Definition | \(f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h}\) | `numerical_derivative(f, x)` |
| Power rule | \(\frac{d}{dx}[x^n] = n x^{n-1}\) | `n * x ** (n - 1)` |
| Constant | \(\frac{d}{dx}[c] = 0\) | `0` |
| Linearity | \(\frac{d}{dx}[af + bg] = af' + bg'\) | `a * f_prime + b * g_prime` |
| Chain rule | \(\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)\) | `outer_prime(inner(x)) * inner_prime(x)` |
| Numerical | \(\frac{f(x+h)-f(x-h)}{2h}\) | `(f(x+h) - f(x-h)) / (2*h)` |

- **Positive derivative** — function increasing as \(x\) increases
- **Negative derivative** — function decreasing as \(x\) increases
- **Zero derivative** — flat tangent (possible min, max, or saddle)

---

## Math ↔ Python

```python
def f(x):
    return x ** 2

def f_prime(x):
    return 2 * x  # power rule

def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

print(f_prime(3))                        # 6 — exact
print(numerical_derivative(f, 3))        # ≈ 6 — estimated
```

PyTorch computes exact derivatives automatically — you still need the intuition.

---

## Shapes at a glance

| Function | Derivative | Key point |
|----------|------------|-----------|
| \(f(x) = 2x\) | \(f'(x) = 2\) | Constant slope everywhere |
| \(f(x) = x^2\) | \(f'(x) = 2x\) | Negative left, zero at 0, positive right |
| \(f(x) = x^3\) | \(f'(x) = 3x^2\) | Always ≥ 0; flat at origin |
| \(f(x) = c\) | \(f'(x) = 0\) | Flat line — no sensitivity |

---

## AI connection

| Idea | Derivative role |
|------|-----------------|
| Gradient descent | Move opposite to \(\frac{dL}{dw}\) |
| Loss landscape | Derivative tells uphill direction |
| Chain rule | Composed layers → multiply local derivatives |
| Learning rate | Scales how far to step, not direction |

**Core for this chapter:** derivative = sensitivity. Training uses that to reduce error.

---

## ML names (preview)

You may see these before they are fully explained. **That is intentional.**

| Name | Enough for now | Learn properly in |
|------|----------------|-------------------|
| **ReLU derivative** | 1 if \(x > 0\), else 0 | [Single Neuron](../03-neural-networks/01-single-neuron.md) |
| **Sigmoid derivative** | Largest near \(x = 0\), tiny at extremes | [Single Neuron](../03-neural-networks/01-single-neuron.md) |
| **`loss.backward()`** | Chain rule through all layers | [Backpropagation](../03-neural-networks/03-backpropagation.md) |
| **Gradient** | Vector of partial derivatives | [Gradients](04-gradients.md) |
| **Autograd** | Framework computes derivatives for you | [Gradients](04-gradients.md) |

```python
# ReLU — derivative is 1 or 0 (preview)
relu_prime = lambda x: 1.0 if x > 0 else 0.0

# MSE loss derivative w.r.t. weight (preview)
# L = (y - w*x)**2  →  dL/dw = -2*x*(y - w*x)
```

**You do not need activation derivative formulas yet.** At the Derivatives stage, remember:

- **Derivative** = slope / sensitivity at one point
- **Chain rule** = multiply derivatives along a composed path
- **ML names** = previews; full treatment in neuron + backprop chapters

---

## Common mistakes

| Wrong | Right |
|-------|-------|
| \(f(x)\) and \(f'(x)\) are the same | \(f(x)\) = height; \(f'(x)\) = slope |
| Derivative must be positive | Negative = decreasing; zero = flat |
| Skip chain rule in composed functions | Multiply local derivatives along the path |
| Use \(h = 0.1\) for numerical derivative | Use \(h \approx 10^{-5}\) for float64 |
| Confident wrong prediction is cheap | Large loss when \(\hat{p}_y \to 0\) (probability chapter) |

---

## Quick examples

**Power rule:** \(\frac{d}{dx}[x^4] = 4x^3\)

**Linear:** \(f(x) = 7x + 3 \Rightarrow f'(x) = 7\) (constant)

**Chain rule:** \(y = (2x+1)^2 \Rightarrow \frac{dy}{dx} = 2(2x+1) \cdot 2 = 4(2x+1)\)

**At minimum:** \(f(x) = x^2\) has \(f'(0) = 0\) — horizontal tangent at the bottom

---

## Interview sound bites

1. **Derivative** = instantaneous rate of change = tangent slope.
2. **Why chain rule?** Networks compose functions; backprop multiplies local derivatives.
3. **ReLU at negative input** — derivative 0, gradient blocked (preview).
4. **Numerical vs autograd** — same concept; autograd is exact and fast.
5. **\(f'(0) = 0\)** — critical point; gradient descent slows near optima.

---

## Stuck?

- Re-read §3 (power rule + chain rule) in the [full chapter](02-derivatives.md).
- Run `numerical_derivative` on \(f(x) = x^2\) and compare to `2*x`.
- Check the [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md) if ML jargon in exercises feels early.

---

## What's next

**Vectors** — lists of numbers with direction; then **Gradients** combine derivatives with vectors for multi-parameter optimization.

→ [Vectors cheatsheet](03-vectors-cheatsheet.md)  
→ [Full chapter](02-derivatives.md)
