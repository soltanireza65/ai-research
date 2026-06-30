# Functions — Cheatsheet

Quick review of [Functions](01-functions.md). Read this when you want the essentials without the full chapter.

---

## One-liner

A **function** is a rule: one input → exactly one output. Same input always gives the same output.

```
x  ──►  f  ──►  y
3  ──►  2x ──►  6
```

---

## Notation

| Symbol | Meaning |
|--------|---------|
| `f` | Function name |
| `x` | Input |
| `f(x)` | "f of x" — apply f to x (not f × x) |
| `=` | "is defined as" |

---

## Key formulas

| Type | Math | Python | Shape |
|------|------|--------|-------|
| Linear | \(f(x) = kx + b\) | `k * x + b` | Straight line |
| Quadratic | \(f(x) = x^2\) | `x ** 2` | U-shaped parabola |
| Composition | \(f(g(x))\) | `f(g(x))` | Chain rules |
| ReLU | \(\max(0, x)\) | `max(0, x)` | Flat below 0, ramp above |

- **Slope `k`** — how steep (rate of change)
- **Intercept `b`** — value at \(x = 0\)

---

## Math ↔ Python

```python
def f(x): return 2 * x      # f(x) = 2x
def g(x): return x ** 2     # g(x) = x²
def h(x): return f(g(x))    # composition
```

NumPy applies functions element-wise to arrays.

---

## Shapes at a glance

| Function | Key property |
|----------|--------------|
| Linear | Constant slope everywhere |
| Quadratic | Symmetric: \(f(-x) = f(x)\); minimum at 0 for \(x^2\) |
| Cubic | Not symmetric; negative in → negative out |
| Composition | Output of inner → input of outer |

---

## AI connection

| Idea | Function form |
|------|----------------|
| Linear regression | \(y = wx + b\) |
| Logistic regression | \(\sigma(wx + b)\) |
| Neural layer | \(\sigma(Wx + b)\) |
| Loss (MSE) | \(\frac{1}{n}\sum(y - \hat{y})^2\) |
| Full network | \(f_L(\cdots f_1(x)\cdots)\) — composed functions |

**Weights ≈ slopes. Biases ≈ intercepts. Activations add nonlinearity.**

Without nonlinear activations, stacking layers is still just one linear function.

---

## ML function names (preview)

You will see these in Chapter 1 before they are fully explained. **That is intentional.** Know the one-liner now; master them later.

| Name | Enough for now | Learn properly in |
|------|----------------|-------------------|
| **ReLU** | `max(0, x)` — zero if negative, else pass through | [Single Neuron](../03-neural-networks/01-single-neuron.md) |
| **Sigmoid** | Squashes any number to (0, 1) — "probability-ish" | [Single Neuron](../03-neural-networks/01-single-neuron.md) |
| **MSE** | Average squared error — how far off predictions are | [Gradients](04-gradients.md) |
| **Cross-entropy** | Punishes confident wrong predictions | [Probability](06-probability.md) |
| **Softmax** | Turns scores into probabilities that sum to 1 | [Probability](06-probability.md) |

```python
# ReLU — activation (after weighted sum)
relu = lambda z: max(0, z)

# Sigmoid — activation (often for binary output)
sigmoid = lambda z: 1 / (1 + 2.718 ** (-z))  # e ≈ 2.718

# MSE — loss (regression: predict a number)
mse = lambda y, y_hat: (y - y_hat) ** 2
```

**You do not need to memorize formulas yet.** At the Functions stage, only remember:

- **Activations** (ReLU, sigmoid) — nonlinear rules applied *after* the linear part of a neuron
- **Losses** (MSE, cross-entropy) — functions that score *how wrong* the model is

---

## Common mistakes

| Wrong | Right |
|-------|-------|
| `f(x)` means f × x | `f(x)` means apply f to x |
| Variable name defines the function | The **rule** defines it |
| Deep net = automatically nonlinear | Only if activations are nonlinear |
| Any input works | Check **domain** (e.g. `1/x` at 0) |

---

## Quick examples

**Evaluate:** \(f(x) = 3x - 1\) → \(f(4) = 11\)

**Symmetry:** \(x^2\) at 3 and -3 both give 9

**Compose:** \(f(x)=2x\), \(h(x)=x+3\) → \(f(h(2)) = f(5) = 10\)

**Neuron:** \(\text{ReLU}(2x - 1)\) — linear then activation

---

## Interview sound bites

1. **Function** = deterministic input→output rule (same as Python `def`).
2. **Why nonlinear activations?** Stacked linear layers collapse to one linear map.
3. **Composition** = how deep networks work; backprop uses chain rule on this.
4. **\(x^2\) bowl** = intuition for loss landscapes and optimization.
5. **ReLU / sigmoid / MSE** — named ML functions; preview here, full treatment in neuron + probability chapters.

---

## What's next

**Derivatives** — how sensitive is the output to a tiny change in input? That's gradient descent.

→ [Derivatives cheatsheet](02-derivatives-cheatsheet.md) *(when available)*  
→ [Full chapter](01-functions.md)
