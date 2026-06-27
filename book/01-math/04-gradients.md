# Gradients

## 1. Introduction

You know derivatives measure sensitivity for **one** input. A neural network has **millions** of inputs — every weight and bias. When you call `loss.backward()` in PyTorch, the framework computes how the loss changes with respect to **each** parameter simultaneously.

That collection of sensitivities is the **gradient**: a vector pointing in the direction of steepest **increase** of the loss. Training walks the opposite direction — downhill — using **gradient descent**.

This chapter bridges single-variable derivatives and the optimization that trains transformers, diffusion models, and every modern neural network.

After this chapter you will be able to:

- Compute **partial derivatives** — "change one variable, hold others fixed."
- Assemble partial derivatives into the **gradient vector** \(\nabla f\).
- Visualize gradient fields on 2D surfaces.
- Implement a simple **gradient descent** loop in Python.
- Explain what `loss.backward()` and `optimizer.step()` do conceptually.
- Connect gradients to learning rate, local minima, and saddle points.

**Where this appears in AI:** `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` is the training heartbeat. Adam, SGD, and every optimizer consumes gradients. Without the gradient, there is no automatic way to improve weights.

---

## 2. Intuition

> 💡 Intuition
>
> Stand on a hillside in fog. You cannot see the whole mountain, but you can feel which way is steepest uphill under your feet. The gradient is that "steepest uphill" direction, expressed as a vector. To reach lower ground (lower loss), walk the opposite way.

For a function of two variables \(f(x, y)\), imagine a heat map viewed from above:

```
  high loss (red)
       ╱╲
      ╱  ╲
     ╱    ╲   ← gradient points uphill (toward red)
    ╱  *   ╲  ← you are here
   ╱        ╲
  low (blue) ╲
```

At each point, the gradient is perpendicular to contour lines and points toward higher values.

**Partial derivative:** Freeze \(y\), wiggle only \(x\). How much does \(f\) change? That is \(\frac{\partial f}{\partial x}\). Swap roles for \(\frac{\partial f}{\partial y}\).

**Gradient vector:** Stack the partials:

\[
\nabla f = \left[\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right]
\]

> 🔬 Deep Dive
>
> In 1D, the derivative is a scalar slope. In \(n\) dimensions, the gradient is an \(n\)-dimensional vector. This is why we needed the vectors chapter first. The gradient lives in the same space as the parameters — you can add a scaled gradient to a weight vector because they have matching dimensions.

---

## 3. Formal Definitions

### Partial derivative

For \(f(x, y)\), the **partial derivative with respect to \(x\)** is:

\[
\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x + h, y) - f(x, y)}{h}
\]

Treat \(y\) as a constant. Compute the ordinary derivative with respect to \(x\).

| Symbol | Meaning |
|--------|---------|
| \(\partial\) | Partial derivative symbol — "hold other variables fixed" |
| \(\frac{\partial f}{\partial x}\) | Rate of change of \(f\) as \(x\) changes, \(y\) fixed |
| \(\frac{\partial f}{\partial y}\) | Rate of change of \(f\) as \(y\) changes, \(x\) fixed |

### Gradient

For \(f: \mathbb{R}^n \to \mathbb{R}\):

\[
\nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]
\]

The **gradient** \(\nabla f\) (read "del f" or "nabla f") is a vector of all partial derivatives.

### Gradient descent update

To **minimize** \(f\), iterate:

\[
\mathbf{x}_{\text{new}} = \mathbf{x}_{\text{old}} - \eta \nabla f(\mathbf{x}_{\text{old}})
\]

| Symbol | Meaning |
|--------|---------|
| \(\mathbf{x}\) | Parameter vector (all weights stacked) |
| \(\eta\) | **Learning rate** — step size (positive scalar) |
| \(-\eta \nabla f\) | Move opposite to gradient (downhill) |

### Jacobian (preview)

When the output is a **vector** (not a scalar loss), partial derivatives organize into a **Jacobian matrix**. Backpropagation through vector-valued layers uses Jacobians; scalar loss reduces to a gradient vector at the end.

---

## 4. Programming Perspective

Partial derivatives in code: differentiate with respect to one variable while others stay fixed.

| Mathematics | Python |
|-------------|--------|
| \(\frac{\partial}{\partial x}(x^2 + y^2)\) | `2 * x` (y treated as constant) |
| \(\nabla f = [2x, 2y]\) | `np.array([2*x, 2*y])` |
| \(x \leftarrow x - \eta \nabla f\) | `x = x - lr * grad` |

```python
import numpy as np

def f(x, y):
    """f(x, y) = x² + y² — bowl centered at origin."""
    return x ** 2 + y ** 2


def gradient_f(x, y):
    """Analytical gradient: [2x, 2y]."""
    return np.array([2 * x, 2 * y])


# Check at a few points
for pt in [(0, 0), (1, 1), (2, -1)]:
    x, y = pt
    print(f"point {pt} -> grad {gradient_f(x, y)}, f = {f(x, y)}")
```

PyTorch autograd computes the same object automatically:

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)
loss = x ** 2 + y ** 2
loss.backward()
print(x.grad, y.grad)  # tensor(6.), tensor(8.) — same as [2x, 2y]
```

`requires_grad=True` tells PyTorch to track operations. `backward()` fills `.grad` with partial derivatives of `loss` with respect to each leaf tensor.

---

## 5. Visualizations

Gradient fields show the direction and strength of steepest ascent at a grid of points.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 12)
y = np.linspace(-2, 2, 12)
X, Y = np.meshgrid(x, y)

# f(x,y) = x² + y², gradient = (2x, 2y)
U = 2 * X
V = 2 * Y

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Contour plot of f
Z = X ** 2 + Y ** 2
axes[0].contour(X, Y, Z, levels=10, cmap="viridis")
axes[0].set_title("Contours of f(x,y) = x² + y²")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_aspect("equal")

# Gradient field (arrows point uphill)
axes[1].quiver(X, Y, U, V, color="coral")
axes[1].set_title("Gradient field (points uphill)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_aspect("equal")

plt.tight_layout()
plt.show()
```

**How to read these plots:**

1. **Left (contours):** Circles centered at origin. Inner contours = lower values. Minimum at \((0, 0)\).

2. **Right (quiver):** Arrows radiate outward from origin — gradient points away from the minimum (uphill). To descend, follow arrows backward (toward origin).

```python
# Gradient descent path on f(x,y) = x² + y²
lr = 0.1
pos = np.array([2.0, 1.5])
path = [pos.copy()]

for _ in range(20):
    grad = 2 * pos
    pos = pos - lr * grad
    path.append(pos.copy())

path = np.array(path)
plt.figure(figsize=(6, 6))
plt.contour(X, Y, Z, levels=10, cmap="viridis")
plt.plot(path[:, 0], path[:, 1], "ro-", markersize=4, label="gradient descent")
plt.legend()
plt.title("GD path converging to (0,0)")
plt.gca().set_aspect("equal")
plt.show()
```

Each step moves opposite to the gradient. Steps shrink as the gradient approaches zero near the minimum.

---

## 6. Worked Examples

### Example 1: Partial derivatives of a simple bowl

\(f(x, y) = x^2 + y^2\)

**Step 1:** \(\frac{\partial f}{\partial x} = 2x\) (treat \(y\) as constant)

**Step 2:** \(\frac{\partial f}{\partial y} = 2y\)

**Step 3:** \(\nabla f = [2x, 2y]\)

At \((1, 2)\): \(\nabla f = [2, 4]\). Loss increases fastest in direction \([2, 4]\); descend toward \([-2, -4]\).

### Example 2: Mixed term

\(f(x, y) = xy\)

**Step 1:** \(\frac{\partial f}{\partial x} = y\) (derivative of \(xy\) w.r.t. \(x\) with \(y\) fixed)

**Step 2:** \(\frac{\partial f}{\partial y} = x\)

At \((2, 3)\): \(\nabla f = [3, 2]\).

```python
print(gradient_f_xy := np.array([3, 2]))  # at (2, 3) for f=xy
```

### Example 3: Gradient descent by hand (one step)

Minimize \(f(x) = x^2\) starting at \(x = 3\), learning rate \(\eta = 0.1\).

**Step 1:** \(f'(x) = 2x\). At \(x = 3\): gradient \(= 6\).

**Step 2:** Update: \(x_{\text{new}} = 3 - 0.1 \times 6 = 3 - 0.6 = 2.4\).

**Step 3:** New loss: \(f(2.4) = 5.76\) (down from 9).

### Example 4: Linear regression gradient

Loss on one point: \(L(w) = (y - wx)^2\). With \(y = 5\), \(x = 2\):

\[
\frac{dL}{dw} = -2x(y - wx)
\]

At \(w = 1\): \(\frac{dL}{dw} = -2(2)(5 - 2) = -16\). Negative → increase \(w\) to reduce loss.

```python
def dL_dw(w, y, x):
    return -2 * x * (y - w * x)

print(dL_dw(1, 5, 2))  # -16
```

### Example 5: Full PyTorch training step

```python
import torch
import torch.nn as nn

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])  # y = 2x

for epoch in range(100):
    optimizer.zero_grad()           # clear old gradients
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    loss.backward()                 # compute gradients
    optimizer.step()                # w <- w - lr * grad

print(list(model.parameters()))  # weight ≈ 2, bias ≈ 0
```

`zero_grad()` prevents gradient accumulation. `backward()` fills `.grad`. `step()` applies the update rule.

---

## 7. AI Connection

> 🧠 AI Insight
>
> Training is repeated gradient descent on a stochastic estimate of the true loss. Each mini-batch gives a noisy gradient — the average gradient over millions of examples is approximated by 32 or 256 examples. Noise helps escape shallow local minima.

**Loss landscape:** For a network with \(N\) parameters, the loss \(L(\theta_1, \ldots, \theta_N)\) is a function from \(\mathbb{R}^N\) to \(\mathbb{R}\). You cannot visualize it — but gradient descent still works locally.

**Optimizers beyond vanilla SGD:**

| Optimizer | Idea |
|-----------|------|
| SGD | \(\theta \leftarrow \theta - \eta \nabla L\) |
| Momentum | Accumulate velocity — damp oscillation |
| Adam | Per-parameter adaptive learning rates from gradient history |

All consume gradients from `backward()`.

**Vanishing/exploding gradients:** In deep networks, repeated chain rule multiplication can shrink or blow up gradients. Residual connections, layer normalization, and careful initialization mitigate this — topics for later neural network chapters.

**`loss.backward()` internals (simplified):**

1. Build computation graph during forward pass.
2. Start at loss node with gradient 1.
3. Walk backward applying chain rule to each operation.
4. Accumulate `.grad` on each leaf parameter tensor.

You rarely compute gradients by hand for real models — but understanding what they **mean** is essential for debugging training.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> **Ascending instead of descending.** The gradient points **uphill**. Update with a **minus** sign: \(\theta - \eta \nabla L\). A plus sign maximizes loss — the opposite of training.

> ⚠️ Common Mistake
>
> **Forgetting `optimizer.zero_grad()`.** Gradients **accumulate** by default in PyTorch. Without zeroing, step 2 uses the sum of step 1 and step 2 gradients — wrong update.

> ⚠️ Common Mistake
>
> **Confusing gradient with loss value.** A large loss does not mean a large gradient. Near a flat minimum, loss can be moderate while gradient is near zero — training appears stuck.

> ⚠️ Common Mistake
>
> **Learning rate too large or too small.** Too large: overshoot, oscillate, diverge. Too small: progress takes forever. Watch loss curves; use learning rate schedulers.

**Correct understanding:** The gradient is a vector of partial derivatives. Minimize by stepping opposite to it. PyTorch automates computation; you control learning rate and optimizer choice.

---

## 9. Exercises

### Easy

1. For \(f(x, y) = x^2 + y^2\), find \(\frac{\partial f}{\partial x}\) and \(\frac{\partial f}{\partial y}\).
2. Evaluate \(\nabla f\) at \((3, 4)\) for the function above.
3. If gradient is \([6, 8]\) and learning rate is 0.1, what is the gradient descent update to position \([1, 1]\)?
4. Run the PyTorch snippet with `requires_grad=True` for \(L = x^2 + y^2\) at \(x=1, y=2\). Verify `.grad` values.

### Medium

5. Find \(\nabla f\) for \(f(x, y) = 3x^2 + xy + y^2\).
6. Plot the gradient field for \(f(x, y) = xy\) on \([-2,2] \times [-2,2]\).
7. Implement 10 steps of gradient descent on \(f(x) = (x-3)^2\) starting at \(x=0\) with \(\eta=0.1\). Print \(x\) each step.
8. Explain why \(\nabla f = [0, 0]\) at a local minimum of a smooth function.

### Hard

9. For \(L(w, b) = (y - (wx + b))^2\), find \(\frac{\partial L}{\partial w}\) and \(\frac{\partial L}{\partial b}\). Evaluate at \(w=1, b=0, x=2, y=5\).
10. Implement 2D gradient descent on \(f(x,y) = x^2 + 4y^2\) from \((3, 3)\) with \(\eta=0.1\). Plot the path. Why does it approach the origin asymmetrically?
11. What does `torch.nn.utils.clip_grad_norm_` do, and when would you use it?

### Challenge

12. **Visualize a saddle:** Plot contours and gradient field for \(f(x,y) = x^2 - y^2\). Run gradient descent from \((1, 1)\). Explain why this point is a saddle, not a minimum.
13. **Manual autograd:** Without `backward()`, use central differences to estimate the gradient of \(f(w) = (y - wx)^2\) for a 1-parameter linear fit on 5 synthetic points. Compare to PyTorch `.grad`.

---

## 10. Mini Project

### Gradient Descent Trainer

Implement a from-scratch trainer for 1D linear regression:

1. Generate synthetic data \(y = 2x + 1 + \text{noise}\).
2. Define \(L(w, b) = \frac{1}{n}\sum (y_i - (wx_i + b))^2\).
3. Compute analytical gradients for \(w\) and \(b\).
4. Run gradient descent for 200 steps; plot data, fitted line, and loss curve.
5. Save figure to `book/assets/04-gradient-descent.png`.

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(0)
x = np.linspace(0, 5, 50)
y = 2 * x + 1 + np.random.randn(50) * 0.5

w, b = 0.0, 0.0
lr = 0.01
losses = []

for step in range(200):
    pred = w * x + b
    error = pred - y
    loss = np.mean(error ** 2)
    losses.append(loss)
    dw = 2 * np.mean(error * x)
    db = 2 * np.mean(error)
    w -= lr * dw
    b -= lr * db

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(x, y, alpha=0.5, label="data")
axes[0].plot(x, w * x + b, "r", label=f"fit: {w:.2f}x + {b:.2f}")
axes[0].legend()
axes[1].plot(losses)
axes[1].set_title("Loss vs step")
out = Path("book/assets/04-gradient-descent.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"w≈{w:.2f}, b≈{b:.2f}, saved to {out}")
```

<details>
<summary>Mini project checklist</summary>

- [ ] Synthetic data generated
- [ ] Analytical gradients implemented
- [ ] Loss decreases over steps
- [ ] Data + fit line + loss curve saved

</details>

---

## 11. Interview Questions

**Q1:** What is the gradient of a scalar-valued function of many variables?

**A1:** The gradient is a vector whose components are the partial derivatives of the function with respect to each input variable. It points in the direction of steepest increase of the function. For a loss function, the negative gradient points toward steepest decrease — the direction used for training updates.

**Q2:** What happens when you call `loss.backward()` in PyTorch?

**A2:** PyTorch traverses the computation graph in reverse topological order, applying the chain rule at each node. It accumulates the partial derivative of the loss with respect to every tensor that has `requires_grad=True`. Leaf parameters end up with their `.grad` fields populated. No weight update happens yet — that is `optimizer.step()`.

**Q3:** Why do we use learning rates?

**A3:** The gradient tells direction, not step size. The learning rate \(\eta\) scales how far to move. Too large: overshoot minima, diverge. Too small: slow convergence. Adaptive optimizers like Adam adjust effective step sizes per parameter based on gradient history.

**Q4:** What is the difference between a partial derivative and a total derivative?

**A4:** A partial derivative varies one input while holding others fixed — \(\partial f / \partial x\). A total derivative accounts for how all inputs change together along a path — relevant when inputs are coupled. In standard training, each parameter is updated independently using its partial derivative, so partial derivatives (assembled as the gradient) are what you need.

**Q5:** Can gradient descent find the global minimum?

**A5:** Not guaranteed. Loss landscapes for neural networks are non-convex with many local minima and saddle points. In practice, SGD with noise often finds **good enough** minima that generalize well. Convex problems (like linear regression with MSE) have a single global minimum where gradient descent converges reliably.

---

## 12. Summary

### Key formulas

| Concept | Formula |
|---------|---------|
| Partial derivative | \(\frac{\partial f}{\partial x_i}\) — derivative w.r.t. \(x_i\), others fixed |
| Gradient | \(\nabla f = [\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}]\) |
| Gradient descent | \(\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x})\) |
| Bowl example | \(f=x^2+y^2 \Rightarrow \nabla f = [2x, 2y]\) |
| MSE gradient (1 param) | \(\frac{d}{dw}(y-wx)^2 = -2x(y-wx)\) |

### Key terminology

- **Partial derivative** — sensitivity to one variable, others held constant
- **Gradient** — vector of all partial derivatives
- **Gradient descent** — iterative optimization moving opposite to the gradient
- **Learning rate** — step size hyperparameter \(\eta\)
- **Autograd** — automatic gradient computation via chain rule
- **Contour line** — curve where \(f\) is constant; gradient is perpendicular
- **Critical point** — where gradient is zero (minimum, maximum, or saddle)

---

## 13. Preview

Gradients tell you how to update **vectors** of parameters. The next chapter — **Matrices** — shows how neural networks apply **linear transformations** to many vectors at once: every `nn.Linear` layer is matrix multiplication, and attention computes \(\mathbf{Q}\mathbf{K}^\top\) — a large grid of dot products stored as a matrix.

Understanding matrices completes the core math toolkit for reading PyTorch code and transformer architectures.

**Next chapter:** [Matrices](05-matrices.md)

---

## Lab

Companion notebook: [`app/math/04_gradients.ipynb`](../../app/math/04_gradients.ipynb)
