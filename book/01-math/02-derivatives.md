# Derivatives

## 1. Introduction

In the previous chapter you learned that a **function** is a rule: input in, output out. Training a model asks a follow-up question: **how sensitive is the output to a tiny change in the input (or in a weight)?**

That sensitivity is measured by the **derivative**. The derivative answers: "If I nudge \(x\) a little, how much does \(f(x)\) move, and in which direction?"

**What you must learn in this chapter (core):**

- Interpret the derivative as **instantaneous rate of change** and **tangent slope**
- Compute simple derivatives using the **power rule** and **linearity**
- Estimate derivatives numerically in Python (the same trick autograd uses internally)
- Use the **chain rule** on composed functions like \((2x+1)^2\)
- Read a derivative plot: where is the slope zero, positive, or negative?

**What you can skip for now (preview):**

Words like **ReLU**, **sigmoid**, **backpropagation**, and **`loss.backward()`** may appear as **preview labels** — a map of where the book goes, not a test of what you should already know. Skip any `📌 Preview` box and use the [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md) when a name bothers you.

After this chapter you will **not** be expected to derive activation derivatives or run full backprop by hand. You **will** understand derivatives — the sensitivity engine behind gradient descent.

**Suggested pacing (3 sessions):**

- Session A: §1–§3 + [cheatsheet](02-derivatives-cheatsheet.md) skim
- Session B: §4–§6 + lab notebook
- Session C: Easy–Medium exercises + readiness checks in §12

---

## 2. Intuition

> 💡 Intuition
>
> Imagine driving on a hilly road. Your position is \(x\); your elevation is \(f(x)\). The derivative at your current location tells you how steep the road is **right now**. Steep uphill → large positive derivative. Steep downhill → large negative derivative. Flat spot → derivative near zero.

Think of zooming in on a smooth curve until it looks like a straight line. That straight line is the **tangent**. The derivative at a point is the **slope of that tangent**.

```
        f(x)
          |        *
          |      *       ← curve
          |    /
          |   /  ← tangent line (slope = derivative at this point)
          |  *
          +---------- x
```

**Constant slope (linear function):** If \(f(x) = 2x\), the road has the same steepness everywhere. The derivative is always 2.

**Changing slope (quadratic):** If \(f(x) = x^2\), the left side slopes downward (negative), the bottom is flat (zero), the right side slopes upward (positive). The derivative **depends on where you stand**.

**Why AI cares:** Training adjusts weights to reduce loss. The loss landscape is a high-dimensional hill system. Derivatives point downhill.

> 🔬 Deep Dive
>
> The formal definition uses a limit: the derivative is what the average rate of change approaches as your measurement interval shrinks to zero. You do not need limit notation to use derivatives in practice, but knowing they come from "zoom in until the curve looks linear" explains why derivatives work only where functions are smooth enough.

---

## 3. Formal Definitions

We introduce every symbol carefully.

### Average rate of change

Between two points \(x\) and \(x + h\):

\[
\text{average rate} = \frac{f(x + h) - f(x)}{h}
\]

| Symbol | Meaning |
|--------|---------|
| \(f(x)\) | Output at starting point |
| \(f(x+h)\) | Output after moving \(h\) units along the input axis |
| \(h\) | A small step in input (not height — unfortunate notation collision) |
| The fraction | "Rise over run" — slope of the secant line connecting two points |

### Derivative

The **derivative** of \(f\) at \(x\) is the **instantaneous** rate of change:

\[
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
\]

| Notation | Read as | Meaning |
|----------|---------|---------|
| \(f'(x)\) | "f prime of x" | Derivative of \(f\) with respect to \(x\) |
| \(\frac{df}{dx}\) | "dee f dee x" | Same thing — "change in output per tiny change in input" |
| \(\frac{d}{dx}[x^2]\) | — | Derivative of \(x^2\) with respect to \(x\) |

If \(f'(x) > 0\), increasing \(x\) increases \(f(x)\). If \(f'(x) < 0\), increasing \(x\) decreases \(f(x)\). If \(f'(x) = 0\), you may be at a flat spot, peak, or valley.

> **Plain English**
> The derivative is the slope of the curve at one point — how steep it is right there.

> **Python**
> `slope = (f(x + h) - f(x - h)) / (2 * h)`  *(numerical estimate)*

### Power rule

For any real exponent \(n\):

\[
\frac{d}{dx}[x^n] = n \cdot x^{n-1}
\]

**Examples:**

- \(\frac{d}{dx}[x^2] = 2x\)
- \(\frac{d}{dx}[x^3] = 3x^2\)
- \(\frac{d}{dx}[x] = 1\)
- \(\frac{d}{dx}[c] = 0\) for constant \(c\)

> **Plain English**
> Bring the exponent down as a multiplier, then reduce the exponent by one: \(x^3 \to 3x^2\).

> **Python**
> `# d/dx of x**3 → 3 * x**2`

### Linearity

\[
\frac{d}{dx}[a f(x) + b g(x)] = a f'(x) + b g'(x)
\]

Constants pull out: \(\frac{d}{dx}[5x^2] = 5 \cdot 2x = 10x\).

### Chain rule (preview)

If \(y = f(g(x))\), then:

\[
\frac{dy}{dx} = f'(g(x)) \cdot g'(x)
\]

The outer function's derivative (evaluated at the inner output) times the inner function's derivative. This is the backbone of backpropagation — you will use it constantly once networks have more than one layer.

> **Plain English**
> Differentiate the outer layer, then multiply by the derivative of the inner layer.

> **Python**
> `# dy/dx = outer_prime(inner(x)) * inner_prime(x)`

---

## 4. Programming Perspective

A derivative in math maps to "how much does this output change?" in code. PyTorch hides the details, but the idea starts with finite differences.

| Mathematics | Python |
|-------------|--------|
| \(f(x) = 2x\) | `def f(x): return 2 * x` |
| \(f'(x) = 2\) | constant — same for all \(x\) |
| \(\frac{f(x+h)-f(x)}{h}\) | `(f(x + h) - f(x)) / h` |

```python
def f(x):
    """f(x) = 2x — linear function."""
    return 2 * x


def numerical_derivative(f, x, h=1e-5):
    """Estimate f'(x) using central difference."""
    return (f(x + h) - f(x - h)) / (2 * h)


x_test = 3.0
print(f"f({x_test}) = {f(x_test)}")                    # 6.0
print(f"f'({x_test}) ≈ {numerical_derivative(f, x_test)}")  # ~2.0
```

For a linear function, the numerical estimate equals the true derivative exactly (up to floating-point noise).

```python
def g(x):
  """g(x) = x² — quadratic."""
  return x ** 2


# True derivative: g'(x) = 2x
for x in [0.0, 1.0, 2.0, 3.0]:
    approx = numerical_derivative(g, x)
    exact = 2 * x
    print(f"x={x}: approx={approx:.6f}, exact={exact}")
```

**Key insight:** `numerical_derivative` is a debugging tool. In production training, frameworks use **symbolic/automatic differentiation** — exact rules like the power rule applied mechanically to every operation. The result is the same concept: sensitivity.

---

## 5. Visualizations

Plots connect slope to the shape of a curve. Read every graph by asking what each axis means.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x ** 2

def f_prime(x):
    return 2 * x

x = np.linspace(-3, 3, 200)
y = f(x)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: function with tangent lines
axes[0].plot(x, y, color="steelblue", linewidth=2, label="f(x) = x²")
for x0 in [-2, 0, 2]:
    slope = f_prime(x0)
    tangent = slope * (x - x0) + f(x0)
    axes[0].plot(x, tangent, "--", linewidth=1.5, label=f"tangent at x={x0}, slope={slope}")
axes[0].set_xlabel("x (input)")
axes[0].set_ylabel("f(x) (output)")
axes[0].set_title("Function and tangent lines")
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].axvline(0, color="black", linewidth=0.5)
axes[0].legend(fontsize=8)

# Right: derivative function
axes[1].plot(x, f_prime(x), color="coral", linewidth=2, label="f'(x) = 2x")
axes[1].set_xlabel("x")
axes[1].set_ylabel("f'(x) (slope)")
axes[1].set_title("Derivative: slope at each x")
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].legend()

plt.tight_layout()
plt.show()
```

**How to read these plots:**

1. **Left panel:** At \(x = -2\), the tangent slopes downward (negative). At \(x = 0\), the tangent is horizontal (slope 0 — the minimum). At \(x = 2\), the tangent slopes upward (positive slope 4).

2. **Right panel:** The derivative function \(f'(x) = 2x\) crosses zero at the minimum. Negative on the left, positive on the right. Optimization moves from positive-derivative regions toward zero-derivative regions.

```python
# Linear function: derivative is constant
x = np.linspace(0, 10, 100)
plt.figure(figsize=(8, 4))
plt.plot(x, 2 * x, label="f(x) = 2x", linewidth=2)
plt.plot(x, np.full_like(x, 2), "--", label="f'(x) = 2", linewidth=2)
plt.xlabel("x")
plt.ylabel("value")
plt.title("Linear function has constant derivative")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

The dashed line stays at height 2: the rate of change never varies.

---

## 6. Worked Examples

### Example 1: Derivative of a linear function

Given \(f(x) = 3x - 1\), find \(f'(x)\).

**Step 1:** Apply the power rule term by term.

- \(\frac{d}{dx}[3x] = 3\)
- \(\frac{d}{dx}[-1] = 0\)

**Step 2:** Combine: \(f'(x) = 3\).

The slope is 3 everywhere. Increasing \(x\) by 0.1 increases \(f(x)\) by about 0.3.

### Example 2: Derivative of a quadratic

Given \(g(x) = x^2\), find \(g'(2)\).

**Step 1:** General derivative: \(g'(x) = 2x\).

**Step 2:** Evaluate at \(x = 2\): \(g'(2) = 2 \times 2 = 4\).

**Interpretation:** At \(x = 2\), the output is \(g(2) = 4\). A tiny increase in \(x\) increases \(g\) roughly 4× as much.

### Example 3: Power rule with coefficient

Given \(h(x) = 5x^3\), find \(h'(x)\).

**Step 1:** Pull out the constant: derivative of \(5 \cdot x^3\).

**Step 2:** Power rule on \(x^3\): \(3x^2\).

**Step 3:** \(h'(x) = 5 \cdot 3x^2 = 15x^2\).

Verify at \(x = 1\):

```python
def h(x):
    return 5 * x ** 3

print(numerical_derivative(h, 1.0))  # ≈ 15.0
```

### Example 4: Chain rule preview

Let \(y = (2x + 1)^2\). Inner function \(u = 2x + 1\); outer \(y = u^2\).

**Step 1:** \(\frac{dy}{du} = 2u\).

**Step 2:** \(\frac{du}{dx} = 2\).

**Step 3:** Chain rule: \(\frac{dy}{dx} = 2u \cdot 2 = 4(2x + 1)\).

At \(x = 1\): \(\frac{dy}{dx} = 4 \times 3 = 12\).

This is exactly what backprop does: multiply local derivatives along the path from output to input.

### Example 5: ReLU derivative (optional preview)

> 📌 Preview — optional for now
>
> **Term:** ReLU  
> **One line:** `max(0, x)` — activation used in neurons  
> **Learn properly in:** [Single Neuron](../03-neural-networks/01-single-neuron.md)  
> This example connects derivatives to a name you saw in Functions. Skip if overwhelming.

\[
\text{ReLU}(x) = \max(0, x)
\]

- For \(x > 0\): behaves like \(x\), derivative is 1.
- For \(x < 0\): output is constant 0, derivative is 0.
- At \(x = 0\): technically undefined; PyTorch picks 0 or 1 by convention.

```python
def relu(x):
    return max(0, x)

# Numerical derivative near 0 from the right vs left
print(numerical_derivative(relu, 0.01))   # ≈ 1
print(numerical_derivative(relu, -0.01))  # ≈ 0
```

Neurons on the "active" side (\(x > 0\)) pass gradients through; inactive neurons block them.

---

## 7. AI Connection

> 🧠 AI Insight
>
> A derivative measures **sensitivity**: nudge an input (or weight) slightly — how much does the output move? Training uses that idea to reduce error. You do not need the full training story yet.

**Core for this chapter:** if \(L(w)\) is loss and \(w\) is a weight, \(\frac{dL}{dw}\) tells you which direction changes \(w\) to increase or decrease loss. The sign tells you which way to move.

> 📌 Preview — optional for now
>
> **Term:** backpropagation / `loss.backward()`  
> **One line:** chain rule applied to all layers to get every gradient  
> **Learn properly in:** [Backpropagation](../03-neural-networks/03-backpropagation.md)  
> You can skip the diagram below until then.

```
input → layer1 → layer2 → ... → loss
         ↑        ↑              ↑
      ∂L/∂W1   ∂L/∂W2         starting point
```

> 📌 Preview — optional for now
>
> **Terms:** ReLU / sigmoid activation derivatives  
> **Learn properly in:** [Single Neuron](../03-neural-networks/01-single-neuron.md) and [Derivatives](02-derivatives.md) worked example 5 (optional)  
> Skip the activation derivative table until you have learned activations.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> **Confusing \(f(x)\) with \(f'(x)\).** \(f(x)\) is the function value (height on the curve). \(f'(x)\) is the slope at that point. At \(x = 3\) for \(f(x) = x^2\): \(f(3) = 9\) but \(f'(3) = 6\).

> ⚠️ Common Mistake
>
> **Thinking the derivative must be positive.** Negative derivative means the function is decreasing as \(x\) increases. Loss should decrease during training — you move **opposite** to the gradient of the loss (which points uphill).

> ⚠️ Common Mistake
>
> **Forgetting the chain rule in composed networks.** The derivative of the final layer alone is not enough. If \(L = f(g(h(x)))\), you need \(\frac{dL}{dx} = f' \cdot g' \cdot h'\). Missing a factor is the most common hand-derived backprop error.

> ⚠️ Common Mistake
>
> **Using too large \(h\) in numerical derivatives.** If \(h = 0.1\), you are measuring average slope over a wide interval, not the tangent. Use \(h \approx 10^{-5}\) for float64. Too small (\(10^{-20}\)) causes catastrophic cancellation — floating-point error dominates.

**Correct understanding:** The derivative measures local sensitivity. Training uses negative derivatives (gradients) of loss with respect to parameters. Composed functions require the chain rule — that is backpropagation.

---

## 9. Exercises

### Easy

1. Using the power rule, find \(\frac{d}{dx}[x^4]\).
2. If \(f(x) = 7x + 3\), what is \(f'(x)\)? Does it depend on \(x\)?
3. For \(g(x) = x^2\), compute \(g'(0)\), \(g'(1)\), and \(g'(-2)\) by hand.
4. Implement `numerical_derivative` and verify that the derivative of \(f(x) = 4x\) at \(x = 10\) is approximately 4.

### Medium

5. Find \(\frac{d}{dx}[3x^2 + 2x - 5]\) using linearity and the power rule.
6. Plot \(f(x) = x^3\) and \(f'(x) = 3x^2\) on separate subplots from \(x = -2\) to \(x = 2\). Where is \(f'(x) = 0\)?
7. Use the chain rule to differentiate \(y = (x^2 + 1)^3\) (hint: let \(u = x^2 + 1\)).
8. Explain in words why the derivative of a constant bias term is zero — and what that means for gradient updates to biases.

### Hard

9. For loss \(L(w) = (y - wx)^2\) with \(y = 5\), \(x = 2\), find \(\frac{dL}{dw}\) when \(w = 1\). Should you increase or decrease \(w\)?

### Challenge (optional — includes ML previews)

10. *(Optional)* The sigmoid is \(\sigma(x) = 1/(1 + e^{-x})\). Without deriving the full formula, explain why \(\sigma'(x)\) is largest near \(x = 0\) and near zero when \(|x|\) is large. Preview: [Single Neuron](../03-neural-networks/01-single-neuron.md)

11. *(Optional)* Plot the ReLU function and its derivative (use \(x = 0.001\) and \(x = -0.001\) to approximate the derivative at 0). Label regions where gradient is blocked. Preview: [Single Neuron](../03-neural-networks/01-single-neuron.md)

12. **Numerical vs analytical:** Implement \(f(x) = x^5 - 2x^2 + 1\). Compute \(f'(x)\) analytically and compare to `numerical_derivative` at 10 random points. Plot the error \(|f'_{\text{num}} - f'_{\text{exact}}|\) as a function of step size \(h\) on a log scale.

13. *(Optional)* **Tiny backprop by hand:** For \(L = (y - w_2 \cdot \text{ReLU}(w_1 x))^2\) with \(x=2\), \(y=3\), \(w_1=1\), \(w_2=0.5\), compute \(\partial L / \partial w_2\) and \(\partial L / \partial w_1\) using the chain rule. Verify with PyTorch autograd. Preview: [Backpropagation](../03-neural-networks/03-backpropagation.md)

---

## 10. Mini Project

### Derivative Visualizer

Build a script that:

1. Accepts a list of functions (at least: \(x^2\), \(x^3\), \(2x + 1\), \(|x|\)).
2. Plots each function and its analytical derivative side by side.
3. Draws tangent lines at three user-chosen \(x\) values.
4. Prints a table comparing numerical vs analytical derivative at those points.
5. Saves the figure to `book/assets/02-derivatives-visualizer.png`.

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_function_and_derivative(f, f_prime, name, x_points, x_range=(-3, 3)):
    x = np.linspace(*x_range, 300)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x, f(x), linewidth=2)
    for x0 in x_points:
        m = f_prime(x0)
        axes[0].plot(x, m * (x - x0) + f(x0), "--")
    axes[0].set_title(f"{name}")
    axes[1].plot(x, f_prime(x), linewidth=2, color="coral")
    axes[1].set_title(f"{name} derivative")
    return fig

# Example usage
f = lambda x: x ** 2
f_prime = lambda x: 2 * x
fig = plot_function_and_derivative(f, f_prime, "x²", [-1, 0, 2])
out = Path("book/assets/02-derivatives-visualizer.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
```

<details>
<summary>Mini project checklist</summary>

- [ ] At least three functions with analytical derivatives
- [ ] Tangent lines at three points per function
- [ ] Numerical vs analytical comparison table
- [ ] Figure saved to `book/assets/`

</details>

---

## 11. Interview Questions

**Q1:** What is a derivative, in plain language?

**A1:** The derivative of a function at a point measures how much the output changes when you make a tiny change to the input — the instantaneous rate of change. Geometrically, it is the slope of the tangent line at that point. In machine learning, derivatives of the loss with respect to weights tell us which direction to adjust each weight to reduce error.

**Q2:** Why is the chain rule essential for training neural networks?

**A2:** A neural network composes many functions: input → layer 1 → layer 2 → … → loss. The chain rule says the derivative of the whole composition is the product of derivatives along the path. Backpropagation implements this efficiently by reusing intermediate results. Without the chain rule, you could not compute how early-layer weights affect the final loss.

**Q3:** What happens to gradients when a ReLU neuron has negative input?

**A3:** ReLU outputs 0 for negative inputs, and its derivative is 0 in that region. During backpropagation, no gradient flows through that neuron to its weights or to earlier layers (via that path). The neuron is "dead" until its input becomes positive again. This is why initialization and learning rate matter — too many dead ReLUs stall learning.

**Q4:** How does a numerical derivative relate to automatic differentiation?

**A4:** Numerical differentiation approximates the derivative by evaluating \(f(x+h)\) and \(f(x-h)\). Automatic differentiation applies exact rules (power rule, chain rule) to each operation in the computation graph. Both compute the same mathematical object; autograd is faster, more accurate, and scales to millions of parameters.

**Q5:** For \(f(x) = x^2\), why is \(f'(0) = 0\), and what does that mean for optimization?

**A5:** At \(x = 0\), the tangent to the parabola is horizontal — slope zero. The function is at a minimum. Gradient descent would stop updating \(x\) if this were the only parameter and learning rate were well-tuned, because the derivative (gradient) is zero at the optimum. In higher dimensions, zero gradient indicates a critical point (minimum, maximum, or saddle).

---

## 12. Summary

### Key formulas

| Rule | Formula |
|------|---------|
| Definition | \(f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h}\) |
| Power rule | \(\frac{d}{dx}[x^n] = n x^{n-1}\) |
| Constant | \(\frac{d}{dx}[c] = 0\) |
| Linearity | \(\frac{d}{dx}[af + bg] = af' + bg'\) |
| Chain rule | \(\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)\) |
| Numerical | \(\frac{f(x+h)-f(x-h)}{2h}\) (central difference) |

### Key terminology

- **Derivative** — instantaneous rate of change of a function
- **Tangent line** — straight line matching the curve's direction at one point
- **Slope** — rise over run; equals the derivative for differentiable functions
- **Chain rule** — how to differentiate composed functions
- **Secant line** — line through two points; slope is average rate of change
- **Autograd** — automatic computation of derivatives in frameworks like PyTorch

### Readiness checks

Before the next chapter, you should be able to:

1. State the power rule and apply it to \(x^4\) and \(5x^3\).
2. Explain the difference between \(f(x)\) and \(f'(x)\) in plain language.
3. Implement `numerical_derivative` and verify it on \(f(x) = 2x\) at any point.
4. Use the chain rule to differentiate \((x^2 + 1)^3\) (or an equivalent composed function).
5. Read a plot of \(f'(x)\) and say where the original function is increasing, decreasing, or flat.

If any item is shaky, reread §3 and the [cheatsheet](02-derivatives-cheatsheet.md).

---

## 13. Preview

Derivatives measure sensitivity for **one input variable**. Real models have **millions** of parameters — millions of inputs to the loss function. Before we assemble those sensitivities into a **gradient**, we need the language for lists of numbers with direction: **vectors**.

The next chapter — **Vectors** — introduces magnitude, addition, dot products, and the connection to embeddings. After that, **Gradients** combines derivatives with vectors to power gradient descent and `loss.backward()`.

**Next chapter:** [Vectors](03-vectors.md)

---

## Lab

Companion notebook: [`app/math/02_derivatives.ipynb`](../../app/math/02_derivatives.ipynb)

## Review

- Cheatsheet: [Derivatives — Cheatsheet](02-derivatives-cheatsheet.md)
- Jargon: [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
