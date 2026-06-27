# Functions

## 1. Introduction

Before you can understand neural networks, transformers, or any machine learning model, you need a precise idea of what a **function** is. Not the vague memory of high-school algebra — a clear, programmer-friendly definition you can connect to Python code immediately.

A function is a rule that takes an **input** and produces exactly one **output**. In machine learning, almost everything is a function:

- A **loss function** takes model predictions and true labels, and outputs a single number measuring error.
- An **activation function** (like ReLU) takes a neuron's pre-activation value and outputs a transformed value.
- A **neural network** is a large composed function: raw pixels in, class probabilities out.

After this chapter you will be able to:

- Read and write mathematical function notation without fear.
- Plot linear, quadratic, and common nonlinear functions with Matplotlib.
- Explain how function shape (slope, curvature) affects learning behavior.
- Map any simple math function to an equivalent Python `def`.

**Where this appears in AI:** Linear regression is a linear function. Logistic regression adds a nonlinear function (sigmoid). Every layer in a neural network applies a function. Understanding functions is the first step toward understanding all of them.

---

## 2. Intuition

> 💡 Intuition
>
> Think of a function as a vending machine. You put something in (input), the machine applies a fixed rule (the function), and you get something out (output). Put in the same input twice — you always get the same output. That predictability is what makes functions useful in code and in math.

Here is a simple mental model:

```
  input x  ──►  [ function f ]  ──►  output y
     3      ──►       f        ──►     6        (if f doubles: f(x) = 2x)
```

**Linear functions** behave like a steady ramp: climb at a constant rate. Double your horizontal step, and the vertical change doubles too.

**Quadratic functions** behave like a bowl or a hill: the steepness changes depending on where you stand. Near the bottom of a bowl, the surface is nearly flat; farther out, it rises sharply.

**Composed functions** chain machines together: the output of one becomes the input of the next. A deep neural network is dozens or hundreds of functions composed in sequence.

```
  x ──► f ──► g ──► h ──► final output
```

> 🔬 Deep Dive
>
> In programming, you already use functions constantly: `len()`, `str.upper()`, your own `def` blocks. Mathematical functions are the same idea — a named rule from input to output — but we often write them as `f(x)` instead of `f(x)` as a function call. The notation is unfamiliar; the concept is not.

---

## 3. Formal Definitions

We introduce notation carefully. Every symbol is defined on first use.

### Function

A **function** `f` maps each input from a set of allowed inputs (the **domain**) to exactly one output in a set of possible outputs (the **codomain**).

We write:

\[
f(x) = \text{(some expression involving } x \text{)}
\]

Breaking this down:

| Symbol | Meaning |
|--------|---------|
| `f` | The name of the function (like a Python function name) |
| `x` | The **input variable** — a placeholder for any allowed value |
| `( )` | Parentheses mean "apply the function to" — `f(x)` means "f evaluated at x" |
| `=` | "equals" or "is defined as" — the output equals the expression on the right |

**Example:** \(f(x) = 2x\)

- `f` is the function name.
- `x` is the input (any real number).
- `f(x)` reads "f of x" — the output when you plug in `x`.
- The rule: multiply the input by 2.

If \(x = 3\), then \(f(3) = 2 \times 3 = 6\).

### Linear function

A **linear function** has the form:

\[
f(x) = kx + b
\]

- `k` is the **slope** — how much `y` changes per unit increase in `x`.
- `b` is the **y-intercept** — the output when \(x = 0\).

When \(b = 0\), we get the simpler form \(f(x) = kx\).

### Quadratic function

A **quadratic function** has the form:

\[
f(x) = ax^2 + bx + c
\]

The simplest case is \(f(x) = x^2\) (here \(a=1, b=0, c=0\)). The graph is a **parabola** — symmetric about the y-axis, U-shaped when \(a > 0\).

### Composition

If \(f(x) = 2x\) and \(g(x) = x + 1\), then the **composition** \(f(g(x))\) means: first apply \(g\), then apply \(f\) to the result.

\[
f(g(x)) = f(x + 1) = 2(x + 1) = 2x + 2
\]

---

## 4. Programming Perspective

Mathematics and Python express the same idea in different syntax.

| Mathematics | Python |
|-------------|--------|
| \(f(x) = 2x\) | `def f(x): return 2 * x` |
| \(f(x) = x^2\) | `def f(x): return x ** 2` |
| \(f(x) = 3x - 1\) | `def f(x): return 3 * x - 1` |
| \(f(g(x))\) | `f(g(x))` |

```python
def f(x):
    """Linear: f(x) = 2x"""
    return 2 * x


def g(x):
    """Quadratic: g(x) = x²"""
    return x ** 2


def h(x):
    """Composition: h(x) = f(g(x)) = 2 * x²"""
    return f(g(x))


# Evaluate at specific points
print(f(3))    # 6
print(g(-2))   # 4
print(h(2))    # 8  (2 * 2² = 8)
```

**Key insight:** `x` in math is just a parameter name. You could write `def f(t): return 2 * t` and it would be the same function. The name of the variable does not matter; the rule does.

```python
import numpy as np

# Functions work on arrays too — element-wise
x = np.array([0, 1, 2, 3])
print(f(x))  # [0 2 4 6]
print(g(x))  # [0 1 4 9]
```

NumPy applies the function to every element. This is exactly how you will evaluate loss over batches of data.

---

## 5. Visualizations

Plots turn abstract rules into shapes you can see. Always read a graph by asking: what does the horizontal axis represent? What does the vertical axis represent? What shape does the rule create?

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 200)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Linear: y = 2x
axes[0].plot(x, 2 * x, color="steelblue", linewidth=2)
axes[0].set_title("Linear: f(x) = 2x")
axes[0].set_xlabel("x (input)")
axes[0].set_ylabel("f(x) (output)")
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].axvline(0, color="black", linewidth=0.5)

# Quadratic: y = x²
axes[1].plot(x, x ** 2, color="coral", linewidth=2)
axes[1].set_title("Quadratic: f(x) = x²")
axes[1].set_xlabel("x (input)")
axes[1].set_ylabel("f(x) (output)")
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].axvline(0, color="black", linewidth=0.5)

# Cubic: y = x³
axes[2].plot(x, x ** 3, color="seagreen", linewidth=2)
axes[2].set_title("Cubic: f(x) = x³")
axes[2].set_xlabel("x (input)")
axes[2].set_ylabel("f(x) (output)")
axes[2].axhline(0, color="black", linewidth=0.5)
axes[2].axvline(0, color="black", linewidth=0.5)

plt.tight_layout()
plt.show()
```

**How to read these plots:**

1. **Linear (`f(x) = 2x`):** A straight line through the origin. Slope 2 means: move right 1 unit, move up 2 units. Constant rate of change everywhere.

2. **Quadratic (`f(x) = x²`):** A U-shaped parabola. Symmetric: `f(-3) = f(3) = 9`. Minimum at `x = 0`. The curve is flat near the bottom and steep at the edges — this changing steepness matters for optimization.

3. **Cubic (`f(x) = x³`):** Not symmetric like the parabola. Negative inputs give negative outputs. Steeper than quadratic for large `|x|`.

```python
# Compare multiple linear functions on one plot
x = np.linspace(0, 5, 100)
plt.figure(figsize=(8, 5))
plt.plot(x, 2 * x, label="y = 2x")
plt.plot(x, -x + 5, label="y = -x + 5")
plt.plot(x, 0.5 * x + 1, label="y = 0.5x + 1")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Different slopes and intercepts")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

The **slope** `k` controls steepness. The **intercept** `b` shifts the line up or down. In a neural network, weights play the role of slopes and biases play the role of intercepts.

---

## 6. Worked Examples

### Example 1: Evaluate a linear function

Given \(f(x) = 3x - 1\), find \(f(0)\), \(f(1)\), and \(f(4)\).

**Step 1:** Substitute each input into the expression.

- \(f(0) = 3(0) - 1 = 0 - 1 = -1\)
- \(f(1) = 3(1) - 1 = 3 - 1 = 2\)
- \(f(4) = 3(4) - 1 = 12 - 1 = 11\)

**Step 2:** Verify in Python.

```python
def f(x):
    return 3 * x - 1

for x in [0, 1, 4]:
    print(f"f({x}) = {f(x)}")
# f(0) = -1
# f(1) = 2
# f(4) = 11
```

### Example 2: Quadratic symmetry

Given \(g(x) = x^2\), compute \(g(-3)\) and \(g(3)\).

- \(g(-3) = (-3)^2 = 9\)
- \(g(3) = 3^2 = 9\)

Squaring removes the sign. The parabola is symmetric about the y-axis. This matters when you see even activation patterns or symmetric loss landscapes.

### Example 3: Function composition

Let \(f(x) = 2x\) and \(h(x) = x + 3\). Find \(f(h(2))\).

**Step 1:** Compute inner function: \(h(2) = 2 + 3 = 5\)

**Step 2:** Apply outer function: \(f(5) = 2 \times 5 = 10\)

**Step 3:** In Python:

```python
def f(x):
    return 2 * x

def h(x):
    return x + 3

print(f(h(2)))  # 10
```

### Example 4: Building a tiny "model"

Suppose a neuron computes \(y = \text{ReLU}(2x - 1)\) where ReLU outputs the input if positive, else 0.

```python
def relu(z):
    return max(0, z)

def neuron(x):
    z = 2 * x - 1  # linear part
    return relu(z)   # activation

for x in [-1, 0, 0.5, 1, 2]:
    print(f"x={x}, output={neuron(x)}")
# x=-1, output=0
# x=0,  output=0
# x=0.5, output=0
# x=1,  output=1
# x=2,  output=3
```

This is already a function composition: linear then nonlinear. Real networks stack many such compositions.

---

## 7. AI Connection

> 🧠 AI Insight
>
> A neural network is not magic — it is a very large function \(f_\theta(x)\) parameterized by weights \(\theta\). Training finds \(\theta\) so that outputs match labels.

**Linear regression** fits \(y = wx + b\). One linear function. The weight `w` is the slope; `b` is the intercept. Loss measures how far predictions are from truth.

**Logistic regression** applies a nonlinear function (sigmoid) to a linear output:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

The sigmoid squashes any real number into the range \((0, 1)\) — interpretable as probability.

**Neural network layer:**

\[
\text{output} = \sigma(Wx + b)
\]

- \(Wx + b\) is a linear function (matrix generalization of \(kx + b\)).
- \(\sigma\) is an activation function (ReLU, sigmoid, GELU).

**Loss functions** are functions too:

- Mean Squared Error: \(L = \frac{1}{n}\sum (y_i - \hat{y}_i)^2\)
- Cross-entropy: measures mismatch between predicted and true probability distributions

When you call `model(x)` in PyTorch, you are evaluating a composed function. When you call `loss.backward()`, you will eventually compute how that function changes with respect to each parameter — which requires derivatives (next chapter).

**Embeddings** map discrete token IDs to vectors, then layers apply functions to those vectors. **Attention** computes weighted sums using softmax — another function. Everything builds on the idea you learned here: input → rule → output.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> **"f(x) means f times x."** No. `f(x)` means "apply the function f to the value x." It is notation for function application, like `f(x)` in Python. `f * x` would mean multiplication.

> ⚠️ Common Mistake
>
> **Confusing the variable name with the function.** `f(x) = 2t` is fine if the input is named `t` instead of `x`. The function is the rule (multiply by 2), not the letter used.

> ⚠️ Common Mistake
>
> **Thinking all functions are linear.** Many beginners assume neural networks are linear because they see matrix multiplication. Linear layers are linear, but activation functions (ReLU, sigmoid) are not. Without nonlinearity, stacking layers would still be one linear function — unable to learn complex patterns.

> ⚠️ Common Mistake
>
> **Ignoring the domain.** \(f(x) = 1/x\) is undefined at \(x = 0\). In code, `1/0` raises an error. Always ask: which inputs are valid?

**Correct understanding:** A function is a deterministic rule. `f(x)` is notation for evaluation. Composition chains rules. Neural networks chain many rules with parameters learned from data.

---

## 9. Exercises

### Easy

1. If \(f(x) = 5x + 2\), compute \(f(0)\), \(f(1)\), and \(f(-1)\).
2. Write a Python function for \(f(x) = x^2 + 1\) and evaluate at \(x = 2\).
3. Which is steeper: \(y = 4x\) or \(y = -2x + 10\)? (Compare absolute slope.)
4. Plot \(y = x\) and \(y = -x\) on the same axes from \(-5\) to \(5\).

### Medium

5. Let \(f(x) = 2x\) and \(g(x) = x - 3\). Find \(f(g(5))\) by hand, then verify in Python.
6. Explain in words why \(f(x) = x^2\) is symmetric about the y-axis.
7. Implement `def compose(f, g):` that returns a new function computing \(f(g(x))\).
8. Plot \(y = x^2\), \(y = x^2 + 2\), and \(y = x^2 - 1\) on the same graph. What changed?

### Hard

9. A ReLU neuron computes \(\max(0, wx + b)\). For \(w = -2\), \(b = 3\), find all inputs \(x\) where the output is zero.
10. Show that composing two linear functions \(f(x) = a x + b\) and \(g(x) = c x + d\) always gives another linear function. What are the new slope and intercept?
11. The sigmoid \(\sigma(x) = 1/(1+e^{-x})\). Plot it from \(-10\) to \(10\). Why do values never reach exactly 0 or 1?

### Challenge

12. **Function Explorer:** Write a program that takes a list of coefficient pairs and plots \(y = kx + b\) for each. Add a slider or loop over at least 5 different lines and save the figure to `book/assets/functions_explorer.png`.
13. **MSE as a function:** For true value \(y = 3\) and prediction \(\hat{y} = wx\), the squared error is \(L(w) = (3 - w \cdot 1)^2\). Plot \(L(w)\) for \(w\) from \(-2\) to \(5\). Where is the minimum? Connect this to "learning the right weight."

---

## 10. Mini Project

### Function Explorer

Build a small Python script or notebook that:

1. Defines at least 5 functions: linear, quadratic, cubic, absolute value \(|x|\), and one of your choice.
2. Plots all five on separate subplots or overlaid with a legend.
3. Labels every axis and titles every subplot.
4. Prints a table of \((x, f(x))\) values for \(x \in \{-2, -1, 0, 1, 2\}\) for each function.
5. Saves the figure to `book/assets/01-functions-explorer.png`.

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

functions = {
    "linear": lambda x: 2 * x,
    "quadratic": lambda x: x ** 2,
    "cubic": lambda x: x ** 3,
    "abs": lambda x: np.abs(x),
    "shifted": lambda x: x ** 2 - 2 * x + 1,
}

x = np.linspace(-3, 3, 200)
fig, ax = plt.subplots(figsize=(10, 6))
for name, fn in functions.items():
    ax.plot(x, fn(x), label=name, linewidth=2)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Function Explorer")
ax.legend()
ax.grid(True, alpha=0.3)
out = Path("book/assets/01-functions-explorer.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
```

<details>
<summary>Mini project checklist</summary>

- [ ] Five distinct function shapes plotted
- [ ] Axes labeled, title present
- [ ] Value table printed for integer inputs
- [ ] Figure saved to `book/assets/`

</details>

---

## 11. Interview Questions

**Q1:** What is a function in mathematics, and how does it relate to a Python function?

**A1:** A function is a rule that assigns exactly one output to each allowed input. In math we write \(f(x) = \text{expression}\); in Python we write `def f(x): return expression`. Both are deterministic: same input always gives same output. Mathematical functions often operate on real numbers; Python functions can operate on any objects, but the core idea — named rule, input, output — is identical.

**Q2:** Why do neural networks need nonlinear activation functions?

**A2:** If you only stack linear functions, the result is still one linear function. Proof sketch: \(f(x) = W_2(W_1 x + b_1) + b_2\) simplifies to \(W'x + b'\) — one matrix multiply. Without nonlinearity, depth adds no expressive power. Nonlinear activations (ReLU, sigmoid) break this collapse, allowing networks to approximate complex patterns.

**Q3:** What is function composition, and where do you see it in deep learning?

**A3:** Composition means applying one function to the result of another: \(f(g(x))\). A neural network is \(f_L(\cdots f_2(f_1(x))\cdots)\). Each layer is one function. The full forward pass is one big composed evaluation. Backpropagation (later) computes how the final output changes with respect to each layer's parameters using the chain rule — which is the calculus of composition.

**Q4:** Explain the difference between slope and y-intercept in \(f(x) = kx + b\).

**A4:** Slope `k` is the rate of change: increase \(x\) by 1, and \(f(x)\) increases by \(k\). Intercept `b` is the value when \(x = 0\): where the graph crosses the y-axis. In `nn.Linear`, learned weights act like slopes (per input feature) and bias acts like the intercept.

**Q5:** How does plotting \(f(x) = x^2\) help you understand optimization?

**A5:** The parabola has a clear minimum at \(x = 0\). Optimization (gradient descent) tries to find such minima in high dimensions. Visualizing \(x^2\) builds intuition: the bowl shape, flat near the bottom, steep away from it. Loss landscapes for real models are vastly more complex, but the idea — move toward valleys — starts here.

---

## 12. Summary

### Key formulas

| Function type | Formula | Python |
|---------------|---------|--------|
| Linear | \(f(x) = kx + b\) | `k * x + b` |
| Quadratic | \(f(x) = x^2\) | `x ** 2` |
| Composition | \(f(g(x))\) | `f(g(x))` |
| ReLU | \(\max(0, x)\) | `max(0, x)` |

### Key terminology

- **Function** — rule mapping each input to exactly one output
- **Domain** — set of allowed inputs
- **Slope** — rate of change in a linear function
- **Intercept** — output when input is zero
- **Composition** — chaining functions: output of one feeds the next
- **Parabola** — graph of a quadratic function
- **Activation function** — nonlinear function applied in neural networks

---

## 13. Preview

Functions tell you **what** happens to inputs. The next chapter — **Derivatives** — tells you **how sensitive** the output is to small changes in the input. That sensitivity is the engine of gradient descent and backpropagation.

When you see \(f(x) = x^2\), the derivative tells you the slope at any point: steep on the sides, flat at the bottom. When you see a loss function \(L(w)\), the derivative tells you which direction to adjust weights to reduce error. Functions give you the landscape; derivatives give you the directions.

**Next chapter:** [Derivatives](02-derivatives.md)

---

## Lab

Companion notebook: [`app/math/01_functions.ipynb`](../../app/math/01_functions.ipynb)
