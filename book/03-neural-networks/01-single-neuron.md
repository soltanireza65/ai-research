# Single Neuron

## 1. Introduction

> **Payoff chapter:** In [Functions](../01-math/01-functions.md) and [Derivatives](../01-math/02-derivatives.md), **ReLU** and **sigmoid** were previews. **Here you learn them properly.**

Every deep learning model — from image classifiers to large language models — is built from the same atomic unit: the **neuron**.

A single neuron takes several numeric inputs, multiplies each input by a **weight**, adds them together with a **bias**, and optionally applies a **nonlinear activation function**. That is the entire story at the smallest scale. A billion-parameter model is millions of these operations arranged in layers.

After this chapter you will be able to:

- Compute the **pre-activation** \(z\) and **output** \(y\) of one neuron by hand.
- Implement the same computation in PyTorch with tensors.
- Explain why **ReLU** is used and what happens when pre-activation is negative.
- Connect the neuron formula to linear regression and to the first layer of any neural network.

**Where this appears in AI:** Logistic regression is one neuron with a sigmoid activation. A fully connected layer is many neurons running in parallel. Every weight in a transformer MLP block started as this same weighted-sum pattern.

**Suggested pacing (3 sessions):**

- Session A: §1–§3 + [cheatsheet](01-single-neuron-cheatsheet.md) skim
- Session B: §4–§6 + lab notebook
- Session C: Easy–Medium exercises + readiness checks in §12

---

## 2. Intuition

> 💡 Intuition
>
> Think of a neuron as a voting booth. Each input feature casts a vote (the input value), and each vote is weighted by how important that feature is (the weight). The bias shifts the threshold — like requiring a minimum number of votes before the booth lights up. The activation function decides whether the booth "fires" or stays silent.

```
  x₁ ──w₁──┐
  x₂ ──w₂──┼──►  Σ + b  ──►  activation  ──►  y
  x₃ ──w₃──┘
     inputs    weighted sum      ReLU         output
```

**Inputs** \(x_1, x_2, \ldots\) are numbers — pixel brightness, word embedding dimensions, sensor readings. **Weights** \(w_1, w_2, \ldots\) are learned parameters: the model adjusts them during training so the neuron responds to useful patterns. **Bias** \(b\) is one extra learnable number that shifts the decision boundary without changing any input.

The **pre-activation** \(z = w_1 x_1 + w_2 x_2 + \cdots + b\) is the raw score before any nonlinearity. The **activation** (here ReLU) transforms \(z\) into the final output \(y\).

> 🔬 Deep Dive
>
> Why multiply inputs by weights instead of adding fixed numbers? Multiplication lets each feature **scale** the output: a large positive weight means "when this feature is on, strongly increase the score." A negative weight means "when this feature is on, decrease the score." Addition alone could only shift the total; multiplication gives **direction and strength** per feature — exactly what you need to learn patterns.

---

## 3. Formal Definitions

We define every symbol on first use.

### Input vector

An **input vector** \(\mathbf{x}\) is an ordered list of numbers:

\[
\mathbf{x} = [x_1, x_2, \ldots, x_n]
\]

Here \(n\) is the number of **input features**. In code, this is a one-dimensional tensor of length \(n\).

### Weight vector

A **weight vector** \(\mathbf{w}\) has the same length as \(\mathbf{x}\):

\[
\mathbf{w} = [w_1, w_2, \ldots, w_n]
\]

Each \(w_i\) scales the corresponding input \(x_i\).

### Bias

The **bias** \(b\) is a single scalar (one number). It is added after the weighted sum.

### Pre-activation

The **pre-activation** \(z\) is the weighted sum plus bias:

\[
z = \mathbf{w} \cdot \mathbf{x} + b = \sum_{i=1}^{n} w_i x_i + b
\]

The dot \(\cdot\) means **dot product**: multiply corresponding entries and add the results.

### Activation function

An **activation function** \(\sigma\) maps the pre-activation to the neuron output:

\[
y = \sigma(z)
\]

**ReLU** (Rectified Linear Unit) is defined as:

\[
\text{ReLU}(z) = \max(0, z)
\]

If \(z > 0\), output equals \(z\). If \(z \leq 0\), output is 0.

| Symbol | Meaning |
|--------|---------|
| \(\mathbf{x}\) | Input vector |
| \(\mathbf{w}\) | Weight vector (learned) |
| \(b\) | Bias (learned) |
| \(z\) | Pre-activation (before activation) |
| \(y\) | Neuron output (after activation) |
| \(\sigma\) | Activation function |

---

## 4. Programming Perspective

The math maps directly to PyTorch. Element-wise multiply, sum, add bias, apply ReLU.

| Mathematics | PyTorch |
|-------------|---------|
| \(\mathbf{x}\) | `x = torch.tensor([...])` |
| \(\mathbf{w}\) | `w = torch.tensor([...])` |
| \(z = \sum w_i x_i + b\) | `(x * w).sum() + b` |
| \(y = \text{ReLU}(z)\) | `torch.maximum(z, torch.tensor(0.0))` |

```python
import torch

def relu(x):
  """ReLU: return x if positive, else 0."""
  return torch.maximum(x, torch.tensor(0.0))

x = torch.tensor([3.0, 4.0])
w = torch.tensor([2.0, -1.0])
b = torch.tensor(0.5)

z = (x * w).sum() + b
y = relu(z)
print("pre-activation:", z.item())
print("output:", y.item())
```

**Reading the output:** With \(x = [3, 4]\), \(w = [2, -1]\), \(b = 0.5\):

- \(z = 3 \times 2 + 4 \times (-1) + 0.5 = 6 - 4 + 0.5 = 2.5\)
- \(y = \text{ReLU}(2.5) = 2.5\) (positive, so unchanged)

```python
# Same computation with explicit steps
products = x * w          # [6.0, -4.0]
weighted_sum = products.sum()  # 2.0
z = weighted_sum + b      # 2.5
y = relu(z)
assert torch.isclose(z, torch.tensor(2.5))
assert torch.isclose(y, torch.tensor(2.5))
```

```python
# What happens when z is negative?
x_neg = torch.tensor([1.0, 5.0])
w_neg = torch.tensor([-2.0, -1.0])
b_neg = torch.tensor(0.0)
z_neg = (x_neg * w_neg).sum() + b_neg  # -2 - 5 = -7
y_neg = relu(z_neg)
print("z:", z_neg.item(), "y:", y_neg.item())  # z: -7.0  y: 0.0
```

ReLU **kills** negative pre-activations. That introduces nonlinearity — essential for deep networks.

```python
import torch.nn as nn

# PyTorch bundles weights and bias in nn.Linear (one neuron: out_features=1)
layer = nn.Linear(in_features=2, out_features=1, bias=True)
layer.weight.data = torch.tensor([[2.0, -1.0]])
layer.bias.data = torch.tensor([0.5])
z_layer = layer(x)
y_layer = relu(z_layer)
print("nn.Linear z:", z_layer.item())
```

`nn.Linear` stores \(\mathbf{w}\) as a row in `weight` and \(b\) in `bias`. This is how production models represent neurons.

---

## 5. Visualizations

A single neuron with one input reduces to a line: \(z = w_1 x_1 + b\). Plotting \(z\) versus \(x_1\) shows how weight and bias shape the response.

```python
import numpy as np
import matplotlib.pyplot as plt

x1 = np.linspace(-5, 5, 200)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# z = 2*x + 1
axes[0].plot(x1, 2 * x1 + 1, color="steelblue", linewidth=2)
axes[0].set_title("z = 2x + 1")
axes[0].set_xlabel("x₁ (input)")
axes[0].set_ylabel("z (pre-activation)")
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].axvline(0, color="black", linewidth=0.5)

# ReLU applied
axes[1].plot(x1, np.maximum(0, 2 * x1 + 1), color="coral", linewidth=2)
axes[1].set_title("y = ReLU(2x + 1)")
axes[1].set_xlabel("x₁ (input)")
axes[1].set_ylabel("y (output)")

# Different weight: z = -x + 3
axes[2].plot(x1, np.maximum(0, -x1 + 3), color="seagreen", linewidth=2)
axes[2].set_title("y = ReLU(-x + 3)")
axes[2].set_xlabel("x₁ (input)")
axes[2].set_ylabel("y (output)")

plt.tight_layout()
plt.show()
```

**How to read these plots:**

1. **Left:** Linear pre-activation. Slope 2 means doubling the input more than doubles \(z\). Intercept 1 shifts the line up.
2. **Middle:** ReLU clips everything below zero. The neuron is **silent** for \(x_1 < -0.5\) and **active** above that.
3. **Right:** Negative weight flips the slope. The neuron fires for small inputs and shuts off for large ones.

```python
# Two-input neuron: visualize z as a plane (no ReLU)
x1_grid, x2_grid = np.meshgrid(
    np.linspace(-3, 3, 50),
    np.linspace(-3, 3, 50),
)
w1, w2, b_val = 2.0, -1.0, 0.5
z_grid = w1 * x1_grid + w2 * x2_grid + b_val

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x1_grid, x2_grid, z_grid, cmap="viridis", alpha=0.8)
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.set_zlabel("z")
ax.set_title("Pre-activation plane: z = 2x₁ - x₂ + 0.5")
plt.show()
```

The horizontal axes are the two inputs; the vertical axis is \(z\). Training slides and tilts this plane until predictions match labels.

---

## 6. Worked Examples

### Example 1: Hand calculation

Given \(\mathbf{x} = [3, 4]\), \(\mathbf{w} = [2, -1]\), \(b = 0.5\), compute \(z\) and \(y = \text{ReLU}(z)\).

**Step 1:** Multiply element-wise: \(3 \times 2 = 6\), \(4 \times (-1) = -4\).

**Step 2:** Sum: \(6 + (-4) = 2\).

**Step 3:** Add bias: \(z = 2 + 0.5 = 2.5\).

**Step 4:** Apply ReLU: \(y = \max(0, 2.5) = 2.5\).

### Example 2: Neuron stays off

Given \(\mathbf{x} = [1, 2]\), \(\mathbf{w} = [-3, -1]\), \(b = 1\).

- \(z = 1 \times (-3) + 2 \times (-1) + 1 = -3 - 2 + 1 = -4\)
- \(y = \text{ReLU}(-4) = 0\)

The neuron produces zero output. During backpropagation, ReLU blocks gradient flow through this path when \(z \leq 0\).

### Example 3: Bias-only shift

If \(\mathbf{x} = [0, 0]\) and \(b = 3\), then \(z = 3\) regardless of weights. Bias lets the neuron fire even when all inputs are zero — like a default threshold.

### Example 4: Changing weights in code

```python
import torch

def neuron_output(x, w, b):
    z = (x * w).sum() + b
    return torch.maximum(z, torch.tensor(0.0)), z

x = torch.tensor([3.0, 4.0])
for w in [torch.tensor([2.0, -1.0]), torch.tensor([0.5, 0.5]), torch.tensor([-1.0, 2.0])]:
    y, z = neuron_output(x, w, torch.tensor(0.5))
    print(f"w={w.tolist()} -> z={z.item():.2f}, y={y.item():.2f}")
```

Different weights route attention to different features. Training discovers which routing helps the loss.

---

## 7. AI Connection

> 🧠 AI Insight
>
> Logistic regression is a single neuron with sigmoid activation: \(y = \sigma(wx + b)\). Binary classification is literally one neuron. Your first hidden layer in ResNet or GPT is thousands of these neurons in parallel — same formula, larger tensors.

**Linear regression** omits the activation: \(y = wx + b\). It is a neuron with identity activation. Useful for predicting continuous values (price, temperature).

**Binary classifier:** One neuron + sigmoid outputs a probability in \((0, 1)\).

**Hidden layers:** Stack many neurons. Each learns a different weighted combination of inputs. ReLU keeps computation sparse (many zeros) and avoids vanishing gradients that plagued sigmoid-heavy deep networks.

**Connection to transformers:** An MLP block inside a transformer layer has two linear transformations with ReLU or GELU between them — fat layers of neurons applied to every token position.

**Training:** Weights \(\mathbf{w}\) and bias \(b\) start random. A **loss function** measures error. **Gradient descent** nudges weights to reduce loss. The next chapter scales this to full layers; the backpropagation chapter explains how gradients reach each weight.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> **Forgetting that weights and inputs must match in length.** If \(\mathbf{x}\) has 4 elements, \(\mathbf{w}\) must have 4 elements. A shape mismatch in PyTorch raises a runtime error. Always check `x.shape` and `w.shape`.

> ⚠️ Common Mistake
>
> **Confusing pre-activation \(z\) with output \(y\).** Papers and code often use `z` for the value before activation and `a` or `y` for after. When debugging, print both. A negative \(z\) with ReLU always gives \(y = 0\).

> ⚠️ Common Mistake
>
> **Thinking ReLU is optional.** Without a nonlinear activation, stacking multiple weighted sums collapses to a single weighted sum. Depth would add no expressive power. ReLU is what makes depth meaningful.

**Correct understanding:** One neuron = dot product + bias + activation. The dot product is linear; the activation is not. Together they form the building block of modern deep learning.

---

## 9. Exercises

### Easy

1. Compute \(z\) and \(\text{ReLU}(z)\) for \(\mathbf{x} = [1, 2]\), \(\mathbf{w} = [1, 1]\), \(b = -2\).
2. Run the lab notebook with \(\mathbf{w} = [1, 0]\). What is the output? Which input mattered?
3. If \(b = 0\) and all inputs are zero, what is \(z\)? What is \(y\) with ReLU?

### Medium

4. Implement `def sigmoid(z):` and compare outputs to ReLU for \(z \in \{-2, 0, 2\}\).
5. Plot \(y = \text{ReLU}(wx + b)\) for \(w \in \{1, -1\}\) and \(b = 0\) on the same axes.
6. Why does a negative weight invert the effect of an input?

### Hard

7. For \(\mathbf{x} = [x_1, x_2]\), find all points where \(z = 2x_1 - x_2 + 0.5 = 0\) (the ReLU boundary).
8. Show that two neurons with ReLU can represent a function one neuron cannot (hint: consider a "bump" shape).
9. Compute by hand what happens to \(z\) if you double every weight and halve every input.

### Challenge

10. **Neuron Explorer:** Write a script that takes random \(\mathbf{w}\) and \(b\), plots the ReLU output for a 1-D input \(x_1 \in [-5, 5]\), and saves the figure. Sweep at least 5 weight pairs and annotate which inputs activate the neuron.

---

## 10. Mini Project

### Single-Neuron Decision Boundary

Build a tiny 2-D classifier using one neuron:

1. Create synthetic data: class A near \((1, 1)\), class B near \((-1, -1)\).
2. Initialize random \(\mathbf{w}\) and \(b\).
3. Compute \(y = \text{ReLU}(\mathbf{w} \cdot \mathbf{x} + b)\) for each point.
4. Plot points colored by class and draw the line where \(\mathbf{w} \cdot \mathbf{x} + b = 0\).
5. Manually adjust \(\mathbf{w}\) and \(b\) until most class-A points have \(z > 0\) and class-B points have \(z < 0\).

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
class_a = rng.normal(loc=[1, 1], scale=0.3, size=(20, 2))
class_b = rng.normal(loc=[-1, -1], scale=0.3, size=(20, 2))

w = np.array([1.0, 1.0])
b = 0.0

def z_fn(points, w, b):
    return points @ w + b

plt.figure(figsize=(7, 6))
plt.scatter(class_a[:, 0], class_a[:, 1], label="class A", alpha=0.7)
plt.scatter(class_b[:, 0], class_b[:, 1], label="class B", alpha=0.7)
x_line = np.linspace(-2, 2, 100)
plt.plot(x_line, -(w[0] * x_line + b) / w[1], "k--", label="z = 0 boundary")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.legend()
plt.title("One-neuron decision boundary")
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.show()
```

<details>
<summary>Mini project checklist</summary>

- [ ] Synthetic 2-D data plotted
- [ ] Decision boundary line drawn
- [ ] Weights adjusted to improve separation
- [ ] Written explanation of what the boundary means

</details>

---

## 11. Interview Questions

**Q1:** What does a single neuron compute before and after activation?

**A1:** Before activation, it computes the pre-activation \(z = \mathbf{w} \cdot \mathbf{x} + b\): a weighted sum of inputs plus bias. After activation, \(y = \sigma(z)\). With ReLU, negative values become zero; positive values pass through unchanged. The pre-activation is linear in the inputs; the activation introduces nonlinearity.

**Q2:** Why do we need a bias term if we already have weights?

**A2:** Weights scale inputs relative to zero. Bias shifts the activation threshold independently of inputs. Without bias, any decision boundary in 2-D must pass through the origin. Bias lets the boundary move anywhere — critical for fitting real data where classes are not centered at zero.

**Q3:** What is the difference between ReLU and sigmoid for a single neuron?

**A3:** Sigmoid squashes \(z\) into \((0, 1)\), interpretable as probability. ReLU outputs zero for negative \(z\) and \(z\) itself for positive \(z\). ReLU is cheaper to compute, sparsifies activations, and avoids saturating gradients for positive inputs. Sigmoid is standard for output neurons in binary classification; ReLU dominates hidden layers.

**Q4:** How does a neuron relate to dot product?

**A4:** The pre-activation is exactly the dot product of \(\mathbf{w}\) and \(\mathbf{x}\), plus bias. Geometrically, the dot product measures alignment: large positive when vectors point the same direction, negative when opposed. A neuron fires (positive \(z\)) when the input aligns with the weight vector.

---

## 12. Summary

### Key formulas

| Concept | Formula |
|---------|---------|
| Pre-activation | \(z = \sum_i w_i x_i + b = \mathbf{w} \cdot \mathbf{x} + b\) |
| ReLU | \(y = \max(0, z)\) |
| Neuron output | \(y = \text{ReLU}(\mathbf{w} \cdot \mathbf{x} + b)\) |

### Key terminology

- **Neuron** — computes weighted sum, bias, and activation
- **Weight** — learned scalar multiplying one input
- **Bias** — learned scalar added to the weighted sum
- **Pre-activation** — value \(z\) before nonlinear activation
- **ReLU** — activation that passes positive values, zeros negatives
- **Dot product** — sum of element-wise products of two vectors

### Readiness checks

Before **Building a Layer**, you should be able to:

1. Compute \(z = \mathbf{w} \cdot \mathbf{x} + b\) by hand for a 2-input neuron.
2. Implement the same computation in PyTorch with tensors.
3. Explain what ReLU does to negative pre-activations.
4. Describe the difference between pre-activation \(z\) and output \(y\).
5. Skim the [cheatsheet](01-single-neuron-cheatsheet.md) without panic at unfamiliar names.

If any item is shaky, reread §6 Worked Examples and rerun the lab.

---

## 13. Preview

One neuron handles one output number from several inputs. Real models need **many outputs at once** — one per hidden unit, per class, per token feature. The next chapter, **Building a Layer**, replaces the dot product with **matrix multiplication** so a single operation computes hundreds of neurons in parallel across an entire batch of examples.

You already know the per-neuron math. The layer is that same math, vectorized.

**Next chapter:** [Building a Layer](02-building-a-layer.md)

---

## Lab

Companion notebook: [`app/neural_networks/01_single_neuron.ipynb`](../../app/neural_networks/01_single_neuron.ipynb)

## Review

- Cheatsheet: [Single Neuron — Cheatsheet](01-single-neuron-cheatsheet.md)
- Jargon: [Vocabulary Roadmap](../../00-intro/04-vocabulary-roadmap.md)
