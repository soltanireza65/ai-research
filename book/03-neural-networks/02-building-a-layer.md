# Building a Layer

## 1. Introduction

A single neuron produces one number. A neural network layer produces **many numbers at once** — one per neuron — for **every example in a batch**. The operation that makes this possible is **matrix multiplication**, the same `@` operator you practiced in the PyTorch fundamentals chapters.

A **fully connected layer** (also called a **dense** or **linear** layer) computes:

\[
\mathbf{Y} = \mathbf{X} \mathbf{W}^\top + \mathbf{b}
\]

where \(\mathbf{X}\) holds a batch of input vectors, \(\mathbf{W}\) holds all neuron weights, and \(\mathbf{b}\) holds one bias per neuron. One matrix multiply replaces thousands of individual dot products.

After this chapter you will be able to:

- Explain the shapes of \(\mathbf{X}\), \(\mathbf{W}\), \(\mathbf{b}\), and \(\mathbf{Y}\) in a linear layer.
- Implement `y = x @ W.T + b` in PyTorch and verify dimensions.
- Apply ReLU element-wise to layer outputs.
- Read `nn.Linear(in_features, out_features)` and know what each argument means.

**Where this appears in AI:** Every `nn.Linear` in PyTorch — from MLP blocks in transformers to classification heads in vision models — is this operation. Understanding layers is understanding 90% of forward passes in deep learning.

---

## 2. Intuition

> 💡 Intuition
>
> Imagine a call center with 3 agents (output neurons) and 4 phone lines (input features). Every customer call (batch row) brings 4 pieces of information. Each agent has their own checklist of how much each piece matters (one row of \(\mathbf{W}\)). All agents read the same call simultaneously, but each produces a different score. That parallel scoring is one layer forward pass.

```
  batch of inputs          weight matrix           output
  ┌─────────┐            ┌─────────┐            ┌─────────┐
  │ x₁ x₂ x₃│            │ row 0   │──neuron 0─►│   y₀    │
  │  ...    │  @  W.T  + │ row 1   │──neuron 1─►│   y₁    │
  │ x₁ x₂ x₃│            │ row 2   │──neuron 2─►│   y₂    │
  └─────────┘            └─────────┘            └─────────┘
     (B, 4)                 (3, 4)                 (B, 3)
```

**B** is **batch size** — how many examples processed together. **in_features** is how many numbers each example has. **out_features** is how many neurons (outputs) the layer has.

> 🔬 Deep Dive
>
> Why \(\mathbf{W}^\top\) (transpose)? PyTorch stores `nn.Linear.weight` with shape `(out_features, in_features)`. Each **row** is one neuron's weights. Input `x` has shape `(batch, in_features)`. The product `x @ W.T` computes all dot products in one shot: row \(i\) of the output is `x` dotted with row \(i\) of \(\mathbf{W}\).

There is a second intuition hiding in the shape notation. A layer does not merely make the model "bigger"; it chooses a new coordinate system for the data. If the input row contains four measurements, the three output neurons are three learned questions about those measurements. One neuron might respond to "large feature 0 and small feature 2." Another might respond to "feature 1 and feature 3 move together." The layer turns raw coordinates into learned features that later layers can combine.

Batching is also more than a performance trick. A batch lets the same layer rule be applied to many examples with one matrix operation. The weights are shared across the entire batch, which means the model is learning one reusable transformation, not a separate transformation per example. This is why a layer trained on one mini-batch can generalize to later mini-batches: every row is processed by the same \(\mathbf{W}\) and \(\mathbf{b}\).

---

## 3. Formal Definitions

### Input batch

\[
\mathbf{X} \in \mathbb{R}^{B \times n_{in}}
\]

- \(B\) = batch size (number of examples)
- \(n_{in}\) = `in_features` (length of each input vector)
- Row \(b\) of \(\mathbf{X}\) is the input for example \(b\)

### Weight matrix

\[
\mathbf{W} \in \mathbb{R}^{n_{out} \times n_{in}}
\]

- \(n_{out}\) = `out_features` (number of neurons)
- Row \(j\) of \(\mathbf{W}\) is the weight vector for neuron \(j\)

### Bias vector

\[
\mathbf{b} \in \mathbb{R}^{n_{out}}
\]

One bias per neuron. Added to each row of the output after multiplication.

### Layer output (pre-activation)

\[
\mathbf{Z} = \mathbf{X} \mathbf{W}^\top + \mathbf{b}
\]

\[
\mathbf{Z} \in \mathbb{R}^{B \times n_{out}}
\]

Element \(Z_{b,j}\) is the pre-activation of neuron \(j\) on example \(b\):

\[
Z_{b,j} = \sum_{i=1}^{n_{in}} X_{b,i} \, W_{j,i} + b_j
\]

### Activation

Apply a nonlinear function element-wise:

\[
\mathbf{Y} = \sigma(\mathbf{Z})
\]

With ReLU: \(Y_{b,j} = \max(0, Z_{b,j})\).

| Symbol | Shape | Meaning |
|--------|-------|---------|
| \(\mathbf{X}\) | \((B, n_{in})\) | Batch of inputs |
| \(\mathbf{W}\) | \((n_{out}, n_{in})\) | All neuron weights |
| \(\mathbf{b}\) | \((n_{out},)\) | All neuron biases |
| \(\mathbf{Z}\) | \((B, n_{out})\) | Pre-activations |
| \(\mathbf{Y}\) | \((B, n_{out})\) | Activations |

---

## 4. Programming Perspective

```python
import torch

in_features, out_features = 4, 3
batch = 2

x = torch.randn(batch, in_features)
W = torch.randn(out_features, in_features)
b = torch.randn(out_features)

y = x @ W.T + b
print("input:", x.shape)    # torch.Size([2, 4])
print("output:", y.shape)   # torch.Size([2, 3])
print(y)
```

**Shape check:** `(B, n_in) @ (n_in, n_out) → (B, n_out)`. Transposing \(\mathbf{W}\) converts `(n_out, n_in)` into `(n_in, n_out)` for valid multiplication.

When debugging layer code, read shapes from left to right. The batch dimension should usually pass through unchanged: two examples in, two examples out. The feature dimension changes from `in_features` to `out_features` because the layer is replacing the old feature representation with a new learned representation. If you see the batch dimension disappear, you probably reduced over the wrong axis. If you see `RuntimeError: mat1 and mat2 shapes cannot be multiplied`, compare the middle dimensions: `(B, n_in)` can multiply `(n_in, n_out)`, but not `(n_out, n_in)`.

This is the same habit you will use in transformer code. A transformer MLP block applies linear layers independently at every token position. The tensor may have shape `(batch, sequence, d_model)`, but the final dimension still plays the role of `in_features`. The idea from this chapter survives; only the number of leading dimensions grows.

```python
# Apply ReLU to the entire layer output at once
y_relu = torch.maximum(y, torch.tensor(0.0))
# or: y_relu = torch.relu(y)
print("after ReLU:", y_relu.shape)  # same shape (2, 3)
```

```python
import torch.nn as nn

layer = nn.Linear(in_features=4, out_features=3)
x = torch.randn(2, 4)
z = layer(x)           # includes bias automatically
y = torch.relu(z)
print("nn.Linear weight shape:", layer.weight.shape)  # (3, 4)
print("nn.Linear bias shape:", layer.bias.shape)      # (3,)
```

```python
# Manual vs nn.Linear — should match
layer = nn.Linear(4, 3)
x = torch.randn(2, 4)
manual = x @ layer.weight.T + layer.bias
auto = layer(x)
print(torch.allclose(manual, auto))  # True
```

```python
# Stacking two layers = composition
layer1 = nn.Linear(4, 8)
layer2 = nn.Linear(8, 3)
x = torch.randn(2, 4)
h = torch.relu(layer1(x))   # hidden layer
out = layer2(h)             # output layer
print("hidden:", h.shape, "output:", out.shape)
```

---

## 5. Visualizations

For one neuron in a layer, the story is the same as the single-neuron chapter. For a layer, visualize how different rows of \(\mathbf{W}\) create different response patterns.

```python
import numpy as np
import matplotlib.pyplot as plt

# One input feature, three neurons = three different lines
x1 = np.linspace(-3, 3, 200)
W = np.array([[2.0], [-1.0], [0.5]])  # (3, 1)
b = np.array([0.0, 1.0, -0.5])

fig, ax = plt.subplots(figsize=(8, 5))
for j in range(3):
    z = W[j, 0] * x1 + b[j]
    y = np.maximum(0, z)
    ax.plot(x1, y, label=f"neuron {j}", linewidth=2)
ax.set_xlabel("x₁ (input)")
ax.set_ylabel("y (ReLU output)")
ax.set_title("Three neurons, one input — different weights and biases")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

Each neuron is a different line (then clipped by ReLU). The layer outputs a 3-vector at every input point.

```python
# Heatmap: layer output across a 2-D input grid (3 neurons)
x1 = np.linspace(-2, 2, 40)
x2 = np.linspace(-2, 2, 40)
g1, g2 = np.meshgrid(x1, x2)
X_grid = np.stack([g1.ravel(), g2.ravel()], axis=1)  # (1600, 2)

W = np.array([[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]])
b = np.array([0.0, 0.0, 0.0])
Z = X_grid @ W.T + b
Y = np.maximum(0, Z)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for j in range(3):
    im = axes[j].imshow(
        Y[:, j].reshape(40, 40),
        extent=[-2, 2, -2, 2],
        origin="lower",
        cmap="viridis",
    )
    axes[j].set_title(f"Neuron {j} activation")
    axes[j].set_xlabel("x₁")
    axes[j].set_ylabel("x₂")
plt.tight_layout()
plt.show()
```

Brighter regions mean higher activation. Each neuron lights up for different regions of input space — the layer **fans out** one input into multiple learned features.

---

## 6. Worked Examples

### Example 1: Tiny numeric forward pass

\(B=1\), \(n_{in}=2\), \(n_{out}=2\).

\[
\mathbf{X} = \begin{bmatrix} 1 & 2 \end{bmatrix}, \quad
\mathbf{W} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
\mathbf{b} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
\]

\[
\mathbf{Z} = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} + \begin{bmatrix} 0 & 1 \end{bmatrix}
= \begin{bmatrix} 1 & 2 \end{bmatrix} + \begin{bmatrix} 0 & 1 \end{bmatrix}
= \begin{bmatrix} 1 & 3 \end{bmatrix}
\]

Neuron 0 outputs \(1\), neuron 1 outputs \(3\). With ReLU, \(\mathbf{Y} = [1, 3]\).

### Example 2: Batch of two

Same \(\mathbf{W}\), \(\mathbf{b}\), but:

\[
\mathbf{X} = \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix}
\]

Row 0: \(Z = [1, 3]\). Row 1: \(Z = [0, 1] + [0, 1] = [0, 2]\).

\[
\mathbf{Z} = \begin{bmatrix} 1 & 3 \\ 0 & 2 \end{bmatrix}
\]

Each row of \(\mathbf{Z}\) is an independent forward pass sharing the same weights.

### Example 3: Verify in PyTorch

```python
import torch

X = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
W = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
b = torch.tensor([0.0, 1.0])
Z = X @ W.T + b
print(Z)
# tensor([[1., 3.],
#         [0., 2.]])
```

### Example 4: Parameter count

For `nn.Linear(4, 3)`: weights \(4 \times 3 = 12\), biases \(3\), total **15** learnable parameters. Parameter count scales as \(n_{in} \times n_{out} + n_{out}\).

### Example 5: Forward pass through a three-layer MLP

Consider `Linear(3, 4) → ReLU → Linear(4, 2) → ReLU → Linear(2, 1)` with batch size \(B=5\).

**Step 1:** Input \(\mathbf{X}\) has shape \((5, 3)\) — five examples, three features each.

**Step 2:** First layer: \(\mathbf{Z}_1 = \mathbf{X}\mathbf{W}_1^\top + \mathbf{b}_1\) → shape \((5, 4)\). Apply ReLU element-wise.

**Step 3:** Second layer: input is activated \(\mathbf{H}_1\) with shape \((5, 4)\). Output \(\mathbf{Z}_2\) has shape \((5, 2)\). ReLU again.

**Step 4:** Third layer: \(\mathbf{Y} = \mathbf{H}_2 \mathbf{W}_3^\top + \mathbf{b}_3\) → shape \((5, 1)\) — one scalar prediction per example.

```python
import torch
import torch.nn as nn

net = nn.Sequential(
    nn.Linear(3, 4), nn.ReLU(),
    nn.Linear(4, 2), nn.ReLU(),
    nn.Linear(2, 1),
)
x = torch.randn(5, 3)
y = net(x)
print(y.shape)  # torch.Size([5, 1])
```

This is exactly how a regression head or a binary logit output is structured inside larger models. The only difference in a 100-layer network is repeating the pattern — the shape logic at each step is identical.

### Example 6: When batch size is 1

If \(B=1\), the layer still works. \(\mathbf{X}\) has shape \((1, n_{in})\). Some beginners squeeze to 1-D and lose the batch dimension, breaking the next layer. Convention: keep batch dimension even when \(B=1\) unless the API explicitly expects a vector.

---

## 7. AI Connection

> 🧠 AI Insight
>
> A transformer block contains two main sublayers: multi-head attention and an MLP. The MLP is typically `Linear → GELU → Linear` — two dense layers with a nonlinearity between. GPT-3's MLP expands dimension 4× in the hidden layer, then projects back. Every parameter in those matrices is trained by backpropagation through the same `y = xW^T + b` pattern.

**Classification head:** Final `nn.Linear(d_model, num_classes)` maps token or image embeddings to class logits.

**Embedding lookup + linear:** Token IDs become vectors via an embedding table; subsequent layers are stacks of linear transforms and activations.

**Batching for GPUs:** Matrix multiply on `(B, n_in) @ (n_in, n_out)` is highly optimized. Large \(B\) keeps hardware busy. That is why training uses mini-batches.

**Width vs depth:** `out_features` is **layer width**. Stacking many layers adds **depth**. Both increase capacity; width parallelizes neurons, depth composes functions.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> **Wrong matrix orientation.** Writing `W @ x` instead of `x @ W.T` when `x` is `(batch, in_features)` and `W` is `(out_features, in_features)`. Always draw shapes before multiplying.

> ⚠️ Common Mistake
>
> **Broadcasting bias incorrectly.** Bias shape must be `(out_features,)`. PyTorch broadcasts it across batch rows. If you use `(1, out_features)` explicitly, that works too — but `(batch, 1)` does not add the right bias per neuron.

> ⚠️ Common Mistake
>
> **Applying activation before the next layer without checking shape.** ReLU preserves shape. Softmax across wrong dimension destroys batch semantics. For hidden layers, element-wise ReLU/GELU on `(B, n_out)` is correct.

**Correct understanding:** One layer = one matrix multiply + bias + optional activation. Shapes: input `(B, in)`, weight `(out, in)`, output `(B, out)`.

---

## 9. Exercises

### Easy

1. If `in_features=5`, `out_features=2`, `batch=4`, what are the shapes of `x`, `W`, `b`, and `y`?
2. Implement `y = x @ W.T + b` from the lab notebook and print shapes.
3. How many parameters does `nn.Linear(10, 20)` have?

### Medium

4. Build a 2-layer network: `Linear(4,8) → ReLU → Linear(8,3)`. Trace shapes for `batch=16`.
5. Without code: compute `Z` for `X=[[2,1]]`, `W=[[1,1],[1,-1]]`, `b=[0,2]`.
6. Why does PyTorch store `weight` as `(out_features, in_features)` rather than the transpose?

### Hard

7. Show that two `Linear` layers without activation between them are equivalent to one `Linear` layer.
8. Implement forward pass using `torch.einsum` instead of `@`.
9. For a batch of 32 images flattened to 784-D vectors, how many multiply-adds in one `Linear(784, 256)` forward pass?

### Challenge

10. **Layer Shape Debugger:** Write a function `trace_shapes(layers, x)` that takes a list of `nn.Linear` modules and prints input/output shape at each step. Add ReLU between layers and verify against a manual forward loop.

---

## 10. Mini Project

### MNIST-Style Linear Classifier Skeleton

Build a classifier for random 784-D inputs and 10 classes (no real dataset required):

1. `nn.Linear(784, 256)` → ReLU → `nn.Linear(256, 10)`.
2. Forward a batch of 64 random vectors.
3. Print output shape — should be `(64, 10)` logits.
4. Apply `softmax` on the class dimension and confirm rows sum to 1.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)

model = TinyClassifier()
x = torch.randn(64, 784)
logits = model(x)
probs = F.softmax(logits, dim=-1)
print("logits:", logits.shape)
print("row sums:", probs.sum(dim=-1)[:3])
```

<details>
<summary>Mini project checklist</summary>

- [ ] Two linear layers with ReLU between
- [ ] Correct output shape `(64, 10)`
- [ ] Softmax probabilities sum to 1 per row
- [ ] Parameter count printed

</details>

---

## 11. Interview Questions

**Q1:** What does `nn.Linear(in_features, out_features)` compute?

**A1:** For input `x` of shape `(batch, in_features)`, it computes `y = x @ weight.T + bias` where `weight` has shape `(out_features, in_features)` and `bias` has shape `(out_features,)`. Output shape is `(batch, out_features)`. Each output column is one neuron's pre-activation across the batch.

**Q2:** Why do we process data in batches?

**A2:** Batching amortizes Python overhead and exploits parallel hardware (GPU matrix units). Gradients are averaged over the batch for a stable estimate of the loss landscape direction. Very large batches can generalize differently; very small batches are noisy but sometimes help escape sharp minima.

**Q3:** What happens if you stack linear layers without a nonlinearity between them?

**A3:** The composition is still a single linear map. Mathematically, \(W_2(W_1 x + b_1) + b_2 = W' x + b'\) for some combined \(W', b'\). Depth adds no new expressive power without nonlinear activations between layers.

**Q4:** How is a layer related to the single neuron from the previous chapter?

**A4:** Each row of the weight matrix is one neuron's weights. The layer forward pass runs all neurons on all batch examples simultaneously via matrix multiplication. Same math, vectorized implementation.

---

## 12. Summary

### Key formulas

| Concept | Formula |
|---------|---------|
| Layer forward | \(\mathbf{Z} = \mathbf{X}\mathbf{W}^\top + \mathbf{b}\) |
| With ReLU | \(\mathbf{Y} = \max(0, \mathbf{Z})\) element-wise |
| Shapes | \(\mathbf{X}\): \((B, n_{in})\), \(\mathbf{W}\): \((n_{out}, n_{in})\), \(\mathbf{Y}\): \((B, n_{out})\) |
| Parameters | \(n_{in} \cdot n_{out} + n_{out}\) |

### Key terminology

- **Fully connected layer** — every input connects to every output neuron
- **Batch** — dimension \(B\); multiple examples processed together
- **in_features / out_features** — input and output dimension of the layer
- **Pre-activation** — matrix multiply result before activation
- **nn.Linear** — PyTorch module implementing the layer forward pass

---

## 13. Preview

You can now **forward** data through layers: inputs flow to outputs. Training requires flowing information **backward** — which weights caused the loss, and in which direction should they change? The next chapter, **Backpropagation**, derives the **chain rule** for composed functions and shows how `loss.backward()` automates gradient computation through arbitrary networks.

Forward pass builds the prediction. Backward pass builds the learning signal.

**Next chapter:** [Backpropagation](03-backpropagation.md)

---

## Lab

Companion notebook: [`app/neural_networks/02_building_a_layer.ipynb`](../../app/neural_networks/02_building_a_layer.ipynb)
