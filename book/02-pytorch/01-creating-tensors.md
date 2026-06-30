# Creating Tensors

## 1. Introduction

Every neural network, every training loop in PyTorch begins with the same primitive: a **tensor**.

> 📌 Preview — optional for now
>
> **Term:** transformer
> **One line:** a neural architecture built from attention and feed-forward blocks
> **Learn properly in:** [Decoder-Only Transformer](../04-transformers/04-decoder-only-transformer.md)
> You can skip the details and keep reading.

**Recap from math:** a 1D tensor is like a [vector](../01-math/03-vectors.md); a 2D tensor is like a [matrix](../01-math/05-matrices.md). If those chapters felt rusty, skim their cheatsheets first.

If NumPy arrays are the workhorse of scientific Python, tensors are the workhorse of deep learning — arrays that can live on a GPU and flow through training graphs.

> 📌 Preview — optional for now
>
> **Term:** autograd
> **One line:** PyTorch's system that tracks operations so gradients can be computed
> **Learn properly in:** [Backpropagation](../03-neural-networks/03-backpropagation.md)
> You can skip the details and keep reading.

You already know lists and NumPy arrays. A tensor is the next step: a multi-dimensional array with a **shape** (how big it is along each axis), a **dtype** (what kind of numbers it holds), and optionally a **device** (CPU or GPU). When you write `model(x)` in PyTorch, `x` is almost always a tensor.

After this chapter you will be able to:

- Create tensors from Python lists, NumPy arrays, and built-in factories.
- Read and interpret `.shape`, `.dtype`, and `.device`.
- Choose the right dtype (`float32` vs `float64`) for ML workloads.
- Explain why tensors are the universal container for weights, activations, and gradients.

**Where this appears in AI:** Model weights are tensors. Input batches are tensors. Loss values are tensors. Mastering tensor creation is the first hands-on step toward building and training any PyTorch model.

**Suggested pacing (3 sessions):**

- Session A: §1–§3 + [cheatsheet](01-creating-tensors-cheatsheet.md) skim
- Session B: §4–§6 + lab notebook
- Session C: Easy–Medium exercises + readiness checks in §12

---

## 2. Intuition

> 💡 Intuition
>
> Think of a tensor as a **labeled box of numbers** organized in a grid. A single number is a 0-dimensional tensor (a scalar). A list of numbers is a 1D tensor (a vector). A spreadsheet is a 2D tensor (a matrix). A stack of spreadsheets is a 3D tensor — and so on. The **rank** (number of dimensions) tells you how many indices you need to reach one element.

```
  Scalar (0D):     7

  Vector (1D):     [1, 2, 3]

  Matrix (2D):     [[1, 2],
                    [3, 4]]

  3D tensor:       [[[1, 2], [3, 4]],
                    [[5, 6], [7, 8]]]
```

In a batched sequence tensor, shape might be `(32, 128, 768)`:

- `32` — batch size (32 sentences processed together)
- `128` — sequence length (128 tokens per sentence)
- `768` — hidden dimension (768 numbers per token)

> 📌 Preview — optional for now
>
> **Term:** embeddings
> **One line:** learned vectors that represent tokens or categories
> **Learn properly in:** [Vectors](../01-math/03-vectors.md) (intuition), [Self-Attention](../04-transformers/02-self-attention.md) (full use)
> You can skip the details and keep reading.

Each slot in that 3D grid is one float. The **shape** tells you how to navigate the grid; the **dtype** tells you the precision of each number.

> 🔬 Deep Dive
>
> PyTorch tensors and NumPy arrays share memory layout ideas (strides, contiguous storage), but tensors add two capabilities critical for deep learning: **GPU placement** (`.to("cuda")`) and **gradient tracking** (`requires_grad=True`). NumPy arrays do neither. That is why PyTorch uses its own type instead of wrapping NumPy everywhere.

---

## 3. Formal Definitions

We define every symbol on first use.

### Tensor

A **tensor** is an ordered multi-dimensional array of numbers of a single **dtype** (data type). We write the shape as a tuple of positive integers:

\[
\text{shape} = (d_0, d_1, \ldots, d_{k-1})
\]

- \(k\) is the **rank** (number of dimensions).
- \(d_i\) is the size along dimension \(i\).
- The **total number of elements** is \(d_0 \times d_1 \times \cdots \times d_{k-1}\).

> **Plain English**
> Shape is a list of sizes along each axis; multiply them to get how many numbers the tensor stores.

> **Python**
> `x.numel() == x.shape[0] * x.shape[1] * ...`

**Example:** A tensor with shape `(2, 3)` has rank 2, two rows and three columns, and \(2 \times 3 = 6\) elements total.

### dtype

The **dtype** specifies how each element is stored in memory:

| PyTorch dtype | Meaning | Typical use |
|---------------|---------|-------------|
| `torch.float32` | 32-bit floating point | Default for ML training |
| `torch.float64` | 64-bit floating point | High-precision numerics |
| `torch.int64` | 64-bit integer | Labels, indices |
| `torch.bool` | Boolean | Masks |

In deep learning, **`float32`** is the standard. It balances precision and speed; GPUs are optimized for it.

### device

The **device** is where the tensor's data lives in memory:

- `cpu` — main system RAM
- `cuda:0` — first GPU

Operations on two tensors require them to be on the **same device**.

---

## 4. Programming Perspective

Creating tensors maps directly to Python data structures.

| Intent | PyTorch |
|--------|---------|
| From a Python list | `torch.tensor([[1, 2], [3, 4]])` |
| Zeros of given shape | `torch.zeros(2, 3)` |
| Ones | `torch.ones(2, 3)` |
| Random normal | `torch.randn(4)` |
| Identity matrix | `torch.eye(3)` |
| From NumPy | `torch.from_numpy(arr)` |

```python
import torch

# From a nested Python list — dtype inferred as int64 by default
a = torch.tensor([[1, 2], [3, 4]])
print(a)
# tensor([[1, 2],
#         [3, 4]])

# Explicit float dtype (standard for ML)
b = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
print(b.shape)   # torch.Size([2, 2])
print(b.dtype)   # torch.float32

# Factory: pre-allocated zeros
c = torch.zeros(2, 3, dtype=torch.float32)
print(c)
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])

# Random tensor — values drawn from standard normal N(0, 1)
d = torch.randn(4)
print(d)  # e.g. tensor([ 0.3367, -0.1288,  0.2345,  0.2303])
```

**Key insight:** `torch.tensor()` **copies** data from the input. `torch.from_numpy()` **shares** memory with the NumPy array — changing one can change the other.

```python
import numpy as np

arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)
arr[0] = 99.0
print(t)  # tensor([99.,  2.,  3.]) — shared memory!
```

For ML, always specify `dtype=torch.float32` when creating tensors from integers or ambiguous sources.

---

## 5. Visualizations

Tensors are hard to draw beyond 2D, but we can visualize small matrices and inspect properties with code.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# A 2D tensor as a heatmap
data = torch.tensor([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Heatmap: rows = y-axis, cols = x-axis, color = value
im = axes[0].imshow(data.numpy(), cmap="Blues", aspect="auto")
axes[0].set_title("2×3 tensor as heatmap")
axes[0].set_xlabel("column index")
axes[0].set_ylabel("row index")
plt.colorbar(im, ax=axes[0])

# Histogram of random tensor values
rand = torch.randn(1000)
axes[1].hist(rand.numpy(), bins=30, color="steelblue", edgecolor="white")
axes[1].set_title("Histogram of torch.randn(1000)")
axes[1].set_xlabel("value")
axes[1].set_ylabel("count")

plt.tight_layout()
plt.show()
```

**How to read these plots:**

1. **Heatmap:** Each cell is one element. Row 0, column 0 holds `1.0`; row 1, column 2 holds `6.0`. Brighter color = larger value. This is how you might visualize a small weight matrix.

2. **Histogram:** `torch.randn(1000)` draws 1000 samples from a bell curve centered at 0. Most values fall between -2 and 2. Weight initialization in neural networks often uses this distribution so starting weights are small and symmetric.

```python
# Compare dtypes visually — float32 vs float64 precision
x32 = torch.tensor([1.0 / 3.0], dtype=torch.float32)
x64 = torch.tensor([1.0 / 3.0], dtype=torch.float64)
print(f"float32: {x32.item():.20f}")
print(f"float64: {x64.item():.20f}")
# float32 shows fewer decimal digits — enough for training, less memory
```

---

## 6. Worked Examples

### Example 1: Create a 2×2 matrix from a list

```python
import torch

matrix = torch.tensor([[1.0, 2.0],
                       [3.0, 4.0]])
print(matrix)
print("shape:", matrix.shape)   # torch.Size([2, 2])
print("dtype:", matrix.dtype)   # torch.float32
print("numel:", matrix.numel()) # 4 elements total
```

**Step by step:** We pass a nested list. PyTorch infers shape `(2, 2)` from the structure — 2 rows, each with 2 elements. Using `1.0` instead of `1` makes the dtype `float32`.

### Example 2: Zeros and ones for initialization

Neural network biases are often initialized to zero; some layers use ones.

```python
bias = torch.zeros(128)          # 128-dimensional bias vector
scale = torch.ones(64, 64)       # 64×64 matrix of ones
print(bias.shape)   # torch.Size([128])
print(scale.shape)  # torch.Size([64, 64])
```

### Example 3: Random weights for a linear layer

A linear layer with 3 inputs and 2 outputs has weight shape `(2, 3)` and bias shape `(2,)`.

```python
torch.manual_seed(42)  # reproducibility
W = torch.randn(2, 3)  # weights
b = torch.zeros(2)     # biases
print("W:\n", W)
print("b:", b)
```

Each weight is a small random number — typical initialization before training adjusts them.

### Example 4: Identity matrix

The identity matrix \(I\) has ones on the diagonal and zeros elsewhere. Multiplying any matrix by \(I\) returns the same matrix.

\[
I_{ij} = 1 \text{ if } i = j \text{, else } 0
\]

> **Plain English**
> Ones on the diagonal, zeros everywhere else — multiplying by \(I\) leaves a matrix unchanged.

> **Python**
> `I = torch.eye(n)`

```python
I = torch.eye(3)
print(I)
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])
```

In residual connections (preview), models often add a sublayer's output back to its input: `output = x + sublayer(x)`.

---

## 7. AI Connection

> 🧠 AI Insight
>
> When you define `nn.Linear(768, 768)` in a transformer block, PyTorch creates two tensors behind the scenes: a weight tensor of shape `(768, 768)` and a bias tensor of shape `(768,)`. Both start as random floats and get updated every training step via gradient descent.

**Model parameters** are tensors with `requires_grad=True`. After `loss.backward()`, each parameter's `.grad` holds the gradient — also a tensor of the same shape.

**Input batches** are tensors. For image classification, shape might be `(64, 3, 224, 224)` — 64 images, 3 color channels, 224×224 pixels. For language modeling, `(batch, seq_len)` integer tensor of token IDs, then embedded to `(batch, seq_len, hidden_dim)`.

**Activations** — outputs of each layer — are tensors flowing forward. **Loss** is a scalar tensor (shape `()`). **Gradients** mirror parameter shapes.

Every operation you will learn in this PyTorch module — multiply, transpose, reshape, slice, concatenate — exists because real models constantly manipulate tensors in these ways. Creating them correctly is step zero.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> **Forgetting `dtype=torch.float32`.** `torch.tensor([1, 2, 3])` creates `int64`. Neural networks expect floats. Always use `torch.tensor([1., 2., 3.])` or `dtype=torch.float32`. Mixed dtypes in the same operation cause errors.

> ⚠️ Common Mistake
>
> **Confusing `torch.tensor()` with `torch.Tensor()`.** `torch.tensor()` is the modern constructor from data. `torch.Tensor()` is a legacy alias that creates an *empty* uninitialized tensor — garbage values. Always prefer `torch.tensor()` or factory functions like `torch.zeros()`.

> ⚠️ Common Mistake
>
> **Assuming `.shape` and `.size()` differ.** They return the same information. `x.shape` is `(2, 3)`; `x.size()` is `torch.Size([2, 3])`. Use whichever reads more clearly.

> ⚠️ Common Mistake
>
> **Modifying a NumPy array after `from_numpy()`.** Because memory is shared, unexpected changes propagate. If you need independence, use `torch.tensor(numpy_array)` to copy.

**Correct understanding:** Specify dtype explicitly. Use factory functions for common patterns. Know whether memory is shared or copied when converting from NumPy.

---

## 9. Exercises

### Easy

1. Create a 1D tensor containing `[10, 20, 30, 40]` with `float32` dtype. Print shape and dtype.
2. Create a 3×4 matrix of zeros using `torch.zeros()`.
3. Create a tensor of shape `(5,)` filled with the value `7.0` using `torch.full()`.
4. How many elements does a tensor of shape `(2, 3, 4)` contain? Verify with `.numel()`.

### Medium

5. Create a 3×3 identity matrix with `torch.eye()`. Multiply it by a random 3×3 matrix `A`. Confirm the result equals `A`.
6. Convert a NumPy array `np.array([1.0, 2.0, 3.0])` to a tensor two ways: `torch.tensor()` and `torch.from_numpy()`. Modify the NumPy array and show the difference.
7. Create `torch.randn(100)` and compute its mean and standard deviation with `.mean()` and `.std()`. Are they close to 0 and 1?
8. Create a tensor on CPU, then move it to GPU with `.to("cuda")` if available. Print the device before and after.

### Hard

9. Explain why model weights use `float32` instead of `float64` for training on a GPU.
10. Create a tensor of shape `(4, 4)` with random values, then cast it to `torch.int64` with `.to(torch.int64)`. What happens to the fractional parts?
11. Write code that creates a batch of 16 "images": shape `(16, 1, 28, 28)` (grayscale 28×28), filled with random normal values. This mimics MNIST batch structure.

### Challenge

12. **Tensor Inspector:** Write a function `inspect(t)` that prints shape, dtype, device, numel, and whether `requires_grad` is True. Test it on five different tensors you create.
13. **Initialization comparison:** Create a `(1000, 1000)` weight matrix three ways: `torch.randn`, `torch.zeros`, and `torch.nn.init.xavier_uniform_` on a zero tensor. Plot histograms of all values for each. Which is appropriate for neural network weights and why?

---

## 10. Mini Project

### Tensor Creation Playground

Build a script that demonstrates every major creation method in one place:

1. From Python list, NumPy array, zeros, ones, randn, eye, arange, linspace.
2. For each tensor, print shape, dtype, and a sample of values.
3. Create a "mini model state": one weight matrix `(10, 5)`, one bias `(10,)`, one input batch `(32, 5)`.
4. Plot the weight matrix as a heatmap and save to `book/assets/01-tensor-playground.png`.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

torch.manual_seed(0)

creators = {
    "from_list": torch.tensor([[1., 2.], [3., 4.]]),
    "zeros": torch.zeros(2, 3),
    "ones": torch.ones(4),
    "randn": torch.randn(5),
    "eye": torch.eye(3),
    "arange": torch.arange(0, 10, 2),
    "linspace": torch.linspace(0, 1, 5),
}

for name, t in creators.items():
    print(f"{name:12s} shape={tuple(t.shape)} dtype={t.dtype} sample={t.flatten()[:3].tolist()}")

W = torch.randn(10, 5)
b = torch.zeros(10)
batch = torch.randn(32, 5)

fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(W.numpy(), cmap="RdBu", aspect="auto")
ax.set_title("Random weight matrix (10×5)")
ax.set_xlabel("input feature")
ax.set_ylabel("output neuron")
out = Path("book/assets/01-tensor-playground.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
```

<details>
<summary>Mini project checklist</summary>

- [ ] At least 7 creation methods demonstrated
- [ ] Shape and dtype printed for each
- [ ] Mini model state (W, b, batch) created
- [ ] Heatmap saved to `book/assets/`

</details>

---

## 11. Interview Questions

**Q1:** What is a PyTorch tensor, and how does it differ from a NumPy array?

**A1:** A tensor is a multi-dimensional array with a dtype and device. Like NumPy arrays, tensors support shape, indexing, and element-wise math. Unlike NumPy, tensors can live on GPU for accelerated computation and can track gradients via autograd (`requires_grad=True`). NumPy is CPU-only and has no autograd. For deep learning, tensors are the standard container; NumPy remains useful for preprocessing and I/O.

**Q2:** Why is `float32` the default dtype for neural network training?

**A2:** `float32` offers sufficient precision for gradient-based optimization while using half the memory of `float64` and running faster on GPUs (which have dedicated float32 hardware). Empirically, training with float32 converges similarly to float64 for most models. Some training uses `float16` or `bfloat16` for further speed, but float32 is the reliable baseline.

**Q3:** What does `torch.tensor()` do versus `torch.from_numpy()`?

**A3:** `torch.tensor(data)` creates a new tensor by **copying** the input data. `torch.from_numpy(arr)` creates a tensor that **shares memory** with the NumPy array — no copy. If you modify the array after `from_numpy`, the tensor changes too. Use `from_numpy` when you want zero-copy conversion and will not mutate the source; use `tensor()` when you need an independent copy.

**Q4:** Explain the meaning of shape `(32, 128, 768)` in a transformer context.

**A4:** This is a batch of embedded token sequences. `32` is the batch size — 32 sequences processed in parallel. `128` is the sequence length — up to 128 tokens per sequence. `768` is the hidden dimension — each token is represented by a 768-dimensional vector (as in BERT-base). Every attention head and feed-forward layer operates on this 3D tensor.

**Q5:** What is the difference between rank, shape, and numel?

**A5:** **Rank** is the number of dimensions (axes). **Shape** is the tuple of sizes along each dimension, e.g. `(2, 3, 4)`. **Numel** is the total element count — the product of shape entries: \(2 \times 3 \times 4 = 24\). Rank tells you how many indices you need; shape tells you the extent along each axis; numel tells you how many numbers are stored.

---

## 12. Summary

### Core takeaways (must know)

- Tensors are multi-dimensional arrays with shape, dtype, and device
- `torch.tensor()` copies; `torch.from_numpy()` shares memory
- Use `float32` for ML weights and activations
- `numel` equals the product of shape entries

### Preview terms (optional until later)

- Embeddings, attention, autograd, transformers — see [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)

### Key formulas

| Concept | Expression |
|---------|------------|
| Total elements | \(\text{numel} = d_0 \times d_1 \times \cdots \times d_{k-1}\) |
| Identity matrix | \(I_{ij} = 1\) if \(i = j\), else \(0\) |
| Random normal | \(X \sim \mathcal{N}(0, 1)\) via `torch.randn` |

### Key terminology

- **Tensor** — multi-dimensional array with dtype and device
- **Shape** — tuple of dimension sizes, e.g. `(2, 3)`
- **Rank** — number of dimensions
- **dtype** — data type (`float32`, `int64`, etc.)
- **device** — where data lives (`cpu`, `cuda`)
- **numel** — total number of elements
- **requires_grad** — whether autograd tracks this tensor

### Readiness checks

Before the next chapter, you should be able to:

1. Create a `(3, 4)` tensor of `float32` zeros and report its shape and numel.
2. Explain the difference between `torch.tensor()` and `torch.from_numpy()`.
3. Choose `float32` vs `int64` for model weights vs class labels.
4. Create a random weight matrix `(out, in)` and bias `(out,)` for a linear layer.
5. Read shape `(32, 128, 768)` as batch × sequence × hidden size.

If any item is shaky, reread §3–§4 and the [cheatsheet](01-creating-tensors-cheatsheet.md).

---

## 13. Preview

You can now create tensors. The next chapter — **Matrix Multiplication** — covers the operation at the heart of every neural network layer: multiplying matrices and batches of matrices. When you see `nn.Linear` or attention scores `Q @ K.T`, you are doing matrix multiplication on tensors you create today.

Shapes must align: `(m, n) @ (n, p) → (m, p)`. Getting this wrong is one of the most common bugs in PyTorch. The next chapter builds the intuition and the code patterns you need.

**Next chapter:** [Matrix Multiplication](02-matrix-multiplication.md)

---

## Lab

Companion notebook: [`app/pytorch/01_creating_tensors.ipynb`](../../app/pytorch/01_creating_tensors.ipynb)

## Review

- Cheatsheet: [Creating Tensors — Cheatsheet](01-creating-tensors-cheatsheet.md)
- Jargon: [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
