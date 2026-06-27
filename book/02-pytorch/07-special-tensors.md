# Special Tensors

## 1. Introduction

Not every tensor comes from data or random weights. PyTorch provides **factory functions** for structured patterns: identity matrices, evenly spaced ranges, linear grids, one-hot encodings, and controlled random distributions. These **special tensors** appear in initialization schemes, positional encodings, masking, debugging, and unit tests.

You already met `zeros`, `ones`, and `randn` in the creating tensors chapter. This chapter goes deeper: `eye`, `arange`, `linspace`, `logspace`, `full`, `ones_like`, and random generators with seeds. Knowing these factories saves time and prevents hand-rolled bugs.

After this chapter you will be able to:

- Create identity, diagonal, and structured tensors with one function call.
- Generate arithmetic sequences with `arange` and `linspace`.
- Control randomness with seeds and generators for reproducible experiments.
- Choose the right special tensor for masks, positions, and initialization.

**Where this appears in AI:** Identity for residual paths. `linspace` for plotting loss curves and learning rate schedules. `arange` for position indices. Random normal/uniform for weight init. One-hot for classification targets.

These factories look trivial compared to attention blocks, yet almost every research codebase uses them daily — in unit tests, learning-rate warmups, synthetic batches, and mask construction. Knowing them by heart removes friction when reading unfamiliar repositories.

**Generator objects:** PyTorch supports explicit `torch.Generator(device='cpu')` passed to `randn(..., generator=g)` for isolated random streams — useful when you need reproducible dropout masks without fixing the global seed for the entire process.

---

## 2. Intuition

> 💡 Intuition
>
> Special tensor factories are **cookie cutters** — you specify the shape and pattern, PyTorch stamps out the numbers. You don't loop in Python; the C++ backend fills memory efficiently. Need a 1000-step learning rate grid? `torch.linspace(1e-5, 1e-2, 1000)`. Need token positions 0..127? `torch.arange(128)`.

```
  eye(3):              arange(0, 10, 2):      linspace(0, 1, 5):

  [1 0 0]              [0, 2, 4, 6, 8]        [0.0, 0.25, 0.5, 0.75, 1.0]
  [0 1 0]
  [0 0 1]
```

In sinusoidal positional encoding (original Transformer), position \(pos\) and dimension \(i\) use angles from a geometric progression — often built with `arange` and powers of 10000.

> 🔬 Deep Dive
>
> `torch.manual_seed(k)` sets the CPU RNG state. For full reproducibility across runs, also set `torch.cuda.manual_seed_all(k)`, NumPy seed, and Python `random` seed — and note that some GPU ops are nondeterministic unless configured.

---

## 3. Formal Definitions

### Identity matrix

\[
I_n \in \mathbb{R}^{n \times n}, \quad I_{ij} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}
\]

`torch.eye(n)` creates \(I_n\).

### arange

Similar to Python `range`: values `start, start+step, ...` while `< end` (when step positive).

### linspace

\(N\) evenly spaced values from `start` to `end` **inclusive**:

\[
x_k = \text{start} + k \cdot \frac{\text{end} - \text{start}}{N - 1}, \quad k = 0, \ldots, N-1
\]

(for \(N > 1\))

### Random normal

\(X \sim \mathcal{N}(\mu, \sigma^2)\): `torch.randn` gives \(\mu=0, \sigma=1\); scale and shift as needed.

---

## 4. Programming Perspective

```python
import torch

print(torch.eye(3))
print(torch.arange(0, 10, 2))
print(torch.linspace(0, 1, 5))
print(torch.randn(3))
print(torch.full((2, 3), 7.0))
```

```python
# ones_like / zeros_like — match another tensor's shape and dtype
x = torch.randn(4, 5)
mask = torch.ones_like(x, dtype=torch.bool)
print(mask.shape, mask.dtype)

# Diagonal matrix from vector
v = torch.tensor([1., 2., 3.])
D = torch.diag(v)
print(D)
```

```python
# Reproducible random
torch.manual_seed(42)
a = torch.randn(3)
torch.manual_seed(42)
b = torch.randn(3)
print(torch.equal(a, b))  # True
```

```python
# 5×5 random normal matrix (lab exercise)
W = torch.randn(5, 5)
print(W.shape)
print(W.mean(), W.std(unbiased=False))  # approx 0 and 1
```

### Common factories table

| Function | Purpose |
|----------|---------|
| `torch.eye(n)` | Identity n×n |
| `torch.diag(v)` | Diagonal from vector |
| `torch.arange(start, end, step)` | Integer/step sequence |
| `torch.linspace(a, b, steps)` | Evenly spaced floats |
| `torch.logspace(a, b, steps)` | Log-spaced |
| `torch.full(shape, val)` | Constant fill |
| `torch.randn(*size)` | Standard normal |
| `torch.rand(*size)` | Uniform [0, 1) |
| `torch.randint(low, high, size)` | Integer uniform |

---

## 5. Visualizations

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Identity
I = torch.eye(5)
axes[0, 0].imshow(I.numpy(), cmap="Greys")
axes[0, 0].set_title("torch.eye(5)")
axes[0, 0].set_xlabel("col")
axes[0, 0].set_ylabel("row")

# linspace
t = torch.linspace(0, 4 * np.pi, 200)
y = torch.sin(t)
axes[0, 1].plot(t.numpy(), y.numpy())
axes[0, 1].set_title("sin(linspace(0, 4π, 200))")
axes[0, 1].set_xlabel("t")
axes[0, 1].set_ylabel("sin(t)")

# randn histogram
r = torch.randn(5000)
axes[1, 0].hist(r.numpy(), bins=40, color="steelblue")
axes[1, 0].set_title("torch.randn(5000)")
axes[1, 0].set_xlabel("value")
axes[1, 0].set_ylabel("count")

# arange as positions
pos = torch.arange(0, 32)
axes[1, 1].stem(pos.numpy(), np.ones(32))
axes[1, 1].set_title("arange(32) — token positions")
axes[1, 1].set_xlabel("position index")
axes[1, 1].set_ylabel("marker")

plt.tight_layout()
plt.show()
```

**How to read:**

1. **Identity:** Diagonal ones — where residual connections conceptually "pass through" the input channel.
2. **linspace + sin:** Smooth sampling for curves — same idea as sampling a loss landscape or activation plot.
3. **randn histogram:** Bell curve — weight initialization distribution.
4. **arange:** Discrete positions 0..31 — like token indices before embedding lookup.

```python
# Learning rate schedule preview
steps = 1000
lr_max, lr_min = 1e-2, 1e-5
lrs = torch.linspace(lr_max, lr_min, steps)
plt.figure(figsize=(8, 3))
plt.plot(lrs.numpy())
plt.xlabel("training step")
plt.ylabel("learning rate")
plt.title("Linear LR decay via linspace")
plt.show()
```

---

## 6. Worked Examples

### Example 1: Identity and matrix multiply

```python
A = torch.randn(3, 3)
I = torch.eye(3)
print(torch.allclose(A, A @ I))
print(torch.allclose(A, I @ A))
```

### Example 2: Positional indices for embedding

```python
seq_len = 128
positions = torch.arange(seq_len)  # [0, 1, ..., 127]
print(positions.shape)  # torch.Size([128])
# Typically passed to nn.Embedding(max_len, dim) or used in sinusoidal formula
```

### Example 3: One-hot via scatter (advanced pattern)

```python
num_classes = 5
labels = torch.tensor([0, 2, 4, 1])
one_hot = torch.zeros(len(labels), num_classes)
one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
print(one_hot)
```

### Example 4: 5×5 random normal matrix

```python
torch.manual_seed(0)
W = torch.randn(5, 5)
print(W)
print("mean:", W.mean().item(), "std:", W.std().item())
```

### Example 5: Learning rate grid with linspace

```python
import torch

total_steps = 500
warmup = 50
base_lr = 3e-4
warmup_lrs = torch.linspace(0, base_lr, warmup)
decay_lrs = torch.linspace(base_lr, base_lr * 0.1, total_steps - warmup)
schedule = torch.cat([warmup_lrs, decay_lrs])
print(schedule.shape, schedule[0], schedule[-1])
```

Training scripts often precompute such schedules once and index by step — no Python loop in the hot path.

Combining `linspace` for warmup and decay segments with `cat` mirrors how production schedulers stitch piecewise schedules into one lookup tensor.

### Example 6: Integer token IDs with randint

```python
batch, seq, vocab = 4, 16, 32000
fake_input_ids = torch.randint(0, vocab, (batch, seq))
print(fake_input_ids.shape, fake_input_ids.min(), fake_input_ids.max())
```

Synthetic batches like this let you profile model forward passes before the data pipeline is ready.

### Example 7: Mask values with full

```python
scores = torch.randn(4, 8, 8)
mask = torch.tril(torch.ones(8, 8, dtype=torch.bool))
scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
print(scores[0, 0, 1])  # masked future position -> large negative
```

Using `torch.finfo(dtype).min` instead of hard-coded `-1e9` respects float16 and bfloat16 ranges.

### Example 8: Xavier-style init with randn

```python
import torch
import math

fan_in, fan_out = 512, 512
W = torch.randn(fan_in, fan_out) * math.sqrt(2.0 / (fan_in + fan_out))
print(W.std().item())  # scaled std — smaller than raw randn
```

Manual init mirrors what `nn.init.xavier_normal_` applies before training starts.

### Example 9: logspace for frequency bins

```python
freqs = torch.logspace(start=0, end=3, steps=4, base=10)
print(freqs)  # tensor([1., 10., 100., 1000.])
```

Log-spaced grids appear in hyperparameter sweeps and spectral feature construction.

---

## 7. AI Connection

> 🧠 AI Insight
>
> Xavier/Glorot and Kaiming initialization formulas set variance based on fan-in and fan-out — implemented with `torch.randn` or `nn.init` helpers that ultimately fill special random tensors.

**Positional encoding:** Sinusoidal PE uses `torch.arange(0, dim, 2)` for dimension pairs and `10000 ** (arange / dim)` for frequencies — classic Transformer building block.

**Masks:** `torch.tril(torch.ones(n, n))` builds causal masks; `torch.full((B, T), pad_id)` for padding comparisons.

**Learning rate warmup:** Schedules often index `torch.linspace` or `torch.logspace` over training steps.

**Debugging:** `torch.eye`, small `arange` tensors make minimal reproducible unit tests for custom layers.

**Diffusion:** Noise timesteps sampled via `torch.randint` over `{0, ..., T-1}`.

**Batch labels:** `torch.randint(0, num_classes, (batch,))` for synthetic classification batches.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> **Off-by-one in `arange`.** `torch.arange(10)` is 0..9, not 1..10. `end` is exclusive like Python `range`.

> ⚠️ Common Mistake
>
> **Expecting `linspace` to exclude endpoint.** It includes both endpoints (for `steps >= 2`). Different from `numpy` legacy behaviors — verify count with small examples.

> ⚠️ Common Mistake
>
> **Forgetting seed when debugging.** Non-deterministic `randn` makes bugs hard to reproduce. Always seed when isolating issues.

> ⚠️ Common Mistake
>
> **Using `rand` vs `randn`.** `rand` is uniform on [0,1); `randn` is standard normal. Weight init almost always wants `randn` (or scaled variant).

**Correct understanding:** For `rand` vs `randn`. Seed for reproducibility. Check inclusive/exclusive endpoints.

Special tensors also appear in **evaluation metrics**: `torch.arange(num_classes)` builds label grids; `torch.eye(num_classes)` helps compute per-class confusion summaries. Small factories keep metric code readable without manual loops.

When benchmarking, `torch.cuda.Event` pairs measure GPU time, but CPU-side setup still uses `arange` and `linspace` to build input shapes and warmup schedules — factories remain the glue between experiment scripts and heavy kernels.

Prefer `torch.full` over multiplying `ones` when filling with arbitrary constants — intent is clearer and the API documents the target value explicitly in one argument list.

---

## 9. Exercises

### Easy

1. Create `3×3` identity and multiply random `3×3` matrix — verify unchanged.
2. `torch.arange(0, 10, 2)` — list all values.
3. `torch.linspace(0, 1, 5)` — print values and confirm endpoints 0 and 1.

### Medium

4. Create 5×5 random normal matrix (lab). Report mean and std.
5. Build `positions = torch.arange(100)` and `dims = torch.arange(0, 64, 2)` for a tiny PE setup.
6. Use `torch.full` to create a `(4, 8)` tensor filled with `-1e9` for masked logits.
7. Set seed 123, create two `randn(10)` — verify equal. Reset seed, verify again.

### Hard

8. Implement diagonal matrix from vector without `torch.diag` — use `zeros` + indexing.
9. Compare `torch.logspace(0, 3, 4)` values to `10**linspace(0, 3, 4)`.
10. Explain why `torch.randn(10000).std()` is close to 1 but not exactly 1.

### Challenge

11. **Sinusoidal PE:** Implement `positional_encoding(seq_len, dim)` using `arange`, `sin`, `cos` only on tensors.
12. **Init study:** Compare histograms of `randn`, `rand`, and `nn.init.xavier_normal_` on `(1000, 1000)` matrix.

---

## 10. Mini Project

### Special Tensor Toolkit

Create a script demonstrating factories used in a mini transformer setup:

```python
import torch
import math

def sinusoidal_pe(seq_len, dim):
    pe = torch.zeros(seq_len, dim)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def causal_mask(n):
    return torch.tril(torch.ones(n, n, dtype=torch.bool))

torch.manual_seed(0)
W = torch.randn(512, 512) * 0.02
pe = sinusoidal_pe(128, 512)
mask = causal_mask(128)

print("W:", W.shape, "PE:", pe.shape, "mask:", mask.shape)
print("PE sample [0, :4]:", pe[0, :4])
```

<details>
<summary>Mini project checklist</summary>

- [ ] Sinusoidal PE from arange/sin/cos
- [ ] Causal mask from ones/tril
- [ ] Seeded random weight matrix

</details>

---

## 11. Interview Questions

**Q1:** What does `torch.eye(n)` create and where might it appear in deep learning?

**A1:** An n×n identity matrix with ones on the diagonal and zeros elsewhere. It represents "no change" under multiplication. Used in theory for residual updates, orthogonal initialization ideas, and tests verifying linear layers preserve identity when configured. Attention and linear algebra derivations reference \(I\) frequently.

**Q2:** What is the difference between `arange` and `linspace`?

**A2:** `arange(start, end, step)` steps by fixed increment until reaching end (exclusive). `linspace(start, end, steps)` specifies the **count** of evenly spaced points including both endpoints. Use `arange` for integer indices; `linspace` for smooth numeric grids (plots, schedules).

**Q3:** How do you make random experiments reproducible in PyTorch?

**A3:** Call `torch.manual_seed(seed)` before creating random tensors. For GPU, also `torch.cuda.manual_seed_all(seed)`. Set NumPy and Python random seeds too. Note some CUDA operations remain nondeterministic unless `torch.use_deterministic_algorithms(True)` and environment variables are set — with possible performance cost.

**Q4:** When would you use `torch.full` instead of `torch.ones * value`?

**A4:** `torch.full((shape), value)` allocates and fills in one call — clearer intent and slightly more efficient than creating ones and multiplying. Common for large negative mask values (`-1e9`) in attention masking before softmax.

**Q5:** What distribution does `torch.randn` sample from?

**A5:** Standard normal \(\mathcal{N}(0, 1)\) — mean zero, variance one. Weight initialization often uses `randn` then scales by fan-in/fan-out formulas (Xavier, Kaiming). For uniform initialization, use `torch.rand` and affine-transform to desired range.

### Factory function quick reference for ML

| Task | Function |
|------|----------|
| Weight init (raw) | `torch.randn(shape) * scale` |
| Bias zeros | `torch.zeros(out_features)` |
| Position indices | `torch.arange(seq_len)` |
| Plot / schedule axis | `torch.linspace(0, 1, steps)` |
| Random token batch | `torch.randint(0, vocab, (B, T))` |
| Attention causal mask | `torch.triu(ones, diagonal=1).bool()` |
| One-hot from labels | `torch.eye(C)[labels]` |
| Reproducibility | `torch.manual_seed(seed)` |

Special tensors are not exotic — they are the scaffolding every training script builds before the first forward pass.

Choosing the right factory is a design decision: `randn` explores; `eye` validates; `arange` indexes; `linspace` schedules; `full` fills masks. Together they cover most non-data tensors in a research codebase.

Before training a new architecture, list every tensor you create that is not loaded from disk — almost every entry will be a factory from this chapter.

---

## 12. Summary

### Key factories

| Function | Output pattern |
|----------|----------------|
| `eye(n)` | Identity matrix |
| `arange(...)` | Regular sequence |
| `linspace(a,b,N)` | N evenly spaced values |
| `randn(...)` | Standard normal |
| `full(shape, v)` | All values = v |
| `ones_like(x)` | Same shape as x, all 1 |

### Key terminology

- **Identity matrix** — diagonal ones, off-diagonal zeros
- **RNG seed** — fixes pseudorandom sequence
- **linspace** — inclusive endpoint spacing
- **positional encoding** — inject order into embeddings
- **factory function** — tensor constructor without input data

---

## 13. Preview

You have completed the PyTorch tensor fundamentals track: create, multiply, transpose, reshape, slice, concatenate, and generate special patterns. The next module applies these operations to **neural networks** — single neurons, layers, and eventually backpropagation through composed tensor graphs.

Every `forward()` you write will be a sequence of the tensor operations from these seven chapters. When you read transformer code, you will recognize each line as something you can build and debug from scratch.

**Next module:** [Neural Networks — Single Neuron](../../03-neural-networks/01-single-neuron.md)

---

## Lab

Companion notebook: [`app/pytorch/07_special_tensors.ipynb`](../../app/pytorch/07_special_tensors.ipynb)
