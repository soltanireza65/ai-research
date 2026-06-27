# Backpropagation

## 1. Introduction

You can forward a batch through `nn.Linear`, apply ReLU, and compute a loss. But **how do the weights learn**? Something must tell each parameter whether to increase or decrease. That signal is the **gradient** — the derivative of the loss with respect to each weight.

**Backpropagation** is an algorithm for computing all those gradients efficiently in a deep network. It applies the **chain rule** from calculus layer by layer, starting at the loss and working backward. PyTorch implements this automatically: call `loss.backward()`, then `optimizer.step()`.

After this chapter you will be able to:

- Apply the **chain rule** to composed functions step by step.
- Derive gradients for a tiny network: linear layer → ReLU → MSE loss.
- Explain what `loss.backward()` and `optimizer.step()` do in a training loop.
- Debug common autograd mistakes (in-place ops, forgetting `zero_grad`).

**Where this appears in AI:** Every trained model — CNNs, transformers, diffusion networks — is trained by gradient-based optimization. Backpropagation is how gradients reach millions of parameters in one backward pass. There is no modern deep learning without it.

---

## 2. Intuition

> 💡 Intuition
>
> Picture a assembly line: raw materials enter, pass through stations, become a finished product. Quality control measures the final product's error. To fix the line, you ask each station: "if I nudge your dial slightly, how much does the final error change?" You start at the end and work backward, because each station's effect on the final product depends on what happened downstream.

```
  x ──► [Layer 1] ──► [ReLU] ──► [Layer 2] ──► ŷ ──► [Loss L]
         w₁, b₁                    w₂, b₂              │
                                                         ▼
  gradients flow backward ◄──────────────────────── ∂L/∂w₂, ∂L/∂w₁
```

**Forward pass:** compute predictions. **Backward pass:** compute \(\frac{\partial L}{\partial w}\) for every parameter \(w\). **Optimizer step:** update \(w \leftarrow w - \eta \frac{\partial L}{\partial w}\) where \(\eta\) is the learning rate.

> 🔬 Deep Dive
>
> Backpropagation is not a separate learning algorithm — it is **efficient gradient computation**. The learning algorithm is **gradient descent** (or Adam, SGD, etc.). Backprop tells you the direction; the optimizer takes the step.

---

## 3. Formal Definitions

### Loss function

A **loss** \(L\) maps predictions and targets to a single scalar measuring error. For one example with prediction \(\hat{y}\) and target \(y\):

\[
L = \frac{1}{2}(\hat{y} - y)^2 \quad \text{(MSE, simplified)}
\]

Lower \(L\) is better. Training minimizes \(L\) over the data distribution.

### Gradient

The **gradient** of \(L\) with respect to scalar parameter \(w\) is:

\[
\frac{\partial L}{\partial w}
\]

Positive gradient means increasing \(w\) increases \(L\) (bad). We step opposite to the gradient to decrease loss.

### Chain rule

If \(L\) depends on \(y\), and \(y\) depends on \(w\), then:

\[
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}
\]

For longer chains \(L \leftarrow y_3 \leftarrow y_2 \leftarrow y_1 \leftarrow w\):

\[
\frac{\partial L}{\partial w} =
\frac{\partial L}{\partial y_3} \cdot
\frac{\partial y_3}{\partial y_2} \cdot
\frac{\partial y_2}{\partial y_1} \cdot
\frac{\partial y_1}{\partial w}
\]

Multiply local derivatives along the path from \(L\) back to \(w\).

### Computational graph

PyTorch builds a **dynamic graph** during the forward pass. Each tensor operation is a node. `backward()` traverses this graph in reverse, applying the chain rule at each node.

| Symbol | Meaning |
|--------|---------|
| \(L\) | Loss scalar |
| \(\frac{\partial L}{\partial w}\) | Gradient of loss w.r.t. weight \(w\) |
| \(\eta\) | Learning rate |
| `requires_grad` | PyTorch flag: track operations for this tensor |

---

## 4. Programming Perspective

### Enable gradients

```python
import torch

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
y = w * x          # y = 6 when w=2
L = (y - 10) ** 2  # target 10
L.backward()
print(w.grad)      # dL/dw = 2(y-10)*x = 2(6-10)*3 = -24
```

`w.grad` holds \(\frac{\partial L}{\partial w}\) after `backward()`.

### Full chain in tiny steps

```python
import torch

# y = w*x, L = (y - t)^2
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
t = torch.tensor(10.0)

y = w * x
L = (y - t) ** 2

L.backward()
# dL/dy = 2(y-t) = -8
# dy/dw = x = 3
# dL/dw = dL/dy * dy/dw = -24
print("gradient:", w.grad.item())
```

```python
# ReLU in the chain: z = w*x, y = relu(z), L = (y-t)^2
w = torch.tensor(-1.0, requires_grad=True)
x = torch.tensor(5.0)
t = torch.tensor(3.0)

z = w * x           # z = -5
y = torch.relu(z)   # y = 0
L = (y - t) ** 2    # L = 9

L.backward()
print(w.grad)       # 0 — ReLU blocks gradient when z <= 0
```

When pre-activation is negative, ReLU output is flat; gradient w.r.t. \(w\) is zero.

```python
import torch.nn as nn

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

x = torch.tensor([[2.0]])
t = torch.tensor([[10.0]])

for step in range(20):
    optimizer.zero_grad()       # clear old gradients
    y = model(x)              # forward
    loss = ((y - t) ** 2).mean()
    loss.backward()           # backward: fill .grad
    optimizer.step()          # w <- w - lr * grad
    if step % 5 == 0:
        print(f"step {step}: loss={loss.item():.4f}, w={model.weight.item():.4f}")
```

### Training loop anatomy

| Step | Code | Purpose |
|------|------|---------|
| 1 | `optimizer.zero_grad()` | Reset gradients from previous step |
| 2 | `y = model(x)` | Forward pass |
| 3 | `loss = criterion(y, t)` | Compute scalar loss |
| 4 | `loss.backward()` | Backpropagate gradients |
| 5 | `optimizer.step()` | Update parameters |

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Synthetic data: y ≈ 3x
X = torch.randn(100, 1)
Y = 3 * X + 0.5 * torch.randn(100, 1)
loader = DataLoader(TensorDataset(X, Y), batch_size=16, shuffle=True)

model = nn.Linear(1, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.05)
criterion = nn.MSELoss()

for epoch in range(10):
    epoch_loss = 0.0
    for xb, yb in loader:
        opt.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    print(f"epoch {epoch}: avg loss {epoch_loss / len(loader):.4f}")
```

---

## 5. Visualizations

Gradients tell you the **slope** of the loss landscape. Plotting \(L(w)\) for a single weight builds intuition for what `backward()` computes.

```python
import numpy as np
import matplotlib.pyplot as plt

# L(w) = (w*x - t)^2 with x=3, t=10
x_val, t_val = 3.0, 10.0
w_range = np.linspace(-1, 6, 200)
L_vals = (w_range * x_val - t_val) ** 2

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(w_range, L_vals, color="steelblue", linewidth=2)
axes[0].set_xlabel("w (weight)")
axes[0].set_ylabel("L (loss)")
axes[0].set_title("Loss landscape for one weight")
w_star = t_val / x_val
axes[0].axvline(w_star, color="coral", linestyle="--", label=f"minimum w={w_star:.2f}")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gradient dL/dw = 2(w*x - t)*x
grad_vals = 2 * (w_range * x_val - t_val) * x_val
axes[1].plot(w_range, grad_vals, color="seagreen", linewidth=2)
axes[1].set_xlabel("w")
axes[1].set_ylabel("dL/dw")
axes[1].set_title("Gradient: zero at minimum, negative left of it")
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Reading the plots:** Left — parabolic loss with minimum at \(w = t/x \approx 3.33\). Right — gradient is negative when \(w\) is too small (increase \(w\) to reduce loss), positive when \(w\) is too large.

```python
# Training trajectory overlay
import torch
import torch.nn as nn

model = nn.Linear(1, 1)
x_t = torch.tensor([[3.0]])
t_t = torch.tensor([[10.0]])
ws, losses = [], []
for _ in range(30):
    ws.append(model.weight.item())
    pred = model(x_t)
    loss = ((pred - t_t) ** 2).item()
    losses.append(loss)
    model.zero_grad()
    ((pred - t_t) ** 2).backward()
    with torch.no_grad():
        model.weight -= 0.1 * model.weight.grad

w_range = np.linspace(-1, 6, 200)
L_curve = (w_range * 3 - 10) ** 2
plt.figure(figsize=(7, 5))
plt.plot(w_range, L_curve, label="L(w)")
plt.scatter(ws, losses, c=range(len(ws)), cmap="plasma", s=30, zorder=5)
plt.xlabel("w")
plt.ylabel("L")
plt.title("Gradient descent path on loss curve")
plt.colorbar(label="step")
plt.grid(True, alpha=0.3)
plt.show()
```

Each point is one optimizer step marching toward the loss minimum.

---

## 6. Worked Examples

### Example 1: Chain rule by hand

\(y = wx\), \(L = \frac{1}{2}(y - t)^2\). Find \(\frac{\partial L}{\partial w}\) when \(w=2\), \(x=3\), \(t=10\).

**Step 1:** Forward: \(y = 2 \times 3 = 6\).

**Step 2:** \(L = \frac{1}{2}(6 - 10)^2 = \frac{1}{2}(16) = 8\).

**Step 3:** \(\frac{\partial L}{\partial y} = y - t = 6 - 10 = -4\).

**Step 4:** \(\frac{\partial y}{\partial w} = x = 3\).

**Step 5:** Chain rule: \(\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w} = (-4)(3) = -12\).

Negative gradient: increase \(w\) to decrease \(L\). SGD with \(\eta = 0.1\): \(w_{\text{new}} = 2 - 0.1 \times (-12) = 3.2\).

### Example 2: Two-layer network

\(h = \text{ReLU}(w_1 x)\), \(\hat{y} = w_2 h\), \(L = (\hat{y} - y)^2\).

Suppose \(x=2\), \(w_1=-1\), so \(z = -2\), \(h = 0\). Then \(\hat{y} = 0\) regardless of \(w_2\).

\(\frac{\partial L}{\partial w_2} = 2(\hat{y} - y) \cdot h\). If \(h=0\), gradient w.r.t. \(w_2\) is **zero**.

\(\frac{\partial L}{\partial w_1}\) also zero through ReLU when \(z \leq 0\). The neuron is **dead** — no learning signal reaches \(w_1\).

### Example 3: Matrix layer gradient (conceptual)

For \(\mathbf{Z} = \mathbf{X}\mathbf{W}^\top + \mathbf{b}\) and scalar loss \(L\):

\[
\frac{\partial L}{\partial W_{j,i}} = \sum_b \frac{\partial L}{\partial Z_{b,j}} \cdot X_{b,i}
\]

Each weight gradient is a sum over the batch of upstream gradient times the corresponding input. PyTorch computes this via `Z.backward()` without you writing the formula.

### Example 4: Verify with `torch.autograd.grad`

```python
import torch

w1 = torch.tensor(2.0, requires_grad=True)
w2 = torch.tensor(-0.5, requires_grad=True)
x = torch.tensor(1.5)
y_true = torch.tensor(1.0)

h = torch.relu(w1 * x)
y_pred = w2 * h
loss = (y_pred - y_true) ** 2

g_w1, g_w2 = torch.autograd.grad(loss, [w1, w2])
print("dL/dw1:", g_w1.item(), "dL/dw2:", g_w2.item())
```

---

## 7. AI Connection

> 🧠 AI Insight
>
> Transformer training runs this same loop at scale: forward through billions of parameters, `loss.backward()` through attention and MLP blocks, `optimizer.step()` with AdamW. Techniques like gradient clipping, mixed precision, and checkpointing exist because backprop through 100+ layers is fragile and memory-hungry — but the core math is identical to the one-weight example above.

**Stochastic gradient descent (SGD):** Average gradients over mini-batches approximate the full-data gradient. Cheaper, noisier, works.

**Adam / AdamW:** Adaptive learning rates per parameter. Default for transformers. Still requires correct gradients from backprop.

**Autograd:** PyTorch records operations on tensors with `requires_grad=True`. `backward()` implements reverse-mode automatic differentiation — generalized backpropagation.

**Fine-tuning LLMs:** Freeze some layers (`requires_grad=False`), backprop only through unfrozen parameters. Same machinery, fewer gradients computed.

**Regularization:** Weight decay adds \(\lambda \|w\|^2\) to loss; gradient includes extra \(2\lambda w\) term. L2 regularization is implemented in optimizers like AdamW.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> **Forgetting `optimizer.zero_grad()`.** Gradients **accumulate** by default. A second `backward()` without zeroing adds to `.grad`, corrupting the update. Always zero before each backward pass.

> ⚠️ Common Mistake
>
> **Calling `backward()` on a non-scalar loss without `grad_tensors`.** `loss.backward()` expects a scalar. For vector losses, use `.mean()` or `.sum()` first, or supply `grad_tensors`.

> ⚠️ Common Mistake
>
> **In-place operations on tensors needed for gradients.** `x.relu_()` or `x += 1` on tensors in the graph can break autograd. Use out-of-place ops during training unless you know the implications.

> ⚠️ Common Mistake
>
> **Updating weights manually while keeping `requires_grad` without `torch.no_grad()`.** Use `optimizer.step()` or wrap manual updates in `with torch.no_grad():` so you do not build a graph through the update.

**Correct understanding:** Forward builds the graph. `backward()` propagates \(\frac{\partial L}{\partial \cdot}\) backward. Optimizer reads `.grad` and updates parameters.

---

## 9. Exercises

### Easy

1. For \(L = (wx - t)^2\), compute \(\frac{\partial L}{\partial w}\) when \(w=1\), \(x=4\), \(t=10\).
2. What are the five steps in a standard PyTorch training iteration?
3. Run `L.backward()` on a scalar loss and print one parameter's `.grad`.

### Medium

4. Derive \(\frac{\partial L}{\partial w}\) for \(L = \frac{1}{2}(wx + b - t)^2\) with respect to both \(w\) and \(b\).
5. Explain why ReLU gives zero gradient when pre-activation is negative.
6. Train `nn.Linear(1,1)` on \(y = 2x\) data and plot loss vs epoch.

### Hard

7. For \(h = \text{ReLU}(W_1 x + b_1)\), \(\hat{y} = W_2 h + b_2\), write the chain rule for \(\frac{\partial L}{\partial W_1}\) symbolically.
8. Implement manual SGD on a 2-parameter quadratic \(L(w_1, w_2) = w_1^2 + w_2^2\) without `nn` — only tensors and `.backward()`.
9. What does `retain_graph=True` do in `backward()`? When is it needed?

### Challenge

10. **Micro-backprop tracer:** Build a 3-layer MLP, run one batch, and after `backward()` print each layer's weight gradient norm (`w.grad.norm()`). Compare norms across layers — do you see vanishing or exploding patterns with random init?

---

## 10. Mini Project

### Train a Linear Regressor From Scratch

1. Generate 200 points: \(y = 2.5x + 1 + \text{noise}\).
2. Define `nn.Linear(1, 1)`.
3. Train with SGD for 100 epochs; log loss every 10 epochs.
4. Plot data, fitted line, and loss curve.
5. Compare learned `weight` and `bias` to true slope 2.5 and intercept 1.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(200, 1)).astype(np.float32)
Y = (2.5 * X + 1 + 0.1 * rng.standard_normal((200, 1))).astype(np.float32)

Xt = torch.from_numpy(X)
Yt = torch.from_numpy(Y)
model = nn.Linear(1, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.05)

losses = []
for epoch in range(100):
    opt.zero_grad()
    pred = model(Xt)
    loss = nn.functional.mse_loss(pred, Yt)
    loss.backward()
    opt.step()
    losses.append(loss.item())

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X, Y, s=10, alpha=0.5)
x_line = np.linspace(-2, 2, 50)
with torch.no_grad():
    w, b = model.weight.item(), model.bias.item()
plt.plot(x_line, w * x_line + b, "r", linewidth=2, label=f"w={w:.2f}, b={b:.2f}")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.tight_layout()
plt.show()
```

<details>
<summary>Mini project checklist</summary>

- [ ] Data generated with known slope/intercept
- [ ] Training loop with zero_grad, backward, step
- [ ] Learned parameters within 0.2 of truth
- [ ] Loss curve decreasing

</details>

---

## 11. Interview Questions

**Q1:** What is backpropagation, and how does it relate to the chain rule?

**A1:** Backpropagation computes gradients of the loss with respect to all parameters by applying the chain rule backward through the computational graph. Each node passes \(\frac{\partial L}{\partial \text{output}}\) downstream multiplied by local derivatives. It is reverse-mode automatic differentiation specialized to neural networks.

**Q2:** What happens when you call `loss.backward()` in PyTorch?

**A2:** PyTorch traverses the graph from the loss node backward. At each operation it applies the chain rule and accumulates gradients into `.grad` fields of leaf tensors (parameters). Non-leaf tensors may have `.grad` populated if `retain_grad()` was used. The graph is freed unless `retain_graph=True`.

**Q3:** Why do we call `optimizer.zero_grad()` before each backward pass?

**A3:** PyTorch accumulates gradients into `.grad` by default. Without zeroing, gradients from multiple batches or multiple backward calls add together, giving wrong update directions. `zero_grad()` resets all parameter gradients to zero before the next computation.

**Q4:** What is the vanishing gradient problem?

**A4:** In deep networks, repeated multiplication of small derivatives (e.g., sigmoid saturation, deep chains) can shrink gradients exponentially toward early layers. Those layers learn slowly or stop learning. ReLU, residual connections, and careful initialization mitigate this. Transformers also use layer normalization and scaled attention to stabilize gradient flow.

**Q5:** How does backpropagation scale to millions of parameters?

**A5:** One backward pass costs roughly the same order as one forward pass. The chain rule reuses intermediate values from the forward pass. Matrix operations batch thousands of gradient computations. GPUs parallelize these ops. Without backprop, computing each parameter's gradient independently would be prohibitively expensive.

---

## 12. Summary

### Key formulas

| Concept | Formula |
|---------|---------|
| Chain rule | \(\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}\) |
| MSE gradient | \(\frac{\partial}{\partial \hat{y}}\frac{1}{2}(\hat{y}-y)^2 = \hat{y} - y\) |
| SGD update | \(w \leftarrow w - \eta \frac{\partial L}{\partial w}\) |
| ReLU derivative | \(1\) if \(z > 0\), else \(0\) |

### Key terminology

- **Backpropagation** — efficient backward pass gradient computation
- **Computational graph** — DAG of operations built during forward pass
- **Autograd** — PyTorch automatic differentiation engine
- **Learning rate** — step size \(\eta\) in parameter updates
- **zero_grad** — reset accumulated gradients before backward
- **Dead ReLU** — neuron with permanently zero gradient when \(z \leq 0\)

---

## 13. Preview

You now understand how layers learn from data. The next module tackles **transformers** — architectures that replaced recurrence with **attention**. Attention is still matrix multiplication and softmax, trained by the same backpropagation loop. The first transformer chapter introduces the **attention mechanism**: queries, keys, values, and weighted sums that let models focus on relevant inputs.

Neural networks taught you *parameterized functions*. Transformers teach you *dynamic, input-dependent routing* between positions.

**Next chapter:** [Attention Mechanism](../04-transformers/01-attention-mechanism.md)

---
