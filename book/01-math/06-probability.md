# Probability

## 1. Introduction

Machine learning models rarely say "the answer is exactly 7." They say "there is a 92% chance this image is a cat" or "the next token is 'learning' with probability 0.15." **Probability** is the language of uncertainty — and modern AI is built on it.

Classification heads output probability distributions. Language models predict the next token from a distribution over the vocabulary. Diffusion models learn noise distributions. Reinforcement learning estimates expected rewards. Even deterministic-looking training uses probabilistic ideas: mini-batch sampling, dropout, data augmentation.

After this chapter you will be able to:

- Compute basic **probabilities** and interpret them as long-run frequencies.
- Work with **discrete distributions** (coin, die, categorical).
- Understand **softmax** — the bridge from raw scores (logits) to probabilities.
- Explain **cross-entropy loss** and why it punishes confident wrong predictions.
- Simulate random experiments in Python and connect simulation to theory.
- Read classification outputs and language model `logits` with confidence.

**Where this appears in AI:** `F.cross_entropy(logits, target)` is the standard classification loss. `F.softmax(logits, dim=-1)` produces attention weights and token probabilities. Generative models sample from learned distributions. Evaluation metrics (accuracy, perplexity) assume probabilistic predictions.

---

## 2. Intuition

> 💡 Intuition
>
> Probability measures how likely something is, on a scale from 0 (impossible) to 1 (certain). Flip a fair coin: P(heads) = 0.5 means "half the time, over many flips, you expect heads." Probability is not a guarantee for one flip — it describes long-run behavior.

```
  Fair die (6 equally likely faces)

  P(1) = P(2) = ... = P(6) = 1/6

  ┌───┬───┬───┬───┬───┬───┐
  │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │   each face: 1/6 of the area
  └───┴───┴───┴───┴───┴───┘
```

**Model output:** A classifier might output `[0.7, 0.2, 0.1]` for three classes — "70% cat, 20% dog, 10% bird." Those numbers must be non-negative and sum to 1. **Softmax** enforces that from arbitrary scores.

**Cross-entropy:** Measures how surprised you are by the true label given your predicted distribution. Predict 99% cat when the truth is bird → huge loss. Predict 40% cat when truth is cat → moderate loss.

> 🔬 Deep Dive
>
> There are two mainstream interpretations: **frequentist** (probability as long-run frequency) and **Bayesian** (probability as degree of belief). For deep learning practice, the frequentist simulation view plus "outputs are distributions we train with cross-entropy" is enough until you study Bayesian neural networks or uncertainty quantification.

---

## 3. Formal Definitions

### Probability of an event

For a finite set of equally likely outcomes, the **probability** of event \(A\):

\[
P(A) = \frac{\text{number of outcomes in } A}{\text{total number of outcomes}}
\]

| Symbol | Meaning |
|--------|---------|
| \(P(A)\) | Probability of event \(A\), a number in \([0, 1]\) |
| \(A^c\) | Complement — \(A\) does not happen; \(P(A^c) = 1 - P(A)\) |

**Axioms (informal):**

1. \(P(A) \geq 0\)
2. \(P(\text{all outcomes}) = 1\)
3. For mutually exclusive events: \(P(A \cup B) = P(A) + P(B)\)

### Random variable

A **random variable** \(X\) maps outcomes to numbers. Example: die roll \(X \in \{1,2,3,4,5,6\}\).

\[
P(X = k) = \text{probability that } X \text{ equals } k
\]

### Discrete distribution

A list of probabilities \(p_1, p_2, \ldots, p_k\) where each \(p_i \geq 0\) and \(\sum_i p_i = 1\).

**Categorical distribution** over \(k\) classes: \(P(y = i) = p_i\).

### Expectation (preview)

\[
\mathbb{E}[X] = \sum_k k \cdot P(X = k)
\]

Average value over many samples — used in reinforcement learning and loss analysis.

### Softmax

Given logits (raw scores) \(z_1, \ldots, z_k\):

\[
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}
\]

- Output is positive.
- Outputs sum to 1.
- Larger logit → larger probability.
- Invariant to adding the same constant to all logits (numerical stability uses this).

### Cross-entropy

For true class \(y\) (integer index) and predicted probabilities \(\hat{p}\):

\[
H(p, \hat{p}) = -\sum_i p_i \log \hat{p}_i
\]

For **one-hot** true label (only class \(y\) has probability 1):

\[
L = -\log \hat{p}_y
\]

Penalizes low probability on the correct class. Log amplifies punishment for very wrong confident predictions.

---

## 4. Programming Perspective

| Concept | Python |
|---------|--------|
| Fair coin flip | `np.random.random() < 0.5` |
| Die roll | `np.random.randint(1, 7)` |
| Softmax | `np.exp(z) / np.exp(z).sum()` or `scipy.special.softmax` |
| Cross-entropy | `-np.log(p_true_class)` |

```python
import numpy as np
from collections import Counter

# Simulate fair coin
n_flips = 10_000
flips = np.random.random(n_flips) < 0.5
p_heads = flips.mean()
print(f"P(heads) ≈ {p_heads:.4f}")  # ~0.5

# Simulate die
rolls = np.random.randint(1, 7, size=10_000)
counts = Counter(rolls)
p_six = counts[6] / len(rolls)
print(f"P(6) ≈ {p_six:.4f}")  # ~1/6 ≈ 0.1667
```

**Softmax in NumPy:**

```python
def softmax(z):
    z = np.array(z, dtype=float)
    z_shifted = z - z.max()  # numerical stability
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum()

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(probs)       # sums to 1.0
print(probs.sum())
```

**Cross-entropy for one example:**

```python
true_class = 0  # index of correct class
probs = softmax(np.array([2.0, 1.0, 0.1]))
loss = -np.log(probs[true_class])
print(f"cross-entropy loss: {loss:.4f}")
```

PyTorch combines them efficiently:

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.1]])
target = torch.tensor([0])  # class 0
loss = F.cross_entropy(logits, target)
print(loss)  # same as -log(softmax(logits)[0])
```

---

## 5. Visualizations

Probability distributions are often shown as bar charts (discrete) or curves (continuous). Start with discrete.

```python
import numpy as np
import matplotlib.pyplot as plt

# Fair die distribution
faces = np.arange(1, 7)
probs = np.ones(6) / 6

plt.figure(figsize=(8, 4))
plt.bar(faces, probs, color="steelblue", edgecolor="black")
plt.xlabel("die face")
plt.ylabel("P(X = face)")
plt.title("Discrete uniform distribution — fair die")
plt.xticks(faces)
plt.ylim(0, 0.25)
plt.show()
```

**How to read this plot:** Each bar has equal height \(1/6 \approx 0.167\). Total area (sum of bar heights) is 1.

```python
# Softmax: effect of logits
labels = ["cat", "dog", "bird"]
for logits in [[1, 1, 1], [3, 1, 0], [5, 1, 0]]:
    p = softmax(np.array(logits, dtype=float))
    plt.bar(labels, p, alpha=0.7, label=str(logits))

plt.ylabel("probability")
plt.title("Softmax sharpens peaks for larger logits")
plt.legend()
plt.show()
```

When one logit dominates, softmax concentrates probability — the model is "confident." Equal logits → uniform distribution.

```python
# Cross-entropy vs predicted probability on true class
p = np.linspace(0.01, 1, 100)
ce = -np.log(p)
plt.figure(figsize=(7, 4))
plt.plot(p, ce, color="coral", linewidth=2)
plt.xlabel("predicted probability on TRUE class")
plt.ylabel("cross-entropy loss")
plt.title("Low probability on truth → high loss")
plt.grid(True, alpha=0.3)
plt.show()
```

As \(\hat{p}_y \to 0\), loss \(\to \infty\). Confident and wrong is punished severely.

---

## 6. Worked Examples

### Example 1: Fair coin

\(P(\text{heads}) = \frac{1}{2}\), \(P(\text{tails}) = \frac{1}{2}\).

Flip 4 times. Probability of all heads?

**Step 1:** Independent flips (assume): \(P = \frac{1}{2} \times \frac{1}{2} \times \frac{1}{2} \times \frac{1}{2} = \frac{1}{16}\).

### Example 2: Fair die

\(P(X = 6) = \frac{1}{6}\). \(P(X \text{ is even}) = P(\{2,4,6\}) = \frac{3}{6} = \frac{1}{2}\).

```python
rolls = np.random.randint(1, 7, size=100_000)
p_even = np.mean(rolls % 2 == 0)
print(f"P(even) ≈ {p_even:.4f}")
```

### Example 3: Softmax by hand (3 classes)

Logits \(z = [2, 1, 0]\).

**Step 1:** \(e^2 \approx 7.389\), \(e^1 \approx 2.718\), \(e^0 = 1\)

**Step 2:** Sum \(\approx 7.389 + 2.718 + 1 = 11.107\)

**Step 3:** Probabilities:

- \(p_0 \approx 7.389 / 11.107 \approx 0.665\)
- \(p_1 \approx 2.718 / 11.107 \approx 0.245\)
- \(p_2 \approx 1 / 11.107 \approx 0.090\)

```python
print(softmax([2, 1, 0]))
```

### Example 4: Cross-entropy

True class 0, predicted probs \([0.665, 0.245, 0.090]\).

\[
L = -\log(0.665) \approx 0.408
\]

If the model predicted \([0.09, 0.245, 0.665]\) (wrong class most likely):

\[
L = -\log(0.09) \approx 2.41
\]

Much higher loss — model was confidently wrong.

### Example 5: Attention softmax (one row)

Attention scores for one query: \(z = [1.0, 0.5, -1.0]\).

```python
scores = np.array([1.0, 0.5, -1.0])
weights = softmax(scores)
print(weights)  # positive, sums to 1 — attention weights
```

These weights multiply value vectors — the probability chapter meets the vectors chapter meets matrices.

---

## 7. AI Connection

> 🧠 AI Insight
>
> Language models output a vector of logits of length `vocab_size` (often 50k–128k). Softmax turns them into a probability distribution over next tokens. Training maximizes log-probability of the true next token — cross-entropy again. Generation samples or greedily picks from that distribution.

**Classification:** Final layer outputs logits \(\mathbf{z} \in \mathbb{R}^k\). Softmax → \(\hat{p}\). Loss = cross-entropy against one-hot label.

**Attention:** Scores \(\mathbf{z}_i\) for one query position → softmax → weights \(\alpha_{ij}\) summing to 1 over keys. Weighted sum of values.

**Temperature sampling:** Divide logits by temperature \(T\) before softmax. \(T > 1\) → softer, more random; \(T < 1\) → sharper, more deterministic.

**Logits vs probabilities:** Frameworks often use `log_softmax` for numerical stability. `F.cross_entropy` expects raw logits and applies log-softmax internally.

**Perplexity:** \(\exp(\text{average cross-entropy})\) — "effective vocabulary size" the model is confused over. Lower is better.

**Calibration:** A model saying "90% confident" should be right 90% of the time. Training optimizes cross-entropy, not calibration — they are related but not identical.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> **Treating model outputs as probabilities without softmax.** Logits can be negative or greater than 1. Only after softmax (or sigmoid for binary) do you get valid probabilities.

> ⚠️ Common Mistake
>
> **Applying softmax twice.** `F.cross_entropy` already includes log-softmax. Passing probabilities to `cross_entropy` double-transforms and gives wrong gradients.

> ⚠️ Common Mistake
>
> **Probabilities that do not sum to 1.** Check `probs.sum()` when debugging. Softmax guarantees sum 1; manual mistakes or wrong axis (`dim=-1` in PyTorch for sequence models) break this.

> ⚠️ Common Mistake
>
> **Confusing likelihood with probability of truth.** High probability on class "cat" means the model bets on cat — not that the image is definitely a cat. Always pair predictions with uncertainty awareness, especially out-of-distribution inputs.

**Correct understanding:** Probabilities are non-negative and sum to 1. Softmax converts logits. Cross-entropy trains classifiers and language models. Simulation approximates theoretical probabilities.

---

## 9. Exercises

### Easy

1. A bag has 3 red and 7 blue marbles. What is \(P(\text{red})\)?
2. Simulate 1000 coin flips. Estimate \(P(\text{heads})\).
3. Apply softmax to `[0, 0, 0]`. What do you get? Why?
4. If the true class is index 1 and \(\hat{p} = [0.2, 0.5, 0.3]\), compute cross-entropy.

### Medium

5. Simulate 10,000 die rolls. Estimate \(P(X \leq 2)\).
6. Plot softmax output for logits `[a, 0, 0]` as `a` ranges from -5 to 5. How does the first class probability change?
7. Explain why subtracting the max logit before `exp` in softmax does not change the result.
8. Three-class classifier: logits `[1, 2, 3]`. Which class has highest probability? Compute exact softmax.

### Hard

9. Show that cross-entropy is minimized when \(\hat{p}_y = 1\) for the true class \(y\) (among valid probability distributions).
10. Implement `softmax` and `cross_entropy` in NumPy. Verify against PyTorch on random logits.
11. For attention, why do we divide scores by \(\sqrt{d_k}\) before softmax? (Hint: dot product variance grows with dimension.)

### Challenge

12. **Temperature study:** Sample logits `[3, 1, 0.5]`, apply softmax with temperatures 0.5, 1.0, 2.0. Plot distributions. Explain effect on language generation diversity.
13. **Perplexity calculator:** Given a sequence of true next-token indices and model logits per step, compute average cross-entropy and perplexity for a tiny 10-word synthetic sequence.

---

## 10. Mini Project

### Probability Simulator

Build a simulator that:

1. Runs coin, die, and categorical sampling experiments.
2. Compares theoretical vs simulated frequencies in a bar chart.
3. Demonstrates softmax + cross-entropy on a fake 5-class classifier output.
4. Plots cross-entropy vs predicted probability on the true class.
5. Saves a summary figure to `book/assets/06-probability-simulator.png`.

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Die simulation
rolls = np.random.randint(1, 7, size=10_000)
sim_probs = np.bincount(rolls, minlength=7)[1:] / len(rolls)
axes[0].bar(range(1, 7), sim_probs, label="simulated")
axes[0].axhline(1/6, color="red", linestyle="--", label="theory 1/6")
axes[0].set_title("Die rolls")
axes[0].legend()

# Softmax demo
labels = ["A", "B", "C"]
p = softmax([2.0, 1.0, 0.1])
axes[1].bar(labels, p, color="steelblue")
axes[1].set_title("Softmax probabilities")

# Cross-entropy curve
p_true = np.linspace(0.01, 1, 100)
axes[2].plot(p_true, -np.log(p_true), color="coral")
axes[2].set_xlabel("P(true class)")
axes[2].set_title("Cross-entropy")

out = Path("book/assets/06-probability-simulator.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
```

<details>
<summary>Mini project checklist</summary>

- [ ] Coin and/or die simulation with theory comparison
- [ ] Softmax bar chart
- [ ] Cross-entropy curve
- [ ] Figure saved to `book/assets/`

</details>

---

## 11. Interview Questions

**Q1:** What is softmax, and why do we use it in classifiers?

**A1:** Softmax maps a vector of real-valued logits to a valid probability distribution: all outputs are positive and sum to 1. Larger logits receive larger probabilities. Classifiers need probabilistic outputs for interpretation, cross-entropy training, and downstream decision-making. It generalizes the sigmoid to multiple classes.

**Q2:** What is cross-entropy loss?

**A2:** Cross-entropy measures the difference between the true distribution (usually one-hot for the correct class) and the predicted distribution. For a single correct class \(y\), it simplifies to \(-\log \hat{p}_y\). It heavily penalizes confident wrong predictions because log blows up as probability on the true class approaches zero. Minimizing cross-entropy maximizes log-likelihood of the data.

**Q3:** Why does `F.cross_entropy` expect logits, not probabilities?

**A3:** It internally applies log-softmax in a numerically stable way, combining two steps into one kernel. Passing probabilities would apply log to already-softmaxed values incorrectly and cause gradient issues. Always pass raw logits from the model's final linear layer.

**Q4:** How does softmax appear in attention mechanisms?

**A4:** Attention scores for a query against all keys form a vector of logits. Softmax converts them to weights that sum to 1 over keys. Those weights scale and sum value vectors. Dividing by \(\sqrt{d_k}\) before softmax keeps score magnitudes stable as embedding dimension grows.

**Q5:** What is the relationship between maximum likelihood and cross-entropy?

**A5:** Maximizing the likelihood of observed labels under the model is equivalent to minimizing cross-entropy between one-hot true labels and predicted probabilities. Training with `cross_entropy` is maximum likelihood estimation for categorical outcomes — the standard objective for classifiers and language models.

---

## 12. Summary

### Key formulas

| Concept | Formula |
|---------|---------|
| Probability (equally likely) | \(P(A) = \frac{\|A\|}{\|\text{outcomes}\|}\) |
| Complement | \(P(A^c) = 1 - P(A)\) |
| Softmax | \(\text{softmax}(z_i) = e^{z_i} / \sum_j e^{z_j}\) |
| Cross-entropy (one-hot) | \(L = -\log \hat{p}_y\) |
| Full cross-entropy | \(H = -\sum_i p_i \log \hat{p}_i\) |
| Perplexity | \(\exp(\text{avg cross-entropy})\) |

### Key terminology

- **Probability** — number in \([0,1]\) measuring likelihood
- **Random variable** — numeric outcome of a random process
- **Distribution** — assignment of probabilities to outcomes
- **Logit** — raw score before softmax
- **Softmax** — maps logits to a probability distribution
- **Cross-entropy** — loss measuring mismatch between true and predicted distributions
- **Categorical distribution** — distribution over finitely many classes
- **Sampling** — drawing outcomes according to a distribution

---

## 13. Preview

You have completed the core math toolkit: **functions**, **derivatives**, **vectors**, **gradients**, **matrices**, and **probability**. The next module — **PyTorch** — puts these ideas into GPU-accelerated tensors: creation, matrix multiply, reshaping, indexing, and the autograd engine that computes gradients automatically.

When you write `torch.softmax` and `F.cross_entropy`, you are using this chapter. When you write `nn.Linear` and attention, you are using matrices and vectors. When you call `backward()`, you are using gradients.

**Next module:** PyTorch — starting with [Creating Tensors](../02-pytorch/01-creating-tensors.md)

---

## Lab

Companion notebook: [`app/math/06_probability.ipynb`](../../app/math/06_probability.ipynb)
