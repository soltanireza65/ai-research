# Decoder-Only Transformer

## 1. Introduction

A **decoder-only transformer** is the architecture family behind GPT-style language models. It reads a sequence of tokens from left to right and learns one central task: predict the next token from the tokens that came before it. That simple training objective scales surprisingly far. With enough data, compute, and careful engineering, next-token prediction produces models that can summarize, translate, write code, reason through problems, call tools, and follow instructions.

The word **decoder** comes from the original transformer paper, where the model had an encoder that read an input sequence and a decoder that generated an output sequence. GPT-style models keep only the decoder side. They do not have a separate encoder; instead, every token position attends only to earlier positions and itself. This constraint is called **causal masking** because the model is not allowed to see the future.

After this chapter you will be able to:

- Explain why decoder-only models use a triangular causal mask.
- Build the core data flow: token IDs → embeddings → masked self-attention → MLP → logits.
- Connect logits and cross-entropy loss to next-token prediction.
- Read a small PyTorch implementation of a decoder block without treating it as magic.
- Understand how this chapter connects single neurons, layers, backpropagation, and attention into a working LLM architecture.

**Where this appears in AI:** GPT, LLaMA, Mistral, Qwen, Gemma, and many other large language models are decoder-only transformers. Their blocks repeat the same pattern: masked multi-head self-attention, residual connection, layer normalization, feed-forward network, another residual connection. The details vary, but the learning problem is the same: use the past context to predict what comes next.

---

## 2. Intuition

> 💡 Intuition
>
> Think of a decoder-only transformer as an autocomplete system reading a sentence through a sliding privacy screen. At position 4, it may read positions 0, 1, 2, 3, and 4, but the screen hides positions 5 and beyond. The model must guess the next token using only information that would be available during real generation.

Here is the core idea:

```
tokens:     The   cat   sat   on   the   mat
position:    0     1     2    3     4     5

query at position 3 ("on") may attend to:

             The   cat   sat   on   the   mat
allowed:     yes   yes   yes   yes   no    no
```

If position 3 could attend to `"the"` or `"mat"` while training, the task would become dishonest. The model would learn to copy future information instead of learning the structure of language. During generation, those future tokens do not exist yet, so the shortcut would fail.

The causal mask is the rule that enforces this:

```
         key position
       0   1   2   3   4
q 0    ✓   ✗   ✗   ✗   ✗
u 1    ✓   ✓   ✗   ✗   ✗
e 2    ✓   ✓   ✓   ✗   ✗
r 3    ✓   ✓   ✓   ✓   ✗
y 4    ✓   ✓   ✓   ✓   ✓
```

Rows are **query positions**: the token currently asking, "What information should I use?" Columns are **key positions**: tokens that may be looked at. The check marks form a lower triangle because each position can look backward, not forward.

> 🔬 Deep Dive
>
> Decoder-only transformers are not just "attention layers stacked together." Each block mixes information in two different ways. Masked self-attention mixes information **across positions**: token 7 can use token 2. The MLP mixes information **within each position**: after attention creates a contextual vector for token 7, the MLP transforms that vector dimension by dimension into a richer representation. The repeated alternation of across-position mixing and per-position computation is a major reason the architecture works so well.

---

## 3. Formal Definitions

Let the input sequence be a list of token IDs:

\[
\mathbf{t} = [t_0, t_1, \ldots, t_{n-1}]
\]

Here:

- \(\mathbf{t}\) is the whole token sequence.
- \(t_i\) is the token ID at position \(i\).
- \(n\) is the sequence length, also called context length for one training example.

A token ID is an integer index into a vocabulary. If the vocabulary size is \(V\), then each token ID is in the range \(0\) to \(V-1\).

### Token and position embeddings

A decoder-only transformer converts token IDs into vectors:

\[
\mathbf{x}_i = \mathbf{E}_{token}[t_i] + \mathbf{E}_{pos}[i]
\]

Breaking this down:

- \(\mathbf{E}_{token}\) is the token embedding table with shape \(V \times d_{model}\).
- \(\mathbf{E}_{pos}\) is the position embedding table with shape \(n_{max} \times d_{model}\).
- \(d_{model}\) is the width of the model representation.
- \(\mathbf{x}_i\) is the vector representation at position \(i\).

The token embedding says what the token is. The position embedding says where the token appears. Without positional information, self-attention would see the same set of tokens but not their order.

### Causal self-attention

For each position, the model computes queries, keys, and values:

\[
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q,\quad
\mathbf{K} = \mathbf{X}\mathbf{W}_K,\quad
\mathbf{V} = \mathbf{X}\mathbf{W}_V
\]

Here \(\mathbf{X}\) is the matrix of input vectors for all positions, with shape \(n \times d_{model}\). The matrices \(\mathbf{W}_Q\), \(\mathbf{W}_K\), and \(\mathbf{W}_V\) are learned parameters.

The raw attention score between query position \(i\) and key position \(j\) is:

\[
s_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}
\]

The causal rule is:

\[
s_{ij} =
\begin{cases}
s_{ij}, & j \le i \\
-\infty, & j > i
\end{cases}
\]

After masking, softmax turns scores into attention weights:

\[
a_{ij} = \frac{e^{s_{ij}}}{\sum_{m=0}^{i} e^{s_{im}}}
\]

Only positions \(m \le i\) appear in the denominator because future positions were set to \(-\infty\), and \(e^{-\infty}\) behaves like zero.

### Next-token logits and loss

At every position, the model outputs a vector of **logits**:

\[
\mathbf{z}_i \in \mathbb{R}^{V}
\]

The logit \(\mathbf{z}_{i,k}\) is the model's raw score for token \(k\) being the next token after position \(i\). Softmax converts logits into probabilities:

\[
p_{i,k} = \frac{e^{z_{i,k}}}{\sum_{\ell=0}^{V-1} e^{z_{i,\ell}}}
\]

If the true next token is \(t_{i+1}\), the cross-entropy loss at position \(i\) is:

\[
L_i = -\log p_{i,t_{i+1}}
\]

Training minimizes the average loss across many positions and many sequences.

---

## 4. Programming Perspective

Mathematically, the decoder-only model is a function:

\[
f([t_0, t_1, \ldots, t_{n-1}]) \rightarrow [\mathbf{z}_0, \mathbf{z}_1, \ldots, \mathbf{z}_{n-1}]
\]

In Python terms, this is a module whose `forward` method receives integer token IDs and returns logits.

```python
import torch
import torch.nn as nn

batch_size = 2
seq_len = 5
vocab_size = 100
d_model = 16

token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

token_embedding = nn.Embedding(vocab_size, d_model)
position_embedding = nn.Embedding(seq_len, d_model)

positions = torch.arange(seq_len)
x = token_embedding(token_ids) + position_embedding(positions)

print(token_ids.shape)  # (batch_size, seq_len)
print(x.shape)          # (batch_size, seq_len, d_model)
```

The input tensor contains integers. The output of the embedding step contains floating-point vectors. This is the first important shift: the model cannot do gradient descent directly on token IDs, but it can learn embedding vectors and all later weights.

Now build the causal mask from the lab notebook:

```python
import torch

seq_len = 5

mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
print(mask)

scores = torch.randn(seq_len, seq_len)
masked_scores = scores.masked_fill(mask, float("-inf"))
weights = torch.softmax(masked_scores, dim=-1)

print(weights)
print(weights.sum(dim=-1))
```

The horizontal axis of the attention matrix is the key position being attended to. The vertical axis is the query position doing the attending. `torch.triu(..., diagonal=1)` selects the upper triangle above the main diagonal: exactly the future positions that must be hidden.

Here is a minimal masked attention function for one batch of vectors:

```python
import torch
import torch.nn.functional as F

def masked_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)

    seq_len = Q.shape[-2]
    future_mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
    scores = scores.masked_fill(future_mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    out = weights @ V
    return out, weights


x = torch.randn(1, 5, 8)
Wq = torch.randn(8, 8)
Wk = torch.randn(8, 8)
Wv = torch.randn(8, 8)

Q = x @ Wq
K = x @ Wk
V = x @ Wv
out, weights = masked_attention(Q, K, V)

print(out.shape)      # (1, 5, 8)
print(weights.shape)  # (1, 5, 5)
```

The output shape matches the input sequence shape, except each position now contains information gathered from earlier positions.

---

## 5. Visualizations

The most important visualization is the causal mask. In the plot below, dark cells mark blocked future attention. The x-axis is the key position. The y-axis is the query position. A dark cell at row 1, column 4 means "token at position 1 may not read token at position 4."

```python
import numpy as np
import matplotlib.pyplot as plt

seq_len = 8
mask = np.triu(np.ones((seq_len, seq_len)), k=1)

plt.figure(figsize=(5, 5))
plt.imshow(mask, cmap="gray_r")
plt.title("Causal Mask: Future Positions Are Blocked")
plt.xlabel("key position (token being read)")
plt.ylabel("query position (token asking)")
plt.xticks(range(seq_len))
plt.yticks(range(seq_len))
plt.colorbar(label="1 = blocked, 0 = allowed")
plt.show()
```

Read this graph row by row. Row 0 has every future column blocked because the first token can only attend to itself. Row 7 has no blocked columns because the last token in this window may attend to all previous tokens and itself.

Now visualize attention weights after masking:

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)
seq_len = 8
scores = torch.randn(seq_len, seq_len)
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
weights = F.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)

plt.figure(figsize=(6, 5))
plt.imshow(weights.numpy(), cmap="viridis")
plt.title("Masked Attention Weights")
plt.xlabel("key position")
plt.ylabel("query position")
plt.colorbar(label="attention probability")
plt.show()
```

The color bar shows probability mass. Each row sums to 1 because softmax normalizes across key positions. All cells above the diagonal are zero after masking, so no probability mass leaks into the future.

A second useful visualization is how a model generates one token at a time:

```python
tokens = ["The", "cat", "sat", "on"]

for step in range(4):
    visible = tokens[: step + 1]
    hidden = ["?"] * (len(tokens) - step - 1)
    print(f"step {step}: visible={visible}, hidden={hidden}")
```

The printed sequence shows the generation constraint. At each step, the model only has the prefix. Training uses the same constraint so the model's training environment matches generation.

---

## 6. Worked Examples

### Example 1: Build a causal mask by hand

For sequence length 4, start with a \(4 \times 4\) grid. Row \(i\) can attend to columns \(j\) when \(j \le i\).

Allowed matrix:

\[
\begin{bmatrix}
1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1
\end{bmatrix}
\]

Blocked matrix, the one we pass to `masked_fill`, is the opposite above the diagonal:

\[
\begin{bmatrix}
0 & 1 & 1 & 1 \\
0 & 0 & 1 & 1 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0
\end{bmatrix}
\]

```python
import torch

seq_len = 4
blocked = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
allowed = ~blocked

print("blocked:")
print(blocked.int())
print("allowed:")
print(allowed.int())
```

### Example 2: Check that future weights are zero

Suppose row 2 tries to attend to positions 0, 1, 2, and 3. Position 3 is future, so it must receive probability 0.

```python
import torch
import torch.nn.functional as F

scores = torch.tensor([0.1, 0.2, 2.0, 10.0])
mask_for_row_2 = torch.tensor([False, False, False, True])

weights = F.softmax(scores.masked_fill(mask_for_row_2, float("-inf")), dim=-1)
print(weights)
print("future probability:", weights[3].item())
```

Even though the future score was large, masking removes it before softmax. This matters because unmasked training would reward the model for using information that generation cannot provide.

### Example 3: Align logits with next tokens

If the input is `[10, 20, 30, 40]`, then the model prediction at position 0 is trained against token `20`, position 1 against `30`, and position 2 against `40`. The final position has no next token inside this short example, so many training loops drop it.

```python
import torch
import torch.nn.functional as F

batch = torch.tensor([[10, 20, 30, 40]])
vocab_size = 50

logits = torch.randn(1, 4, vocab_size)

pred_logits = logits[:, :-1, :]  # positions 0, 1, 2
targets = batch[:, 1:]           # tokens 20, 30, 40

loss = F.cross_entropy(
    pred_logits.reshape(-1, vocab_size),
    targets.reshape(-1),
)

print(pred_logits.shape)
print(targets.shape)
print(loss.item())
```

This shift is the heart of language model training. The input and target come from the same text, just offset by one position.

### Example 4: A tiny decoder block

A production block has more details, but the skeleton is compact:

```python
import torch
import torch.nn as nn

class TinyDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.mlp(x))
        return x


block = TinyDecoderBlock(d_model=16, num_heads=4)
x = torch.randn(2, 5, 16)
print(block(x).shape)
```

The residual additions `x + attn_out` and `x + self.mlp(x)` preserve a direct path for information and gradients. Layer normalization stabilizes the scale of activations before the next computation.

---

## 7. AI Connection

> 🧠 AI Insight
>
> Decoder-only transformers turn language modeling into supervised learning without hand-labeled examples. Every document becomes training data because each token supplies the label for the previous position. This is why next-token prediction scales: the internet contains enormous amounts of raw text, and the target is already embedded in the sequence.

In machine learning terms, a decoder-only transformer learns a conditional distribution:

\[
P(t_{i+1} \mid t_0, t_1, \ldots, t_i)
\]

This reads: the probability of the next token given the prefix. During generation, the model samples or selects from this distribution, appends the chosen token, and repeats.

In neural network terms, the model is a large differentiable function. Token embeddings, attention projections, MLP weights, and output projection weights are all learned by gradient descent. Backpropagation flows from the cross-entropy loss through logits, through every decoder block, and back into embeddings.

In transformer terms, causal masking is what separates decoder-only self-attention from bidirectional encoder attention. A BERT-style encoder can look left and right because it is usually trained for representation learning. A GPT-style decoder must preserve left-to-right generation because it produces text autoregressively.

In optimization terms, the training loop is familiar:

```python
import torch
import torch.nn.functional as F

optimizer = torch.optim.AdamW(block.parameters(), lr=3e-4)
lm_head = torch.nn.Linear(16, 100)
tokens = torch.randint(0, 100, (2, 6))
x = torch.randn(2, 6, 16)

hidden = block(x)
logits = lm_head(hidden)
loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, 100), tokens[:, 1:].reshape(-1))

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(loss.item())
```

This example uses random hidden states rather than a full model, but the learning mechanics are the same: compute logits, compare them to next-token targets, call `loss.backward()`, and update parameters.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> A frequent mistake is to create the causal mask with the wrong triangle. If you block the lower triangle instead of the upper triangle, every token can see the future but not the past. The code may still run, and the loss may even drop, but the model is learning an invalid task for autoregressive generation.

Another mistake is confusing **sequence length** with **vocabulary size**. Sequence length is how many token positions are in the context window. Vocabulary size is how many possible token IDs the model can predict at each position. Attention matrices are usually \(n \times n\). Logit matrices are \(n \times V\).

A third mistake is treating the final position's logits as useless in every setting. During fixed-window training, the final position may not have a target inside the window, so we often drop it. During generation, the final position is exactly the one we use to choose the next token.

A fourth mistake is thinking the model "understands" text because attention weights are easy to visualize. Attention weights are useful diagnostics, but they are not a complete explanation of model behavior. The MLP, residual stream, layer norms, embeddings, and output projection all participate in the final prediction.

---

## 9. Exercises

### Easy

1. Create a causal mask for sequence lengths 3, 5, and 8. Print the mask as integers and identify which positions are blocked for row 2.
2. Given token IDs `[4, 9, 1, 7]`, write the next-token targets used for positions 0, 1, and 2.
3. Run the notebook mask example and verify that each attention row sums to 1 after softmax.

### Medium

1. Write a function `make_causal_mask(seq_len, device)` that returns a boolean upper-triangular mask.
2. Modify the masked attention function so it accepts a batch of inputs with shape `(batch, seq_len, d_model)`.
3. Plot random masked attention weights and explain what the x-axis, y-axis, and color intensity mean.

### Hard

1. Implement a tiny decoder block using separate linear layers for \(Q\), \(K\), and \(V\) instead of `nn.MultiheadAttention`.
2. Create a toy vocabulary of characters, turn a string into token IDs, and build input-target pairs for next-character prediction.
3. Compare the loss when you train with a correct causal mask versus no mask on a tiny synthetic dataset. Explain why the unmasked version can cheat.

### Challenge

1. Implement a two-layer decoder-only transformer for character-level modeling on a short text file.
2. Add greedy generation: feed a prefix, take the argmax next token, append it, and repeat for 100 steps.
3. Replace learned positional embeddings with sinusoidal or rotary-style positional information. Explain what changed and what stayed the same.

---

## 10. Mini Project

Build a **tiny character-level decoder-only transformer**.

Project requirements:

1. Choose a short text corpus such as a README file or a few paragraphs of public-domain text.
2. Build a character vocabulary and encode the text into integer IDs.
3. Sample random windows of length `block_size + 1`. Use the first `block_size` characters as input and the next `block_size` characters as targets.
4. Implement token embeddings, position embeddings, two decoder blocks, layer normalization, and an output projection.
5. Train with cross-entropy loss and `AdamW`.
6. Generate text from a short prefix every few hundred steps.

Success criteria: the generated text will not be polished, but it should gradually move from random characters toward recognizable local patterns such as spaces, punctuation, common letters, and repeated words. The goal is not to build a production model. The goal is to see the full language-modeling loop end to end.

<details>
<summary>Solution sketch</summary>

Start with a single file script. Keep `d_model=64`, `num_heads=4`, `num_layers=2`, and `block_size=64`. Use batches of random contiguous windows. In `forward`, return both logits and optional loss. During generation, crop the context to the last `block_size` tokens before each forward pass, then sample from `softmax(logits[:, -1, :])`.

</details>

---

## 11. Interview Questions

**Q1:** Why does a decoder-only transformer need a causal mask?

**A1:** It needs a causal mask because the model is trained to predict the next token from the previous tokens. If position \(i\) could attend to position \(i+1\) or later, training would leak the answer. The model would learn a shortcut that is unavailable during generation. The causal mask keeps training aligned with inference by allowing each position to attend only to itself and earlier positions.

**Q2:** What is the difference between attention weights and logits?

**A2:** Attention weights are probabilities over positions in the input sequence. They answer, "Which previous tokens should this position read from?" Logits are raw scores over vocabulary items. They answer, "Which token should come next?" Attention weights usually have shape `(batch, heads, seq_len, seq_len)`, while logits usually have shape `(batch, seq_len, vocab_size)`.

**Q3:** Why are input tokens shifted when computing language-model loss?

**A3:** The model receives tokens up to a position and predicts the following token. If the input is `[t0, t1, t2, t3]`, then the prediction at position 0 is compared with `t1`, position 1 with `t2`, and position 2 with `t3`. This one-token shift turns ordinary text into supervised examples without external labels.

**Q4:** What role do residual connections play in decoder blocks?

**A4:** Residual connections let each block add a learned update to the existing representation instead of replacing it completely. This helps preserve information across many layers and gives gradients a more direct route backward during training. Deep transformers would be much harder to optimize without this pathway.

---

## 12. Summary

Decoder-only transformers are GPT-style language models trained with next-token prediction. The architecture uses token embeddings, position embeddings, masked multi-head self-attention, MLPs, residual connections, layer normalization, and an output projection to vocabulary logits.

### Key formulas

- Token and position representation:
  \[
  \mathbf{x}_i = \mathbf{E}_{token}[t_i] + \mathbf{E}_{pos}[i]
  \]
- Scaled attention score:
  \[
  s_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}
  \]
- Causal rule:
  \[
  j > i \Rightarrow s_{ij} = -\infty
  \]
- Softmax probability:
  \[
  p_{i,k} = \frac{e^{z_{i,k}}}{\sum_{\ell=0}^{V-1} e^{z_{i,\ell}}}
  \]
- Next-token loss:
  \[
  L_i = -\log p_{i,t_{i+1}}
  \]

### Key terminology

- **Decoder-only transformer:** Autoregressive transformer architecture used by GPT-style models.
- **Causal mask:** Upper-triangular mask that prevents attention to future positions.
- **Autoregressive generation:** Producing one token at a time, feeding each new token back into the context.
- **Logits:** Raw vocabulary scores before softmax.
- **Cross-entropy loss:** Loss that penalizes low probability assigned to the true next token.
- **Residual stream:** The main representation pathway updated by attention and MLP blocks.

---

## 13. Preview

This chapter completes the first pass through transformer mechanics: attention, self-attention, multi-head attention, and decoder-only generation. From here, you are ready to study training details that make real LLMs work at scale: normalization variants, initialization, optimizers, learning-rate schedules, tokenization choices, efficient attention kernels, inference-time sampling, and alignment methods such as supervised fine-tuning and reinforcement learning from feedback.

---

## Lab

Companion notebook: [`app/transformers/04_decoder_only_transformer.ipynb`](../../app/transformers/04_decoder_only_transformer.ipynb)
