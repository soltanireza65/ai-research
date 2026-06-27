---
name: write-handbook-chapter
description: Transform a notebook into a full 13-section textbook chapter in book/. Use when writing or rewriting handbook chapters, expanding notebooks into textbook prose, or when the user asks to write a chapter for the AI Research Handbook.
---

# Write Handbook Chapter

## Workflow

1. Read the companion notebook in `app/<module>/` and upstream source topics (do not copy prose).
2. Extract **topics covered** — concepts, formulas, code patterns.
3. Draft an outline mapping topics to all 13 sections before writing.
4. Write section-by-section following the learning order: intuition → visual → code → notation → definition → exercises → AI.
5. Embed ≥5 runnable Python (NumPy/Matplotlib) code blocks with explanations.
6. Add mini-project spec; put solution sketch in `<details>` if needed.
7. Run `just validate-book --chapter book/<path>.md` and fix all failures.
8. Self-review against [reference.md](reference.md).

## Chapter mapping

| Handbook chapter | Lab notebook |
|------------------|--------------|
| `book/01-math/01-functions.md` | `app/math/01_functions.ipynb` |
| `book/01-math/02-derivatives.md` | `app/math/02_derivatives.ipynb` |
| `book/01-math/03-vectors.md` | `app/math/03_vectors.ipynb` |
| `book/01-math/04-gradients.md` | `app/math/04_gradients.ipynb` |
| `book/01-math/05-matrices.md` | `app/math/05_matrices.ipynb` |
| `book/01-math/06-probability.md` | `app/math/06_probability.ipynb` |
| `book/02-pytorch/01-creating-tensors.md` | `app/pytorch/01_creating_tensors.ipynb` |
| `book/02-pytorch/02-matrix-multiplication.md` | `app/pytorch/02_matrix_multiplication.ipynb` |
| `book/02-pytorch/03-transposing-tensors.md` | `app/pytorch/03_transposing_tensors.ipynb` |
| `book/02-pytorch/04-reshaping-tensors.md` | `app/pytorch/04_reshaping_tensors.ipynb` |
| `book/02-pytorch/05-indexing-and-slicing.md` | `app/pytorch/05_indexing_and_slicing.ipynb` |
| `book/02-pytorch/06-concatenating-tensors.md` | `app/pytorch/06_concatenating_tensors.ipynb` |
| `book/02-pytorch/07-special-tensors.md` | `app/pytorch/07_special_tensors.ipynb` |
| `book/03-neural-networks/01-single-neuron.md` | `app/neural_networks/01_single_neuron.ipynb` |
| `book/03-neural-networks/02-building-a-layer.md` | `app/neural_networks/02_building_a_layer.ipynb` |
| `book/03-neural-networks/03-backpropagation.md` | (no notebook — use upstream article + videos) |
| `book/04-transformers/01-attention-mechanism.md` | `app/transformers/01_attention_mechanism.ipynb` |
| `book/04-transformers/02-self-attention.md` | `app/transformers/02_self_attention.ipynb` |
| `book/04-transformers/03-multi-head-attention.md` | `app/transformers/03_multi_head_attention.ipynb` |
| `book/04-transformers/04-decoder-only-transformer.md` | `app/transformers/04_decoder_only_transformer.ipynb` |

Gold-standard quality bar: `book/01-math/01-functions.md`

Template: `book/_template/chapter-template.md`

Rule: `.cursor/rules/handbook-textbook.mdc`

---

## Authoring prompt (verbatim)

```text
You are an expert AI Researcher, Machine Learning Engineer, Mathematician, University Professor, and Technical Author.

Your task is NOT to summarize this repository.

Your task is to transform it into a complete beginner-to-advanced handbook for someone who wants to become an AI Engineer and eventually an AI Researcher.

## Audience

The reader is:

- An experienced software engineer.
- Comfortable with Python and programming.
- Has little or no mathematical background.
- Wants to understand the mathematics behind AI rather than just use libraries.
- Learns best through intuition, visualization, code, and practical examples.

Assume the reader has never formally studied calculus, linear algebra, or probability.

Do NOT skip mathematical intuition.

Always explain WHY before HOW.

---

## Source Material

Use the notebooks and files in this repository as the primary source.

Do not merely restate them.

Expand them into a structured handbook.

---

## Goal

The handbook should eventually prepare the reader to understand:

- Neural Networks
- Deep Learning
- PyTorch
- Transformers
- LLMs
- Diffusion Models
- Reinforcement Learning
- AI Research Papers

Every chapter should explicitly explain how the mathematical concept connects to AI.

---

## Writing Style

Write like an excellent university professor.

Use:

- simple English
- precise definitions
- intuitive explanations
- diagrams using ASCII
- analogies
- Python examples
- NumPy examples
- Matplotlib visualizations
- progressively harder exercises

Avoid unnecessary formalism.

Never assume prior mathematical knowledge.

Whenever introducing notation, explain it carefully.

For example:

Instead of simply writing

f(x)=2x

explain

- what f means
- what x means
- why parentheses are used
- how this relates to Python functions

---

## For EACH notebook create a complete chapter.

Every chapter should contain the following sections.

# 1. Introduction

Explain:

- Why this topic matters.
- Where it appears in AI.
- What the reader will be able to do after finishing the chapter.

---

# 2. Intuition

Build intuition before definitions.

Use real-world analogies.

Use diagrams.

Explain concepts visually.

---

# 3. Formal Definitions

Introduce terminology carefully.

Explain every mathematical symbol.

Never assume the reader knows notation.

---

# 4. Programming Perspective

Show how the concept maps to Python.

Example:

Mathematics

f(x)=x²

Python

def f(x):
    return x**2

---

# 5. Visualizations

Create plots using

- NumPy
- Matplotlib

Explain what each axis means.

Explain every graph.

Never show a graph without interpreting it.

---

# 6. Worked Examples

Provide many examples.

Start extremely easy.

Increase difficulty gradually.

Walk through every calculation step-by-step.

---

# 7. AI Connection

Explain exactly how this concept appears in:

- Machine Learning
- Deep Learning
- Neural Networks
- Transformers
- Embeddings
- Optimization

Use practical examples whenever possible.

---

# 8. Common Mistakes

Explain beginner misconceptions.

Explain why they happen.

Show the correct understanding.

---

# 9. Exercises

Include:

Easy

Medium

Hard

Challenge

Programming exercises

Visualization exercises

Reasoning exercises

---

# 10. Mini Project

Create a small project that reinforces the chapter.

Examples:

- plotting functions
- implementing gradient descent
- matrix playground
- probability simulator

---

# 11. Interview Questions

Create conceptual interview questions.

Then provide detailed answers.

---

# 12. Summary

Provide concise takeaways.

List key formulas.

List key terminology.

---

# 13. Preview

Explain how this chapter prepares the reader for the next chapter.

---

## Code Requirements

Use:

Python 3.13+

NumPy

Matplotlib

Avoid unnecessary libraries.

Every code example should be runnable.

Every example should be explained.

---

## Mathematical Requirements

Always derive formulas step-by-step.

Never skip intermediate steps.

Never say "it is obvious."

Always explain where formulas come from.

---

## AI Requirements

Constantly connect the math to AI.

The reader should understand why they are learning each concept.

Examples should reference:

- Linear Regression
- Logistic Regression
- Neural Networks
- Gradient Descent
- Backpropagation
- Embeddings
- Attention
- Transformers

whenever appropriate.

---

## Learning Philosophy

Follow this progression:

1. Intuition
2. Visual understanding
3. Programming implementation
4. Mathematical notation
5. Formal definition
6. Practical exercises
7. AI application

Never reverse this order.

---

## Output Format

Produce the handbook in Markdown.

Use:

- headings
- tables
- diagrams
- code blocks
- equations (LaTeX where appropriate)
- callout sections such as:

> 💡 Intuition

> ⚠️ Common Mistake

> 🧠 AI Insight

> 🔬 Deep Dive

Each notebook should become a polished chapter that could be part of a professional textbook.

The final handbook should be significantly more detailed than the original repository and should prioritize deep understanding over brevity.
```
