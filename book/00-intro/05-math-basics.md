# Math Basics

> Shorter intro chapter. Read this **before** [Functions](../01-math/01-functions.md) if school math feels rusty or scary.

## Learning Objectives

- Use Python to practice arithmetic habits the handbook assumes (exponents, negatives, fractions).
- Read a simple graph: what the x-axis and y-axis mean.
- Recognize `e`, `log`, `sin`, and `cos` as names you will meet later — without mastering them yet.
- Pass the **Readiness for Functions** checklist at the end.

## Content

You do not need a math degree to learn AI. You **do** need a handful of school-math habits so symbols like \(x^2\) and `x ** 2` do not stop you on page one. This chapter is that bridge — entirely in plain language with Python as ground truth.

If you can write Python, you can learn these ideas the same way you learned syntax: small steps, run the code, observe the output.

### Order of operations (PEMDAS)

When an expression mixes `+`, `-`, `*`, and `**`, Python follows standard rules:

1. Parentheses first
2. Exponents (`**`)
3. Multiply and divide (left to right)
4. Add and subtract (left to right)

```python
print(2 + 3 * 4)      # 14, not 20 — multiply before add
print((2 + 3) * 4)    # 20 — parentheses first
print(2 ** 3)         # 8 — two cubed
```

> 💡 Intuition
>
> Think of PEMDAS as Python's evaluation order. If you are unsure, add parentheses and run the code. Parentheses are always allowed.

### Negative numbers

A negative sign means “opposite direction” on a number line. Multiplying two negatives gives a positive.

```python
print(-3 + 5)         # 2
print(-3 * -2)        # 6
print((-3) ** 2)      # 9 — square removes the sign
```

Slopes and derivatives use negatives constantly (“going downhill”). Getting comfortable here saves pain in Chapter 2.

### Exponents

\(x^2\) means `x * x`. \(x^3\) means `x * x * x`. In Python we write `x ** 2`, `x ** 3`.

```python
x = 4
print(x ** 2)   # 16
print(x ** 0.5) # 2.0 — square root is a fractional exponent
```

The handbook uses \(f(x) = x^2\) and the power rule \(d/dx\, x^n = n x^{n-1}\). Exponents are not optional — but `**` in Python makes them concrete.

> **Plain English**
> Squaring a number means multiplying it by itself.

> **Python**
> `y = x ** 2`

### Fractions and “rise over run”

A fraction is one number divided by another. **Slope** is “rise over run”: how much vertical change per one step horizontal.

```python
rise = 6
run = 3
slope = rise / run
print(slope)  # 2.0
```

Derivatives are a limit of rise-over-run on a curve. You do not need limits yet — just know that `/` between two changes is a rate.

### Coordinates and axes

A **graph** plots input (usually **x**, horizontal) against output (usually **y**, vertical).

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 50)
y = 2 * x

plt.plot(x, y)
plt.xlabel("x (input)")
plt.ylabel("y = 2x (output)")
plt.title("A line: double the input")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.show()
```

**How to read the plot:** pick a point on the line. Move right (larger x) — does y go up or down? Here, y always goes up: slope is positive.

> ⚠️ Common Mistake
>
> Treating the vertical axis as the input. In this handbook, **x is almost always input** (bottom) and **y or f(x) is output** (left side). When in doubt, read the axis labels.

### The number \(e\) and `np.exp` (preview)

Later chapters use \(e^x\) in sigmoid and softmax. **\(e\)** is a special constant ≈ 2.718. In Python:

```python
import numpy as np

print(np.e)           # 2.718...
print(np.exp(1))      # e^1 ≈ 2.718
print(np.exp(2))      # e^2 ≈ 7.389
```

> 📌 Preview — optional for now
>
> **Term:** \(e\), exponential
> **One line:** A constant base for growth/decay; `np.exp(x)` computes \(e^x\)
> **Learn properly in:** [Probability](../01-math/06-probability.md) (softmax) and [Single Neuron](../03-neural-networks/01-single-neuron.md) (sigmoid)
> You can skip formulas using \(e\) until those chapters.

### Logarithms (preview)

**Log** answers: “what exponent gives this number?” Cross-entropy loss uses `np.log`. For now:

```python
import numpy as np

p = 0.5
print(np.log(p))   # negative — log of a probability below 1
```

> 📌 Preview — optional for now
>
> **Term:** logarithm, `log`
> **One line:** Inverse of exp; punishes tiny probabilities in classification loss
> **Learn properly in:** [Probability](../01-math/06-probability.md)

### Sine and cosine (preview)

`sin` and `cos` are **wavy** functions that repeat. Positional encodings in transformers use them. You do not need trig identities — just know they exist and you can plot them:

```python
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 2 * np.pi, 200)
plt.plot(t, np.sin(t), label="sin(t)")
plt.plot(t, np.cos(t), label="cos(t)")
plt.xlabel("t")
plt.legend()
plt.title("Wavy functions (preview)")
plt.show()
```

> 📌 Preview — optional for now
>
> **Term:** sin, cos, trigonometry
> **One line:** Smooth waves used to encode position in sequences
> **Learn properly in:** [Special Tensors](../02-pytorch/07-special-tensors.md) and [Decoder-Only Transformer](../04-transformers/04-decoder-only-transformer.md)

> 🔬 Deep Dive
>
> \(\pi\) (pi) ≈ 3.14159 is the ratio of a circle's circumference to its diameter. `np.pi` in Python. One full wave of `sin` spans \(0\) to \(2\pi\). You do not need to memorize this — plots are enough for Part I.

> 🧠 AI Insight
>
> Every “scary” math symbol in later chapters is either arithmetic (`+`, `*`, `**`), a named function (`exp`, `log`, `sin`), or shorthand for “do this to every element in an array.” Python makes the last case explicit. When notation confuses you, write the Python equivalent with small numbers.

### Readiness for Functions

Before starting [Functions](../01-math/01-functions.md), you should be able to say **yes** to each:

1. I can predict the output of `2 + 3 * 4` without a calculator.
2. I know `x ** 2` means `x * x`.
3. I can explain which axis is horizontal (input) on a plot.
4. I can write `plt.xlabel` and `plt.ylabel` with meaningful words.
5. I know negative slope means “going downhill” on a graph.
6. I have seen `np.exp` and `np.log` and know they are previews for later.
7. I have seen `sin`/`cos` plots and know they are previews for later.
8. I will use the [Vocabulary Roadmap](04-vocabulary-roadmap.md) when a word bothers me.

If any item is **no**, reread the matching section above and run the code cells in [`app/math/00_math_basics.ipynb`](../../app/math/00_math_basics.ipynb).

## Summary

- **PEMDAS** and parentheses — trust Python or add `()` when unsure.
- **Exponents** — `x ** n` in Python, \(x^n\) on paper.
- **Graphs** — x is input (horizontal), y is output (vertical); always read labels.
- **\(e\), log, sin, cos** — previews only; full lessons come in Probability, PyTorch, and Transformers.
- **Next step:** [Functions](../01-math/01-functions.md) when the readiness list above is solid.

## Further Reading

- [Prerequisites](02-prerequisites.md) — Python and environment setup
- [Vocabulary Roadmap](04-vocabulary-roadmap.md) — jargon map for the whole book
- [Functions cheatsheet](../01-math/01-functions-cheatsheet.md) — skim after Functions, not before
- [Math Basics cheatsheet](05-math-basics-cheatsheet.md) — one-page review of this chapter
