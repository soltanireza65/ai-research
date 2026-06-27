"""Generate handbook lab notebooks from definitions."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


NOTEBOOKS: dict[str, list[dict]] = {
    "app/math/01_functions.ipynb": [
        md("# Functions Lab\n\nCompanion to `book/01-math/01-functions.md`."),
        code(
            "import numpy as np\nimport matplotlib.pyplot as plt\n\n"
            "from app.utils.plot import setup_plot_style\n\nsetup_plot_style()"
        ),
        md("## Linear: y = 2x"),
        code(
            "x = np.linspace(0, 10, 100)\ny = 2 * x\n\n"
            "plt.plot(x, y, label='y = 2x')\nplt.xlabel('x')\nplt.ylabel('y')\n"
            "plt.title('Linear function')\nplt.legend()\nplt.show()"
        ),
        md("## Quadratic: y = x²"),
        code(
            "x = np.linspace(-3, 3, 100)\ny = x**2\n\n"
            "plt.plot(x, y, label='y = x²')\nplt.xlabel('x')\nplt.ylabel('y')\n"
            "plt.title('Quadratic function')\nplt.legend()\nplt.show()"
        ),
        md("## Try it\n\nPlot `y = 3x - 1` from x = 0 to 10."),
        code("# Your code here\n"),
    ],
    "app/math/02_derivatives.ipynb": [
        md("# Derivatives Lab\n\nCompanion to `book/01-math/02-derivatives.md`."),
        code(
            "import numpy as np\nimport matplotlib.pyplot as plt\n\n"
            "from app.utils.plot import setup_plot_style\n\nsetup_plot_style()\n\n\n"
            "def numerical_derivative(f, x, h=0.001):\n"
            "    return (f(x + h) - f(x - h)) / (2 * h)"
        ),
        md("## Derivative of y = 2x is constant 2"),
        code(
            "def f(x):\n    return 2 * x\n\nx = np.linspace(0, 10, 100)\n"
            "plt.plot(x, f(x), label='f(x)=2x')\n"
            "plt.plot(x, np.full_like(x, 2), '--', label=\"f'(x)=2\")\n"
            "plt.legend()\nplt.show()\n\nprint('numerical at 3:', numerical_derivative(f, 3.0))"
        ),
        md("## Try it\n\nEstimate the derivative of `x³` at `x=2` numerically."),
        code("# Your code here\n"),
    ],
    "app/math/03_vectors.ipynb": [
        md("# Vectors Lab\n\nCompanion to `book/01-math/03-vectors.md`."),
        code(
            "import numpy as np\nimport matplotlib.pyplot as plt\n\n"
            "from app.utils.plot import setup_plot_style\n\nsetup_plot_style()"
        ),
        md("## 2D vectors"),
        code(
            "v1 = np.array([3, 4])\nv2 = np.array([-2, 1])\n\n"
            "fig, ax = plt.subplots()\n"
            "ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='b', label='v1')\n"
            "ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='r', label='v2')\n"
            "ax.set_xlim(-4, 5)\nax.set_ylim(-1, 5)\nax.set_aspect('equal')\n"
            "ax.legend()\nplt.show()\n\nprint('||v1|| =', np.linalg.norm(v1))"
        ),
        md("## Try it\n\nAdd v1 and v2, then plot the result."),
        code("# Your code here\n"),
    ],
    "app/math/04_gradients.ipynb": [
        md("# Gradients Lab\n\nCompanion to `book/01-math/04-gradients.md`."),
        code(
            "import numpy as np\nimport matplotlib.pyplot as plt\n\n"
            "from app.utils.plot import setup_plot_style\n\nsetup_plot_style()"
        ),
        md("## Gradient of f(x,y) = x² + y²"),
        code(
            "def gradient_f(x, y):\n    return np.array([2 * x, 2 * y])\n\n"
            "for pt in [(0, 0), (1, 1), (2, -1)]:\n"
            "    print(pt, '->', gradient_f(*pt))\n\n"
            "x = np.linspace(-2, 2, 8)\ny = np.linspace(-2, 2, 8)\n"
            "X, Y = np.meshgrid(x, y)\nU, V = 2 * X, 2 * Y\n\n"
            "plt.quiver(X, Y, U, V)\nplt.gca().set_aspect('equal')\n"
            "plt.title('Gradient field of x² + y²')\nplt.show()"
        ),
        md("## Try it\n\nCompute the gradient of f(x,y) = xy at (2, 3)."),
        code("# Your code here\n"),
    ],
    "app/math/05_matrices.ipynb": [
        md("# Matrices Lab\n\nCompanion to `book/01-math/05-matrices.md`."),
        code("import numpy as np"),
        md("## Shapes and multiplication"),
        code(
            "A = np.array([[1, 2], [3, 4]])\nB = np.array([[5, 6]])\n"
            "print('A shape:', A.shape)\nprint('B shape:', B.shape)\n"
            "print('A @ B.T =\\n', A @ B.T)"
        ),
        md("## Try it\n\nMultiply a 2×3 matrix by a 3×2 matrix."),
        code("# Your code here\n"),
    ],
    "app/math/06_probability.ipynb": [
        md("# Probability Lab\n\nCompanion to `book/01-math/06-probability.md`."),
        code("import numpy as np\nfrom collections import Counter"),
        md("## Coin and die"),
        code(
            "p_heads = 1 / 2\np_six = 1 / 6\n"
            "print('P(heads) =', p_heads)\nprint('P(die=6) =', p_six)\n\n"
            "rolls = np.random.randint(1, 7, size=10_000)\ncounts = Counter(rolls)\n"
            "print('Simulated P(6) ≈', counts[6] / len(rolls))"
        ),
        md("## Try it\n\nSimulate 1000 coin flips."),
        code("# Your code here\n"),
    ],
    "app/pytorch/01_creating_tensors.ipynb": [
        md("# Creating Tensors Lab"),
        code("import torch"),
        code(
            "a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])\n"
            "b = torch.zeros(2, 3)\nc = torch.randn(4)\n"
            "print(a.shape, a.dtype)\nprint(b)\nprint(c)"
        ),
        md("## Try it\n\nCreate a 3×3 identity tensor."),
        code("# Your code here\n"),
    ],
    "app/pytorch/02_matrix_multiplication.ipynb": [
        md("# Matrix Multiplication Lab"),
        code("import torch"),
        code(
            "A = torch.tensor([[1., 2., 3.], [4., 5., 6.]])\n"
            "B = torch.tensor([[1.], [2.], [3.]])\n"
            "print('A @ B =\\n', A @ B)\nprint('shape:', (A @ B).shape)"
        ),
        md("## Try it\n\nBatch matmul: `(2, 3, 4) @ (4, 5)`."),
        code("# Your code here\n"),
    ],
    "app/pytorch/03_transposing_tensors.ipynb": [
        md("# Transposing Tensors Lab"),
        code("import torch"),
        code(
            "A = torch.arange(6).reshape(2, 3)\n"
            "print(A)\nprint('A.T =\\n', A.T)\nprint('shape:', A.T.shape)"
        ),
        md("## Try it\n\nTranspose a 3D tensor along dims 0 and 2."),
        code("# Your code here\n"),
    ],
    "app/pytorch/04_reshaping_tensors.ipynb": [
        md("# Reshaping Tensors Lab"),
        code("import torch"),
        code(
            "x = torch.arange(12).reshape(2, 2, 3)\n"
            "print('original:', x.shape)\nprint('flatten:', x.flatten().shape)\n"
            "print('unsqueeze:', x.unsqueeze(0).shape)\n"
            "print('squeeze:', torch.tensor([[[1], [2]]]).squeeze().shape)"
        ),
        md("## Try it\n\nReshape `(4, 6)` to `(2, 3, 4)`."),
        code("# Your code here\n"),
    ],
    "app/pytorch/05_indexing_and_slicing.ipynb": [
        md("# Indexing and Slicing Lab"),
        code("import torch"),
        code(
            "x = torch.arange(20).reshape(4, 5)\nprint(x)\n"
            "print('row 1:', x[1])\nprint('col 2:', x[:, 2])\n"
            "print('mask:', x[x > 10])"
        ),
        md("## Try it\n\nSelect the 2×2 top-left submatrix."),
        code("# Your code here\n"),
    ],
    "app/pytorch/06_concatenating_tensors.ipynb": [
        md("# Concatenating Tensors Lab"),
        code("import torch"),
        code(
            "a = torch.ones(2, 3)\nb = torch.zeros(2, 3)\n"
            "print('cat dim=0:', torch.cat([a, b], dim=0).shape)\n"
            "print('stack dim=0:', torch.stack([a, b], dim=0).shape)"
        ),
        md("## Try it\n\nConcatenate three vectors of length 4."),
        code("# Your code here\n"),
    ],
    "app/pytorch/07_special_tensors.ipynb": [
        md("# Special Tensors Lab"),
        code("import torch"),
        code(
            "print(torch.eye(3))\nprint(torch.linspace(0, 1, 5))\n"
            "print(torch.arange(0, 10, 2))\nprint(torch.randn(3))"
        ),
        md("## Try it\n\nCreate a 5×5 random normal matrix."),
        code("# Your code here\n"),
    ],
    "app/neural_networks/01_single_neuron.ipynb": [
        md("# Single Neuron Lab"),
        code("import torch\n\ndef relu(x):\n    return torch.maximum(x, torch.tensor(0.0))"),
        code(
            "x = torch.tensor([3.0, 4.0])\nw = torch.tensor([2.0, -1.0])\n"
            "b = torch.tensor(0.5)\n\nz = (x * w).sum() + b\ny = relu(z)\n"
            "print('pre-activation:', z.item())\nprint('output:', y.item())"
        ),
        md("## Try it\n\nChange weights and observe the output."),
        code("# Your code here\n"),
    ],
    "app/neural_networks/02_building_a_layer.ipynb": [
        md("# Building a Layer Lab"),
        code("import torch"),
        code(
            "in_features, out_features = 4, 3\nbatch = 2\n\n"
            "x = torch.randn(batch, in_features)\n"
            "W = torch.randn(out_features, in_features)\n"
            "b = torch.randn(out_features)\n\ny = x @ W.T + b\n"
            "print('input:', x.shape)\nprint('output:', y.shape)\nprint(y)"
        ),
        md("## Try it\n\nImplement ReLU on the layer output."),
        code("# Your code here\n"),
    ],
    "app/transformers/01_attention_mechanism.ipynb": [
        md("# Attention Mechanism Lab"),
        code("import torch\nimport torch.nn.functional as F"),
        code(
            "Q = torch.tensor([[1., 0.], [0., 1.]])\n"
            "K = torch.tensor([[1., 0.], [0., 1.]])\n"
            "V = torch.tensor([[2., 3.], [4., 5.]])\n\n"
            "d_k = Q.shape[-1]\nscores = Q @ K.T / (d_k ** 0.5)\n"
            "weights = F.softmax(scores, dim=-1)\nout = weights @ V\n"
            "print('weights:\\n', weights)\nprint('output:\\n', out)"
        ),
        md("## Try it\n\nChange Q so row 0 attends more to key 1."),
        code("# Your code here\n"),
    ],
    "app/transformers/02_self_attention.ipynb": [
        md("# Self-Attention Lab"),
        code("import torch\nimport torch.nn.functional as F"),
        code(
            "seq_len, d_model = 4, 8\nx = torch.randn(seq_len, d_model)\n"
            "Wq = Wk = Wv = torch.randn(d_model, d_model)\n\n"
            "Q, K, V = x @ Wq, x @ Wk, x @ Wv\n"
            "scores = Q @ K.T / (d_model ** 0.5)\n"
            "weights = F.softmax(scores, dim=-1)\nout = weights @ V\n"
            "print('attention weights shape:', weights.shape)\n"
            "print('output shape:', out.shape)"
        ),
        md("## Try it\n\nPrint which position each query attends to most."),
        code("# Your code here\n"),
    ],
    "app/transformers/03_multi_head_attention.ipynb": [
        md("# Multi-Head Attention Lab"),
        code("import torch\nimport torch.nn.functional as F"),
        code(
            "seq_len, d_model, num_heads = 4, 8, 2\nd_head = d_model // num_heads\n"
            "x = torch.randn(1, seq_len, d_model)\n\n"
            "Q = x.view(1, seq_len, num_heads, d_head).transpose(1, 2)\n"
            "K = Q.clone()\nV = Q.clone()\n"
            "scores = Q @ K.transpose(-2, -1) / (d_head ** 0.5)\n"
            "weights = F.softmax(scores, dim=-1)\nout = weights @ V\n"
            "print('per-head weights shape:', weights.shape)"
        ),
        md("## Try it\n\nConcatenate head outputs and project with a linear layer."),
        code("# Your code here\n"),
    ],
    "app/transformers/04_decoder_only_transformer.ipynb": [
        md("# Decoder-Only Transformer Lab"),
        code("import torch\nimport torch.nn.functional as F"),
        code(
            "seq_len = 5\nd_model = 8\n\n"
            "mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()\n"
            "scores = torch.randn(seq_len, seq_len)\n"
            "scores = scores.masked_fill(mask, float('-inf'))\n"
            "weights = F.softmax(scores, dim=-1)\n"
            "print('causal attention weights (row 0):', weights[0])\n"
            "print('weights sum per row:', weights.sum(dim=-1))"
        ),
        md("## Try it\n\nVerify position 3 cannot attend to position 4."),
        code("# Your code here\n"),
    ],
}


def main() -> None:
    for rel_path, cells in NOTEBOOKS.items():
        path = ROOT / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(nb(cells), indent=1) + "\n")
        print("wrote", rel_path)


if __name__ == "__main__":
    main()
