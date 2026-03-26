<p align="center">
  <h1 align="center">Activation Steering for LLMs</h1>
  <p align="center">
    <em>Steer language model behavior by intervening on internal representations — no fine-tuning required.</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white" alt="Python 3.9+">
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white" alt="Transformers">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  </p>
</p>

---

## What is Activation Steering?

**Activation Engineering** (also called *Representation Engineering*) is a technique that modifies the internal activations of a language model at inference time to influence its outputs — without retraining or fine-tuning.

This project provides a clean, modular Python toolkit to:

1. **Extract** a steering vector by contrasting two opposing concepts (e.g. *"love"* vs *"hate"*)
2. **Inject** that vector into the model's residual stream during generation
3. **Control** the steering intensity with a single scalar multiplier

### How it works

```
Positive prompts  ──►  Activation(pos)  ─┐
                                          ├──  diff  ──►  Steering Vector  ──►  Inject at Layer N
Negative prompts  ──►  Activation(neg)  ─┘                                      during generation
```

1. **Activations as Representations** — Internal hidden states at specific layers encode semantic concepts (sentiment, genre, tone, ...).
2. **Vector Arithmetic** — The difference `act(positive) − act(negative)` isolates the direction of that concept in latent space.
3. **Forward Hooking** — A PyTorch forward hook adds `vector × multiplier` to the residual stream at a chosen layer, steering every subsequent token.

## The Neural Analogy

A useful way to think about this: in neuroscience, techniques like **Transcranial Magnetic Stimulation (TMS)** and **Deep Brain Stimulation (DBS)** modulate neural activity without changing the brain's structure.

Activation steering does the same to a transformer:

| Neuroscience | Activation Steering |
|---|---|
| Magnetic/electric pulse | Steering vector |
| Target brain region | Target transformer layer |
| Stimulation intensity | Multiplier parameter |
| No structural change | No weight updates |

We act as *digital neurosurgeons* — identifying the layer where a concept is most salient and applying a precise vectorial impulse.

## Quick Start

```python
from steering_lib import SteerableModel

# Load model (auto-detects GPU)
model = SteerableModel("gpt2")

# Extract a steering vector at layer 7
vector = model.extract_vector(
    positive_text="I love this movie because",
    negative_text="I hate this movie because",
    layer_idx=7,
)

# Generate with steering
print(model.generate("I think this film is",
                      steering_vector=vector, layer_idx=7, multiplier=40.0))
```

For robust vectors, pass **lists of prompts** — the activations are averaged to reduce noise:

```python
import prompts

vector = model.extract_vector(
    positive_text=prompts.FANTASY_PROMPTS,
    negative_text=prompts.SCIFI_PROMPTS,
    layer_idx=8,
)
```

## Example Results

> **Prompt:** *"In the box, I found a"* — Layer 8, Force 40.0, Fantasy vs Sci-Fi

| | Output |
|---|---|
| **Baseline** | *"...a set of a variety of accessories with a small space."* |
| **+ Fantasy** | *"...a very nice long, deep wooden sword. It's carved from stone, and I can see the hilt coming out."* |
| **- Fantasy (Sci-Fi)** | *"...a few basic features, including: USB-C for USB devices to connect to and communicate with devices."* |

> **Prompt:** *"Hey, I need"* — Layer 7, Force 40.0, Formal vs Street

| | Output |
|---|---|
| **+ Formal** | *"...some assistance with the information that you have reported on our site and the information that has been provided under this Policy."* |
| **- Formal (Street)** | *"...a bed!" "I was talking on my cellphone before you came out!" "Just go, I'll be okay!"* |

## Installation

**Prerequisites:** Python 3.9+ and an NVIDIA GPU with CUDA support (recommended).

```bash
git clone https://github.com/EuBa-Code/Steering-Project-by-EuBa-Code.git
cd Steering-Project-by-EuBa-Code
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

> **Note:** If you need a specific CUDA version of PyTorch, install it first following the [official instructions](https://pytorch.org/get-started/locally/), then run `pip install -e ".[dev]"`.

## Usage

The project includes two Jupyter notebooks:

| Notebook | Description |
|---|---|
| `demo.ipynb` | Minimal end-to-end example (extract vector, steer generation) |
| `playground.ipynb` | Advanced experiments: robust multi-prompt vectors, layer sweeps |

```bash
jupyter notebook demo.ipynb
```

## Experimental Insights

Findings from empirical testing on **GPT-2 Small** (12 layers, 768-dim hidden state):

### Optimal Layers

**Layers 7–8** consistently perform best for semantic steering. Early layers (0–3) focus on syntax; final layers (10–11) are biased toward immediate next-token prediction. The middle layers encode the *conceptual core*.

### Vector Quality

Short, **symmetric prompt pairs** produce the cleanest vectors. Longer sentences introduce grammatical noise that dilutes the target concept. Using **lists of 5–10 prompts** and averaging activations further improves signal-to-noise ratio.

### Multiplier Scaling

Since vectors are normalized to unit length, the multiplier directly controls intensity:

| Range | Effect |
|---|---|
| < 10 | Subtle — often overridden by the model's natural bias |
| **30 – 50** | **Sweet spot** — strong steering with coherent output |
| > 80 | Degradation — repetitive or nonsensical text |

## Project Structure

```
steering_lib/
    __init__.py          # Public API: SteerableModel, SteeringVector
    model.py             # High-level model wrapper and generation
    hooks.py             # PyTorch forward-hook management
    vectors.py           # Steering vector representation and math
prompts.py               # Curated prompt collections for experiments
demo.ipynb               # Quick-start notebook
playground.ipynb         # Advanced experiments and layer sweeps
```

## References

This project builds on recent work in **Mechanistic Interpretability**:

### Foundational Theory
- **Representation Engineering (RepE)** — Zou et al. (2023) — [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)
- **Activation Addition (ActAdd)** — Turner et al. (2023) — [arXiv:2308.10248](https://arxiv.org/abs/2308.10248)
- **Scaling Monosemanticity** — Templeton et al., Anthropic (2024) — SAEs isolating high-level concepts in latent space

### Technical Deep Dives
- **The Geometry of Truth** — Marks & Tegmark (2023) — [arXiv:2310.18166](https://arxiv.org/abs/2310.18166)
- **Concept Activation Vectors (CAV)** — LessWrong (2023) — Mathematical intuition behind steering behaviors

### Practical Implementations
- **LLM-Steer-Instruct** — Microsoft (2024) — Steering instruction-tuned models
- **Eiffel Tower Llama** — David Louapre — Community experiment on forced concept injection
- **mHC (Manifold-Constrained Hyper-Connections)** — DeepSeek-AI (2025) — Stabilizing residual stream architectures

## License

[MIT](LICENSE) — Created for educational purposes in the field of AI Safety and Interpretability.
