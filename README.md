# LLM Steering with Activation Engineering

This project demonstrates the technique of **Activation Engineering** (also known as *Representation Engineering*) on Large Language Models (LLMs) using PyTorch and Hugging Face Transformers. 

It is designed as an educational tool to understand how to intervene on the internal state of a model to influence its output without the need for fine-tuning or retraining.

## 🚀 How it works

1. **Activations as Representation**: We assume that the internal activations of an LLM at specific layers represent complex concepts (e.g., sentiment, honesty, topics).
2. **Vector Arithmetic**: By contrasting the activations of two opposing prompts (e.g., "Love" vs "Hate"), we extract a "direction" vector that represents that specific difference in the model's latent space.
3. **Forward Hooking**: During the generation process, we use PyTorch hooks to inject this vector (addition or subtraction) into the model's residual stream, "steering" the generation towards the desired concept.

## 📚 References & Scientific Background

This project is built upon recent breakthroughs in **Mechanistic Interpretability**. For a deeper understanding of the theory, please refer to these foundational papers:

* **Activation Addition (ActAdd)**: *Turner et al. (2023)* - [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248). This work demonstrates how adding steering vectors to the residual stream can control model behavior.
* **Representation Engineering (RepE)**: *Zou et al. (2023)* - [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405). A comprehensive framework for extracting and manipulating concept vectors.
* **The Geometry of Truth**: *Marks & Tegmark (2023)* - [The Geometry of Truth: Emergent Learned Representations of Truthfulness in LLMs](https://arxiv.org/abs/2310.18166). This study provides evidence that concepts like "truth" are represented as linear directions in activation space.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd Steering

2. **Prerequisites**:

An NVIDIA GPU (e.g., RTX 4070 Super) with CUDA support is highly recommended.

Python 3.9+

3. **Setup Environment**:
   ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On Linux/Mac

4. Install Dependencies: This project requires PyTorch optimized for CUDA 12.4.
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
    pip install transformers datasets ipykernel jupyter

💻 Usage
Explore the implementation using the provided Jupyter notebook:
    ```bash
    jupyter notebook demo.ipynb

The notebook guides you through:

1. Loading a pre-trained model (e.g., GPT-2).

2. Computing a steering vector for a specific concept.

3. Generating text and comparing the steered vs. unsteered output.

🧪 Example Results
Prompt: "I think that this film is"

No Steering: "...okay, but it lacks depth in the second act."

Positive Steering (+Vector): "...absolutely fantastic, a true masterpiece of modern cinema!"

Negative Steering (-Vector): "...a complete disaster and a total waste of time."

Created for educational purposes in the field of AI Safety and Interpretability.



