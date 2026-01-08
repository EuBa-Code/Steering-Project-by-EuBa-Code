# LLM Steering with Activation Engineering

This project demonstrates the technique of **Activation Engineering** (also known as *Representation Engineering*) on Large Language Models (LLMs) using PyTorch and Hugging Face Transformers. 

It is designed as an educational tool to understand how to intervene on the internal state of a model to influence its output without the need for fine-tuning or retraining.

## 🚀 How it works

1. **Activations as Representation**: We assume that the internal activations of an LLM at specific layers represent complex concepts (e.g., sentiment, honesty, topics).
2. **Vector Arithmetic**: By contrasting the activations of two opposing prompts (e.g., "Love" vs "Hate"), we extract a "direction" vector that represents that specific difference in the model's latent space.
3. **Forward Hooking**: During the generation process, we use PyTorch hooks to inject this vector (addition or subtraction) into the model's residual stream, "steering" the generation towards the desired concept.

## 🧠 The Neural Analogy: AI Steering as Neurostimulation

A compelling way to understand Activation Engineering is through the lens of neuroscience. In biological brains, techniques like **Transcranial Magnetic Stimulation (TMS)** or **Deep Brain Stimulation (DBS)** are used to modulate neural activity.

* **Non-Invasive Modulation**: Just as TMS influences specific brain regions with magnetic pulses without changing the brain's physical structure, Steering modulates the "residual stream" without changing the model's weights.
* **Real-time Intervention**: We are not "teaching" the model new things (Learning); we are "stimulating" existing latent representations to change the model's behavioral state in real-time.
* **Circuit Level Control**: We act as "digital neurosurgeons," identifying the specific layers (circuits) where a concept is most salient and applying a precise "vectorial impulse."

## 📚 References & Scientific Background

This project is built upon recent breakthroughs in **Mechanistic Interpretability**. For a deeper understanding of the theory, please refer to these foundational papers:

* **[Representation Engineering (RepE)](https://arxiv.org/abs/2310.01405)**: *Zou et al. (2023)*. This is arguably the most important paper on the subject, introducing a top-down approach to AI transparency.
* **[Activation Addition (ActAdd)](https://arxiv.org/abs/2308.10248)**: *Turner et al. (2023)*. The foundational paper that popularized steering language models without optimization using simple vector addition.
* **[The Geometry of Truth](https://arxiv.org/abs/2310.18166)**: *Marks & Tegmark (2023)*. A key study showing how "truthfulness" is represented as a linear direction in the activation space of LLMs.

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

----------------------------------------------------------------------------------------------------

💻 Usage
Explore the implementation using the provided Jupyter notebook:
    ```bash
    jupyter notebook demo.ipynb

The notebook guides you through:

1. Loading a pre-trained model (e.g., GPT-2).

2. Computing a steering vector for a specific concept.

3. Generating text and comparing the steered vs. unsteered output.

------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------

### 🧪 Experimental Insights & Best Practices

Through empirical testing on **GPT-2 Small**, we have identified several key factors that influence the effectiveness of activation steering:

#### 1. The "Golden" Layers
*   **Layers 7 and 8**: These layers consistently perform best for semantic concepts like sentiment or genre.
*   **Rationale**: Early layers focus heavily on syntax and low-level token features, while the final layers are too biased towards immediate token prediction. Middle layers (7-8) represent the "conceptual core" where abstract meanings are solidified.

#### 2. Vector Purity (Signal-to-Noise)
*   **Symmetry is key**: Using short, symmetric pairs of prompts (e.g., `"I love it"` vs `"I hate it"`) produces much cleaner steering vectors. 
*   **Grammar Noise**: Longer, varied sentences often introduce "grammatical noise" into the vector, which can lead to incoherent generations or failure to steer the model effectively.

#### 3. Power Scaling (Magnitude)
Since the implementation normalizes the steering vector to a unit length (norm = 1.0), the `multiplier` (force) becomes a predictable control:
*   **Force < 10.0**: Subtle effect, often overcome by the model's natural bias.
*   **Force 30.0 - 50.0**: The "Sweet Spot". The model follows the steering direction while maintaining high linguistic coherence.
*   **Force > 80.0**: Excessive perturbation. The model may begin generating repetitive or nonsensical text as activations are pushed too far out of their natural distribution.

--------------------------------------------------------------------------------------------------------

Created for educational purposes in the field of AI Safety and Interpretability.



