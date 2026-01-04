# LLM Steering with Activation Engineering

This project demonstrates the technique of **Activation Engineering** (a form of "Steering") on Large Language Models (LLMs) using PyTorch and Hugging Face Transformers.

It is designed as an educational tool to understand how to intervene on the internal state of a model to influence its output without fine-tuning.

## How it works

1.  **Activations as Representation**: We assume that the internal activations of an LLM at specific layers represent concepts (e.g., sentiment, topics).
2.  **Vector Arithmetic**: By contrasting the activations of two opposing prompts (e.g., "Love" vs "Hate"), we can extract a "direction" vector that represents that difference.
3.  **Forward Hooking**: During generation, we use PyTorch hooks to inject this vector (add or subtract) into the model's activations at the same layer, "steering" the generation towards one concept or the other.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd Steering
    ```

2.  **Prerequisites**:
    - An NVIDIA GPU with CUDA support is highly recommended for performance (tested with CUDA 12.4).
    - Python 3.9+

3.  **Setup Environment**:
    It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On Linux/Mac
    ```

4.  **Install Dependencies**:
    This project requires PyTorch with CUDA support. The `requirements.txt` is configured to automatically download the correct version for CUDA 12.4.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The download may be large (~2.5GB) as it includes the CUDA-enabled PyTorch binaries.*

## Usage

Open the `demo.ipynb` notebook in Jupyter:

```bash
jupyter notebook demo.ipynb
```

Follow the steps to:
1.  Load GPT-2.
2.  Compute a steering vector (e.g., for sentiment).
3.  Generate text with the steered model.

## Example

**Prompt**: "I think that this film is"

*   **No Steering**: "...okay, but it lacks depth..."
*   **Positive Steering**: "...absolutely fantastic and a masterpiece..."
*   **Negative Steering**: "...a complete disaster and a waste of time..."

## Credits

This project explores concepts popularized by researchers in the AI Alignment and Interpretability community (e.g., work on Activation Engineering by Turner et al.).
