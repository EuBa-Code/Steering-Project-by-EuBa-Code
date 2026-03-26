"""High-level wrapper that makes any GPT-2-family model steerable."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .hooks import HookManager
from .vectors import SteeringVector


class SteerableModel:
    """Load a causal LM and steer its generation via activation engineering.

    Typical workflow::

        model = SteerableModel("gpt2")
        vec   = model.extract_vector("I love it", "I hate it", layer_idx=7)
        out   = model.generate("I think this is", steering_vector=vec,
                               layer_idx=7, multiplier=40.0)

    Args:
        model_name: Any Hugging Face causal-LM identifier (default ``"gpt2"``).
        device: ``"cuda"``, ``"cpu"``, or ``None`` for auto-detection.
    """

    def __init__(self, model_name: str = "gpt2", device: str | None = None) -> None:
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"

        print(f"Loading {model_name} on {self.device} ...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

        self._num_layers: int = len(self.model.transformer.h)
        self._hook_manager = HookManager()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_vector(
        self,
        positive_text: str | list[str],
        negative_text: str | list[str],
        layer_idx: int,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Compute a steering vector from contrasting concepts.

        When lists are provided the activations are averaged, which produces
        a cleaner, more robust direction than a single prompt pair.

        Args:
            positive_text: Prompt(s) representing the target concept.
            negative_text: Prompt(s) representing the opposite concept.
            layer_idx: Transformer block index to extract activations from.
            normalize: If ``True`` (recommended), normalize to unit length.

        Returns:
            A 1-D tensor of shape ``(hidden_dim,)``.

        Raises:
            ValueError: If inputs are empty or of unexpected type.
            IndexError: If *layer_idx* is out of range.
        """
        self._validate_layer(layer_idx)

        act_pos = self._avg_activation(positive_text, layer_idx)
        act_neg = self._avg_activation(negative_text, layer_idx)

        vector = SteeringVector.from_diff(act_pos, act_neg)

        if normalize:
            vector = F.normalize(vector, dim=-1)

        return vector

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        steering_vector: torch.Tensor | None = None,
        layer_idx: int | None = None,
        multiplier: float = 0.0,
    ) -> str:
        """Generate text, optionally steered by a vector.

        Args:
            prompt: The input text.
            max_new_tokens: Maximum number of tokens to generate.
            steering_vector: Direction tensor (from :meth:`extract_vector`).
            layer_idx: Layer at which to inject the vector.
            multiplier: Scaling factor — positive steers *toward* the concept,
                negative steers *away*. Recommended range: 30–50.

        Returns:
            The decoded generated string (prompt + continuation).
        """
        self._hook_manager.remove_all_hooks()

        if steering_vector is not None and layer_idx is not None and multiplier != 0:
            self._validate_layer(layer_idx)
            layer = self._get_layer(layer_idx)
            self._hook_manager.register_steering_hook(
                layer, layer_idx, steering_vector, multiplier
            )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        self._hook_manager.remove_all_hooks()
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    @property
    def num_layers(self) -> int:
        """Total number of transformer blocks in the loaded model."""
        return self._num_layers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_layer(self, layer_idx: int) -> torch.nn.Module:
        """Return the transformer block at *layer_idx* (GPT-2 layout)."""
        return self.model.transformer.h[layer_idx]

    def _run_and_capture(self, text: str, layer_idx: int) -> torch.Tensor:
        """Run a forward pass and return the last-token activation at *layer_idx*."""
        self._hook_manager.remove_all_hooks()

        layer = self._get_layer(layer_idx)
        self._hook_manager.register_capture_hook(layer, layer_idx)

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model(**inputs)

        activation = self._hook_manager.get_activation(layer_idx)
        self._hook_manager.remove_all_hooks()
        return activation

    def _avg_activation(
        self, text_input: str | list[str], layer_idx: int
    ) -> torch.Tensor:
        """Return the (averaged) last-token activation for *text_input*."""
        if isinstance(text_input, str):
            return self._run_and_capture(text_input, layer_idx)

        if isinstance(text_input, list):
            if not text_input:
                raise ValueError("Prompt list must not be empty.")
            activations = [self._run_and_capture(t, layer_idx) for t in text_input]
            return torch.stack(activations).mean(dim=0)

        raise TypeError(
            f"Expected str or list[str], got {type(text_input).__name__}"
        )

    def _validate_layer(self, layer_idx: int) -> None:
        """Raise ``IndexError`` if *layer_idx* is out of range."""
        if not 0 <= layer_idx < self._num_layers:
            raise IndexError(
                f"layer_idx={layer_idx} is out of range for a model "
                f"with {self._num_layers} layers (0–{self._num_layers - 1})."
            )
