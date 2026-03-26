"""Low-level PyTorch forward-hook management for activation capture and injection."""

from __future__ import annotations

from typing import Any

import torch


class HookManager:
    """Register, fire, and clean up PyTorch forward hooks.

    This class is the low-level engine behind :class:`SteerableModel`.  It
    handles two operations:

    * **Capture** — record the last-token hidden state at a given layer.
    * **Inject** — add a scaled steering vector to the hidden state during
      generation.
    """

    def __init__(self) -> None:
        self._registered_hooks: list[torch.utils.hooks.RemovableHook] = []
        self._captured_activations: dict[int, torch.Tensor] = {}

        # Active steering state
        self._active_steering_vector: torch.Tensor | None = None
        self._target_layer_idx: int | None = None
        self._multiplier: float = 0.0

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def capture_activation(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        layer_idx: int,
    ) -> None:
        """Forward-hook callback that saves the last-token hidden state.

        GPT-2 blocks return ``(hidden_state, present_key_values)``; we
        extract ``hidden_state[0, -1, :]`` (batch 0, last token).
        """
        hidden_state = output[0] if isinstance(output, tuple) else output
        self._captured_activations[layer_idx] = hidden_state[0, -1, :].detach().cpu()

    def inject_steering(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        layer_idx: int,
    ) -> Any:
        """Forward-hook callback that adds the steering perturbation.

        The perturbation is ``vector * multiplier`` and is broadcast across
        the full ``[batch, seq_len, hidden_dim]`` tensor.
        """
        if self._active_steering_vector is None or layer_idx != self._target_layer_idx:
            return output

        if isinstance(output, tuple):
            hidden_state, *rest = output
        else:
            hidden_state = output
            rest = []

        perturbation = (
            self._active_steering_vector.to(hidden_state.device) * self._multiplier
        )
        steered = hidden_state + perturbation

        return (steered, *rest) if isinstance(output, tuple) else steered

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def register_capture_hook(
        self, layer: torch.nn.Module, layer_idx: int
    ) -> None:
        """Attach a capture hook to *layer*."""
        handle = layer.register_forward_hook(
            lambda m, i, o: self.capture_activation(m, i, o, layer_idx)
        )
        self._registered_hooks.append(handle)

    def register_steering_hook(
        self,
        layer: torch.nn.Module,
        layer_idx: int,
        vector: torch.Tensor,
        multiplier: float,
    ) -> None:
        """Attach a steering-injection hook to *layer*."""
        self._active_steering_vector = vector
        self._target_layer_idx = layer_idx
        self._multiplier = multiplier

        handle = layer.register_forward_hook(
            lambda m, i, o: self.inject_steering(m, i, o, layer_idx)
        )
        self._registered_hooks.append(handle)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_all_hooks(self) -> None:
        """Remove every registered hook and reset internal state."""
        for handle in self._registered_hooks:
            handle.remove()
        self._registered_hooks.clear()
        self._captured_activations.clear()
        self._active_steering_vector = None
        self._target_layer_idx = None
        self._multiplier = 0.0

    def get_activation(self, layer_idx: int) -> torch.Tensor | None:
        """Return the captured activation for *layer_idx*, or ``None``."""
        return self._captured_activations.get(layer_idx)
