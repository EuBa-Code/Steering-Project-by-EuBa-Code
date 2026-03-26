"""Steering vector representation and arithmetic operations."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class SteeringVector:
    """A directional vector extracted from model activations, used to steer generation.

    The vector represents the difference between two concepts in the model's
    latent space (e.g., "love" vs "hate"). Once normalized, its magnitude is
    controlled entirely by the multiplier at injection time.
    """

    def __init__(self, tensor: torch.Tensor, strength: float = 1.0) -> None:
        """Initialize a SteeringVector.

        Args:
            tensor: The activation-difference tensor representing a concept direction.
            strength: Default scaling factor (typically overridden at generation time).
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")
        self.tensor = tensor
        self.strength = strength

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def from_diff(
        activations_pos: torch.Tensor,
        activations_neg: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a steering direction as ``act_pos - act_neg``.

        Both tensors are moved to the same device before subtraction.

        Args:
            activations_pos: Activation tensor for the positive concept.
            activations_neg: Activation tensor for the negative concept.

        Returns:
            The raw difference tensor (not yet normalized).
        """
        return activations_pos - activations_neg.to(activations_pos.device)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def to(self, device: str | torch.device) -> SteeringVector:
        """Move the internal tensor to *device* (CPU / CUDA)."""
        self.tensor = self.tensor.to(device)
        return self

    def norm(self) -> torch.Tensor:
        """Return the L2 norm (magnitude) of the vector."""
        return self.tensor.norm()

    def normalize(self) -> SteeringVector:
        """Normalize the vector to unit length (L2 norm = 1).

        This makes the ``multiplier`` parameter the sole control for
        steering intensity, which is the recommended workflow.
        """
        self.tensor = F.normalize(self.tensor, dim=-1)
        return self

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SteeringVector(shape={tuple(self.tensor.shape)}, "
            f"norm={self.norm().item():.4f}, device={self.tensor.device})"
        )
