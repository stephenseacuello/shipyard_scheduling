"""Adaptive entropy tuning for RL algorithms.

Implements SAC-style automatic temperature adjustment for entropy-regularized
reinforcement learning. The entropy coefficient (alpha/temperature) is learned
to maintain a target entropy level.

Reference: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy
Deep Reinforcement Learning with a Stochastic Actor" (2018)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class EntropyTuner:
    """Automatic entropy temperature tuning following SAC principles.

    Learns a temperature parameter (alpha) that controls the entropy
    regularization strength. The target entropy is computed based on
    the action space dimensionality.

    Args:
        action_dim: Dimension of the action space (for target entropy computation).
        device: Device for tensor operations.
        lr: Learning rate for the temperature optimizer.
        target_entropy: Target entropy value. If None, computed from action_dim.
        initial_alpha: Initial value for the temperature parameter.
    """

    def __init__(
        self,
        action_dim: int,
        device: str = "cpu",
        lr: float = 3e-4,
        target_entropy: Optional[float] = None,
        initial_alpha: float = 0.2,
    ):
        self.device = device

        # Target entropy: typically -dim(A) or some fraction thereof
        # For discrete actions, use -log(1/|A|) * 0.98
        if target_entropy is None:
            self.target_entropy = -np.log(1.0 / action_dim) * 0.98
        else:
            self.target_entropy = target_entropy

        # Learnable log temperature (log for stability)
        self.log_alpha = torch.tensor(
            np.log(initial_alpha),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        # Optimizer for temperature parameter
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        # Statistics tracking
        self._update_count = 0
        self._alpha_history = []

    @property
    def alpha(self) -> torch.Tensor:
        """Current temperature value (entropy coefficient)."""
        return self.log_alpha.exp()

    def get_alpha(self) -> float:
        """Get current temperature as a Python float."""
        return self.alpha.item()

    def update(self, entropy: torch.Tensor) -> float:
        """Update the temperature parameter based on current policy entropy.

        Minimizes the loss: alpha * (entropy - target_entropy)

        When entropy < target: loss is negative, gradient pushes alpha down
        When entropy > target: loss is positive, gradient pushes alpha up

        Args:
            entropy: Current policy entropy (can be batched).

        Returns:
            The alpha loss value.
        """
        # Detach entropy since we only want gradients w.r.t. alpha
        alpha_loss = -(self.log_alpha * (entropy.detach() - self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Track statistics
        self._update_count += 1
        self._alpha_history.append(self.get_alpha())

        return alpha_loss.item()

    def get_stats(self) -> dict:
        """Get statistics about entropy tuning."""
        return {
            "alpha": self.get_alpha(),
            "target_entropy": self.target_entropy,
            "update_count": self._update_count,
            "alpha_mean": np.mean(self._alpha_history[-100:]) if self._alpha_history else 0.0,
        }

    def state_dict(self) -> dict:
        """Get state dictionary for checkpointing."""
        return {
            "log_alpha": self.log_alpha.detach().cpu().numpy(),
            "target_entropy": self.target_entropy,
            "update_count": self._update_count,
            "optimizer_state": self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        self.log_alpha = torch.tensor(
            state_dict["log_alpha"],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.target_entropy = state_dict["target_entropy"]
        self._update_count = state_dict["update_count"]
        # Recreate optimizer with new parameter
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha_optimizer.load_state_dict(state_dict["optimizer_state"])


class MultiHeadEntropyTuner:
    """Entropy tuning for multi-head action distributions.

    Maintains separate temperature parameters for each action head,
    useful for hierarchical action spaces like in shipyard scheduling.

    Args:
        head_dims: Dictionary mapping head names to action dimensions.
        device: Device for tensor operations.
        lr: Learning rate for temperature optimizers.
    """

    def __init__(
        self,
        head_dims: dict,
        device: str = "cpu",
        lr: float = 3e-4,
    ):
        self.device = device
        self.head_dims = head_dims

        self.tuners = {
            name: EntropyTuner(dim, device=device, lr=lr)
            for name, dim in head_dims.items()
        }

    def get_alpha(self, head_name: str) -> float:
        """Get temperature for a specific action head."""
        return self.tuners[head_name].get_alpha()

    def get_alphas(self) -> dict:
        """Get all temperature values."""
        return {name: tuner.get_alpha() for name, tuner in self.tuners.items()}

    def update(self, entropies: dict) -> dict:
        """Update temperatures for all heads.

        Args:
            entropies: Dictionary mapping head names to entropy tensors.

        Returns:
            Dictionary of alpha losses per head.
        """
        losses = {}
        for name, entropy in entropies.items():
            if name in self.tuners:
                losses[name] = self.tuners[name].update(entropy)
        return losses

    def state_dict(self) -> dict:
        """Get state dictionary for checkpointing."""
        return {name: tuner.state_dict() for name, tuner in self.tuners.items()}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        for name, tuner_state in state_dict.items():
            if name in self.tuners:
                self.tuners[name].load_state_dict(tuner_state)
