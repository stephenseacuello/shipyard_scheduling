"""Adversarial training for robust RL policies.

Implements adversarial perturbations to train policies that are
robust to worst-case state perturbations and environment dynamics.

References:
- Pinto et al., "Robust Adversarial Reinforcement Learning" (2017)
- Tessler et al., "Action Robust RL with Uncertainty" (2019)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any


class StateAdversary(nn.Module):
    """Adversary network that generates state perturbations.

    Learns to find perturbations within an epsilon ball that
    minimize the policy's expected return.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        epsilon: float = 0.1,
    ):
        super().__init__()

        self.epsilon = epsilon

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Generate perturbation for state.

        Args:
            state: Original state [batch, state_dim].

        Returns:
            Perturbation [batch, state_dim] scaled by epsilon.
        """
        raw_perturbation = self.network(state)
        return self.epsilon * raw_perturbation

    def perturb(self, state: torch.Tensor) -> torch.Tensor:
        """Apply perturbation to state.

        Args:
            state: Original state.

        Returns:
            Perturbed state.
        """
        perturbation = self.forward(state)
        return state + perturbation


class ActionAdversary(nn.Module):
    """Adversary that modifies actions.

    Learns to inject noise into actions to find worst-case
    action perturbations.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        epsilon: float = 0.2,
    ):
        super().__init__()

        self.epsilon = epsilon
        self.n_actions = n_actions

        self.network = nn.Sequential(
            nn.Linear(state_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(
        self,
        state: torch.Tensor,
        action_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Generate action probability perturbation.

        Args:
            state: State tensor.
            action_probs: Current action probabilities.

        Returns:
            Perturbed action probabilities.
        """
        x = torch.cat([state, action_probs], dim=-1)
        perturbation = self.network(x)

        # Apply softmax to get valid probabilities
        perturbed_logits = torch.log(action_probs + 1e-8) + self.epsilon * perturbation
        return F.softmax(perturbed_logits, dim=-1)


class AdversarialTrainer:
    """Trainer for adversarial robust RL.

    Alternates between:
    1. Training the adversary to find worst-case perturbations
    2. Training the policy to be robust against the adversary
    """

    def __init__(
        self,
        policy: nn.Module,
        encoder: nn.Module,
        state_adversary: Optional[StateAdversary] = None,
        action_adversary: Optional[ActionAdversary] = None,
        lr_policy: float = 3e-4,
        lr_adversary: float = 1e-4,
        adversary_train_freq: int = 5,
        state_dim: int = 512,
        epsilon: float = 0.1,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.encoder = encoder.to(device)
        self.device = device
        self.adversary_train_freq = adversary_train_freq
        self._step_count = 0

        # Create adversaries if not provided
        if state_adversary is None:
            self.state_adversary = StateAdversary(state_dim, epsilon=epsilon).to(device)
        else:
            self.state_adversary = state_adversary.to(device)

        if action_adversary is not None:
            self.action_adversary = action_adversary.to(device)
        else:
            self.action_adversary = None

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(encoder.parameters()),
            lr=lr_policy,
        )
        self.adversary_optimizer = torch.optim.Adam(
            self.state_adversary.parameters(),
            lr=lr_adversary,
        )

    def generate_adversarial_state(self, state: torch.Tensor) -> torch.Tensor:
        """Generate adversarial perturbation for state.

        Args:
            state: Original encoded state.

        Returns:
            Perturbed state.
        """
        return self.state_adversary.perturb(state)

    def train_adversary_step(
        self,
        states: torch.Tensor,
        values: torch.Tensor,
    ) -> float:
        """Train adversary to minimize policy value.

        The adversary tries to find perturbations that make
        the policy's value estimates lower.

        Args:
            states: Batch of states.
            values: Corresponding value estimates.

        Returns:
            Adversary loss.
        """
        self.adversary_optimizer.zero_grad()

        # Generate perturbed states
        perturbed_states = self.state_adversary.perturb(states)

        # Get policy values on perturbed states
        with torch.no_grad():
            # Assuming policy has a forward method that returns distributions and values
            _, perturbed_values = self.policy.forward(perturbed_states, None)

        # Adversary loss: maximize the drop in value (minimize perturbed value)
        adversary_loss = perturbed_values.mean()

        adversary_loss.backward()
        self.adversary_optimizer.step()

        return adversary_loss.item()

    def train_policy_step(
        self,
        states: torch.Tensor,
        actions: Dict[str, torch.Tensor],
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ) -> Dict[str, float]:
        """Train policy on adversarially perturbed states.

        Args:
            states: Batch of encoded states.
            actions: Batch of actions.
            advantages: GAE advantages.
            old_log_probs: Log probs from rollout.
            returns: Discounted returns.
            masks: Action masks.
            clip_epsilon: PPO clip parameter.
            entropy_coef: Entropy bonus weight.
            value_coef: Value loss weight.

        Returns:
            Dictionary of training metrics.
        """
        self.policy_optimizer.zero_grad()

        # Mix clean and adversarial states (50/50)
        with torch.no_grad():
            adv_states = self.state_adversary.perturb(states)

        mix_mask = torch.rand(states.size(0), 1, device=self.device) < 0.5
        mixed_states = torch.where(mix_mask, adv_states, states)

        # Policy forward
        log_probs, entropy, values = self.policy.evaluate_action(
            mixed_states, actions, masks
        )

        # PPO loss
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values.squeeze(-1), returns)

        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

        total_loss.backward()
        self.policy_optimizer.step()

        self._step_count += 1

        # Train adversary periodically
        if self._step_count % self.adversary_train_freq == 0:
            adv_loss = self.train_adversary_step(states, values.detach())
        else:
            adv_loss = 0.0

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "adversary_loss": adv_loss,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save trainer state."""
        torch.save({
            "policy": self.policy.state_dict(),
            "encoder": self.encoder.state_dict(),
            "state_adversary": self.state_adversary.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "adversary_optimizer": self.adversary_optimizer.state_dict(),
            "step_count": self._step_count,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load trainer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.state_adversary.load_state_dict(checkpoint["state_adversary"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.adversary_optimizer.load_state_dict(checkpoint["adversary_optimizer"])
        self._step_count = checkpoint["step_count"]


class RobustPolicyWrapper:
    """Wrapper that adds robustness testing to policy evaluation.

    At test time, evaluates policy on both clean and perturbed states
    to measure robustness.
    """

    def __init__(
        self,
        policy: nn.Module,
        adversary: StateAdversary,
        n_perturbations: int = 5,
    ):
        self.policy = policy
        self.adversary = adversary
        self.n_perturbations = n_perturbations

    def evaluate_robustness(
        self,
        state: torch.Tensor,
        mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Evaluate policy robustness on a state.

        Args:
            state: Encoded state.
            mask: Action mask.

        Returns:
            Dictionary with clean and robust performance metrics.
        """
        with torch.no_grad():
            # Clean evaluation
            clean_dist, clean_value = self.policy.forward(state, mask)
            clean_action = {k: d.probs.argmax(dim=-1) for k, d in clean_dist.items()}

            # Perturbed evaluations
            perturbed_actions = []
            perturbed_values = []

            for _ in range(self.n_perturbations):
                perturbed_state = self.adversary.perturb(state)
                dist, value = self.policy.forward(perturbed_state, mask)
                action = {k: d.probs.argmax(dim=-1) for k, d in dist.items()}
                perturbed_actions.append(action)
                perturbed_values.append(value.item())

            # Compute consistency
            action_consistency = sum(
                1 for a in perturbed_actions
                if a["action_type"].item() == clean_action["action_type"].item()
            ) / self.n_perturbations

            value_stability = 1.0 - abs(
                clean_value.item() - sum(perturbed_values) / len(perturbed_values)
            ) / max(abs(clean_value.item()), 1e-8)

        return {
            "clean_action": clean_action,
            "clean_value": clean_value.item(),
            "action_consistency": action_consistency,
            "value_stability": value_stability,
            "mean_perturbed_value": sum(perturbed_values) / len(perturbed_values),
        }
