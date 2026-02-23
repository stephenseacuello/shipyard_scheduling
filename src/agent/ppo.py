"""Proximal Policy Optimization (PPO) trainer for shipyard scheduling RL.

This class manages experience collection, return and advantage computation
using Generalized Advantage Estimation (GAE) and performs mini-batch updates
on the actor and critic networks. The encoder is re-run on stored graph data
during updates so that encoder gradients flow through backprop.

Enhancements for graduate-level research:
- Cosine annealing learning rate schedule
- Return normalization with running statistics
- KL divergence early stopping
- Explained variance tracking
- Gradient statistics monitoring
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from typing import Dict, List, Tuple, Any, Optional

from .action_masking import (
    flatten_env_mask_to_policy_mask,
    batch_masks,
    to_torch_mask,
)
from .entropy_tuning import EntropyTuner
from .reward_shaping import PotentialBasedRewardShaper


class RunningMeanStd:
    """Tracks running mean and standard deviation for return normalization.

    Uses Welford's online algorithm for numerical stability.
    """

    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # Small epsilon to avoid division by zero
        self.epsilon = epsilon

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a batch of values."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        """Update from batch moments using parallel algorithm."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize values using running statistics."""
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + self.epsilon)


class PPOTrainer:
    """PPO trainer with research-grade enhancements.

    Features:
    - Cosine annealing learning rate schedule
    - Return normalization with running statistics
    - KL divergence early stopping
    - Explained variance tracking
    - Comprehensive training metrics

    Args:
        policy: Actor-critic policy network.
        encoder: GNN encoder for graph observations.
        lr: Initial learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        clip_epsilon: PPO clipping parameter.
        entropy_coef: Entropy bonus coefficient (initial value if using schedule).
        value_coef: Value loss coefficient.
        max_grad_norm: Gradient clipping threshold.
        n_epochs: Number of PPO update epochs per rollout.
        batch_size: Mini-batch size for updates.
        device: Computation device (cpu/cuda).
        total_epochs: Total training epochs (for LR scheduling).
        target_kl: Target KL divergence for early stopping (None to disable).
        normalize_returns: Whether to normalize returns.
        lr_schedule: Learning rate schedule type ('constant', 'cosine', 'linear').
        entropy_schedule: Entropy annealing schedule ('constant', 'linear', 'exponential', 'cosine').
        entropy_coef_final: Final entropy coefficient (defaults to 2% of initial).
    """

    def __init__(
        self,
        policy: nn.Module,
        encoder: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
        total_epochs: int = 100,
        target_kl: Optional[float] = 0.015,
        normalize_returns: bool = True,
        lr_schedule: str = "cosine",
        entropy_schedule: str = "cosine",
        entropy_coef_final: Optional[float] = None,
        use_adaptive_entropy: bool = False,
        encoder_lr_scale: float = 0.1,
        use_reward_shaping: bool = False,
        reward_shaping_weight: float = 0.2,
    ) -> None:
        self.policy = policy.to(device)
        self.encoder = encoder.to(device)
        self.device = device
        self.lr = lr
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.lr_schedule = lr_schedule

        # Differential learning rates: encoder learns slower for stable representations
        # Phase 3B: encoder_lr = lr * 0.1 to prevent representation drift
        self.optimizer = torch.optim.Adam([
            {"params": self.encoder.parameters(), "lr": lr * encoder_lr_scale},
            {"params": self.policy.parameters(), "lr": lr},
        ], weight_decay=1e-5)

        # Learning rate scheduler (applied to both param groups)
        if lr_schedule == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_epochs, eta_min=lr * 0.01
            )
        elif lr_schedule == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.01, total_iters=total_epochs
            )
        else:
            self.scheduler = None

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl
        self.normalize_returns = normalize_returns

        # Entropy annealing parameters
        self.entropy_coef_initial = entropy_coef
        # Use 20% of initial (was 2% - too aggressive, caused entropy collapse in ~6 epochs)
        self.entropy_coef_final = entropy_coef_final if entropy_coef_final is not None else entropy_coef * 0.2
        self.entropy_schedule = entropy_schedule

        # Return normalizer (running statistics)
        self.return_normalizer = RunningMeanStd() if normalize_returns else None

        # Experience storage
        self.buffer: Dict[str, List] = {
            "graph_data": [],  # raw HeteroData objects for encoder re-encoding
            "actions": [],
            "rewards": [],
            "dones": [],
            "log_probs": [],
            "values": [],
            "masks": [],  # per-head policy masks (flattened)
        }

        # Training statistics
        self._training_stats: Dict[str, List[float]] = {
            "clip_fractions": [],
            "kl_divergences": [],
            "explained_variances": [],
        }

        # Adaptive entropy tuning (SAC-style automatic temperature)
        self.use_adaptive_entropy = use_adaptive_entropy
        if use_adaptive_entropy:
            # action_dim = 4 for the action_type head
            self.entropy_tuner = EntropyTuner(
                action_dim=policy.n_action_types,
                device=device,
                lr=3e-4,
                target_entropy=None,  # Auto-computed as -log(1/action_dim) * 0.98
            )
        else:
            self.entropy_tuner = None

        # Potential-based reward shaping for denser learning signal
        self.use_reward_shaping = use_reward_shaping
        self.reward_shaping_weight = reward_shaping_weight
        if use_reward_shaping:
            self.reward_shaper = PotentialBasedRewardShaper(gamma=gamma)
        else:
            self.reward_shaper = None

    def collect_rollout(self, env, n_steps: int, reset: bool = True) -> Dict[str, Any]:
        """Collect `n_steps` interactions from the environment.

        Args:
            env: The environment to collect from.
            n_steps: Number of steps to collect.
            reset: If True, reset the environment at start. If False, continue from current state.
        """
        if reset:
            obs, info = env.reset()
        else:
            obs = env._get_observation()
        for _ in range(n_steps):
            graph_data = env.get_graph_data()
            graph_data = graph_data.to(self.device)

            # Encode state
            with torch.no_grad():
                state_emb = self.encoder(graph_data)

            # Build per-head policy masks from environment mask
            env_mask = env.get_action_mask()
            policy_mask = flatten_env_mask_to_policy_mask(
                env_mask,
                n_spmts=self.policy.n_spmts,
                n_cranes=self.policy.n_cranes,
                max_requests=self.policy.max_requests,
            )
            torch_mask = to_torch_mask(policy_mask, device=self.device)

            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(state_emb, torch_mask)

            # Capture state snapshot for reward shaping (before action)
            if self.use_reward_shaping and self.reward_shaper is not None:
                snapshot_before = self.reward_shaper.get_state_snapshot(env)

            # Convert action tensors to python ints for env.step
            action_cpu = {k: int(v.item()) for k, v in action.items()}
            next_obs, reward, terminated, truncated, info = env.step(action_cpu)
            done = terminated or truncated

            # Apply reward shaping if enabled
            if self.use_reward_shaping and self.reward_shaper is not None:
                snapshot_after = self.reward_shaper.get_state_snapshot(env)
                shaped_reward = self.reward_shaper.shape_reward(
                    snapshot_before, snapshot_after, reward
                )
                # Blend raw and shaped rewards
                reward = (1 - self.reward_shaping_weight) * reward + self.reward_shaping_weight * shaped_reward

            # Store experience — keep raw graph data for re-encoding during update
            self.buffer["graph_data"].append(graph_data.cpu())
            self.buffer["actions"].append(action)
            self.buffer["rewards"].append(reward)
            self.buffer["dones"].append(done)
            self.buffer["log_probs"].append(log_prob)
            self.buffer["values"].append(value.squeeze(0).detach())
            self.buffer["masks"].append(torch_mask)

            if done:
                obs, info = env.reset()
        return self._compute_returns_and_advantages()

    def _compute_returns_and_advantages(self) -> Dict[str, Any]:
        """Compute GAE advantages and discounted returns."""
        rewards = self.buffer["rewards"]
        values = torch.stack(self.buffer["values"])
        dones = self.buffer["dones"]
        advantages = []
        returns = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, device=self.device, dtype=torch.float32)

        # Update return normalizer and normalize returns if enabled
        if self.normalize_returns and self.return_normalizer is not None:
            returns_np = returns_tensor.cpu().numpy()
            self.return_normalizer.update(returns_np)
            returns_normalized = self.return_normalizer.normalize(returns_np)
            returns_tensor = torch.tensor(
                returns_normalized, device=self.device, dtype=torch.float32
            )

        # Normalize advantages (always do this for stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "graph_data": self.buffer["graph_data"],
            "actions": self.buffer["actions"],
            "log_probs": torch.stack(self.buffer["log_probs"]),
            "returns": returns_tensor,
            "advantages": advantages,
            "masks": self.buffer["masks"],
            "values": values,  # For explained variance calculation
        }

    def update(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """Perform PPO updates using collected rollout data.

        Returns:
            Dictionary of training metrics including:
            - policy_loss: Mean policy loss
            - value_loss: Mean value loss
            - entropy: Mean entropy
            - kl_divergence: Approximate KL divergence
            - clip_fraction: Fraction of clipped ratios
            - explained_variance: Explained variance of value function
            - learning_rate: Current learning rate
            - grad_norm: Gradient norm before clipping
        """
        graph_data_list = rollout_data["graph_data"]
        actions = rollout_data["actions"]
        old_log_probs = rollout_data["log_probs"]
        returns = rollout_data["returns"]
        advantages = rollout_data["advantages"]
        masks_list = rollout_data["masks"]
        old_values = rollout_data["values"]

        n_samples = len(graph_data_list)
        metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl_divergence": [],
            "clip_fraction": [],
            "grad_norm": [],
        }

        # Compute explained variance before updates
        explained_var = self._compute_explained_variance(
            old_values.cpu().numpy(), returns.cpu().numpy()
        )

        for epoch_idx in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            epoch_kl = []

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Batch graph data and re-encode through encoder (gradients flow)
                batch_graphs = Batch.from_data_list(
                    [graph_data_list[i] for i in batch_indices]
                ).to(self.device)
                batch_states = self.encoder(batch_graphs)

                # Batch actions
                batch_actions = {
                    k: torch.stack([actions[i][k] for i in batch_indices])
                    for k in actions[0]
                }
                # Batch masks
                batch_mask = batch_masks([masks_list[i] for i in batch_indices])

                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Policy evaluate with masks
                log_probs, entropy, values = self.policy.evaluate_action(
                    batch_states, batch_actions, batch_mask
                )

                # Compute approximate KL divergence
                with torch.no_grad():
                    log_ratio = log_probs - batch_old_log_probs
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    epoch_kl.append(approx_kl)

                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)

                # Compute entropy loss with adaptive or scheduled coefficient
                if self.use_adaptive_entropy and self.entropy_tuner is not None:
                    # SAC-style adaptive entropy: use learnable temperature
                    # get_alpha() returns float, alpha property returns tensor
                    alpha = self.entropy_tuner.alpha  # Use tensor for gradient flow
                    entropy_loss = -alpha * entropy.mean()
                    # Update alpha based on current entropy (target vs actual)
                    self.entropy_tuner.update(entropy.mean())
                    current_entropy_coef = alpha.item()
                else:
                    # Use scheduled entropy coefficient
                    current_entropy_coef = self.get_entropy_coef()
                    entropy_loss = -current_entropy_coef * entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()

                # Compute gradient norm before clipping
                grad_norm = self._compute_grad_norm()

                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.encoder.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # Track clip fraction
                with torch.no_grad():
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean().item()
                    )

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.mean().item())
                metrics["kl_divergence"].append(approx_kl)
                metrics["clip_fraction"].append(clip_fraction)
                metrics["grad_norm"].append(grad_norm)

            # KL early stopping
            mean_epoch_kl = np.mean(epoch_kl)
            if self.target_kl is not None and mean_epoch_kl > 1.5 * self.target_kl:
                # Early stop if KL divergence is too high
                break

        # Step learning rate scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        self.current_epoch += 1

        # Clear buffer
        for k in self.buffer:
            self.buffer[k] = []

        # Compile final metrics
        result = {k: float(np.mean(v)) for k, v in metrics.items()}
        result["explained_variance"] = explained_var
        result["learning_rate"] = self.get_learning_rate()
        result["entropy_coef"] = self.get_entropy_coef()
        result["epochs_completed"] = epoch_idx + 1  # How many PPO epochs ran (for KL early stop tracking)

        return result

    def _compute_explained_variance(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> float:
        """Compute explained variance of value predictions.

        Explained variance = 1 - Var(y_true - y_pred) / Var(y_true)

        Values close to 1 indicate the value function is learning well.
        Values close to 0 indicate poor value predictions.
        Negative values indicate the value function is worse than predicting the mean.
        """
        var_true = np.var(y_true)
        if var_true < 1e-8:
            return 0.0
        return 1 - np.var(y_true - y_pred) / var_true

    def _compute_grad_norm(self) -> float:
        """Compute the total gradient norm before clipping."""
        total_norm = 0.0
        for p in list(self.policy.parameters()) + list(self.encoder.parameters()):
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def get_learning_rate(self) -> float:
        """Get current policy learning rate (encoder uses scaled rate)."""
        # param_groups[0] = encoder, param_groups[1] = policy
        return self.optimizer.param_groups[1]["lr"]

    def get_learning_rates(self) -> Dict[str, float]:
        """Get learning rates for both encoder and policy."""
        return {
            "encoder_lr": self.optimizer.param_groups[0]["lr"],
            "policy_lr": self.optimizer.param_groups[1]["lr"],
        }

    def get_entropy_coef(self) -> float:
        """Get current entropy coefficient based on annealing schedule.

        Supports four schedule types:
        - 'constant': Fixed entropy coefficient throughout training
        - 'linear': Linear decay from initial to final
        - 'exponential': Exponential decay with rate 0.99 per epoch
        - 'cosine': Cosine annealing (smooth decay with warmup-like behavior)
        """
        if self.entropy_schedule == "constant":
            return self.entropy_coef_initial

        # Compute progress through training (0.0 to 1.0)
        progress = min(1.0, self.current_epoch / max(1, self.total_epochs))

        if self.entropy_schedule == "linear":
            return self.entropy_coef_initial + progress * (
                self.entropy_coef_final - self.entropy_coef_initial
            )
        elif self.entropy_schedule == "exponential":
            decay = 0.99 ** self.current_epoch
            return max(self.entropy_coef_final, self.entropy_coef_initial * decay)
        elif self.entropy_schedule == "cosine":
            # Cosine annealing: smooth transition from initial to final
            return self.entropy_coef_final + 0.5 * (
                self.entropy_coef_initial - self.entropy_coef_final
            ) * (1 + np.cos(np.pi * progress))
        else:
            # Fallback to constant
            return self.entropy_coef_initial

    def save_checkpoint(self, path: str) -> None:
        """Save trainer state to checkpoint."""
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "encoder_state_dict": self.encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "current_epoch": self.current_epoch,
            "return_normalizer": {
                "mean": self.return_normalizer.mean if self.return_normalizer else None,
                "var": self.return_normalizer.var if self.return_normalizer else None,
                "count": self.return_normalizer.count if self.return_normalizer else None,
            },
            "use_adaptive_entropy": self.use_adaptive_entropy,
            "entropy_tuner_state": self.entropy_tuner.state_dict() if self.entropy_tuner else None,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load trainer state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["current_epoch"]
        if self.return_normalizer and checkpoint["return_normalizer"]["mean"] is not None:
            self.return_normalizer.mean = checkpoint["return_normalizer"]["mean"]
            self.return_normalizer.var = checkpoint["return_normalizer"]["var"]
            self.return_normalizer.count = checkpoint["return_normalizer"]["count"]
        # Restore entropy tuner state if present
        if self.entropy_tuner and checkpoint.get("entropy_tuner_state") is not None:
            self.entropy_tuner.load_state_dict(checkpoint["entropy_tuner_state"])
