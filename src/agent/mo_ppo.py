"""Multi-Objective Proximal Policy Optimization (MO-PPO).

This module implements multi-objective RL for shipyard scheduling, allowing
optimization of multiple potentially conflicting objectives simultaneously.

Objectives:
1. Throughput maximization (blocks completed per time)
2. Tardiness minimization (on-time delivery)
3. Breakdown prevention (equipment health maintenance)
4. Cost minimization (operational efficiency)

Approaches implemented:
- Weighted sum scalarization with random/adaptive weights
- Chebyshev scalarization for Pareto-optimal policies
- Hypernetwork for weight-conditioned policies
- Pareto archive for storing non-dominated solutions

Reference:
- Mossalam et al. "Multi-Objective Reinforcement Learning: A Selective Overview"
- Van Moffaert & Nowé "Multi-Objective RL using Sets of Pareto Dominating Policies"
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from .action_masking import (
    flatten_env_mask_to_policy_mask,
    batch_masks,
    to_torch_mask,
)


@dataclass
class ParetoSolution:
    """A solution in the Pareto archive."""
    policy_state: Dict[str, torch.Tensor]
    encoder_state: Dict[str, torch.Tensor]
    objectives: np.ndarray  # Vector of objective values
    weight_vector: np.ndarray  # Weight vector used to find this solution


class ParetoArchive:
    """Archive of non-dominated (Pareto-optimal) solutions.

    Maintains a set of solutions where no solution dominates another.
    A solution A dominates B if A is at least as good as B on all objectives
    and strictly better on at least one.
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.solutions: List[ParetoSolution] = []

    def add(self, solution: ParetoSolution) -> bool:
        """Add a solution if it's non-dominated.

        Returns True if the solution was added (is Pareto-optimal).
        """
        # Check if new solution is dominated by any existing solution
        for existing in self.solutions:
            if self._dominates(existing.objectives, solution.objectives):
                return False  # New solution is dominated, don't add

        # Remove solutions dominated by the new one
        self.solutions = [
            s for s in self.solutions
            if not self._dominates(solution.objectives, s.objectives)
        ]

        # Add the new solution
        self.solutions.append(solution)

        # If archive is too large, remove solutions with most similar objectives
        if len(self.solutions) > self.max_size:
            self._prune()

        return True

    def _dominates(self, obj_a: np.ndarray, obj_b: np.ndarray) -> bool:
        """Check if objective vector A dominates B.

        Assumes all objectives are to be maximized.
        """
        return np.all(obj_a >= obj_b) and np.any(obj_a > obj_b)

    def _prune(self) -> None:
        """Remove solutions to maintain max_size."""
        # Use crowding distance to remove least diverse solutions
        if len(self.solutions) <= self.max_size:
            return

        n_objectives = len(self.solutions[0].objectives)
        crowding = np.zeros(len(self.solutions))

        for obj_idx in range(n_objectives):
            # Sort by this objective
            sorted_indices = np.argsort([s.objectives[obj_idx] for s in self.solutions])

            # Boundary solutions get infinite crowding distance
            crowding[sorted_indices[0]] = np.inf
            crowding[sorted_indices[-1]] = np.inf

            # Compute crowding for middle solutions
            obj_range = (
                self.solutions[sorted_indices[-1]].objectives[obj_idx] -
                self.solutions[sorted_indices[0]].objectives[obj_idx]
            )
            if obj_range > 0:
                for i in range(1, len(sorted_indices) - 1):
                    prev_obj = self.solutions[sorted_indices[i-1]].objectives[obj_idx]
                    next_obj = self.solutions[sorted_indices[i+1]].objectives[obj_idx]
                    crowding[sorted_indices[i]] += (next_obj - prev_obj) / obj_range

        # Keep solutions with highest crowding distance
        keep_indices = np.argsort(crowding)[-self.max_size:]
        self.solutions = [self.solutions[i] for i in sorted(keep_indices)]

    def get_solution_for_weight(self, weight: np.ndarray) -> Optional[ParetoSolution]:
        """Get solution closest to a given weight preference."""
        if not self.solutions:
            return None

        # Find solution with best scalarized value for this weight
        best_value = -np.inf
        best_solution = None
        for sol in self.solutions:
            value = np.dot(weight, sol.objectives)
            if value > best_value:
                best_value = value
                best_solution = sol
        return best_solution

    def get_hypervolume(self, reference_point: np.ndarray) -> float:
        """Compute hypervolume indicator for the archive.

        The hypervolume is the volume of objective space dominated by
        the Pareto front and bounded by the reference point.
        """
        if not self.solutions:
            return 0.0

        # Simple 2D hypervolume calculation (extend for higher dimensions)
        n_obj = len(self.solutions[0].objectives)
        if n_obj != 2:
            # For higher dimensions, use approximation
            return self._approximate_hypervolume(reference_point)

        # Sort by first objective
        sorted_sols = sorted(self.solutions, key=lambda s: s.objectives[0])
        hv = 0.0
        prev_obj1 = reference_point[1]

        for sol in sorted_sols:
            if sol.objectives[0] > reference_point[0]:
                continue
            if sol.objectives[1] > reference_point[1]:
                continue

            width = reference_point[0] - sol.objectives[0]
            height = prev_obj1 - sol.objectives[1]
            if height > 0:
                hv += width * height
            prev_obj1 = min(prev_obj1, sol.objectives[1])

        return hv

    def _approximate_hypervolume(self, reference_point: np.ndarray) -> float:
        """Monte Carlo approximation for higher-dimensional hypervolume."""
        n_samples = 10000
        n_obj = len(reference_point)

        # Sample random points in the bounding box
        min_point = np.min([s.objectives for s in self.solutions], axis=0)
        max_point = reference_point

        samples = np.random.uniform(min_point, max_point, size=(n_samples, n_obj))

        # Count samples dominated by at least one solution
        dominated_count = 0
        for sample in samples:
            for sol in self.solutions:
                if np.all(sol.objectives >= sample):
                    dominated_count += 1
                    break

        # Estimate hypervolume
        box_volume = np.prod(max_point - min_point)
        return box_volume * dominated_count / n_samples


class HyperNetwork(nn.Module):
    """Hypernetwork that generates policy parameters conditioned on weights.

    Instead of training separate policies for each weight vector, the
    hypernetwork learns to generate appropriate policy parameters for
    any weight preference, enabling continuous coverage of the Pareto front.
    """

    def __init__(
        self,
        weight_dim: int,
        policy: nn.Module,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.weight_dim = weight_dim
        self.policy = policy

        # Count total parameters in policy
        self.n_params = sum(p.numel() for p in policy.parameters())

        # Build hypernetwork
        self.hypernet = nn.Sequential(
            nn.Linear(weight_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Parameter generators for each policy layer
        self.param_generators = nn.ModuleDict()
        for name, param in policy.named_parameters():
            # Replace dots with underscores for valid key names
            key = name.replace(".", "_")
            self.param_generators[key] = nn.Linear(hidden_dim, param.numel())

        self._param_shapes = {
            name.replace(".", "_"): param.shape
            for name, param in policy.named_parameters()
        }

    def forward(self, weight: torch.Tensor) -> nn.Module:
        """Generate policy parameters for given weight vector.

        Args:
            weight: Weight vector [weight_dim] or [batch, weight_dim].

        Returns:
            Policy module with generated parameters.
        """
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)

        # Encode weight
        hidden = self.hypernet(weight)  # [batch, hidden_dim]

        # Generate parameters
        generated_params = {}
        for key, generator in self.param_generators.items():
            flat_params = generator(hidden)  # [batch, param_size]
            # Reshape to original parameter shape (use first batch item)
            shape = self._param_shapes[key]
            generated_params[key] = flat_params[0].view(shape)

        # Create new policy with generated parameters
        new_policy = copy.deepcopy(self.policy)
        with torch.no_grad():
            for name, param in new_policy.named_parameters():
                key = name.replace(".", "_")
                param.copy_(generated_params[key])

        return new_policy


class MultiObjectivePPO:
    """Multi-Objective PPO trainer for Pareto-optimal policies.

    Supports multiple scalarization methods:
    - 'weighted_sum': Linear combination of objectives
    - 'chebyshev': Chebyshev (L-infinity) scalarization
    - 'hypernetwork': Weight-conditioned policy via hypernetwork

    Args:
        policy: Actor-critic policy network.
        encoder: GNN encoder for graph observations.
        n_objectives: Number of objectives to optimize.
        objective_names: Names of objectives (for logging).
        scalarization: Scalarization method.
        lr: Learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_epsilon: PPO clipping parameter.
        entropy_coef: Entropy bonus coefficient.
        value_coef: Value loss coefficient.
        max_grad_norm: Gradient clipping threshold.
        n_epochs: PPO epochs per update.
        batch_size: Mini-batch size.
        device: Computation device.
        weight_sampling: How to sample weights ('uniform', 'dirichlet', 'adaptive').
        archive_size: Maximum Pareto archive size.
    """

    def __init__(
        self,
        policy: nn.Module,
        encoder: nn.Module,
        n_objectives: int = 4,
        objective_names: Optional[List[str]] = None,
        scalarization: str = "weighted_sum",
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
        weight_sampling: str = "uniform",
        archive_size: int = 50,
    ) -> None:
        self.policy = policy.to(device)
        self.encoder = encoder.to(device)
        self.device = device
        self.n_objectives = n_objectives
        self.objective_names = objective_names or [f"obj_{i}" for i in range(n_objectives)]
        self.scalarization = scalarization
        self.weight_sampling = weight_sampling

        # Optimizer
        params = list(self.policy.parameters()) + list(self.encoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        # Hypernetwork if using that scalarization
        if scalarization == "hypernetwork":
            self.hypernet = HyperNetwork(n_objectives, policy).to(device)
            self.optimizer = torch.optim.Adam(
                list(self.hypernet.parameters()) + list(self.encoder.parameters()),
                lr=lr,
            )

        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Multi-objective components
        self.pareto_archive = ParetoArchive(max_size=archive_size)
        self.current_weight = self._sample_weight()
        self.utopia_point = np.zeros(n_objectives)  # Best known value per objective
        self.nadir_point = np.ones(n_objectives) * np.inf  # Worst value per objective

        # Experience buffer (stores multi-objective rewards)
        self.buffer: Dict[str, List] = {
            "graph_data": [],
            "actions": [],
            "rewards": [],  # Now [n_objectives] per step
            "dones": [],
            "log_probs": [],
            "values": [],
            "masks": [],
        }

    def _sample_weight(self) -> np.ndarray:
        """Sample a weight vector for scalarization."""
        if self.weight_sampling == "uniform":
            # Uniform on simplex
            w = np.random.exponential(1, self.n_objectives)
            return w / w.sum()
        elif self.weight_sampling == "dirichlet":
            # Dirichlet with concentration 1 (uniform on simplex, but smoother)
            return np.random.dirichlet(np.ones(self.n_objectives))
        elif self.weight_sampling == "adaptive":
            # Sample weights to explore under-represented regions
            if len(self.pareto_archive.solutions) < 5:
                return np.random.dirichlet(np.ones(self.n_objectives))
            # Find region with fewest solutions
            # (simplified: just sample away from existing weights)
            existing_weights = np.array([s.weight_vector for s in self.pareto_archive.solutions])
            candidate = np.random.dirichlet(np.ones(self.n_objectives))
            min_dist = np.min(np.linalg.norm(existing_weights - candidate, axis=1))
            for _ in range(10):
                new_candidate = np.random.dirichlet(np.ones(self.n_objectives))
                new_dist = np.min(np.linalg.norm(existing_weights - new_candidate, axis=1))
                if new_dist > min_dist:
                    candidate = new_candidate
                    min_dist = new_dist
            return candidate
        else:
            raise ValueError(f"Unknown weight sampling: {self.weight_sampling}")

    def scalarize_reward(
        self,
        reward_vector: np.ndarray,
        weight: np.ndarray,
    ) -> float:
        """Scalarize multi-objective reward to single value.

        Args:
            reward_vector: Vector of objective values [n_objectives].
            weight: Weight vector [n_objectives].

        Returns:
            Scalarized reward value.
        """
        if self.scalarization == "weighted_sum":
            return float(np.dot(weight, reward_vector))

        elif self.scalarization == "chebyshev":
            # Normalize objectives using utopia/nadir points
            if np.any(self.nadir_point == np.inf):
                normalized = reward_vector
            else:
                range_vec = self.nadir_point - self.utopia_point + 1e-8
                normalized = (reward_vector - self.utopia_point) / range_vec

            # Chebyshev scalarization: min over weighted deviations from utopia
            # We maximize, so return negative of max deviation
            deviations = weight * (1 - normalized)  # 1 = utopia (best)
            return float(-np.max(deviations))

        elif self.scalarization == "hypernetwork":
            # For hypernetwork, use weighted sum for reward
            return float(np.dot(weight, reward_vector))

        else:
            raise ValueError(f"Unknown scalarization: {self.scalarization}")

    def collect_rollout(self, env, n_steps: int) -> Dict[str, Any]:
        """Collect experience with multi-objective rewards."""
        # Sample new weight for this rollout
        self.current_weight = self._sample_weight()
        weight_tensor = torch.tensor(self.current_weight, device=self.device, dtype=torch.float32)

        # Get policy (possibly weight-conditioned)
        if self.scalarization == "hypernetwork":
            active_policy = self.hypernet(weight_tensor)
        else:
            active_policy = self.policy

        obs, info = env.reset()
        episode_objectives = np.zeros(self.n_objectives)

        for _ in range(n_steps):
            graph_data = env.get_graph_data().to(self.device)

            with torch.no_grad():
                state_emb = self.encoder(graph_data)

            env_mask = env.get_action_mask()
            policy_mask = flatten_env_mask_to_policy_mask(
                env_mask,
                n_spmts=active_policy.n_spmts,
                n_cranes=active_policy.n_cranes,
                max_requests=active_policy.max_requests,
            )
            torch_mask = to_torch_mask(policy_mask, device=self.device)

            with torch.no_grad():
                action, log_prob, value = active_policy.get_action(state_emb, torch_mask)

            action_cpu = {k: int(v.item()) for k, v in action.items()}
            next_obs, reward, terminated, truncated, info = env.step(action_cpu)
            done = terminated or truncated

            # Extract multi-objective rewards from environment
            reward_vector = self._extract_objectives(env, reward, info)
            episode_objectives += reward_vector

            # Store with scalarized reward
            scalarized_reward = self.scalarize_reward(reward_vector, self.current_weight)

            self.buffer["graph_data"].append(graph_data.cpu())
            self.buffer["actions"].append(action)
            self.buffer["rewards"].append(scalarized_reward)
            self.buffer["dones"].append(done)
            self.buffer["log_probs"].append(log_prob)
            self.buffer["values"].append(value.squeeze(0).detach())
            self.buffer["masks"].append(torch_mask)

            if done:
                # Update reference points
                self.utopia_point = np.maximum(self.utopia_point, episode_objectives)
                if np.any(self.nadir_point == np.inf):
                    self.nadir_point = episode_objectives.copy()
                else:
                    self.nadir_point = np.minimum(self.nadir_point, episode_objectives)

                episode_objectives = np.zeros(self.n_objectives)
                obs, info = env.reset()

        return self._compute_returns_and_advantages()

    def _extract_objectives(
        self,
        env,
        scalar_reward: float,
        info: Dict[str, Any],
    ) -> np.ndarray:
        """Extract individual objective values from environment.

        Override this method for custom objective definitions.

        Default objectives:
        0. Throughput reward (from scalar reward)
        1. Tardiness penalty (negative of tardiness)
        2. Health reward (equipment health preservation)
        3. Efficiency (negative of idle time)
        """
        objectives = np.zeros(self.n_objectives)

        # Objective 0: Throughput (base reward)
        objectives[0] = scalar_reward

        # Objective 1: Tardiness (from info or metrics)
        if hasattr(env, "metrics"):
            tardiness = env.metrics.get("total_tardiness", 0)
            objectives[1] = -tardiness / 100  # Normalize and negate (minimize)

        # Objective 2: Health preservation
        if hasattr(env, "spmts") and hasattr(env, "cranes"):
            avg_health = np.mean([
                *[s.health for s in env.spmts],
                *[c.health for c in env.cranes],
            ])
            objectives[2] = avg_health

        # Objective 3: Efficiency (from utilization)
        if hasattr(env, "metrics"):
            utilization = env.metrics.get("utilization", 0.5)
            objectives[3] = utilization

        return objectives

    def _compute_returns_and_advantages(self) -> Dict[str, Any]:
        """Compute GAE advantages for scalarized rewards."""
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
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "graph_data": self.buffer["graph_data"],
            "actions": self.buffer["actions"],
            "log_probs": torch.stack(self.buffer["log_probs"]),
            "returns": returns,
            "advantages": advantages,
            "masks": self.buffer["masks"],
            "weight": self.current_weight,
        }

    def update(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """Perform PPO update."""
        graph_data_list = rollout_data["graph_data"]
        actions = rollout_data["actions"]
        old_log_probs = rollout_data["log_probs"]
        returns = rollout_data["returns"]
        advantages = rollout_data["advantages"]
        masks_list = rollout_data["masks"]
        weight = rollout_data["weight"]

        # Get active policy
        if self.scalarization == "hypernetwork":
            weight_tensor = torch.tensor(weight, device=self.device, dtype=torch.float32)
            active_policy = self.hypernet(weight_tensor)
        else:
            active_policy = self.policy

        n_samples = len(graph_data_list)
        metrics = {"policy_loss": [], "value_loss": [], "entropy": []}

        for _ in range(self.n_epochs):
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_graphs = Batch.from_data_list(
                    [graph_data_list[i] for i in batch_indices]
                ).to(self.device)
                batch_states = self.encoder(batch_graphs)

                batch_actions = {
                    k: torch.stack([actions[i][k] for i in batch_indices])
                    for k in actions[0]
                }
                batch_mask = batch_masks([masks_list[i] for i in batch_indices])
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                log_probs, entropy, values = active_policy.evaluate_action(
                    batch_states, batch_actions, batch_mask
                )

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(active_policy.parameters()) + list(self.encoder.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.mean().item())

        # Clear buffer
        for k in self.buffer:
            self.buffer[k] = []

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def evaluate_and_archive(self, env, n_episodes: int = 5) -> np.ndarray:
        """Evaluate current policy and add to Pareto archive if non-dominated.

        Returns:
            Mean objective values across evaluation episodes.
        """
        if self.scalarization == "hypernetwork":
            weight_tensor = torch.tensor(self.current_weight, device=self.device, dtype=torch.float32)
            active_policy = self.hypernet(weight_tensor)
        else:
            active_policy = self.policy

        all_objectives = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_obj = np.zeros(self.n_objectives)

            while not done:
                graph_data = env.get_graph_data().to(self.device)
                with torch.no_grad():
                    state = self.encoder(graph_data)
                    env_mask = env.get_action_mask()
                    policy_mask = flatten_env_mask_to_policy_mask(
                        env_mask, active_policy.n_spmts, active_policy.n_cranes, active_policy.max_requests
                    )
                    torch_mask = to_torch_mask(policy_mask, device=self.device)
                    action, _, _ = active_policy.get_action(state, torch_mask)

                action_cpu = {k: int(v.item()) for k, v in action.items()}
                _, reward, terminated, truncated, info = env.step(action_cpu)
                done = terminated or truncated
                episode_obj += self._extract_objectives(env, reward, info)

            all_objectives.append(episode_obj)

        mean_objectives = np.mean(all_objectives, axis=0)

        # Add to Pareto archive
        solution = ParetoSolution(
            policy_state=copy.deepcopy(active_policy.state_dict()),
            encoder_state=copy.deepcopy(self.encoder.state_dict()),
            objectives=mean_objectives,
            weight_vector=self.current_weight.copy(),
        )
        self.pareto_archive.add(solution)

        return mean_objectives

    def get_pareto_front(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get the current Pareto front.

        Returns:
            List of (objectives, weight) tuples for each solution in the archive.
        """
        return [
            (sol.objectives, sol.weight_vector)
            for sol in self.pareto_archive.solutions
        ]

    def load_pareto_solution(self, weight: np.ndarray) -> None:
        """Load the Pareto solution closest to given weight preference."""
        solution = self.pareto_archive.get_solution_for_weight(weight)
        if solution is not None:
            self.policy.load_state_dict(solution.policy_state)
            self.encoder.load_state_dict(solution.encoder_state)
