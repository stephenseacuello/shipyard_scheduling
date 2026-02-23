"""Potential-based reward shaping for shipyard scheduling.

Implements reward shaping that preserves optimal policies while providing
denser learning signals. Based on Ng, Harada & Russell (1999).

The shaped reward is: F(s, s') = gamma * Phi(s') - Phi(s)
where Phi is a potential function representing "progress" towards the goal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from simulation.environment import ShipyardEnv


@dataclass
class StateSnapshot:
    """Snapshot of environment state for potential computation."""
    block_stages: np.ndarray  # Stage values for all blocks
    block_completions: np.ndarray  # Completion percentages
    block_urgencies: np.ndarray  # Time to due date
    equipment_health: np.ndarray  # Health values for all equipment
    queue_lengths: np.ndarray  # Queue lengths at facilities
    blocks_completed: int
    sim_time: float


class PotentialBasedRewardShaper:
    """Potential-based reward shaping for shipyard scheduling.

    Computes a potential function based on:
    - Block production progress (stage + completion)
    - Due date urgency (reward progress on urgent blocks more)
    - Equipment health preservation
    - Queue balance (penalize long queues)

    Args:
        gamma: Discount factor (must match RL algorithm's gamma).
        stage_weight: Weight for stage progression potential.
        completion_weight: Weight for completion percentage potential.
        urgency_weight: Weight for urgency-based potential.
        health_weight: Weight for equipment health potential.
        queue_weight: Weight for queue length penalty.
        max_stage: Maximum production stage value.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        stage_weight: float = 0.5,
        completion_weight: float = 0.3,
        urgency_weight: float = 0.2,
        health_weight: float = 0.1,
        queue_weight: float = 0.05,
        max_stage: int = 7,
    ):
        self.gamma = gamma
        self.stage_weight = stage_weight
        self.completion_weight = completion_weight
        self.urgency_weight = urgency_weight
        self.health_weight = health_weight
        self.queue_weight = queue_weight
        self.max_stage = max_stage

    def get_state_snapshot(self, env: "ShipyardEnv") -> StateSnapshot:
        """Extract a snapshot of the current environment state.

        Args:
            env: The shipyard environment.

        Returns:
            StateSnapshot containing relevant state information.
        """
        blocks = env.entities.get("blocks", [])
        spmts = env.entities.get("spmts", [])
        cranes = env.entities.get("cranes", [])

        # Block information
        block_stages = np.array([b.current_stage.value for b in blocks]) if blocks else np.array([])
        block_completions = np.array([b.completion_pct for b in blocks]) if blocks else np.array([])

        # Urgency: higher value for blocks closer to due date
        block_urgencies = np.array([
            1.0 / (1.0 + max(0, b.due_date - env.sim_time) / 100.0)
            for b in blocks
        ]) if blocks else np.array([])

        # Equipment health
        health_values = []
        for spmt in spmts:
            if hasattr(spmt, "get_min_health"):
                health_values.append(spmt.get_min_health() / 100.0)
            elif hasattr(spmt, "health"):
                health_values.append(spmt.health / 100.0)
        for crane in cranes:
            if hasattr(crane, "get_min_health"):
                health_values.append(crane.get_min_health() / 100.0)
            elif hasattr(crane, "health_hoist"):
                health_values.append(min(crane.health_hoist, crane.health_trolley, crane.health_gantry) / 100.0)
        equipment_health = np.array(health_values) if health_values else np.array([1.0])

        # Queue lengths
        queue_lengths = np.array([
            len(q) for q in env.facility_queues.values()
        ]) if hasattr(env, "facility_queues") else np.array([0])

        return StateSnapshot(
            block_stages=block_stages,
            block_completions=block_completions,
            block_urgencies=block_urgencies,
            equipment_health=equipment_health,
            queue_lengths=queue_lengths,
            blocks_completed=env.metrics.get("blocks_completed", 0),
            sim_time=env.sim_time,
        )

    def compute_potential(self, snapshot: StateSnapshot) -> float:
        """Compute the potential value for a state snapshot.

        Higher potential = closer to goal state.

        Args:
            snapshot: State snapshot.

        Returns:
            Potential value (higher is better).
        """
        potential = 0.0

        if len(snapshot.block_stages) > 0:
            # Stage progress potential (normalized to [0, 1])
            stage_progress = snapshot.block_stages / self.max_stage
            potential += self.stage_weight * np.mean(stage_progress)

            # Completion potential
            completion_progress = snapshot.block_completions / 100.0
            potential += self.completion_weight * np.mean(completion_progress)

            # Urgency-weighted progress (reward progress on urgent blocks more)
            urgency_weighted_progress = stage_progress * snapshot.block_urgencies
            potential += self.urgency_weight * np.mean(urgency_weighted_progress)

        # Equipment health potential (preserve health)
        if len(snapshot.equipment_health) > 0:
            potential += self.health_weight * np.mean(snapshot.equipment_health)

        # Queue length penalty (shorter queues = higher potential)
        if len(snapshot.queue_lengths) > 0:
            # Normalize queue lengths (assume max reasonable queue is 20)
            normalized_queue = np.clip(snapshot.queue_lengths / 20.0, 0, 1)
            potential -= self.queue_weight * np.mean(normalized_queue)

        # Bonus for completed blocks
        potential += 0.01 * snapshot.blocks_completed

        return potential

    def shape_reward(
        self,
        snapshot_before: StateSnapshot,
        snapshot_after: StateSnapshot,
        original_reward: float,
    ) -> float:
        """Compute shaped reward using potential difference.

        F(s, s') = gamma * Phi(s') - Phi(s)

        This preserves optimal policies (Ng et al., 1999).

        Args:
            snapshot_before: State before action.
            snapshot_after: State after action.
            original_reward: Original environment reward.

        Returns:
            Shaped reward = original_reward + shaping_bonus.
        """
        phi_before = self.compute_potential(snapshot_before)
        phi_after = self.compute_potential(snapshot_after)

        shaping_bonus = self.gamma * phi_after - phi_before

        return original_reward + shaping_bonus

    def get_shaping_info(
        self,
        snapshot_before: StateSnapshot,
        snapshot_after: StateSnapshot,
    ) -> Dict[str, float]:
        """Get detailed information about the shaping bonus.

        Useful for debugging and understanding what drives the shaping signal.
        """
        phi_before = self.compute_potential(snapshot_before)
        phi_after = self.compute_potential(snapshot_after)
        shaping_bonus = self.gamma * phi_after - phi_before

        return {
            "phi_before": phi_before,
            "phi_after": phi_after,
            "shaping_bonus": shaping_bonus,
            "gamma": self.gamma,
        }


class HindsightRewardShaper:
    """Hindsight Experience Replay (HER) inspired reward modification.

    After an episode, relabels goals to provide additional learning signal
    from failed trajectories. Adapted for shipyard scheduling where we can
    treat completed blocks as "achieved goals".
    """

    def __init__(
        self,
        k_goals: int = 4,
        strategy: str = "future",
    ):
        """
        Args:
            k_goals: Number of additional goals to sample per transition.
            strategy: Goal sampling strategy ('future', 'episode', 'random').
        """
        self.k_goals = k_goals
        self.strategy = strategy

    def relabel_episode(
        self,
        episode_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Relabel an episode with hindsight goals.

        For shipyard scheduling, we can relabel with blocks that were
        eventually completed as if they were the "target" blocks at
        earlier timesteps.

        Args:
            episode_data: Dictionary containing episode transitions.

        Returns:
            Augmented episode data with relabeled transitions.
        """
        # This is a placeholder for the full HER implementation
        # In practice, this would involve:
        # 1. Identifying completed blocks at end of episode
        # 2. Sampling past states where those blocks were in earlier stages
        # 3. Relabeling with the completed block as the "goal"
        # 4. Computing new rewards based on goal achievement

        return episode_data


class NStepRewardAccumulator:
    """Accumulator for n-step returns computation.

    Collects rewards over n steps and computes n-step returns
    for better credit assignment.
    """

    def __init__(self, n_steps: int = 3, gamma: float = 0.99):
        """
        Args:
            n_steps: Number of steps for n-step returns.
            gamma: Discount factor.
        """
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = []

    def add(self, reward: float, done: bool, value: float) -> Optional[float]:
        """Add a transition and potentially return an n-step return.

        Args:
            reward: Immediate reward.
            done: Whether episode terminated.
            value: Value estimate for bootstrapping.

        Returns:
            N-step return if buffer is full, None otherwise.
        """
        self.buffer.append((reward, done, value))

        if len(self.buffer) >= self.n_steps:
            return self._compute_n_step_return()
        elif done:
            # Compute partial return at episode end
            return self._compute_n_step_return()

        return None

    def _compute_n_step_return(self) -> float:
        """Compute n-step return from buffer."""
        n_step_return = 0.0

        for i, (reward, done, value) in enumerate(self.buffer):
            n_step_return += (self.gamma ** i) * reward
            if done:
                break

        # Bootstrap with value if not terminated
        if not self.buffer[-1][1]:  # not done
            n_step_return += (self.gamma ** len(self.buffer)) * self.buffer[-1][2]

        # Remove oldest transition
        self.buffer.pop(0)

        return n_step_return

    def reset(self) -> None:
        """Reset the buffer."""
        self.buffer = []
