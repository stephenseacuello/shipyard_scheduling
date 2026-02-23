"""Noise injection for robust policy training.

Adds observation and action noise to train policies that are
robust to sensor noise and action execution errors.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from simulation.environment import ShipyardEnv


@dataclass
class NoiseConfig:
    """Configuration for noise injection."""
    # Observation noise
    obs_noise_type: str = "gaussian"  # 'gaussian', 'uniform', 'none'
    obs_noise_std: float = 0.05
    obs_noise_clip: float = 0.2

    # Action noise (execution errors)
    action_noise_prob: float = 0.05  # Probability of random action
    action_delay_prob: float = 0.02  # Probability of action delay

    # Environment dynamics noise
    process_time_noise_std: float = 0.1
    travel_time_noise_std: float = 0.1

    # Enable/disable
    training_only: bool = True  # Only inject noise during training


class ObservationNoiseInjector:
    """Injects noise into observations."""

    def __init__(self, config: NoiseConfig):
        self.config = config

    def inject(self, obs: np.ndarray) -> np.ndarray:
        """Add noise to observation.

        Args:
            obs: Original observation array.

        Returns:
            Noisy observation.
        """
        if self.config.obs_noise_type == "none":
            return obs

        if self.config.obs_noise_type == "gaussian":
            noise = np.random.normal(0, self.config.obs_noise_std, obs.shape)
        elif self.config.obs_noise_type == "uniform":
            noise = np.random.uniform(
                -self.config.obs_noise_std,
                self.config.obs_noise_std,
                obs.shape
            )
        else:
            return obs

        # Clip noise
        noise = np.clip(noise, -self.config.obs_noise_clip, self.config.obs_noise_clip)

        return obs + noise

    def inject_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        """Add noise to observation tensor.

        Args:
            obs: Original observation tensor.

        Returns:
            Noisy observation tensor.
        """
        if self.config.obs_noise_type == "none":
            return obs

        if self.config.obs_noise_type == "gaussian":
            noise = torch.randn_like(obs) * self.config.obs_noise_std
        elif self.config.obs_noise_type == "uniform":
            noise = (torch.rand_like(obs) * 2 - 1) * self.config.obs_noise_std
        else:
            return obs

        # Clip
        noise = torch.clamp(noise, -self.config.obs_noise_clip, self.config.obs_noise_clip)

        return obs + noise


class ActionNoiseInjector:
    """Injects noise into actions (execution errors)."""

    def __init__(
        self,
        config: NoiseConfig,
        n_action_types: int = 4,
        n_spmts: int = 12,
        n_cranes: int = 3,
        max_requests: int = 200,
    ):
        self.config = config
        self.n_action_types = n_action_types
        self.n_spmts = n_spmts
        self.n_cranes = n_cranes
        self.max_requests = max_requests

    def inject(self, action: Dict[str, int]) -> Dict[str, int]:
        """Add noise to action.

        Args:
            action: Original action dictionary.

        Returns:
            Potentially modified action.
        """
        # Random action with probability
        if np.random.random() < self.config.action_noise_prob:
            return self._random_action()

        return action

    def _random_action(self) -> Dict[str, int]:
        """Generate a random valid action."""
        action_type = np.random.randint(0, self.n_action_types)

        return {
            "action_type": action_type,
            "spmt": np.random.randint(0, self.n_spmts),
            "request": np.random.randint(0, max(1, self.max_requests)),
            "crane": np.random.randint(0, self.n_cranes),
            "lift": np.random.randint(0, max(1, self.max_requests)),
            "equipment": np.random.randint(0, self.n_spmts + self.n_cranes),
        }


class NoisyEnvWrapper:
    """Environment wrapper that injects noise.

    Wraps a shipyard environment to add observation and action
    noise for robust training.
    """

    def __init__(
        self,
        env: "ShipyardEnv",
        config: Optional[NoiseConfig] = None,
        training: bool = True,
    ):
        self.env = env
        self.config = config or NoiseConfig()
        self.training = training

        self.obs_injector = ObservationNoiseInjector(self.config)
        self.action_injector = ActionNoiseInjector(
            self.config,
            n_action_types=4,
            n_spmts=len(env.entities.get("spmts", [])),
            n_cranes=len(env.entities.get("cranes", [])),
            max_requests=len(getattr(env, "transport_requests", [])) or 200,
        )

    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset with optional noise injection."""
        obs, info = self.env.reset(**kwargs)

        if self.training or not self.config.training_only:
            obs = self.obs_injector.inject(obs)

        return obs, info

    def step(self, action: Dict[str, int]) -> Tuple[Any, float, bool, bool, Dict]:
        """Step with noise injection.

        Args:
            action: Action dictionary.

        Returns:
            Standard gym step outputs with noisy observation.
        """
        # Inject action noise
        if self.training or not self.config.training_only:
            action = self.action_injector.inject(action)

        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Inject observation noise
        if self.training or not self.config.training_only:
            obs = self.obs_injector.inject(obs)

        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        """Forward attribute access to wrapped environment."""
        return getattr(self.env, name)


class ProcessTimeNoise:
    """Adds noise to processing times during simulation.

    Injects multiplicative noise to processing times to model
    uncertainty in production durations.
    """

    def __init__(self, std: float = 0.1, seed: Optional[int] = None):
        self.std = std
        if seed is not None:
            np.random.seed(seed)

    def sample_multiplier(self) -> float:
        """Sample a processing time multiplier.

        Returns:
            Multiplier in [1-3*std, 1+3*std] with std deviation std.
        """
        return np.clip(
            np.random.lognormal(0, self.std),
            1 - 3 * self.std,
            1 + 3 * self.std
        )

    def apply(self, base_time: float) -> float:
        """Apply noise to a processing time.

        Args:
            base_time: Base processing time.

        Returns:
            Noisy processing time.
        """
        return base_time * self.sample_multiplier()


class TravelTimeNoise:
    """Adds noise to travel times during simulation.

    Models traffic, congestion, and path variations.
    """

    def __init__(self, std: float = 0.1, congestion_factor: float = 0.0):
        self.std = std
        self.congestion_factor = congestion_factor

    def apply(
        self,
        base_time: float,
        n_active_spmts: int = 0,
        total_spmts: int = 1,
    ) -> float:
        """Apply noise to travel time.

        Args:
            base_time: Base travel time.
            n_active_spmts: Number of SPMTs currently traveling.
            total_spmts: Total number of SPMTs.

        Returns:
            Noisy travel time.
        """
        # Base noise
        noise_multiplier = np.random.lognormal(0, self.std)

        # Congestion effect
        if total_spmts > 0:
            utilization = n_active_spmts / total_spmts
            congestion_multiplier = 1 + self.congestion_factor * utilization
        else:
            congestion_multiplier = 1.0

        return base_time * noise_multiplier * congestion_multiplier


def create_noisy_env(
    env: "ShipyardEnv",
    obs_noise_std: float = 0.05,
    action_noise_prob: float = 0.05,
    training: bool = True,
) -> NoisyEnvWrapper:
    """Convenience function to create a noisy environment.

    Args:
        env: Base environment.
        obs_noise_std: Observation noise standard deviation.
        action_noise_prob: Probability of random action.
        training: Whether in training mode.

    Returns:
        Wrapped noisy environment.
    """
    config = NoiseConfig(
        obs_noise_std=obs_noise_std,
        action_noise_prob=action_noise_prob,
    )
    return NoisyEnvWrapper(env, config, training)
