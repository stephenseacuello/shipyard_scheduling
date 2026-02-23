"""Domain randomization for robust policy training.

Implements parameter randomization to train policies that generalize
across different environment conditions. This helps with sim-to-real
transfer and robustness to uncertainty.

Randomizes:
- Processing times
- Travel times
- Equipment degradation rates
- Block weights
- Due date slack
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
from copy import deepcopy


@dataclass
class RandomizationRange:
    """Range for parameter randomization."""
    min_scale: float = 1.0
    max_scale: float = 1.0
    distribution: str = "uniform"  # 'uniform', 'normal', 'log_uniform'

    def sample(self) -> float:
        """Sample a scale factor from the range."""
        if self.distribution == "uniform":
            return np.random.uniform(self.min_scale, self.max_scale)
        elif self.distribution == "normal":
            mean = (self.min_scale + self.max_scale) / 2
            std = (self.max_scale - self.min_scale) / 4
            return np.clip(np.random.normal(mean, std), self.min_scale, self.max_scale)
        elif self.distribution == "log_uniform":
            log_min = np.log(self.min_scale)
            log_max = np.log(self.max_scale)
            return np.exp(np.random.uniform(log_min, log_max))
        else:
            return 1.0


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization."""
    # Processing time randomization
    processing_time: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.7, 1.3)
    )

    # Travel time randomization
    travel_time: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.8, 1.2)
    )

    # Equipment degradation rate randomization
    # Reduced from (0.5, 2.0) to (0.8, 1.2) - ±100% variance was destabilizing value function
    degradation_rate: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.8, 1.2)
    )

    # Block weight randomization
    block_weight: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.9, 1.1)
    )

    # Due date slack randomization
    due_date_slack: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.8, 1.2)
    )

    # Initial health randomization
    initial_health: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.8, 1.0)
    )

    # Breakdown probability randomization
    breakdown_prob: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.5, 1.5)
    )

    # Enable/disable specific randomizations
    enabled: Dict[str, bool] = field(default_factory=lambda: {
        "processing_time": True,
        "travel_time": True,
        "degradation_rate": True,
        "block_weight": True,
        "due_date_slack": True,
        "initial_health": True,
        "breakdown_prob": False,  # Disabled by default (can destabilize training)
    })


class DomainRandomizer:
    """Domain randomization for environment parameters.

    Applies randomization to environment configuration to train
    robust policies that generalize across different conditions.

    Args:
        config: Randomization configuration.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        config: Optional[DomainRandomizationConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or DomainRandomizationConfig()
        if seed is not None:
            np.random.seed(seed)

        self._current_scales: Dict[str, float] = {}

    def randomize_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a randomized environment configuration.

        Args:
            base_config: Base configuration dictionary.

        Returns:
            Randomized configuration.
        """
        config = deepcopy(base_config)
        self._current_scales = {}

        # Randomize processing times
        if self.config.enabled.get("processing_time", False):
            scale = self.config.processing_time.sample()
            self._current_scales["processing_time"] = scale
            config = self._scale_processing_times(config, scale)

        # Randomize travel times
        if self.config.enabled.get("travel_time", False):
            scale = self.config.travel_time.sample()
            self._current_scales["travel_time"] = scale
            config = self._scale_travel_times(config, scale)

        # Randomize degradation rates
        if self.config.enabled.get("degradation_rate", False):
            scale = self.config.degradation_rate.sample()
            self._current_scales["degradation_rate"] = scale
            config = self._scale_degradation_rates(config, scale)

        # Randomize block weights
        if self.config.enabled.get("block_weight", False):
            scale = self.config.block_weight.sample()
            self._current_scales["block_weight"] = scale
            config = self._scale_block_weights(config, scale)

        # Randomize due date slack
        if self.config.enabled.get("due_date_slack", False):
            scale = self.config.due_date_slack.sample()
            self._current_scales["due_date_slack"] = scale
            config = self._scale_due_dates(config, scale)

        return config

    def _scale_processing_times(
        self, config: Dict[str, Any], scale: float
    ) -> Dict[str, Any]:
        """Scale processing times in config."""
        for zone in ["steel_processing", "panel_assembly", "block_assembly", "pre_erection"]:
            if zone in config:
                zone_config = config[zone]
                for facility in zone_config.get("facilities", []):
                    if "processing_time_mean" in facility:
                        facility["processing_time_mean"] *= scale
                    if "processing_time" in facility:
                        facility["processing_time"] *= scale

        return config

    def _scale_travel_times(
        self, config: Dict[str, Any], scale: float
    ) -> Dict[str, Any]:
        """Scale travel times in config."""
        if "transport_network" in config:
            network = config["transport_network"]
            if "travel_times" in network:
                for key in network["travel_times"]:
                    network["travel_times"][key] *= scale

        return config

    def _scale_degradation_rates(
        self, config: Dict[str, Any], scale: float
    ) -> Dict[str, Any]:
        """Scale equipment degradation rates."""
        if "equipment" in config:
            equip = config["equipment"]
            if "degradation_rate" in equip:
                equip["degradation_rate"] *= scale
            if "spmts" in equip:
                for spmt in equip["spmts"]:
                    if "degradation_rate" in spmt:
                        spmt["degradation_rate"] *= scale

        return config

    def _scale_block_weights(
        self, config: Dict[str, Any], scale: float
    ) -> Dict[str, Any]:
        """Scale block weights in config."""
        if "blocks" in config:
            blocks = config["blocks"]
            if "weight_mean" in blocks:
                blocks["weight_mean"] *= scale
            if "weight_range" in blocks:
                blocks["weight_range"] = (
                    blocks["weight_range"][0] * scale,
                    blocks["weight_range"][1] * scale,
                )

        return config

    def _scale_due_dates(
        self, config: Dict[str, Any], scale: float
    ) -> Dict[str, Any]:
        """Scale due date slack in config."""
        if "blocks" in config:
            blocks = config["blocks"]
            if "due_date_slack" in blocks:
                blocks["due_date_slack"] *= scale

        return config

    def get_current_scales(self) -> Dict[str, float]:
        """Get the current randomization scales (for logging)."""
        return self._current_scales.copy()

    def reset_scales(self) -> None:
        """Reset to no randomization (all scales = 1.0)."""
        self._current_scales = {}


class RobustEnvWrapper:
    """Environment wrapper that applies domain randomization on reset.

    Wraps an environment and randomizes its configuration each time
    it is reset, to train more robust policies.

    Args:
        env_class: Environment class to wrap.
        base_config: Base configuration.
        randomizer: Domain randomizer instance.
        randomize_on_reset: If True, randomize on each reset.
    """

    def __init__(
        self,
        env_class,
        base_config: Dict[str, Any],
        randomizer: Optional[DomainRandomizer] = None,
        randomize_on_reset: bool = True,
    ):
        self.env_class = env_class
        self.base_config = base_config
        self.randomizer = randomizer or DomainRandomizer()
        self.randomize_on_reset = randomize_on_reset

        # Create initial environment
        self.env = self.env_class(base_config)
        self._last_scales: Dict[str, float] = {}

    def reset(self, **kwargs):
        """Reset environment with optional randomization."""
        if self.randomize_on_reset:
            randomized_config = self.randomizer.randomize_config(self.base_config)
            self._last_scales = self.randomizer.get_current_scales()
            self.env = self.env_class(randomized_config)
        return self.env.reset(**kwargs)

    def step(self, action):
        """Step the environment."""
        return self.env.step(action)

    def get_randomization_info(self) -> Dict[str, float]:
        """Get info about the current randomization."""
        return self._last_scales

    def __getattr__(self, name):
        """Forward attribute access to wrapped environment."""
        return getattr(self.env, name)


class AdversarialRandomizer(DomainRandomizer):
    """Adversarial domain randomization.

    Focuses randomization on parameters that challenge the current policy,
    creating a curriculum of increasingly difficult environments.

    Args:
        config: Randomization configuration.
        difficulty_schedule: Function mapping training step to difficulty level.
    """

    def __init__(
        self,
        config: Optional[DomainRandomizationConfig] = None,
        difficulty_schedule: Optional[callable] = None,
    ):
        super().__init__(config)

        self.difficulty_schedule = difficulty_schedule or (lambda step: min(1.0, step / 10000))
        self.training_step = 0

    def update_difficulty(self, step: int) -> None:
        """Update the difficulty level based on training progress."""
        self.training_step = step
        difficulty = self.difficulty_schedule(step)

        # Expand randomization ranges based on difficulty
        base_range = 0.1  # Minimum range
        max_range = 0.5  # Maximum range at full difficulty

        current_range = base_range + (max_range - base_range) * difficulty

        # Update ranges
        for attr in ["processing_time", "travel_time", "degradation_rate"]:
            rand_range = getattr(self.config, attr)
            rand_range.min_scale = 1.0 - current_range
            rand_range.max_scale = 1.0 + current_range

    def randomize_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Randomize with current difficulty level."""
        self.update_difficulty(self.training_step)
        return super().randomize_config(base_config)
