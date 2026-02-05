"""Curriculum learning scheduler for shipyard scheduling RL.

Provides both fixed milestone-based curriculum and adaptive curriculum
that adjusts difficulty based on agent performance.

Enhancements for graduate-level research:
- Adaptive curriculum based on success rate
- Automatic difficulty regression when struggling
- Smooth difficulty transitions
- Detailed curriculum state tracking
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np


class CurriculumScheduler:
    """Fixed milestone-based curriculum scheduler (legacy).

    Gradually increases problem difficulty over training episodes.
    This scheduler adjusts the environment configuration to
    increase the number of blocks and equipment as training progresses.
    """

    def __init__(self, milestones: Dict[int, Dict[str, Any]]) -> None:
        """Initialize with a mapping from episode index to config overrides.

        Parameters
        ----------
        milestones : dict
            Keys are episode numbers, values are dictionaries specifying
            configuration overrides (e.g. ``{"n_blocks": 20}``). When the
            current episode reaches a milestone, the corresponding overrides
            are applied.
        """
        self.milestones = dict(sorted(milestones.items()))

    def get_config(self, base_config: Dict[str, Any], episode: int) -> Dict[str, Any]:
        """Return an updated config for the given episode."""
        config = dict(base_config)
        for ep, overrides in self.milestones.items():
            if episode >= ep:
                config.update(overrides)
        return config

    @classmethod
    def from_config(cls, curriculum_cfg: Dict[str, Any]) -> "CurriculumScheduler":
        """Build from YAML config section.

        Expected format::

            curriculum:
              milestones:
                0: {n_blocks: 20, max_time: 2000}
                5: {n_blocks: 50, max_time: 5000}
                10: {n_blocks: 100, max_time: 10000}
        """
        milestones = {int(k): v for k, v in curriculum_cfg.get("milestones", {}).items()}
        return cls(milestones)


@dataclass
class DifficultyLevel:
    """Defines a curriculum difficulty level."""
    name: str
    n_blocks: int
    n_spmts: int
    n_cranes: int
    max_time: int
    extra_overrides: Dict[str, Any] = field(default_factory=dict)

    def to_config_overrides(self) -> Dict[str, Any]:
        """Convert to config override dictionary."""
        overrides = {
            "n_blocks": self.n_blocks,
            "n_spmts": self.n_spmts,
            "n_cranes": self.n_cranes,
            "max_time": self.max_time,
        }
        overrides.update(self.extra_overrides)
        return overrides


# Default difficulty levels for shipyard scheduling
DEFAULT_DIFFICULTY_LEVELS = [
    DifficultyLevel("trivial", n_blocks=10, n_spmts=2, n_cranes=1, max_time=1000),
    DifficultyLevel("easy", n_blocks=20, n_spmts=3, n_cranes=2, max_time=2000),
    DifficultyLevel("medium", n_blocks=50, n_spmts=5, n_cranes=3, max_time=5000),
    DifficultyLevel("hard", n_blocks=100, n_spmts=8, n_cranes=4, max_time=10000),
    DifficultyLevel("expert", n_blocks=200, n_spmts=12, n_cranes=6, max_time=20000),
]


class AdaptiveCurriculum:
    """Adaptive curriculum learning based on agent performance.

    Advances difficulty when the agent demonstrates consistent success,
    and regresses when the agent struggles. This enables more efficient
    learning by matching task difficulty to agent capability.

    Features:
    - Performance-based advancement (not just epochs)
    - Automatic regression on consistent failure
    - Hysteresis to prevent oscillation
    - Smooth difficulty interpolation (optional)
    - Detailed state tracking for analysis

    Args:
        levels: List of difficulty levels in increasing order.
        advance_threshold: Success rate needed to advance (default: 0.8).
        regress_threshold: Success rate below which to regress (default: 0.3).
        window_size: Number of episodes to consider for success rate (default: 20).
        advance_patience: Consecutive windows above threshold before advancing (default: 3).
        regress_patience: Consecutive windows below threshold before regressing (default: 5).
        min_episodes_per_level: Minimum episodes at each level before considering advancement.
        interpolation_steps: Steps to interpolate between levels (0 for instant transition).
    """

    def __init__(
        self,
        levels: Optional[List[DifficultyLevel]] = None,
        advance_threshold: float = 0.8,
        regress_threshold: float = 0.3,
        window_size: int = 20,
        advance_patience: int = 3,
        regress_patience: int = 5,
        min_episodes_per_level: int = 50,
        interpolation_steps: int = 0,
    ) -> None:
        self.levels = levels if levels is not None else DEFAULT_DIFFICULTY_LEVELS
        self.advance_threshold = advance_threshold
        self.regress_threshold = regress_threshold
        self.window_size = window_size
        self.advance_patience = advance_patience
        self.regress_patience = regress_patience
        self.min_episodes_per_level = min_episodes_per_level
        self.interpolation_steps = interpolation_steps

        # State tracking
        self.current_level_idx = 0
        self.episodes_at_current_level = 0
        self.total_episodes = 0

        # Performance history
        self.success_history: deque = deque(maxlen=window_size * 2)  # Track more for analysis
        self.window_success_rates: List[float] = []

        # Patience counters
        self.consecutive_advance_windows = 0
        self.consecutive_regress_windows = 0

        # Interpolation state
        self.interpolation_progress = 0
        self.is_interpolating = False
        self.interpolation_from_level = 0
        self.interpolation_to_level = 0

        # History for analysis
        self.level_history: List[Tuple[int, int]] = [(0, 0)]  # (episode, level_idx)

    @property
    def current_level(self) -> DifficultyLevel:
        """Get current difficulty level."""
        return self.levels[self.current_level_idx]

    @property
    def max_level_idx(self) -> int:
        """Get maximum level index."""
        return len(self.levels) - 1

    @property
    def current_success_rate(self) -> float:
        """Compute success rate over recent window."""
        if len(self.success_history) == 0:
            return 0.0
        recent = list(self.success_history)[-self.window_size:]
        return np.mean(recent)

    def record_episode(self, success: bool, metrics: Optional[Dict[str, float]] = None) -> None:
        """Record episode result and update curriculum state.

        Args:
            success: Whether the episode was successful (e.g., on_time_rate > threshold).
            metrics: Optional additional metrics for logging.
        """
        self.success_history.append(float(success))
        self.episodes_at_current_level += 1
        self.total_episodes += 1

        # Only evaluate after enough episodes
        if len(self.success_history) < self.window_size:
            return

        # Compute window success rate
        success_rate = self.current_success_rate
        self.window_success_rates.append(success_rate)

        # Check for advancement
        if self._should_advance(success_rate):
            self._advance_level()
        # Check for regression
        elif self._should_regress(success_rate):
            self._regress_level()

    def _should_advance(self, success_rate: float) -> bool:
        """Check if conditions are met for advancing difficulty."""
        if self.current_level_idx >= self.max_level_idx:
            return False

        if self.episodes_at_current_level < self.min_episodes_per_level:
            return False

        if success_rate >= self.advance_threshold:
            self.consecutive_advance_windows += 1
            self.consecutive_regress_windows = 0
        else:
            self.consecutive_advance_windows = 0

        return self.consecutive_advance_windows >= self.advance_patience

    def _should_regress(self, success_rate: float) -> bool:
        """Check if conditions are met for regressing difficulty."""
        if self.current_level_idx <= 0:
            return False

        if success_rate <= self.regress_threshold:
            self.consecutive_regress_windows += 1
        else:
            self.consecutive_regress_windows = 0

        return self.consecutive_regress_windows >= self.regress_patience

    def _advance_level(self) -> None:
        """Advance to next difficulty level."""
        if self.interpolation_steps > 0:
            self._start_interpolation(self.current_level_idx, self.current_level_idx + 1)
        else:
            self.current_level_idx += 1
            self._reset_level_state()
        self.level_history.append((self.total_episodes, self.current_level_idx))

    def _regress_level(self) -> None:
        """Regress to previous difficulty level."""
        if self.interpolation_steps > 0:
            self._start_interpolation(self.current_level_idx, self.current_level_idx - 1)
        else:
            self.current_level_idx -= 1
            self._reset_level_state()
        self.level_history.append((self.total_episodes, self.current_level_idx))

    def _reset_level_state(self) -> None:
        """Reset state when changing levels."""
        self.episodes_at_current_level = 0
        self.consecutive_advance_windows = 0
        self.consecutive_regress_windows = 0

    def _start_interpolation(self, from_idx: int, to_idx: int) -> None:
        """Start smooth interpolation between levels."""
        self.is_interpolating = True
        self.interpolation_from_level = from_idx
        self.interpolation_to_level = to_idx
        self.interpolation_progress = 0

    def _step_interpolation(self) -> None:
        """Advance interpolation by one step."""
        if not self.is_interpolating:
            return

        self.interpolation_progress += 1
        if self.interpolation_progress >= self.interpolation_steps:
            self.current_level_idx = self.interpolation_to_level
            self.is_interpolating = False
            self._reset_level_state()

    def get_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get environment config for current curriculum state.

        Args:
            base_config: Base environment configuration.

        Returns:
            Updated configuration with current difficulty level applied.
        """
        config = dict(base_config)

        if self.is_interpolating:
            # Interpolate between levels
            from_level = self.levels[self.interpolation_from_level]
            to_level = self.levels[self.interpolation_to_level]
            alpha = self.interpolation_progress / self.interpolation_steps

            config.update({
                "n_blocks": int(from_level.n_blocks * (1 - alpha) + to_level.n_blocks * alpha),
                "n_spmts": int(from_level.n_spmts * (1 - alpha) + to_level.n_spmts * alpha),
                "n_cranes": int(from_level.n_cranes * (1 - alpha) + to_level.n_cranes * alpha),
                "max_time": int(from_level.max_time * (1 - alpha) + to_level.max_time * alpha),
            })
            self._step_interpolation()
        else:
            config.update(self.current_level.to_config_overrides())

        return config

    def get_state(self) -> Dict[str, Any]:
        """Get current curriculum state for logging/checkpointing."""
        return {
            "current_level_idx": self.current_level_idx,
            "current_level_name": self.current_level.name,
            "episodes_at_level": self.episodes_at_current_level,
            "total_episodes": self.total_episodes,
            "success_rate": self.current_success_rate,
            "consecutive_advance_windows": self.consecutive_advance_windows,
            "consecutive_regress_windows": self.consecutive_regress_windows,
            "is_interpolating": self.is_interpolating,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load curriculum state from checkpoint."""
        self.current_level_idx = state["current_level_idx"]
        self.episodes_at_current_level = state["episodes_at_level"]
        self.total_episodes = state["total_episodes"]
        self.consecutive_advance_windows = state["consecutive_advance_windows"]
        self.consecutive_regress_windows = state["consecutive_regress_windows"]
        self.is_interpolating = state.get("is_interpolating", False)

    @classmethod
    def from_config(cls, curriculum_cfg: Dict[str, Any]) -> "AdaptiveCurriculum":
        """Build from YAML config section.

        Expected format::

            curriculum:
              type: adaptive
              advance_threshold: 0.8
              regress_threshold: 0.3
              window_size: 20
              advance_patience: 3
              regress_patience: 5
              min_episodes_per_level: 50
              levels:
                - name: easy
                  n_blocks: 20
                  n_spmts: 3
                  n_cranes: 2
                  max_time: 2000
                - name: medium
                  n_blocks: 50
                  n_spmts: 5
                  n_cranes: 3
                  max_time: 5000
        """
        levels = None
        if "levels" in curriculum_cfg:
            levels = [
                DifficultyLevel(
                    name=lvl.get("name", f"level_{i}"),
                    n_blocks=lvl["n_blocks"],
                    n_spmts=lvl["n_spmts"],
                    n_cranes=lvl["n_cranes"],
                    max_time=lvl["max_time"],
                    extra_overrides={k: v for k, v in lvl.items()
                                    if k not in ["name", "n_blocks", "n_spmts", "n_cranes", "max_time"]},
                )
                for i, lvl in enumerate(curriculum_cfg["levels"])
            ]

        return cls(
            levels=levels,
            advance_threshold=curriculum_cfg.get("advance_threshold", 0.8),
            regress_threshold=curriculum_cfg.get("regress_threshold", 0.3),
            window_size=curriculum_cfg.get("window_size", 20),
            advance_patience=curriculum_cfg.get("advance_patience", 3),
            regress_patience=curriculum_cfg.get("regress_patience", 5),
            min_episodes_per_level=curriculum_cfg.get("min_episodes_per_level", 50),
            interpolation_steps=curriculum_cfg.get("interpolation_steps", 0),
        )


def create_curriculum(curriculum_cfg: Dict[str, Any]) -> CurriculumScheduler | AdaptiveCurriculum:
    """Factory function to create the appropriate curriculum scheduler.

    Args:
        curriculum_cfg: Curriculum configuration dictionary.

    Returns:
        Either CurriculumScheduler (milestone-based) or AdaptiveCurriculum (performance-based).
    """
    curriculum_type = curriculum_cfg.get("type", "milestone")

    if curriculum_type == "adaptive":
        return AdaptiveCurriculum.from_config(curriculum_cfg)
    else:
        return CurriculumScheduler.from_config(curriculum_cfg)
