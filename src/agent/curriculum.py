"""Curriculum learning scheduler.

Curriculum learning gradually increases problem difficulty over training
episodes. This scheduler adjusts the environment configuration to
increase the number of blocks and equipment as training progresses.
"""

from __future__ import annotations

from typing import Dict, Any


class CurriculumScheduler:
    def __init__(self, milestones: Dict[int, Dict[str, Any]]) -> None:
        """Initialize with a mapping from episode index to config overrides.

        Parameters
        ----------
        milestones : dict
            Keys are episode numbers, values are dictionaries specifying
            configuration overrides (e.g. ``{"n_blocks": 100}``). When the
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
