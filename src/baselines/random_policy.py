"""Random baseline scheduler for shipyard scheduling.

Selects uniformly at random among valid actions using the environment's
action mask.  Useful as a lower-bound comparison for learned policies.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any

from .rule_based import _hold_action


class RandomScheduler:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def decide(self, env) -> Dict[str, Any]:
        """Return a random valid action."""
        mask = env.get_action_mask()

        valid_types = np.where(mask["action_type"])[0]
        if len(valid_types) == 0:
            return _hold_action()

        action_type = int(self.rng.choice(valid_types))

        if action_type == 0:  # transport
            valid = np.argwhere(mask["spmt_dispatch"])
            if len(valid) == 0:
                return _hold_action()
            pick = valid[self.rng.integers(len(valid))]
            return {
                "action_type": 0,
                "spmt_idx": int(pick[0]),
                "request_idx": int(pick[1]),
                "crane_idx": 0, "lift_idx": 0,
                "erection_idx": 0, "equipment_idx": 0,
            }

        if action_type == 1:  # crane
            valid = np.argwhere(mask["crane_dispatch"])
            if len(valid) == 0:
                return _hold_action()
            pick = valid[self.rng.integers(len(valid))]
            return {
                "action_type": 1,
                "crane_idx": int(pick[0]),
                "lift_idx": int(pick[1]),
                "erection_idx": int(pick[1]),
                "spmt_idx": 0, "request_idx": 0, "equipment_idx": 0,
            }

        if action_type == 2:  # maintenance
            valid = np.where(mask["maintenance"])[0]
            if len(valid) == 0:
                return _hold_action()
            pick = int(self.rng.choice(valid))
            return {
                "action_type": 2,
                "equipment_idx": pick,
                "spmt_idx": 0, "request_idx": 0,
                "crane_idx": 0, "lift_idx": 0,
            }

        return _hold_action()
