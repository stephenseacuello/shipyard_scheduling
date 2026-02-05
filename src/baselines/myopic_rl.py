"""Myopic reinforcement‑learning‑style scheduler without health awareness.

This baseline ignores equipment health when making decisions. It chooses
random valid actions from the available mask, without learning long‑term
consequences. It serves as a lower bound on performance relative to the
trained RL agent.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any


class MyopicRLScheduler:
    def decide(self, env) -> Dict[str, Any]:
        mask = env.get_action_mask()
        # Choose action type randomly among valid ones
        valid_types = [i for i, v in enumerate(mask["action_type"]) if v]
        if not valid_types:
            return {"action_type": 3, "spmt_idx": 0, "request_idx": 0, "crane_idx": 0, "lift_idx": 0, "equipment_idx": 0}
        action_type = int(np.random.choice(valid_types))
        if action_type == 0 and mask["spmt_dispatch"].size > 0 and mask["spmt_dispatch"].any():
            coords = np.argwhere(mask["spmt_dispatch"])
            spmt_idx, req_idx = coords[np.random.randint(len(coords))]
            return {"action_type": 0, "spmt_idx": int(spmt_idx), "request_idx": int(req_idx), "crane_idx": 0, "lift_idx": 0, "equipment_idx": 0}
        if action_type == 1 and mask["crane_dispatch"].size > 0 and mask["crane_dispatch"].any():
            coords = np.argwhere(mask["crane_dispatch"])
            crane_idx, req_idx = coords[np.random.randint(len(coords))]
            return {"action_type": 1, "crane_idx": int(crane_idx), "lift_idx": int(req_idx), "spmt_idx": 0, "request_idx": 0, "equipment_idx": 0}
        if action_type == 2 and mask["maintenance"].any():
            equip_ids = np.argwhere(mask["maintenance"]).flatten()
            equip_idx = int(np.random.choice(equip_ids))
            return {"action_type": 2, "equipment_idx": equip_idx, "spmt_idx": 0, "request_idx": 0, "crane_idx": 0, "lift_idx": 0}
        # Default hold
        return {"action_type": 3, "spmt_idx": 0, "request_idx": 0, "crane_idx": 0, "lift_idx": 0, "equipment_idx": 0}