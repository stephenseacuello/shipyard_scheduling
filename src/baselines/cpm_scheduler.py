"""Critical-path-method (CPM) baseline scheduler for shipyard scheduling.

Prioritises requests by slack: blocks with the least remaining slack
(due_date - current_time - estimated_remaining_processing) are dispatched
first.  Also performs preventive maintenance on equipment below a health
threshold.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple

from .rule_based import _is_idle, _hold_action

# Rough estimate: each remaining stage takes ~20 time-units on average.
_HOURS_PER_STAGE = 20


def _slack(env, block_id: str) -> float:
    """Compute scheduling slack for a block.

    slack = due_date - sim_time - remaining_processing_estimate
    Lower (more negative) slack means more critical.
    """
    block = env._get_block(block_id)
    stage_value = (block.current_stage.value
                   if hasattr(block.current_stage, "value")
                   else int(block.current_stage))
    remaining_stages = max(10 - stage_value, 0)
    remaining_processing = remaining_stages * _HOURS_PER_STAGE
    sim_time = getattr(env, "sim_time", 0)
    return block.due_date - sim_time - remaining_processing


def _rank_requests(env, requests: List[dict]) -> List[Tuple[float, int, dict]]:
    """Return requests sorted by slack (ascending = most critical first)."""
    scored = []
    for idx, req in enumerate(requests):
        s = _slack(env, req["block_id"])
        scored.append((s, idx, req))
    scored.sort(key=lambda x: x[0])
    return scored


class CPMScheduler:
    def decide(self, env) -> Dict[str, Any]:
        """Return an action prioritising the most critical block."""
        spmts = env.entities.get("spmts", [])
        cranes = (env.entities.get("cranes", [])
                  or env.entities.get("goliath_cranes", []))
        lift_requests = (getattr(env, "lift_requests", None)
                         or getattr(env, "erection_requests", []))

        # PRIORITY 1: Crane dispatch (lowest slack first)
        if lift_requests:
            ranked = _rank_requests(env, lift_requests)
            for _slack_val, _orig_idx, req in ranked:
                if hasattr(env, "_get_ship") and "ship_id" in req:
                    ship = env._get_ship(req["ship_id"])
                    if not ship or not ship.assigned_dock:
                        continue
                    for ci, crane in enumerate(cranes):
                        if (_is_idle(crane)
                                and hasattr(crane, "assigned_dock")
                                and crane.assigned_dock == ship.assigned_dock):
                            ri = lift_requests.index(req)
                            return {
                                "action_type": 1,
                                "crane_idx": ci, "lift_idx": ri,
                                "erection_idx": ri,
                                "spmt_idx": 0, "request_idx": 0,
                                "equipment_idx": 0,
                            }
                else:
                    for ci, crane in enumerate(cranes):
                        if _is_idle(crane):
                            ri = lift_requests.index(req)
                            return {
                                "action_type": 1,
                                "crane_idx": ci, "lift_idx": ri,
                                "erection_idx": ri,
                                "spmt_idx": 0, "request_idx": 0,
                                "equipment_idx": 0,
                            }

        # PRIORITY 2: Transport dispatch (lowest slack first, nearest SPMT)
        if env.transport_requests:
            ranked = _rank_requests(env, env.transport_requests)
            best_req = ranked[0][2]
            block = env._get_block(best_req["block_id"])
            best_idx = None
            best_time = float("inf")
            for si, spmt in enumerate(spmts):
                if not _is_idle(spmt):
                    continue
                tt = env.shipyard.get_travel_time(
                    spmt.current_location, block.location)
                if tt < best_time:
                    best_time = tt
                    best_idx = si
            if best_idx is not None:
                ri = env.transport_requests.index(best_req)
                return {
                    "action_type": 0,
                    "spmt_idx": best_idx, "request_idx": ri,
                    "crane_idx": 0, "lift_idx": 0,
                    "equipment_idx": 0,
                }

        # PRIORITY 3: Preventive maintenance (health < 40%)
        for i, spmt in enumerate(spmts):
            if _is_idle(spmt) and spmt.get_min_health() < 40.0:
                return {
                    "action_type": 2,
                    "equipment_idx": i,
                    "spmt_idx": 0, "request_idx": 0,
                    "crane_idx": 0, "lift_idx": 0,
                }
        for i, crane in enumerate(cranes):
            if _is_idle(crane) and crane.get_min_health() < 40.0:
                return {
                    "action_type": 2,
                    "equipment_idx": len(spmts) + i,
                    "spmt_idx": 0, "request_idx": 0,
                    "crane_idx": 0, "lift_idx": 0,
                }

        return _hold_action()
