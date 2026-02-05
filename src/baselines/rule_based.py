"""Rule‑based heuristic for shipyard scheduling.

This scheduler implements a simple earliest‑due‑date (EDD) dispatch rule
combined with a nearest‑vehicle assignment. It ignores long‑term
consequences and acts greedily at each decision epoch.
"""

from __future__ import annotations

from typing import Dict, Any

from shipyard_scheduling.simulation.entities import SPMTStatus, CraneStatus


def _hold_action() -> Dict[str, Any]:
    return {
        "action_type": 3,
        "spmt_idx": 0,
        "request_idx": 0,
        "crane_idx": 0,
        "lift_idx": 0,
        "equipment_idx": 0,
    }


class RuleBasedScheduler:
    def decide(self, env) -> Dict[str, Any]:
        """Return a dictionary describing the next action for the environment."""
        spmts = env.entities.get("spmts", [])
        cranes = env.entities.get("cranes", [])

        # SPMT dispatch: EDD request, nearest idle SPMT
        if env.transport_requests:
            reqs = sorted(
                env.transport_requests,
                key=lambda r: env._get_block(r["block_id"]).due_date,
            )
            best_req = reqs[0]
            block = env._get_block(best_req["block_id"])
            # Find nearest idle SPMT by travel time
            best_spmt_idx = None
            best_time = float("inf")
            for idx, spmt in enumerate(spmts):
                if spmt.status != SPMTStatus.IDLE:
                    continue
                tt = env.shipyard.get_travel_time(spmt.current_location, block.location)
                if tt < best_time:
                    best_time = tt
                    best_spmt_idx = idx
            if best_spmt_idx is not None:
                req_idx = env.transport_requests.index(best_req)
                return {
                    "action_type": 0,
                    "spmt_idx": best_spmt_idx,
                    "request_idx": req_idx,
                    "crane_idx": 0,
                    "lift_idx": 0,
                    "equipment_idx": 0,
                }

        # Crane dispatch: EDD lift request, nearest idle crane
        if env.lift_requests:
            reqs = sorted(
                env.lift_requests,
                key=lambda r: env._get_block(r["block_id"]).due_date,
            )
            best_req = reqs[0]
            best_crane_idx = None
            for idx, crane in enumerate(cranes):
                if crane.status == CraneStatus.IDLE:
                    best_crane_idx = idx
                    break
            if best_crane_idx is not None:
                req_idx = env.lift_requests.index(best_req)
                return {
                    "action_type": 1,
                    "crane_idx": best_crane_idx,
                    "lift_idx": req_idx,
                    "spmt_idx": 0,
                    "request_idx": 0,
                    "equipment_idx": 0,
                }

        # Preventive maintenance if any equipment below health threshold
        for i, spmt in enumerate(spmts):
            if spmt.status == SPMTStatus.IDLE and spmt.get_min_health() < 40.0:
                return {
                    "action_type": 2,
                    "equipment_idx": i,
                    "spmt_idx": 0,
                    "request_idx": 0,
                    "crane_idx": 0,
                    "lift_idx": 0,
                }
        for i, crane in enumerate(cranes):
            if crane.status == CraneStatus.IDLE and crane.get_min_health() < 40.0:
                return {
                    "action_type": 2,
                    "equipment_idx": len(spmts) + i,
                    "spmt_idx": 0,
                    "request_idx": 0,
                    "crane_idx": 0,
                    "lift_idx": 0,
                }

        return _hold_action()
