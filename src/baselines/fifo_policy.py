"""FIFO (first-in, first-out) baseline scheduler for shipyard scheduling.

Dispatches blocks in order of their block_id (lowest first), assigning
the nearest idle equipment.  A simple deterministic baseline that ignores
due dates and criticality.
"""

from __future__ import annotations

from typing import Dict, Any

from .rule_based import _is_idle, _hold_action


class FIFOScheduler:
    def decide(self, env) -> Dict[str, Any]:
        """Return an action that dispatches the earliest-created block first."""
        spmts = env.entities.get("spmts", [])
        cranes = (env.entities.get("cranes", [])
                  or env.entities.get("goliath_cranes", []))
        lift_requests = (getattr(env, "lift_requests", None)
                         or getattr(env, "erection_requests", []))

        # PRIORITY 1: Crane dispatch (lowest block_id first)
        if lift_requests:
            reqs = sorted(lift_requests,
                          key=lambda r: r["block_id"])
            for req in reqs:
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

        # PRIORITY 2: Transport dispatch (lowest block_id first, nearest SPMT)
        if env.transport_requests:
            reqs = sorted(env.transport_requests,
                          key=lambda r: r["block_id"])
            best_req = reqs[0]
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

        return _hold_action()
