"""Rule‑based heuristic for shipyard scheduling.

This scheduler implements a simple earliest‑due‑date (EDD) dispatch rule
combined with a nearest‑vehicle assignment. It ignores long‑term
consequences and acts greedily at each decision epoch.
"""

from __future__ import annotations

from typing import Dict, Any


def _is_idle(entity) -> bool:
    """Check if entity is idle, handling different enum import paths."""
    status = entity.status
    # Handle enum comparison across different import paths
    if hasattr(status, 'name'):
        return status.name == 'IDLE'
    return str(status) == 'IDLE' or 'IDLE' in str(status)


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
        # Support both ShipyardEnv (cranes) and HHIShipyardEnv (goliath_cranes)
        cranes = env.entities.get("cranes", []) or env.entities.get("goliath_cranes", [])
        # Support both lift_requests (ShipyardEnv) and erection_requests (HHIShipyardEnv)
        lift_requests = getattr(env, "lift_requests", None) or getattr(env, "erection_requests", [])

        # PRIORITY 1: Crane dispatch (erection is closest to completion - critical path)
        # Dispatch cranes first to prevent backlog at pre-erection staging
        if lift_requests:
            # Sort by due date (earliest first)
            reqs = sorted(
                lift_requests,
                key=lambda r: env._get_block(r["block_id"]).due_date,
            )
            # Find a request with a matching idle crane
            for req in reqs:
                # Check if env has _get_ship (HHIShipyardEnv) for dock-based matching
                if hasattr(env, '_get_ship') and "ship_id" in req:
                    ship = env._get_ship(req["ship_id"])
                    if not ship or not ship.assigned_dock:
                        continue
                    # Find the crane that serves this ship's dock
                    for crane_idx, crane in enumerate(cranes):
                        if _is_idle(crane) and hasattr(crane, 'assigned_dock'):
                            if crane.assigned_dock == ship.assigned_dock:
                                req_idx = lift_requests.index(req)
                                return {
                                    "action_type": 1,
                                    "crane_idx": crane_idx,
                                    "lift_idx": req_idx,
                                    "erection_idx": req_idx,
                                    "spmt_idx": 0,
                                    "request_idx": 0,
                                    "equipment_idx": 0,
                                }
                else:
                    # Fallback for ShipyardEnv: just find any idle crane
                    for crane_idx, crane in enumerate(cranes):
                        if _is_idle(crane):
                            req_idx = lift_requests.index(req)
                            return {
                                "action_type": 1,
                                "crane_idx": crane_idx,
                                "lift_idx": req_idx,
                                "erection_idx": req_idx,
                                "spmt_idx": 0,
                                "request_idx": 0,
                                "equipment_idx": 0,
                            }

        # PRIORITY 1.5: Urgent procurement when any material is in stockout
        if hasattr(env, 'enable_suppliers') and env.enable_suppliers:
            for inv_idx, inv in enumerate(env.entities.get("inventory", [])):
                if inv.is_stockout() or inv.quantity < inv.reorder_point * 0.5:
                    best_sup_idx = None
                    best_price = float("inf")
                    for sup_idx, sup in enumerate(env.entities.get("suppliers", [])):
                        avail_cap = sup.capacity_per_period - sup.current_backlog
                        if (inv.material_type.value in sup.specializations
                                and avail_cap > 0
                                and sup.price_per_unit < best_price):
                            best_price = sup.price_per_unit
                            best_sup_idx = sup_idx
                    if best_sup_idx is not None:
                        action = _hold_action()
                        action["action_type"] = env._action_type_map["PLACE_ORDER"]
                        action["supplier_idx"] = best_sup_idx
                        action["material_idx"] = inv_idx
                        return action

        # PRIORITY 2: SPMT dispatch - EDD request, nearest idle SPMT
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
                if not _is_idle(spmt):
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

        # PRIORITY 3: Preventive maintenance if any equipment below health threshold
        for i, spmt in enumerate(spmts):
            if _is_idle(spmt) and spmt.get_min_health() < 40.0:
                return {
                    "action_type": 2,
                    "equipment_idx": i,
                    "spmt_idx": 0,
                    "request_idx": 0,
                    "crane_idx": 0,
                    "lift_idx": 0,
                }
        for i, crane in enumerate(cranes):
            if _is_idle(crane) and crane.get_min_health() < 40.0:
                return {
                    "action_type": 2,
                    "equipment_idx": len(spmts) + i,
                    "spmt_idx": 0,
                    "request_idx": 0,
                    "crane_idx": 0,
                    "lift_idx": 0,
                }

        # PRIORITY 4: Procurement when material below reorder point (cheapest capable supplier)
        if hasattr(env, 'enable_suppliers') and env.enable_suppliers:
            for inv_idx, inv in enumerate(env.entities.get("inventory", [])):
                if inv.is_below_reorder_point():
                    best_sup_idx = None
                    best_price = float("inf")
                    for sup_idx, sup in enumerate(env.entities.get("suppliers", [])):
                        # Order qty capped by supplier available capacity
                        avail_cap = sup.capacity_per_period - sup.current_backlog
                        if (inv.material_type.value in sup.specializations
                                and avail_cap > 0
                                and sup.price_per_unit < best_price):
                            best_price = sup.price_per_unit
                            best_sup_idx = sup_idx
                    if best_sup_idx is not None:
                        action = _hold_action()
                        action["action_type"] = env._action_type_map["PLACE_ORDER"]
                        action["supplier_idx"] = best_sup_idx
                        action["material_idx"] = inv_idx
                        return action

        # PRIORITY 5: Worker assignment for blocks in processing (skill match)
        if hasattr(env, 'enable_labor') and env.enable_labor:
            for pool_idx, pool in enumerate(env.entities.get("labor_pools", [])):
                if pool.can_assign():
                    for blk_idx, block in enumerate(env.entities.get("blocks", [])):
                        if block.status.name == "IN_PROCESS":
                            required_skills = env._STAGE_SKILLS.get(block.current_stage, [])
                            if pool.skill_type in required_skills:
                                action = _hold_action()
                                action["action_type"] = env._action_type_map["ASSIGN_WORKER"]
                                action["labor_pool_idx"] = pool_idx
                                action["target_block_idx"] = blk_idx
                                return action

        return _hold_action()
