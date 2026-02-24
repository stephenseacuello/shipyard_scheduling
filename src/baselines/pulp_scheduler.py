"""Mixed-Integer Programming scheduler using PuLP (CBC solver).

Solves a rolling-horizon MIP that optimizes block-to-SPMT assignment,
crane scheduling, and maintenance decisions jointly.

Mirrors the interface of RollingHorizonMPC (src/baselines/mpc_scheduler.py)
but uses PuLP's CBC solver instead of OR-Tools CP-SAT.

Decision variables:
- x[b,s,t]: binary — SPMT s transports block b at time step t
- y[c,r,t]: binary — crane c erects request r at time step t
- m[e,t]: binary — equipment e under maintenance at time step t

Objective: minimize weighted_tardiness + transport_cost + idle_dock_time
Subject to: one-assignment-per-block, SPMT/crane capacity per period,
            maintenance prevents operation

Uses plate-count processing times when available (via block.n_plates).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from simulation.shipyard_env import HHIShipyardEnv

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


def _hold_action() -> Dict[str, Any]:
    return {
        "action_type": 3,
        "spmt_idx": 0,
        "request_idx": 0,
        "crane_idx": 0,
        "lift_idx": 0,
        "erection_idx": 0,
        "equipment_idx": 0,
    }


def _is_idle(entity) -> bool:
    """Check if entity is idle, handling different enum import paths."""
    status = entity.status
    if hasattr(status, 'name'):
        return status.name == 'IDLE'
    return 'IDLE' in str(status)


class PuLPMIPScheduler:
    """Mixed-Integer Programming scheduler using PuLP (CBC solver).

    Uses a rolling-horizon approach: solves a MIP over a planning window,
    executes the first few actions, then re-solves.

    Args:
        config: Optional configuration dictionary with keys:
            horizon: Planning window length (default 50)
            replan_interval: Steps between re-solves (default 10)
            solver_time_limit: Max solver time in seconds (default 5.0)
            objective_weights: Dict of weight names to floats
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        config = config or {}
        self.horizon: int = config.get("horizon", 20)
        self.replan_interval: int = config.get("replan_interval", 5)
        self.solver_time_limit: float = config.get("solver_time_limit", 0.5)
        self.objective_weights: Dict[str, float] = config.get("objective_weights", {
            "tardiness": 10.0,
            "transport_cost": 1.0,
            "idle_time": 5.0,
            "maintenance_opportunity": 0.5,
        })
        self._steps_since_plan: int = 0
        self._cached_plan: List[Dict[str, Any]] = []
        self._plan_index: int = 0
        self._last_n_reqs: int = 0  # Track request count for reactive replanning

    def reset(self) -> None:
        """Reset planner state for a new episode."""
        self._steps_since_plan = 0
        self._cached_plan = []
        self._plan_index = 0
        self._last_n_reqs = 0

    def decide(self, env) -> Dict[str, Any]:
        """Select next action using MIP solution.

        Same interface as RuleBasedScheduler, SiloedOptimizationScheduler, etc.
        Falls back to EDD dispatch if PuLP is unavailable or solver fails.
        """
        if not HAS_PULP:
            return self._fallback_edd(env)

        # Count current requests to detect new work
        trans_reqs = getattr(env, "transport_requests", [])
        erect_reqs = getattr(env, "erection_requests", []) or getattr(env, "lift_requests", [])
        n_erect = len(erect_reqs)
        current_n_reqs = len(trans_reqs) + n_erect

        # When erection requests exist, use fast EDD path for immediate crane
        # dispatch — avoids solver overhead on time-critical erection decisions
        if n_erect > 0:
            edd_action = self._fallback_edd(env)
            if edd_action.get("action_type") == 1:
                self._steps_since_plan += 1
                return edd_action

        # Replan on interval or when request count changes
        needs_replan = (
            not self._cached_plan
            or self._plan_index >= len(self._cached_plan)
            or self._steps_since_plan >= self.replan_interval
            or current_n_reqs != self._last_n_reqs
        )

        if needs_replan:
            self._cached_plan = self._solve(env)
            self._plan_index = 0
            self._steps_since_plan = 0
            self._last_n_reqs = current_n_reqs

        # Execute next action from plan
        if self._plan_index < len(self._cached_plan):
            action = self._cached_plan[self._plan_index]
            self._plan_index += 1
            self._steps_since_plan += 1
            return self._validate_action(action, env)

        self._steps_since_plan += 1
        return self._fallback_edd(env)

    # ------------------------------------------------------------------
    # MIP construction
    # ------------------------------------------------------------------
    def _solve(self, env) -> List[Dict[str, Any]]:
        """Build and solve the MIP, return action sequence."""
        try:
            model = self._build_model(env)
            if model is None:
                return self._greedy_plan(env)
            return self._extract_actions(model, env)
        except Exception:
            return self._greedy_plan(env)

    def _build_model(self, env) -> Optional["pulp.LpProblem"]:
        """Build the rolling-horizon MIP from current environment state."""
        transport_reqs = list(getattr(env, "transport_requests", []))
        erection_reqs = list(getattr(env, "erection_requests", []) or getattr(env, "lift_requests", []))
        spmts = list(env.entities.get("spmts", []))
        cranes = env.entities.get("cranes", []) or env.entities.get("goliath_cranes", [])
        cranes = list(cranes)

        n_trans = len(transport_reqs)
        n_erect = len(erection_reqs)
        n_spmts = len(spmts)
        n_cranes = len(cranes)

        # Nothing to schedule
        if n_trans == 0 and n_erect == 0:
            return None

        H = min(self.horizon, max(n_trans + n_erect + 5, 10))
        w = self.objective_weights

        prob = pulp.LpProblem("ShipyardMIP", pulp.LpMinimize)

        # ---- Decision variables ----

        # x[b, s, t] = 1 if SPMT s dispatched for transport request b at time t
        x = {}
        for b in range(n_trans):
            for s in range(n_spmts):
                for t in range(H):
                    x[b, s, t] = pulp.LpVariable(f"x_{b}_{s}_{t}", cat="Binary")

        # y[r, c, t] = 1 if crane c assigned to erection request r at time t
        y = {}
        for r in range(n_erect):
            for c in range(n_cranes):
                for t in range(H):
                    y[r, c, t] = pulp.LpVariable(f"y_{r}_{c}_{t}", cat="Binary")

        # m[e, t] = 1 if equipment e is under maintenance at time t
        # Only create maintenance vars for equipment below health threshold
        n_equip = n_spmts + n_cranes
        maint_candidates = set()
        for e in range(n_equip):
            if e < n_spmts:
                health = spmts[e].get_min_health()
            else:
                health = cranes[e - n_spmts].get_min_health()
            if health < 30.0:
                maint_candidates.add(e)

        m = {}
        for e in range(n_equip):
            for t in range(H):
                if e in maint_candidates:
                    m[e, t] = pulp.LpVariable(f"m_{e}_{t}", cat="Binary")
                else:
                    m[e, t] = pulp.LpVariable(f"m_{e}_{t}", cat="Binary", upBound=0)

        # ---- Constraints ----

        # Each transport request assigned to at most one SPMT at one time
        for b in range(n_trans):
            prob += (
                pulp.lpSum(x[b, s, t] for s in range(n_spmts) for t in range(H)) <= 1,
                f"trans_assign_{b}",
            )

        # Each erection request assigned to at most one crane at one time
        for r in range(n_erect):
            prob += (
                pulp.lpSum(y[r, c, t] for c in range(n_cranes) for t in range(H)) <= 1,
                f"erect_assign_{r}",
            )

        # SPMT can do at most one transport per time step
        for s in range(n_spmts):
            for t in range(H):
                prob += (
                    pulp.lpSum(x[b, s, t] for b in range(n_trans)) + m[s, t] <= 1,
                    f"spmt_cap_{s}_{t}",
                )

        # Crane can do at most one erection per time step
        for c in range(n_cranes):
            for t in range(H):
                prob += (
                    pulp.lpSum(y[r, c, t] for r in range(n_erect)) + m[n_spmts + c, t] <= 1,
                    f"crane_cap_{c}_{t}",
                )

        # Only idle equipment can be assigned (non-idle SPMTs/cranes get 0)
        for s in range(n_spmts):
            if not _is_idle(spmts[s]):
                for b in range(n_trans):
                    for t in range(min(2, H)):  # Lock out for first few steps
                        prob += (x[b, s, t] == 0, f"spmt_busy_{s}_{b}_{t}")

        for c in range(n_cranes):
            if not _is_idle(cranes[c]):
                for r in range(n_erect):
                    for t in range(min(2, H)):
                        prob += (y[r, c, t] == 0, f"crane_busy_{c}_{r}_{t}")

        # ---- Objective ----
        obj_terms = []

        # 1) Tardiness: penalize late transport/erection by time step
        for b in range(n_trans):
            block = env._get_block(transport_reqs[b]["block_id"])
            urgency = max(0.0, 1.0 - (block.due_date - env.sim_time) / max(block.due_date, 1.0))
            for s in range(n_spmts):
                for t in range(H):
                    # Penalize later scheduling of urgent blocks
                    obj_terms.append(
                        w.get("tardiness", 10.0) * (urgency + 0.01) * t * x[b, s, t]
                    )

        for r in range(n_erect):
            block = env._get_block(erection_reqs[r]["block_id"])
            urgency = max(0.0, 1.0 - (block.due_date - env.sim_time) / max(block.due_date, 1.0))
            for c in range(n_cranes):
                for t in range(H):
                    obj_terms.append(
                        w.get("tardiness", 10.0) * (urgency + 0.01) * t * y[r, c, t]
                    )

        # 2) Transport cost: estimated travel time for SPMT dispatch
        for b in range(n_trans):
            req = transport_reqs[b]
            block = env._get_block(req["block_id"])
            for s in range(n_spmts):
                travel = env.shipyard.get_travel_time(
                    spmts[s].current_location, block.location
                )
                obj_terms.append(
                    w.get("transport_cost", 1.0) * travel * x[b, s, 0]
                )

        # 3) Idle time: encourage productive assignments (transport + erection only)
        for t in range(H):
            productive_assigned = (
                pulp.lpSum(x[b, s, t] for b in range(n_trans) for s in range(n_spmts))
                + pulp.lpSum(y[r, c, t] for r in range(n_erect) for c in range(n_cranes))
            )
            # Negate to penalize fewer productive assignments (minimize negative = maximize)
            obj_terms.append(-w.get("idle_time", 5.0) * productive_assigned)

        # 4) Maintenance opportunity: only encourage for genuinely low-health equipment
        for e in range(n_equip):
            if e < n_spmts:
                health = spmts[e].get_min_health()
            else:
                health = cranes[e - n_spmts].get_min_health()
            # Only incentivize maintenance below 30% health
            if health < 30.0:
                maint_benefit = (30.0 - health) / 30.0
                for t in range(H):
                    obj_terms.append(
                        -w.get("maintenance_opportunity", 0.5) * maint_benefit * m[e, t]
                    )

        if obj_terms:
            prob += pulp.lpSum(obj_terms)

        # Solve
        solver = pulp.PULP_CBC_CMD(
            msg=0,
            timeLimit=self.solver_time_limit,
        )
        status = prob.solve(solver)

        if status != pulp.constants.LpStatusOptimal:
            # Accept feasible solutions too
            if prob.status != 1:  # Not optimal
                # Check if any variables were set
                any_set = any(
                    v.varValue is not None and v.varValue > 0.5
                    for v in prob.variables()
                )
                if not any_set:
                    return None

        # Store solution data for extraction
        prob._x = x
        prob._y = y
        prob._m = m
        prob._n_trans = n_trans
        prob._n_erect = n_erect
        prob._n_spmts = n_spmts
        prob._n_cranes = n_cranes
        prob._n_equip = n_equip
        prob._H = H
        prob._transport_reqs = transport_reqs
        prob._erection_reqs = erection_reqs

        return prob

    def _extract_actions(self, prob: "pulp.LpProblem", env) -> List[Dict[str, Any]]:
        """Extract an ordered action sequence from the solved MIP."""
        x = prob._x
        y = prob._y
        m = prob._m
        H = prob._H
        n_trans = prob._n_trans
        n_erect = prob._n_erect
        n_spmts = prob._n_spmts
        n_cranes = prob._n_cranes
        n_equip = prob._n_equip
        transport_reqs = prob._transport_reqs
        erection_reqs = prob._erection_reqs

        schedule: List[Dict[str, Any]] = []

        for t in range(H):
            action_found = False

            # Check crane/erection assignments first (higher priority)
            for r in range(n_erect):
                for c in range(n_cranes):
                    val = y[r, c, t].varValue
                    if val is not None and val > 0.5:
                        schedule.append({
                            "action_type": 1,
                            "crane_idx": c,
                            "lift_idx": r,
                            "erection_idx": r,
                            "spmt_idx": 0,
                            "request_idx": 0,
                            "equipment_idx": 0,
                        })
                        action_found = True
                        break
                if action_found:
                    break

            if action_found:
                continue

            # Check SPMT transport assignments
            for b in range(n_trans):
                for s in range(n_spmts):
                    val = x[b, s, t].varValue
                    if val is not None and val > 0.5:
                        schedule.append({
                            "action_type": 0,
                            "spmt_idx": s,
                            "request_idx": b,
                            "crane_idx": 0,
                            "lift_idx": 0,
                            "equipment_idx": 0,
                        })
                        action_found = True
                        break
                if action_found:
                    break

            if action_found:
                continue

            # Check maintenance assignments
            for e in range(n_equip):
                val = m[e, t].varValue
                if val is not None and val > 0.5:
                    schedule.append({
                        "action_type": 2,
                        "equipment_idx": e,
                        "spmt_idx": 0,
                        "request_idx": 0,
                        "crane_idx": 0,
                        "lift_idx": 0,
                    })
                    action_found = True
                    break

            if not action_found:
                schedule.append(_hold_action())

        return schedule

    # ------------------------------------------------------------------
    # Validation and fallbacks
    # ------------------------------------------------------------------
    def _validate_action(self, action: Dict[str, Any], env) -> Dict[str, Any]:
        """Validate an action against current env state; fall back to EDD if invalid."""
        atype = action.get("action_type", 3)

        if atype == 0:
            # Transport: check request and SPMT still valid
            req_idx = action.get("request_idx", 0)
            spmt_idx = action.get("spmt_idx", 0)
            trans_reqs = getattr(env, "transport_requests", [])
            spmts = env.entities.get("spmts", [])
            if (req_idx < len(trans_reqs)
                    and spmt_idx < len(spmts)
                    and _is_idle(spmts[spmt_idx])):
                return action

        elif atype == 1:
            # Erection: check request and crane still valid
            req_idx = action.get("lift_idx", action.get("erection_idx", 0))
            crane_idx = action.get("crane_idx", 0)
            erect_reqs = getattr(env, "erection_requests", []) or getattr(env, "lift_requests", [])
            cranes = env.entities.get("cranes", []) or env.entities.get("goliath_cranes", [])
            if (req_idx < len(erect_reqs)
                    and crane_idx < len(cranes)
                    and _is_idle(cranes[crane_idx])):
                return action

        elif atype == 2:
            # Maintenance: check equipment exists and is idle
            equip_idx = action.get("equipment_idx", 0)
            spmts = env.entities.get("spmts", [])
            cranes = env.entities.get("cranes", []) or env.entities.get("goliath_cranes", [])
            if equip_idx < len(spmts) and _is_idle(spmts[equip_idx]):
                return action
            elif equip_idx < len(spmts) + len(cranes):
                c_idx = equip_idx - len(spmts)
                if _is_idle(cranes[c_idx]):
                    return action

        elif atype == 3:
            return action

        # Invalid action — fall back to EDD for this step
        return self._fallback_edd(env)

    def _fallback_edd(self, env) -> Dict[str, Any]:
        """Earliest Due Date fallback when PuLP is unavailable or solver fails."""
        spmts = env.entities.get("spmts", [])
        cranes = env.entities.get("cranes", []) or env.entities.get("goliath_cranes", [])
        erect_reqs = getattr(env, "erection_requests", []) or getattr(env, "lift_requests", [])
        trans_reqs = getattr(env, "transport_requests", [])

        # Priority 1: crane dispatch
        if erect_reqs:
            sorted_reqs = sorted(
                range(len(erect_reqs)),
                key=lambda i: env._get_block(erect_reqs[i]["block_id"]).due_date,
            )
            for r_idx in sorted_reqs:
                req = erect_reqs[r_idx]
                if hasattr(env, '_get_ship') and "ship_id" in req:
                    ship = env._get_ship(req["ship_id"])
                    if not ship or not ship.assigned_dock:
                        continue
                    for c_idx, crane in enumerate(cranes):
                        if _is_idle(crane) and hasattr(crane, 'assigned_dock'):
                            if crane.assigned_dock == ship.assigned_dock:
                                return {
                                    "action_type": 1,
                                    "crane_idx": c_idx,
                                    "lift_idx": r_idx,
                                    "erection_idx": r_idx,
                                    "spmt_idx": 0,
                                    "request_idx": 0,
                                    "equipment_idx": 0,
                                }

        # Priority 2: SPMT dispatch
        if trans_reqs:
            sorted_reqs = sorted(
                range(len(trans_reqs)),
                key=lambda i: env._get_block(trans_reqs[i]["block_id"]).due_date,
            )
            for b_idx in sorted_reqs:
                block = env._get_block(trans_reqs[b_idx]["block_id"])
                best_spmt = None
                best_time = float("inf")
                for s_idx, spmt in enumerate(spmts):
                    if _is_idle(spmt):
                        tt = env.shipyard.get_travel_time(spmt.current_location, block.location)
                        if tt < best_time:
                            best_time = tt
                            best_spmt = s_idx
                if best_spmt is not None:
                    return {
                        "action_type": 0,
                        "spmt_idx": best_spmt,
                        "request_idx": b_idx,
                        "crane_idx": 0,
                        "lift_idx": 0,
                        "equipment_idx": 0,
                    }

        # Priority 3: maintenance (only for genuinely degraded equipment)
        for i, spmt in enumerate(spmts):
            if _is_idle(spmt) and spmt.get_min_health() < 30.0:
                return {
                    "action_type": 2,
                    "equipment_idx": i,
                    "spmt_idx": 0,
                    "request_idx": 0,
                    "crane_idx": 0,
                    "lift_idx": 0,
                }

        return _hold_action()

    def _greedy_plan(self, env) -> List[Dict[str, Any]]:
        """Build a simple greedy plan when MIP fails."""
        plan = []
        for _ in range(self.replan_interval):
            plan.append(self._fallback_edd(env))
        return plan

    def _get_processing_estimate(self, block, facility_name: str) -> float:
        """Estimate processing time — uses plate count if available."""
        if hasattr(block, 'n_plates') and block.n_plates > 0:
            # Rough estimate: base + per_plate * n_plates
            return 5.0 + 0.2 * block.n_plates
        return 10.0  # Default estimate
