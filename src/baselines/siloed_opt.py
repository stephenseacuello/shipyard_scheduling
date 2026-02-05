"""Siloed optimization baseline.

This baseline decomposes the integrated scheduling problem into three
independent sub-optimizers run in sequence:
  1. Production sequencing: EDD/SPT priority on facility queues
  2. SPMT routing: greedy nearest-neighbor assignment
  3. Maintenance: threshold-based condition-based maintenance (CBM)

Each sub-optimizer is unaware of the others, serving as a reference
for measuring the benefit of integrated decision making.
"""

from __future__ import annotations

from typing import Dict, Any, List

from shipyard_scheduling.simulation.entities import SPMTStatus, CraneStatus


class _ProductionSequencer:
    """Selects the highest-priority transport/lift request by EDD."""

    def select_transport_request(self, env) -> int | None:
        if not env.transport_requests:
            return None
        reqs = sorted(
            range(len(env.transport_requests)),
            key=lambda i: env._get_block(env.transport_requests[i]["block_id"]).due_date,
        )
        return reqs[0]

    def select_lift_request(self, env) -> int | None:
        if not env.lift_requests:
            return None
        reqs = sorted(
            range(len(env.lift_requests)),
            key=lambda i: env._get_block(env.lift_requests[i]["block_id"]).due_date,
        )
        return reqs[0]


class _TransportRouter:
    """Assigns SPMTs using greedy nearest-neighbor from distance matrix."""

    def select_spmt(self, env, block_location: str) -> int | None:
        spmts = env.entities.get("spmts", [])
        best_idx = None
        best_time = float("inf")
        for idx, spmt in enumerate(spmts):
            if spmt.status != SPMTStatus.IDLE:
                continue
            tt = env.shipyard.get_travel_time(spmt.current_location, block_location)
            if tt < best_time:
                best_time = tt
                best_idx = idx
        return best_idx

    def select_crane(self, env) -> int | None:
        cranes = env.entities.get("cranes", [])
        for idx, crane in enumerate(cranes):
            if crane.status == CraneStatus.IDLE:
                return idx
        return None


class _MaintenanceScheduler:
    """Independent CBM: triggers maintenance when health drops below threshold."""

    def __init__(self, threshold: float = 40.0) -> None:
        self.threshold = threshold

    def get_maintenance_target(self, env) -> int | None:
        spmts = env.entities.get("spmts", [])
        cranes = env.entities.get("cranes", [])
        for i, spmt in enumerate(spmts):
            if spmt.status == SPMTStatus.IDLE and spmt.get_min_health() < self.threshold:
                return i
        for i, crane in enumerate(cranes):
            if crane.status == CraneStatus.IDLE and crane.get_min_health() < self.threshold:
                return len(spmts) + i
        return None


class SiloedOptimizationScheduler:
    def __init__(self, maintenance_threshold: float = 40.0) -> None:
        self.sequencer = _ProductionSequencer()
        self.router = _TransportRouter()
        self.maintenance = _MaintenanceScheduler(threshold=maintenance_threshold)

    def decide(self, env) -> Dict[str, Any]:
        # 1. Maintenance check first (independent of production)
        maint_target = self.maintenance.get_maintenance_target(env)
        if maint_target is not None:
            return {
                "action_type": 2,
                "equipment_idx": maint_target,
                "spmt_idx": 0,
                "request_idx": 0,
                "crane_idx": 0,
                "lift_idx": 0,
            }

        # 2. Production sequencing + transport routing
        req_idx = self.sequencer.select_transport_request(env)
        if req_idx is not None:
            block = env._get_block(env.transport_requests[req_idx]["block_id"])
            spmt_idx = self.router.select_spmt(env, block.location)
            if spmt_idx is not None:
                return {
                    "action_type": 0,
                    "spmt_idx": spmt_idx,
                    "request_idx": req_idx,
                    "crane_idx": 0,
                    "lift_idx": 0,
                    "equipment_idx": 0,
                }

        # 3. Crane dispatch
        lift_idx = self.sequencer.select_lift_request(env)
        if lift_idx is not None:
            crane_idx = self.router.select_crane(env)
            if crane_idx is not None:
                return {
                    "action_type": 1,
                    "crane_idx": crane_idx,
                    "lift_idx": lift_idx,
                    "spmt_idx": 0,
                    "request_idx": 0,
                    "equipment_idx": 0,
                }

        # 4. Hold
        return {
            "action_type": 3,
            "spmt_idx": 0,
            "request_idx": 0,
            "crane_idx": 0,
            "lift_idx": 0,
            "equipment_idx": 0,
        }
