"""Priority dispatch rules for shipyard scheduling baseline comparisons.

Implements classic scheduling heuristics:
- EDD: Earliest Due Date
- SPT: Shortest Processing Time
- CR: Critical Ratio
- WSJF: Weighted Shortest Job First
- FIFO: First In, First Out
- WSPT: Weighted Shortest Processing Time

These serve as baselines for comparing RL performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from simulation.environment import ShipyardEnv


class DispatchRule(Enum):
    """Available dispatch rules."""
    EDD = "edd"           # Earliest Due Date
    SPT = "spt"           # Shortest Processing Time
    CR = "cr"             # Critical Ratio
    WSJF = "wsjf"         # Weighted Shortest Job First
    FIFO = "fifo"         # First In, First Out
    WSPT = "wspt"         # Weighted Shortest Processing Time
    SLACK = "slack"       # Minimum Slack Time
    COVERT = "covert"     # Cost Over Time (weighted tardiness)


@dataclass
class DispatchDecision:
    """A dispatch decision from a priority rule."""
    action_type: int  # 0=SPMT dispatch, 1=crane dispatch, 2=maintenance, 3=hold
    spmt_idx: Optional[int] = None
    request_idx: Optional[int] = None
    crane_idx: Optional[int] = None
    lift_idx: Optional[int] = None
    equipment_idx: Optional[int] = None
    priority_score: float = 0.0


class PriorityDispatchScheduler:
    """Scheduler using priority dispatch rules.

    Implements classic scheduling heuristics for baseline comparisons.
    Each rule assigns priorities to pending transport requests, then
    dispatches the highest-priority request to the nearest available
    equipment.

    Args:
        rule: The dispatch rule to use.
        maintenance_threshold: Health threshold for triggering maintenance.
        tiebreaker: Secondary rule for breaking ties.
    """

    def __init__(
        self,
        rule: DispatchRule = DispatchRule.EDD,
        maintenance_threshold: float = 30.0,
        tiebreaker: Optional[DispatchRule] = None,
    ):
        self.rule = rule
        self.maintenance_threshold = maintenance_threshold
        self.tiebreaker = tiebreaker

        # Rule implementations
        self._priority_functions: Dict[DispatchRule, Callable] = {
            DispatchRule.EDD: self._priority_edd,
            DispatchRule.SPT: self._priority_spt,
            DispatchRule.CR: self._priority_cr,
            DispatchRule.WSJF: self._priority_wsjf,
            DispatchRule.FIFO: self._priority_fifo,
            DispatchRule.WSPT: self._priority_wspt,
            DispatchRule.SLACK: self._priority_slack,
            DispatchRule.COVERT: self._priority_covert,
        }

    def _priority_edd(
        self, request: Dict, current_time: float, **kwargs
    ) -> float:
        """Earliest Due Date: prioritize blocks due soonest."""
        due_date = request.get("due_date", float("inf"))
        return -due_date  # Negative so earlier = higher priority

    def _priority_spt(
        self, request: Dict, current_time: float, **kwargs
    ) -> float:
        """Shortest Processing Time: prioritize quick jobs."""
        processing_time = request.get("processing_time", 1.0)
        return -processing_time  # Negative so shorter = higher priority

    def _priority_cr(
        self, request: Dict, current_time: float, **kwargs
    ) -> float:
        """Critical Ratio: (due_date - now) / remaining_processing_time.

        Lower CR = more critical = higher priority.
        """
        due_date = request.get("due_date", current_time + 1000)
        remaining_time = request.get("remaining_processing_time", 1.0)
        slack = due_date - current_time
        cr = slack / max(remaining_time, 0.1)
        return -cr  # Negative so lower CR = higher priority

    def _priority_wsjf(
        self, request: Dict, current_time: float, **kwargs
    ) -> float:
        """Weighted Shortest Job First (SAFe methodology).

        Priority = Cost of Delay / Job Size
        Cost of Delay increases as due date approaches.
        """
        due_date = request.get("due_date", current_time + 1000)
        processing_time = request.get("processing_time", 1.0)
        weight = request.get("weight", 1.0)

        slack = max(1.0, due_date - current_time)
        cost_of_delay = weight / slack  # Higher weight and lower slack = higher cost
        job_size = processing_time

        return cost_of_delay / max(job_size, 0.1)  # Higher = higher priority

    def _priority_fifo(
        self, request: Dict, current_time: float, **kwargs
    ) -> float:
        """First In, First Out: prioritize by arrival time."""
        arrival_time = request.get("arrival_time", 0.0)
        return -arrival_time  # Earlier arrival = higher priority

    def _priority_wspt(
        self, request: Dict, current_time: float, **kwargs
    ) -> float:
        """Weighted Shortest Processing Time.

        Priority = Weight / Processing Time
        """
        weight = request.get("weight", 1.0)
        processing_time = request.get("processing_time", 1.0)
        return weight / max(processing_time, 0.1)

    def _priority_slack(
        self, request: Dict, current_time: float, **kwargs
    ) -> float:
        """Minimum Slack: prioritize jobs with least slack time."""
        due_date = request.get("due_date", current_time + 1000)
        remaining_time = request.get("remaining_processing_time", 1.0)
        slack = (due_date - current_time) - remaining_time
        return -slack  # Lower slack = higher priority

    def _priority_covert(
        self, request: Dict, current_time: float, **kwargs
    ) -> float:
        """Cost Over Time (COVERT): expected tardiness cost per time unit.

        Prioritizes jobs likely to be tardy with high tardiness cost.
        """
        due_date = request.get("due_date", current_time + 1000)
        processing_time = request.get("processing_time", 1.0)
        weight = request.get("weight", 1.0)  # Tardiness cost weight

        slack = due_date - current_time - processing_time

        if slack >= 0:
            # Won't be tardy if started now
            urgency = max(0, 1 - slack / max(processing_time, 0.1))
        else:
            # Already late
            urgency = 1.0 + abs(slack) / max(processing_time, 0.1)

        return (weight * urgency) / max(processing_time, 0.1)

    def compute_priority(
        self,
        request: Dict,
        current_time: float,
    ) -> float:
        """Compute priority score for a transport request.

        Args:
            request: Dictionary with request information.
            current_time: Current simulation time.

        Returns:
            Priority score (higher = higher priority).
        """
        priority_fn = self._priority_functions[self.rule]
        primary_priority = priority_fn(request, current_time)

        # Apply tiebreaker if specified
        if self.tiebreaker is not None:
            tiebreak_fn = self._priority_functions[self.tiebreaker]
            tiebreak_priority = tiebreak_fn(request, current_time) * 1e-6
            return primary_priority + tiebreak_priority

        return primary_priority

    def select_request(
        self,
        requests: List[Dict],
        current_time: float,
    ) -> Optional[int]:
        """Select the highest-priority request.

        Args:
            requests: List of pending transport requests.
            current_time: Current simulation time.

        Returns:
            Index of selected request, or None if no requests.
        """
        if not requests:
            return None

        priorities = [
            self.compute_priority(req, current_time)
            for req in requests
        ]

        return int(np.argmax(priorities))

    def select_equipment(
        self,
        equipment: List[Dict],
        request: Dict,
    ) -> Optional[int]:
        """Select equipment for the request (nearest idle).

        Args:
            equipment: List of available equipment (SPMTs or cranes).
            request: The transport request.

        Returns:
            Index of selected equipment, or None if none available.
        """
        idle_equipment = [
            (i, eq) for i, eq in enumerate(equipment)
            if eq.get("status") == "idle" or eq.get("status", "").lower() == "idle"
        ]

        if not idle_equipment:
            return None

        # Find nearest by travel time
        request_location = request.get("location", request.get("from_location", ""))

        best_idx = idle_equipment[0][0]
        best_time = float("inf")

        for idx, eq in idle_equipment:
            eq_location = eq.get("location", eq.get("current_location", ""))
            travel_time = eq.get("travel_times", {}).get(request_location, 10.0)
            if travel_time < best_time:
                best_time = travel_time
                best_idx = idx

        return best_idx

    def needs_maintenance(self, equipment: Dict) -> bool:
        """Check if equipment needs maintenance.

        Args:
            equipment: Equipment state dictionary.

        Returns:
            True if maintenance should be triggered.
        """
        health = equipment.get("health", equipment.get("min_health", 100.0))
        return health < self.maintenance_threshold

    def decide(self, env: "ShipyardEnv") -> DispatchDecision:
        """Make a dispatch decision based on the current environment state.

        Args:
            env: The shipyard environment.

        Returns:
            DispatchDecision containing the action to take.
        """
        current_time = env.sim_time

        # Check for maintenance needs first
        for i, spmt in enumerate(env.entities.get("spmts", [])):
            spmt_dict = self._entity_to_dict(spmt)
            if self.needs_maintenance(spmt_dict):
                return DispatchDecision(
                    action_type=2,  # Maintenance
                    equipment_idx=i,
                    priority_score=float("inf"),
                )

        # Get transport requests
        requests = []
        for i, req in enumerate(getattr(env, "transport_requests", [])):
            req_dict = self._request_to_dict(req, i)
            requests.append(req_dict)

        if not requests:
            # No pending requests, hold
            return DispatchDecision(action_type=3, priority_score=0.0)

        # Select highest-priority request
        request_idx = self.select_request(requests, current_time)
        if request_idx is None:
            return DispatchDecision(action_type=3, priority_score=0.0)

        selected_request = requests[request_idx]
        priority = self.compute_priority(selected_request, current_time)

        # Select equipment
        spmts = [
            self._entity_to_dict(spmt)
            for spmt in env.entities.get("spmts", [])
        ]
        spmt_idx = self.select_equipment(spmts, selected_request)

        if spmt_idx is None:
            # No available SPMT, hold
            return DispatchDecision(action_type=3, priority_score=0.0)

        return DispatchDecision(
            action_type=0,  # SPMT dispatch
            spmt_idx=spmt_idx,
            request_idx=request_idx,
            priority_score=priority,
        )

    def _entity_to_dict(self, entity) -> Dict:
        """Convert an entity object to a dictionary."""
        if isinstance(entity, dict):
            return entity

        result = {}
        if hasattr(entity, "status"):
            result["status"] = str(entity.status.name if hasattr(entity.status, "name") else entity.status)
        if hasattr(entity, "current_location"):
            result["location"] = entity.current_location
        if hasattr(entity, "get_min_health"):
            result["health"] = entity.get_min_health()
        elif hasattr(entity, "health"):
            result["health"] = entity.health

        return result

    def _request_to_dict(self, request, idx: int) -> Dict:
        """Convert a transport request to a dictionary."""
        if isinstance(request, dict):
            return {**request, "index": idx}

        result = {"index": idx}

        if hasattr(request, "block"):
            block = request.block
            if hasattr(block, "due_date"):
                result["due_date"] = block.due_date
            if hasattr(block, "weight"):
                result["weight"] = block.weight
            if hasattr(block, "creation_time"):
                result["arrival_time"] = block.creation_time

        if hasattr(request, "from_location"):
            result["from_location"] = request.from_location
        if hasattr(request, "to_location"):
            result["to_location"] = request.to_location
        if hasattr(request, "processing_time"):
            result["processing_time"] = request.processing_time

        return result

    def to_action_dict(self, decision: DispatchDecision) -> Dict[str, int]:
        """Convert decision to environment action dictionary."""
        return {
            "action_type": decision.action_type,
            "spmt": decision.spmt_idx or 0,
            "request": decision.request_idx or 0,
            "crane": decision.crane_idx or 0,
            "lift": decision.lift_idx or 0,
            "equipment": decision.equipment_idx or 0,
        }


def run_episode_with_rule(
    env: "ShipyardEnv",
    rule: DispatchRule = DispatchRule.EDD,
    max_steps: int = 10000,
) -> Dict[str, Any]:
    """Run a full episode using a dispatch rule.

    Args:
        env: The shipyard environment.
        rule: The dispatch rule to use.
        max_steps: Maximum steps before truncation.

    Returns:
        Dictionary with episode statistics.
    """
    scheduler = PriorityDispatchScheduler(rule)

    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    for step in range(max_steps):
        decision = scheduler.decide(env)
        action = scheduler.to_action_dict(decision)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    return {
        "rule": rule.value,
        "total_reward": total_reward,
        "steps": steps,
        "blocks_completed": env.metrics.get("blocks_completed", 0),
        "throughput": env.metrics.get("blocks_completed", 0) / max(env.sim_time, 1),
        "total_tardiness": env.metrics.get("total_tardiness", 0),
        "breakdowns": env.metrics.get("breakdowns", 0),
    }


def compare_rules(
    env: "ShipyardEnv",
    rules: List[DispatchRule] = None,
    n_episodes: int = 5,
    max_steps: int = 10000,
) -> Dict[str, Dict[str, float]]:
    """Compare multiple dispatch rules.

    Args:
        env: The shipyard environment.
        rules: List of rules to compare (default: all).
        n_episodes: Number of episodes per rule.
        max_steps: Maximum steps per episode.

    Returns:
        Dictionary mapping rule names to average statistics.
    """
    if rules is None:
        rules = list(DispatchRule)

    results = {}

    for rule in rules:
        rule_results = []
        for _ in range(n_episodes):
            episode_result = run_episode_with_rule(env, rule, max_steps)
            rule_results.append(episode_result)

        # Average results
        avg_result = {}
        for key in rule_results[0]:
            if key == "rule":
                avg_result[key] = rule.value
            else:
                avg_result[key] = np.mean([r[key] for r in rule_results])
                avg_result[f"{key}_std"] = np.std([r[key] for r in rule_results])

        results[rule.value] = avg_result

    return results
