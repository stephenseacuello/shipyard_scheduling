"""Monte Carlo Tree Search (MCTS) agent for shipyard scheduling.

MCTS naturally handles masked action spaces by only expanding valid children,
sidestepping the entropy collapse problem that kills PPO/SAC on this domain.

Usage::

    from src.agent.mcts import MCTSScheduler
    scheduler = MCTSScheduler(n_simulations=50, max_rollout_depth=30)
    action = scheduler.decide(env)
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List, Optional

import numpy as np

from baselines.rule_based import RuleBasedScheduler

_HOLD: Dict[str, int] = {
    "action_type": 3, "spmt_idx": 0, "request_idx": 0,
    "crane_idx": 0, "lift_idx": 0, "erection_idx": 0, "equipment_idx": 0,
}


def _hold_action() -> Dict[str, Any]:
    """Return a fresh copy of the default hold (no-op) action."""
    return dict(_HOLD)


class MCTSNode:
    """A node in the MCTS search tree with UCB1 selection."""

    __slots__ = ("parent", "children", "action", "visits",
                 "total_value", "untried_actions")

    def __init__(self, parent: Optional[MCTSNode] = None,
                 action: Optional[Dict[str, Any]] = None,
                 valid_actions: Optional[List[Dict[str, Any]]] = None) -> None:
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.action = action
        self.visits: int = 0
        self.total_value: float = 0.0
        self.untried_actions: List[Dict[str, Any]] = valid_actions or []

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @property
    def value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def ucb1(self, c: float = 1.414) -> float:
        """Upper Confidence Bound for Trees score."""
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        return (self.total_value / self.visits
                + c * math.sqrt(math.log(parent_visits) / self.visits))

    def best_child(self, c: float = 1.414) -> MCTSNode:
        """Select child with highest UCB1 score."""
        return max(self.children, key=lambda ch: ch.ucb1(c))

    def best_action_child(self) -> MCTSNode:
        """Select most-visited child (robust final decision)."""
        return max(self.children, key=lambda ch: ch.visits)

    def add_child(self, action: Dict[str, Any],
                  valid_actions: List[Dict[str, Any]]) -> MCTSNode:
        """Remove *action* from untried list and create a child node."""
        self.untried_actions = [a for a in self.untried_actions if a is not action]
        child = MCTSNode(parent=self, action=action, valid_actions=valid_actions)
        self.children.append(child)
        return child


class MCTSScheduler:
    """MCTS-based scheduler for the HHI shipyard environment.

    Parameters
    ----------
    n_simulations : int
        MCTS iterations (select-expand-rollout-backup) per ``decide()`` call.
    max_rollout_depth : int
        Maximum environment steps during the rollout phase.
    c_explore : float
        Exploration constant for UCB1.
    max_transport_actions : int
        Cap on SPMT dispatch actions enumerated from the mask.
    max_crane_actions : int
        Cap on crane dispatch actions enumerated from the mask.
    """

    _expert = RuleBasedScheduler()  # shared EDD expert for rollouts (~0.005ms)

    def __init__(self, n_simulations: int = 50, max_rollout_depth: int = 30,
                 c_explore: float = 1.414, max_transport_actions: int = 20,
                 max_crane_actions: int = 10) -> None:
        self.n_simulations = n_simulations
        self.max_rollout_depth = max_rollout_depth
        self.c_explore = c_explore
        self.max_transport_actions = max_transport_actions
        self.max_crane_actions = max_crane_actions

    # ------------------------------------------------------------------ #
    # Public API (matches RuleBasedScheduler.decide)
    # ------------------------------------------------------------------ #
    def decide(self, env: Any) -> Dict[str, Any]:
        """Return a single action dict for the current environment state.

        Runs *n_simulations* iterations of MCTS and returns the action of the
        most-visited root child.
        """
        valid_actions = self._get_valid_actions(env)
        if len(valid_actions) <= 1:
            return valid_actions[0] if valid_actions else _hold_action()

        root = MCTSNode(valid_actions=valid_actions)

        for _ in range(self.n_simulations):
            sim_env = copy.deepcopy(env)  # fresh copy per simulation

            # 1. Selection — traverse tree via UCB1
            node = root
            while node.is_fully_expanded and node.children:
                node = node.best_child(self.c_explore)
                sim_env.step(node.action)

            # 2. Expansion — add one untried child
            if node.untried_actions:
                action = random.choice(node.untried_actions)
                sim_env.step(action)
                child_valid = self._get_valid_actions(sim_env)
                node = node.add_child(action, child_valid)

            # 3. Rollout — EDD expert for fast, informed simulation
            rollout_value = self._rollout(sim_env)

            # 4. Back-propagation
            while node is not None:
                node.visits += 1
                node.total_value += rollout_value
                node = node.parent

        if not root.children:
            return _hold_action()
        return root.best_action_child().action  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Rollout
    # ------------------------------------------------------------------ #
    def _rollout(self, env: Any) -> float:
        """Run EDD expert for up to *max_rollout_depth* steps; return cumulative reward."""
        total_reward = 0.0
        for _ in range(self.max_rollout_depth):
            action = self._expert.decide(env)
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward

    # ------------------------------------------------------------------ #
    # Action enumeration from mask
    # ------------------------------------------------------------------ #
    def _get_valid_actions(self, env: Any) -> List[Dict[str, Any]]:
        """Enumerate valid actions from the environment's action mask.

        Large action sets are randomly sub-sampled to keep the branching
        factor tractable.
        """
        mask = env.get_action_mask()
        actions: List[Dict[str, Any]] = []

        # Action type 0: SPMT dispatch
        spmt_mask = mask.get("spmt_dispatch")
        if spmt_mask is not None and mask["action_type"][0]:
            transport: List[Dict[str, Any]] = []
            for s, r in zip(*np.where(spmt_mask)):
                transport.append({"action_type": 0, "spmt_idx": int(s),
                                  "request_idx": int(r), "crane_idx": 0,
                                  "lift_idx": 0, "erection_idx": 0,
                                  "equipment_idx": 0})
            if len(transport) > self.max_transport_actions:
                transport = random.sample(transport, self.max_transport_actions)
            actions.extend(transport)

        # Action type 1: Crane dispatch
        crane_mask = mask.get("crane_dispatch")
        if crane_mask is not None and mask["action_type"][1]:
            crane_acts: List[Dict[str, Any]] = []
            for c, r in zip(*np.where(crane_mask)):
                crane_acts.append({"action_type": 1, "crane_idx": int(c),
                                   "lift_idx": int(r), "erection_idx": int(r),
                                   "spmt_idx": 0, "request_idx": 0,
                                   "equipment_idx": 0})
            if len(crane_acts) > self.max_crane_actions:
                crane_acts = random.sample(crane_acts, self.max_crane_actions)
            actions.extend(crane_acts)

        # Action type 2: Maintenance
        maint_mask = mask.get("maintenance")
        if maint_mask is not None and mask["action_type"][2]:
            for e in range(len(maint_mask)):
                if maint_mask[e]:
                    actions.append({"action_type": 2, "equipment_idx": int(e),
                                    "spmt_idx": 0, "request_idx": 0,
                                    "crane_idx": 0, "lift_idx": 0})

        # Action type 3: Hold (always valid)
        actions.append(_hold_action())
        return actions
