"""Actor-critic policy with hierarchical action heads and masking.

This policy consumes a state embedding (from a GNN encoder) and outputs
distributions over action types and sub-action indices. Only the heads
relevant to the selected action type contribute to the log probability.
Invalid actions are masked by setting logits to -inf before sampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Dict, Any, Tuple

from .action_masking import head_relevance_mask, ALL_HEADS


def _apply_mask(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mask logits: set positions where mask is False to -1e9."""
    if mask is None:
        return logits
    if mask.dtype != torch.bool:
        mask = mask.bool()
    # Broadcast: mask may be (head_size,) or (batch, head_size)
    return logits.masked_fill(~mask, -1e9)


class ActorCriticPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_action_types: int = 4,
        n_spmts: int = 1,
        n_cranes: int = 1,
        max_requests: int = 1,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.n_action_types = n_action_types
        self.n_spmts = n_spmts
        self.n_cranes = n_cranes
        self.max_requests = max_requests
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Action heads
        self.action_type_head = nn.Linear(hidden_dim, n_action_types)
        self.spmt_head = nn.Linear(hidden_dim, n_spmts)
        self.request_head = nn.Linear(hidden_dim, max_requests)
        self.crane_head = nn.Linear(hidden_dim, n_cranes)
        self.lift_head = nn.Linear(hidden_dim, max_requests)
        self.equipment_head = nn.Linear(hidden_dim, n_spmts + n_cranes)
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, state: torch.Tensor, mask: Dict[str, torch.Tensor] | None = None
    ) -> Tuple[Dict[str, Categorical], torch.Tensor]:
        features = self.shared(state)
        # Compute logits for all heads
        logits = {
            "action_type": self.action_type_head(features),
            "spmt": self.spmt_head(features),
            "request": self.request_head(features),
            "crane": self.crane_head(features),
            "lift": self.lift_head(features),
            "equipment": self.equipment_head(features),
        }
        # Apply per-head masks
        if mask is not None:
            for key in logits:
                m = mask.get(key)
                if m is not None:
                    logits[key] = _apply_mask(logits[key], m)
        # Build distributions
        action_dist = {k: Categorical(logits=v) for k, v in logits.items()}
        value = self.critic(features)
        return action_dist, value

    def _compute_log_prob(
        self,
        action_dist: Dict[str, Categorical],
        action: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute log probability summing only heads relevant to the action type.

        For each sample in the batch, only the heads that matter for that
        sample's action_type contribute. Irrelevant heads are zeroed out.
        """
        at = action["action_type"]  # (batch,) or scalar
        lp = action_dist["action_type"].log_prob(at)
        for head_name in ALL_HEADS:
            if head_name == "action_type":
                continue
            head_lp = action_dist[head_name].log_prob(action[head_name])
            relevance = head_relevance_mask(at, head_name).float()
            lp = lp + head_lp * relevance
        return lp

    def _compute_entropy(
        self,
        action_dist: Dict[str, Categorical],
        action_types: torch.Tensor,
    ) -> torch.Tensor:
        """Compute entropy summing only relevant heads."""
        ent = action_dist["action_type"].entropy()
        for head_name in ALL_HEADS:
            if head_name == "action_type":
                continue
            head_ent = action_dist[head_name].entropy()
            relevance = head_relevance_mask(action_types, head_name).float()
            ent = ent + head_ent * relevance
        return ent

    def get_action(
        self,
        state: torch.Tensor,
        mask: Dict[str, Any] | None = None,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        action_dist, value = self.forward(state, mask)
        if deterministic:
            action = {k: d.probs.argmax(dim=-1) for k, d in action_dist.items()}
        else:
            action = {k: d.sample() for k, d in action_dist.items()}
        log_prob = self._compute_log_prob(action_dist, action)
        return action, log_prob, value

    def evaluate_action(
        self,
        state: torch.Tensor,
        action: Dict[str, torch.Tensor],
        mask: Dict[str, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_dist, value = self.forward(state, mask)
        log_prob = self._compute_log_prob(action_dist, action)
        entropy = self._compute_entropy(action_dist, action["action_type"])
        return log_prob, entropy, value
