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


class AttentionActionHead(nn.Module):
    """Attention-based action head for variable-size entity selection.

    Instead of a fixed-size linear layer, this head uses cross-attention
    between the state embedding (query) and per-entity embeddings (keys/values).
    This scales naturally to any number of entities without retraining.

    Used for request_head and lift_head where the number of valid targets
    varies across instances (10 to 1600 blocks).
    """

    def __init__(self, state_dim: int, entity_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.query_proj = nn.Linear(state_dim, entity_dim)
        self.key_proj = nn.Linear(entity_dim, entity_dim)
        self.scale = entity_dim ** 0.5
        self.n_heads = n_heads

    def forward(
        self, state: torch.Tensor, entity_embeddings: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute attention scores over entities.

        Args:
            state: (batch, state_dim) or (state_dim,) state embedding
            entity_embeddings: (batch, n_entities, entity_dim) or None.
                If None, falls back to learned linear projection (no attention).

        Returns:
            logits: (batch, n_entities) attention scores as action logits
        """
        if entity_embeddings is None:
            # Fallback: no entity embeddings available, use linear projection
            # This maintains backward compatibility with existing code
            return self._fallback(state)

        query = self.query_proj(state)  # (batch, entity_dim) or (entity_dim,)
        keys = self.key_proj(entity_embeddings)  # (batch, n, entity_dim)

        if query.dim() == 1:
            query = query.unsqueeze(0)  # (1, entity_dim)
        if keys.dim() == 2:
            keys = keys.unsqueeze(0)  # (1, n, entity_dim)

        # Scaled dot-product attention scores
        # query: (batch, 1, entity_dim) @ keys: (batch, entity_dim, n) → (batch, 1, n)
        logits = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
        return logits / self.scale

    def _fallback(self, state: torch.Tensor) -> torch.Tensor:
        """Linear fallback when entity embeddings are unavailable."""
        return self.query_proj(state)


def _apply_mask(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mask logits: set positions where mask is False to a large negative value.

    Uses -20.0 instead of -1e9 to allow some probability mass on masked actions,
    enabling exploration while still making them unlikely. This helps prevent
    entropy collapse where the policy becomes completely deterministic.
    """
    if mask is None:
        return logits
    if mask.dtype != torch.bool:
        mask = mask.bool()
    # Handle size mismatch between attention-head logits (variable n_entities)
    # and fixed-size masks from flatten_env_mask_to_policy_mask.
    logit_dim = logits.shape[-1]
    mask_dim = mask.shape[-1]
    if logit_dim != mask_dim:
        if logit_dim < mask_dim:
            # Attention head produced fewer logits than mask size; truncate mask
            mask = mask[..., :logit_dim]
        else:
            # Attention head produced more logits than mask size; pad mask with False
            pad_shape = list(mask.shape)
            pad_shape[-1] = logit_dim - mask_dim
            pad = torch.zeros(pad_shape, dtype=torch.bool, device=mask.device)
            mask = torch.cat([mask, pad], dim=-1)
    # Broadcast: mask may be (head_size,) or (batch, head_size)
    return logits.masked_fill(~mask, -20.0)


class ActorCriticPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_action_types: int = 4,
        n_spmts: int = 1,
        n_cranes: int = 1,
        max_requests: int = 1,
        hidden_dim: int = 256,
        epsilon: float = 0.1,  # Epsilon for epsilon-greedy exploration
        temperature: float = 1.0,  # Logit temperature (>1 = more exploration)
        n_suppliers: int = 0,
        n_inventory: int = 0,
        n_labor_pools: int = 0,
        entity_dim: int = 64,  # Embedding dim for attention-based heads
    ) -> None:
        super().__init__()
        self.n_action_types = n_action_types
        self.n_spmts = n_spmts
        self.n_cranes = n_cranes
        self.max_requests = max_requests
        self.epsilon = epsilon  # Probability of random action
        self.temperature = temperature  # Logit scaling for exploration
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
        # Attention-based heads for variable-size entity selection.
        # Used when entity_embeddings are provided (e.g., per-block GNN
        # embeddings), replacing the fixed-size linear request/lift/target heads.
        self.attn_request_head = AttentionActionHead(hidden_dim, entity_dim)
        self.attn_lift_head = AttentionActionHead(hidden_dim, entity_dim)
        self.attn_target_block_head = AttentionActionHead(hidden_dim, entity_dim)
        # Supply chain heads (created even if size=1 for uniform interface)
        self.supplier_head = nn.Linear(hidden_dim, max(n_suppliers, 1))
        self.material_head = nn.Linear(hidden_dim, max(n_inventory, 1))
        self.labor_pool_head = nn.Linear(hidden_dim, max(n_labor_pools, 1))
        self.target_block_head = nn.Linear(hidden_dim, max(max_requests, 1))
        # Readiness gate: binary logistic regression classifier.
        # Predicts P(block is ready to advance to next stage) as a single
        # sigmoid output.  Used to modulate dispatch probabilities — if the
        # gate predicts low readiness, transport/erection actions for that
        # block are down-weighted.
        self.readiness_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),  # single logit → sigmoid = logistic regression
        )
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        mask: Dict[str, torch.Tensor] | None = None,
        entity_embeddings: torch.Tensor | None = None,
    ) -> Tuple[Dict[str, Categorical], torch.Tensor]:
        features = self.shared(state)

        # Readiness gate: logistic regression P(ready to dispatch)
        # Positive readiness boosts dispatch actions (types 0, 1);
        # negative readiness suppresses them in favor of hold/maintenance.
        readiness_logit = self.readiness_gate(features)  # (batch, 1) or (1,)
        self._last_readiness_prob = torch.sigmoid(readiness_logit)

        # Compute logits for all heads.
        # When entity_embeddings are provided, use attention-based heads for
        # request/lift/target_block (variable-size selection over entities).
        # Otherwise fall back to fixed-size linear heads.
        if entity_embeddings is not None:
            request_logits = self.attn_request_head(features, entity_embeddings)
            lift_logits = self.attn_lift_head(features, entity_embeddings)
            target_block_logits = self.attn_target_block_head(
                features, entity_embeddings
            )
        else:
            request_logits = self.request_head(features)
            lift_logits = self.lift_head(features)
            target_block_logits = self.target_block_head(features)

        logits = {
            "action_type": self.action_type_head(features),
            "spmt": self.spmt_head(features),
            "request": request_logits,
            "crane": self.crane_head(features),
            "lift": lift_logits,
            "equipment": self.equipment_head(features),
            "supplier": self.supplier_head(features),
            "material": self.material_head(features),
            "labor_pool": self.labor_pool_head(features),
            "target_block": target_block_logits,
        }

        # Modulate dispatch action logits by readiness gate.
        # When readiness is low (sigmoid → 0), the logit bias is negative,
        # discouraging dispatch; when high (sigmoid → 1), bias is ~0.
        readiness_bias = readiness_logit.squeeze(-1)  # (batch,) or scalar
        at_logits = logits["action_type"]
        if at_logits.dim() == 1:
            # Single sample: at_logits is (n_action_types,)
            at_logits = at_logits.clone()
            at_logits[0] = at_logits[0] + readiness_bias  # SPMT dispatch
            if at_logits.shape[0] > 1:
                at_logits[1] = at_logits[1] + readiness_bias  # crane dispatch
        else:
            # Batch: at_logits is (batch, n_action_types)
            at_logits = at_logits.clone()
            at_logits[:, 0] = at_logits[:, 0] + readiness_bias
            if at_logits.shape[1] > 1:
                at_logits[:, 1] = at_logits[:, 1] + readiness_bias
        logits["action_type"] = at_logits

        # Apply per-head masks
        if mask is not None:
            for key in logits:
                m = mask.get(key)
                if m is not None:
                    logits[key] = _apply_mask(logits[key], m)
        # Apply temperature scaling to increase exploration (T>1 = more uniform)
        if self.temperature != 1.0:
            logits = {k: v / self.temperature for k, v in logits.items()}
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
        entity_embeddings: torch.Tensor | None = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        action_dist, value = self.forward(state, mask, entity_embeddings)
        if deterministic:
            action = {k: d.probs.argmax(dim=-1) for k, d in action_dist.items()}
        else:
            # Epsilon-greedy exploration: with probability epsilon, sample uniformly
            # from valid actions instead of from the learned distribution
            if self.epsilon > 0 and torch.rand(1).item() < self.epsilon:
                # Random action from valid options (respecting mask)
                action = {}
                for k, d in action_dist.items():
                    # Sample uniformly from non-zero probability actions
                    probs = d.probs
                    if probs.dim() == 1:
                        # Single sample case
                        valid_mask = probs > 1e-8
                        if valid_mask.any():
                            uniform_probs = valid_mask.float() / valid_mask.sum()
                            action[k] = torch.multinomial(uniform_probs, 1).squeeze(0)
                        else:
                            action[k] = d.sample()
                    else:
                        # Batch case - sample for each item in batch
                        batch_size = probs.shape[0]
                        samples = []
                        for i in range(batch_size):
                            p = probs[i]
                            valid_mask = p > 1e-8
                            if valid_mask.any():
                                uniform_probs = valid_mask.float() / valid_mask.sum()
                                samples.append(torch.multinomial(uniform_probs, 1).squeeze(0))
                            else:
                                samples.append(d.sample()[i])
                        action[k] = torch.stack(samples)
            else:
                action = {k: d.sample() for k, d in action_dist.items()}
        log_prob = self._compute_log_prob(action_dist, action)
        return action, log_prob, value

    def evaluate_action(
        self,
        state: torch.Tensor,
        action: Dict[str, torch.Tensor],
        mask: Dict[str, torch.Tensor] | None = None,
        entity_embeddings: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_dist, value = self.forward(state, mask, entity_embeddings)
        log_prob = self._compute_log_prob(action_dist, action)
        entropy = self._compute_entropy(action_dist, action["action_type"])
        return log_prob, entropy, value
