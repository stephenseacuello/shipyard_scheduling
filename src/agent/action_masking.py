"""Utility functions for action masking.

This module contains helpers to transform boolean masks produced by the
environment into PyTorch tensors compatible with the policy. It handles
hierarchical masking where only certain action heads are relevant depending
on the selected action type, and provides batching utilities for PPO updates.
"""

from __future__ import annotations

from typing import Dict, Any, List
import numpy as np
import torch


# Maps action_type index to which policy heads are relevant
# action_type 0 = dispatch SPMT: needs spmt + request heads
# action_type 1 = dispatch crane: needs crane + lift heads
# action_type 2 = maintenance: needs equipment head
# action_type 3 = hold: no sub-heads needed
# action_type 4 = place order: needs supplier + material heads
# action_type 5 = assign worker: needs labor_pool + target_block heads
# action_type 6 = transfer material: needs material + target_block heads
ACTION_TYPE_TO_HEADS: Dict[int, List[str]] = {
    0: ["action_type", "spmt", "request"],
    1: ["action_type", "crane", "lift"],
    2: ["action_type", "equipment"],
    3: ["action_type"],
    4: ["action_type", "supplier", "material"],
    5: ["action_type", "labor_pool", "target_block"],
    6: ["action_type", "material", "target_block"],
}

ALL_HEADS = [
    "action_type", "spmt", "request", "crane", "lift", "equipment",
    "supplier", "material", "labor_pool", "target_block",
]


def to_torch_mask(mask: Dict[str, Any], device: str | torch.device = "cpu") -> Dict[str, torch.Tensor]:
    """Convert a mask dictionary with boolean arrays into torch tensors."""
    return {k: torch.tensor(v, dtype=torch.bool, device=device) for k, v in mask.items()}


def flatten_env_mask_to_policy_mask(
    env_mask: Dict[str, Any],
    n_spmts: int,
    n_cranes: int,
    max_requests: int,
    n_suppliers: int = 0,
    n_inventory: int = 0,
    n_labor_pools: int = 0,
) -> Dict[str, np.ndarray]:
    """Convert environment mask format to per-head 1D boolean masks.

    The environment produces:
      - action_type: (n_action_types,) bool
      - spmt_dispatch: (n_spmts, n_transport_requests) bool
      - crane_dispatch: (n_cranes, n_lift_requests) bool
      - maintenance: (n_spmts + n_cranes,) bool
      - supplier_order: (n_suppliers, n_inventory) bool  [if enabled]
      - labor_assign: (n_labor_pools,) bool  [if enabled]
      - inventory_transfer: (n_inventory,) bool  [if enabled]

    This function produces masks matching policy head output sizes.
    """
    at_mask = env_mask["action_type"]

    spmt_dispatch = env_mask["spmt_dispatch"]  # (n_spmts, n_requests)
    crane_dispatch = env_mask["crane_dispatch"]  # (n_cranes, n_lifts)
    maint_mask = env_mask["maintenance"]  # (n_spmts + n_cranes,)

    # SPMT head: True if this SPMT can serve any request
    spmt_mask = np.zeros(n_spmts, dtype=bool)
    if spmt_dispatch.size > 0:
        spmt_mask[: spmt_dispatch.shape[0]] = spmt_dispatch.any(axis=1)

    # Request head: True if this request can be served by any SPMT
    request_mask = np.zeros(max_requests, dtype=bool)
    if spmt_dispatch.size > 0 and spmt_dispatch.shape[1] > 0:
        n_req = min(spmt_dispatch.shape[1], max_requests)
        request_mask[:n_req] = spmt_dispatch.any(axis=0)[:n_req]

    # Crane head: True if this crane can serve any lift request
    crane_mask = np.zeros(n_cranes, dtype=bool)
    if crane_dispatch.size > 0:
        crane_mask[: crane_dispatch.shape[0]] = crane_dispatch.any(axis=1)

    # Lift head: True if this lift request can be served by any crane
    lift_mask = np.zeros(max_requests, dtype=bool)
    if crane_dispatch.size > 0 and crane_dispatch.shape[1] > 0:
        n_lift = min(crane_dispatch.shape[1], max_requests)
        lift_mask[:n_lift] = crane_dispatch.any(axis=0)[:n_lift]

    # Equipment head: same as maintenance mask
    equipment_mask = np.array(maint_mask, dtype=bool)

    result = {
        "action_type": np.array(at_mask, dtype=bool),
        "spmt": spmt_mask,
        "request": request_mask,
        "crane": crane_mask,
        "lift": lift_mask,
        "equipment": equipment_mask,
    }

    # Supply chain masks
    n_sup = max(n_suppliers, 1)
    n_inv = max(n_inventory, 1)
    n_lp = max(n_labor_pools, 1)

    supplier_order = env_mask.get("supplier_order")
    if supplier_order is not None and supplier_order.size > 0:
        result["supplier"] = np.zeros(n_sup, dtype=bool)
        result["supplier"][:supplier_order.shape[0]] = supplier_order.any(axis=1)
        result["material"] = np.zeros(n_inv, dtype=bool)
        result["material"][:supplier_order.shape[1]] = supplier_order.any(axis=0)
    else:
        result["supplier"] = np.ones(n_sup, dtype=bool)
        result["material"] = np.ones(n_inv, dtype=bool)

    labor_assign = env_mask.get("labor_assign")
    if labor_assign is not None and labor_assign.size > 0:
        result["labor_pool"] = np.zeros(n_lp, dtype=bool)
        result["labor_pool"][:len(labor_assign)] = labor_assign
    else:
        result["labor_pool"] = np.ones(n_lp, dtype=bool)

    # target_block: allow all for now; action execution validates
    result["target_block"] = np.ones(max_requests, dtype=bool)

    inv_transfer = env_mask.get("inventory_transfer")
    if inv_transfer is not None and inv_transfer.size > 0:
        # For transfer: material head already set from supplier_order or this
        inv_mask = np.zeros(n_inv, dtype=bool)
        inv_mask[:len(inv_transfer)] = inv_transfer
        result["material"] = result["material"] | inv_mask

    return result


def head_relevance_mask(action_types: torch.Tensor, head_name: str) -> torch.Tensor:
    """Return a boolean mask of shape (batch,) indicating which samples need this head.

    Used to zero out log_probs for irrelevant heads during evaluate_action.
    """
    # Ensure 1D: stacking scalar tensors from sample() can yield (batch, 1)
    if action_types.dim() > 1:
        action_types = action_types.squeeze(-1)
    if head_name == "action_type":
        return torch.ones(action_types.shape[0], dtype=torch.bool, device=action_types.device)
    relevance = torch.zeros(action_types.shape[0], dtype=torch.bool, device=action_types.device)
    for at_idx, heads in ACTION_TYPE_TO_HEADS.items():
        if head_name in heads:
            relevance |= (action_types == at_idx)
    return relevance


def batch_masks(mask_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of per-step mask dicts into batched tensors.

    Each element in mask_list is a dict of 1D boolean tensors (per head).
    Returns a dict of 2D tensors with shape (batch_size, head_size).
    """
    if not mask_list:
        return {}
    keys = mask_list[0].keys()
    return {k: torch.stack([m[k] for m in mask_list]) for k in keys}
