"""Tests for action masking utilities."""

import numpy as np
import torch
from shipyard_scheduling.agent.action_masking import (
    flatten_env_mask_to_policy_mask,
    head_relevance_mask,
    batch_masks,
    to_torch_mask,
    ACTION_TYPE_TO_HEADS,
)


def test_flatten_env_mask():
    env_mask = {
        "action_type": np.array([True, True, False, True]),
        "spmt_dispatch": np.array([[True, False], [False, True]]),
        "crane_dispatch": np.array([[True, True]]),
        "maintenance": np.array([True, False, True]),
    }
    result = flatten_env_mask_to_policy_mask(env_mask, n_spmts=2, n_cranes=1, max_requests=3)
    assert result["action_type"].shape == (4,)
    assert result["spmt"].shape == (2,)
    assert result["request"].shape == (3,)
    assert result["crane"].shape == (1,)
    assert result["lift"].shape == (3,)
    assert result["equipment"].shape == (3,)
    # SPMT 0 has one valid request, SPMT 1 has one valid request
    assert result["spmt"][0] and result["spmt"][1]


def test_head_relevance_mask():
    action_types = torch.tensor([0, 1, 2, 3])
    # SPMT head relevant for action_type 0 only
    mask = head_relevance_mask(action_types, "spmt")
    assert mask.tolist() == [True, False, False, False]
    # Crane head relevant for action_type 1 only
    mask = head_relevance_mask(action_types, "crane")
    assert mask.tolist() == [False, True, False, False]
    # Equipment relevant for action_type 2
    mask = head_relevance_mask(action_types, "equipment")
    assert mask.tolist() == [False, False, True, False]
    # action_type head always relevant
    mask = head_relevance_mask(action_types, "action_type")
    assert all(mask)


def test_head_relevance_mask_2d_input():
    # Stacking scalar tensors from sample() may yield (batch, 1)
    action_types = torch.tensor([[0], [1], [2], [3]])
    mask = head_relevance_mask(action_types, "spmt")
    assert mask.shape == (4,)
    assert mask[0] and not mask[1]


def test_batch_masks():
    masks = [
        {"action_type": torch.ones(4, dtype=torch.bool), "spmt": torch.zeros(3, dtype=torch.bool)},
        {"action_type": torch.ones(4, dtype=torch.bool), "spmt": torch.ones(3, dtype=torch.bool)},
    ]
    batched = batch_masks(masks)
    assert batched["action_type"].shape == (2, 4)
    assert batched["spmt"].shape == (2, 3)


def test_to_torch_mask():
    mask = {"a": np.array([True, False]), "b": np.array([False, True])}
    result = to_torch_mask(mask)
    assert result["a"].dtype == torch.bool
    assert result["a"].tolist() == [True, False]
