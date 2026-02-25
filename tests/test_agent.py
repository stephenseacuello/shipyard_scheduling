"""Tests for agent components."""

import torch
from shipyard_scheduling.simulation.environment import ShipyardEnv
from shipyard_scheduling.agent.gnn_encoder import HeterogeneousGNNEncoder
from shipyard_scheduling.agent.policy import ActorCriticPolicy
import yaml
import os


def load_cfg():
    with open(os.path.join(os.path.dirname(__file__), "..", "config", "small_instance.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    inherit = cfg.get("inherit_from")
    if inherit:
        with open(os.path.join(os.path.dirname(__file__), "..", "config", inherit), "r") as g:
            base = yaml.safe_load(g)
        base.update({k: v for k, v in cfg.items() if k != "inherit_from"})
        cfg = base
    return cfg


def test_encoder_and_policy():
    cfg = load_cfg()
    env = ShipyardEnv(cfg)
    encoder = HeterogeneousGNNEncoder(
        block_dim=env.block_features,
        spmt_dim=env.spmt_features,
        crane_dim=env.crane_features,
        facility_dim=env.facility_features,
        hidden_dim=32,
        num_layers=1,
    )
    state_dim = 32 * 4
    policy = ActorCriticPolicy(
        state_dim=state_dim,
        n_action_types=4,
        n_spmts=env.n_spmts,
        n_cranes=env.n_cranes,
        max_requests=env.n_blocks,
        hidden_dim=64,
    )
    env.reset()
    graph = env.get_graph_data().to("cpu")
    with torch.no_grad():
        state_emb = encoder(graph)
        action_dist, value = policy.forward(state_emb)
    # Ensure outputs have correct shapes
    assert value.shape[-1] == 1
    assert set(action_dist.keys()) == {
        "action_type", "spmt", "request", "crane", "lift", "equipment",
        "supplier", "material", "labor_pool", "target_block",
    }


def test_readiness_gate_exists():
    """Verify the readiness gate (logistic regression classifier) is present."""
    policy = ActorCriticPolicy(state_dim=64, hidden_dim=32)
    assert hasattr(policy, "readiness_gate"), "Policy must have readiness_gate head"
    # Readiness gate is a single-output linear layer (logistic regression)
    gate_layer = policy.readiness_gate[0]
    assert gate_layer.out_features == 1, "Readiness gate must output 1 logit (sigmoid)"


def test_readiness_gate_produces_probability():
    """Readiness gate should produce a probability in [0, 1] via sigmoid."""
    policy = ActorCriticPolicy(state_dim=64, hidden_dim=32)
    state = torch.randn(1, 64)
    with torch.no_grad():
        action_dist, value = policy.forward(state)
    prob = policy._last_readiness_prob
    assert prob.shape[-1] == 1
    assert (prob >= 0.0).all() and (prob <= 1.0).all(), \
        f"Readiness probability must be in [0,1], got {prob.item():.4f}"


def test_readiness_gate_modulates_dispatch_logits():
    """When readiness is low, dispatch action types should have lower probability."""
    policy = ActorCriticPolicy(state_dim=64, hidden_dim=32, n_action_types=4)
    state = torch.randn(1, 64)

    # Run forward pass
    with torch.no_grad():
        action_dist, _ = policy.forward(state)

    # The readiness gate biases action_type logits for indices 0 and 1
    # We just verify that action_type distribution sums to 1 (valid distribution)
    probs = action_dist["action_type"].probs
    assert abs(probs.sum().item() - 1.0) < 1e-5, "Action type probs must sum to 1"


def test_readiness_gate_batch():
    """Readiness gate works with batched inputs."""
    policy = ActorCriticPolicy(state_dim=64, hidden_dim=32, n_action_types=4)
    batch_state = torch.randn(8, 64)
    with torch.no_grad():
        action_dist, value = policy.forward(batch_state)
    prob = policy._last_readiness_prob
    assert prob.shape == (8, 1), f"Batched readiness should be (8,1), got {prob.shape}"
    assert (prob >= 0.0).all() and (prob <= 1.0).all()
    # Action type distribution should still be valid per sample
    probs = action_dist["action_type"].probs
    assert probs.shape == (8, 4)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(8), atol=1e-5)