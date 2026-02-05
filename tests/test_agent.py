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
    assert set(action_dist.keys()) == {"action_type", "spmt", "request", "crane", "lift", "equipment"}